import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_ci import __version__
from research_ci.cli import common
from research_ci.cli.main import app
from research_ci.render.common import card_id
from research_ci.workflow.idea import Idea
from toy import label_contract

CONTRACT = label_contract(budget=20)
runner = CliRunner()


def _derived(tmp_path: Path, name: str = "tox") -> Path:
    idea = tmp_path / name
    assert runner.invoke(app, ["init", str(idea), "-t", "factor"]).exit_code == 0
    (idea / "contract.py").write_text(CONTRACT)
    assert runner.invoke(app, ["contract", "derive", "-i", str(idea)]).exit_code == 0
    return idea


def _card(idea: Path, stage: str) -> str:
    return card_id(Idea(idea).card(stage))


@pytest.fixture
def at_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "interactive", lambda: True)


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_init_then_status_and_next(tmp_path: Path) -> None:
    idea = tmp_path / "tox"
    assert runner.invoke(app, ["init", str(idea), "--template", "factor"]).exit_code == 0
    actions = json.loads(runner.invoke(app, ["next", "-i", str(idea), "--json"]).stdout)
    assert [(a["who"], a["command"]) for a in actions] == [("agent", "rci contract derive")]
    status = runner.invoke(app, ["status", "-i", str(idea)]).stdout
    assert "contract  OPEN" in status


def test_unknown_template_is_an_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path / "x"), "-t", "momentum"])
    assert result.exit_code == 1
    assert "unknown template" in result.output


def test_the_card_id_derive_prints_is_the_one_approval_checks(tmp_path: Path) -> None:
    idea = tmp_path / "tox"
    assert runner.invoke(app, ["init", str(idea), "-t", "factor"]).exit_code == 0
    (idea / "contract.py").write_text(CONTRACT)
    printed = runner.invoke(app, ["contract", "derive", "-i", str(idea)]).stdout
    assert f"card {_card(idea, 'contract')}" in printed


def test_approve_refuses_without_a_terminal(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    result = runner.invoke(app, ["approve", "--stage", "contract", "-i", str(idea)], input="y\n")
    assert result.exit_code == 1
    assert "interactive terminal" in result.output


@pytest.mark.usefixtures("at_terminal")
def test_a_person_approves_the_contract(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    result = runner.invoke(app, ["approve", "--stage", "contract", "-i", str(idea)], input="y\n")
    assert result.exit_code == 0, result.output
    assert "cross-venue funding spreads persist" in result.output
    assert "ContractCard" in result.output
    assert "frozen contract v1" in result.output
    assert "git add" in result.output
    actions = json.loads(runner.invoke(app, ["next", "-i", str(idea), "--json"]).stdout)
    assert [(a["who"], a["stage"]) for a in actions] == [("agent", "data")]


@pytest.mark.usefixtures("at_terminal")
def test_declining_leaves_the_stage_unfrozen(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    result = runner.invoke(app, ["approve", "--stage", "contract", "-i", str(idea)], input="n\n")
    assert result.exit_code == 1
    assert "contract  REGISTERED" in runner.invoke(app, ["status", "-i", str(idea)]).stdout


@pytest.mark.usefixtures("at_terminal")
def test_out_of_order_approve_is_rejected(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    result = runner.invoke(app, ["approve", "--stage", "feat", "-i", str(idea)], input="y\n")
    assert result.exit_code == 1
    assert "waiting for contract, data" in result.output


def test_a_tampered_ledger_is_refused_everywhere(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    ledger = idea / ".rci" / "ledger.jsonl"
    ledger.write_text(ledger.read_text().replace('"template":"factor"', '"template":"rule"'))
    for command in (["verify"], ["status"], ["next"]):
        result = runner.invoke(app, [*command, "-i", str(idea)])
        assert result.exit_code == 1
        assert "line 1" in result.output


def test_inbox_lists_only_what_waits_for_a_person(tmp_path: Path) -> None:
    _derived(tmp_path, "ready")
    assert runner.invoke(app, ["init", str(tmp_path / "fresh"), "-t", "rule"]).exit_code == 0
    items = json.loads(runner.invoke(app, ["inbox", str(tmp_path), "--json"]).stdout)
    assert [(Path(i["idea"]).name, i["command"]) for i in items] == [("ready", "rci approve --stage contract")]


def test_an_agent_can_relay_a_persons_approval_and_the_ledger_says_so(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    card = _card(idea, "contract")
    args = ["approve", "--stage", "contract", "-i", str(idea), "--relayed", "ok, freeze it", "--card", card]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "relayed by an agent" in result.output
    status = json.loads(runner.invoke(app, ["status", "-i", str(idea), "--json"]).stdout)
    assert "ok, freeze it" in status["stages"][0]["note"]
    record = [json.loads(line) for line in (idea / ".rci" / "ledger.jsonl").read_text().splitlines()][-1]
    assert record["payload"]["approval"] == {
        "by": "human",
        "channel": "relayed",
        "words": "ok, freeze it",
        "card": card,
    }


def test_a_relayed_approval_must_quote_the_person_and_budgets_still_need_a_terminal(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    assert runner.invoke(app, ["approve", "--stage", "contract", "-i", str(idea), "--relayed", " "]).exit_code == 1
    assert runner.invoke(app, ["approve", "--stage", "data", "-i", str(idea), "--auto", "--extend", "3"]).exit_code == 1
    auto = runner.invoke(app, ["approve", "--stage", "contract", "-i", str(idea), "--auto"])
    assert auto.exit_code == 0 and "approved by the agent" in auto.output


def test_the_data_layer_from_the_command_line(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    unshown = runner.invoke(app, ["approve", "--stage", "contract", "-i", str(idea), "--relayed", "ok"])
    assert unshown.exit_code == 1 and "must quote the card" in unshown.output
    card = ["--card", _card(idea, "contract")]
    assert (
        runner.invoke(app, ["approve", "--stage", "contract", "-i", str(idea), "--relayed", "ok", *card]).exit_code == 0
    )
    first = runner.invoke(app, ["data", "check", "-i", str(idea)])
    assert first.exit_code == 0, first.output
    assert "SourceCard · toy.data" in first.output and "DataCard · tox · v1" in first.output
    assert "blocked" in first.output
    assert f"card {_card(idea, 'source')}" in first.output
    admitted = runner.invoke(
        app, ["source", "approve", "-i", str(idea), "--relayed", "admit it", "--card", _card(idea, "source")]
    )
    assert admitted.exit_code == 0, admitted.output
    assert "relayed by an agent: 'admit it'" in admitted.output
    second = runner.invoke(app, ["data", "check", "-i", str(idea)])
    assert "READY" in second.output and "SourceCard" not in second.output
    assert "DataCard" in runner.invoke(app, ["data", "show", "-i", str(idea)]).output
    data_card = _card(idea, "data")
    assert f"card {data_card}" in second.output
    frozen = runner.invoke(
        app, ["approve", "--stage", "data", "-i", str(idea), "--relayed", "freeze", "--card", data_card]
    )
    assert frozen.exit_code == 0 and "DataCard" in frozen.output
    assert runner.invoke(app, ["data", "submit", "-i", str(idea)]).output.strip() == "data PASSED"
    assert "data      PASSED" in runner.invoke(app, ["status", "-i", str(idea)]).output
