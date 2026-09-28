import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_ci import __version__
from research_ci.cli import main
from research_ci.cli.main import app

CONTRACT = (
    "from research_ci.api.contract import Contract\n\n"
    "CONTRACT = Contract(template='factor', claim='toxicity predicts markouts', "
    "explore_end='2025-06-30T00:00:00Z', seal_end='2025-12-31T00:00:00Z')\n"
)
runner = CliRunner()


def _derived(tmp_path: Path, name: str = "tox") -> Path:
    idea = tmp_path / name
    assert runner.invoke(app, ["init", str(idea), "-t", "factor"]).exit_code == 0
    (idea / "contract.py").write_text(CONTRACT)
    assert runner.invoke(app, ["contract", "derive", "-i", str(idea)]).exit_code == 0
    return idea


@pytest.fixture
def at_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_interactive", lambda: True)


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
    assert "toxicity predicts markouts" in result.output
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
