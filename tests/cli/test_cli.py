import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_ci import __version__
from research_ci.cli import common
from research_ci.cli.main import app
from research_ci.render.common import card_id
from research_ci.store.ledger import Ledger
from research_ci.workflow.idea import Idea
from toy import carry_contract, lab

CONTRACT = carry_contract(budget=20)
runner = CliRunner()


def _derived(tmp_path: Path, name: str = "tox") -> Path:
    idea = tmp_path / name
    assert runner.invoke(app, ["init", str(idea)]).exit_code == 0
    (idea / "contract.py").write_text(CONTRACT)
    assert runner.invoke(app, ["contract", "derive", "-i", str(idea)]).exit_code == 0
    lab(idea)
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
    created = runner.invoke(app, ["init", str(idea)])
    assert created.exit_code == 0 and created.stdout.startswith(f"created {idea}; read {idea / 'START.md'} first")
    actions = json.loads(runner.invoke(app, ["next", "-i", str(idea), "--json"]).stdout)
    assert [(a["who"], a["command"]) for a in actions] == [("agent", "rci contract derive")]
    status = runner.invoke(app, ["status", "-i", str(idea)]).stdout
    assert status.splitlines()[0] == "tox" and "contract  OPEN" in status


def test_init_takes_no_template(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path / "x"), "--template", "factor"])
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_the_card_id_derive_prints_is_the_one_approval_checks(tmp_path: Path) -> None:
    idea = tmp_path / "tox"
    assert runner.invoke(app, ["init", str(idea)]).exit_code == 0
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
    assert "contract v1 PASSED: freeze" in result.output
    assert "git add" in result.output
    actions = json.loads(runner.invoke(app, ["next", "-i", str(idea), "--json"]).stdout)
    assert [(a["who"], a["stage"]) for a in actions] == [("agent", "data")]


@pytest.mark.usefixtures("at_terminal")
def test_declining_leaves_the_stage_unfrozen(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    result = runner.invoke(app, ["approve", "--stage", "contract", "-i", str(idea)], input="n\n")
    assert result.exit_code == 1
    assert "contract  REVIEW" in runner.invoke(app, ["status", "-i", str(idea)]).stdout


@pytest.mark.usefixtures("at_terminal")
def test_out_of_order_approve_is_rejected(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    result = runner.invoke(app, ["approve", "--stage", "feat", "-i", str(idea)], input="y\n")
    assert result.exit_code == 1
    assert "waiting for contract, data" in result.output


def test_an_old_format_ledger_is_refused_with_a_way_out(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    old = Ledger(tmp_path / "old" / ".rci" / "ledger.jsonl")
    old.append("init", "idea", {})
    for command in (["verify"], ["status"], ["next"]):
        result = runner.invoke(app, [*command, "-i", str(tmp_path / "old")])
        assert result.exit_code == 1
        assert "older Research CI" in result.output and "rci init" in result.output
    assert runner.invoke(app, ["verify", "-i", str(idea)]).output.startswith("ledger ok")


def test_a_tampered_ledger_is_refused_everywhere(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    ledger = idea / ".rci" / "ledger.jsonl"
    ledger.write_text(ledger.read_text().replace('"stage":"idea"', '"stage":"ideas"'))
    for command in (["verify"], ["status"], ["next"]):
        result = runner.invoke(app, [*command, "-i", str(idea)])
        assert result.exit_code == 1
        assert "line 1" in result.output


def test_inbox_lists_only_what_waits_for_a_person(tmp_path: Path) -> None:
    _derived(tmp_path, "ready")
    assert runner.invoke(app, ["init", str(tmp_path / "fresh")]).exit_code == 0
    items = json.loads(runner.invoke(app, ["inbox", str(tmp_path), "--json"]).stdout)
    assert [(Path(i["idea"]).name, i["command"]) for i in items] == [("ready", "rci approve --stage contract")]
    assert "review the ContractCard and approve it" in items[0]["reason"]


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
    checked = runner.invoke(app, ["data", "check", "-i", str(idea)])
    assert checked.exit_code == 0, checked.output
    assert "DataCard · tox · v1" in checked.output and "not yet admitted" in checked.output
    assert "READY" in checked.output and "noise from one symbol: Z" in checked.output
    assert "up for review" in checked.output
    assert "DataCard" in runner.invoke(app, ["data", "show", "-i", str(idea)]).output
    data_card = _card(idea, "data")
    assert f"card {data_card}" in checked.output
    inbox = json.loads(runner.invoke(app, ["inbox", str(tmp_path), "--json"]).stdout)
    assert [i["command"] for i in inbox] == ["rci approve --stage data"]
    approved = runner.invoke(
        app, ["approve", "--stage", "data", "-i", str(idea), "--relayed", "freeze", "--card", data_card]
    )
    assert approved.exit_code == 0 and "DataCard" in approved.output and "data v1 PASSED" in approved.output
    assert "data      PASSED" in runner.invoke(app, ["status", "-i", str(idea)]).output
    for gone in (["source", "approve"], ["data", "submit"], ["feat", "register"]):
        assert runner.invoke(app, [*gone, "-i", str(idea)]).exit_code == 2


def test_feat_show_says_when_nothing_was_submitted(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    result = runner.invoke(app, ["feat", "show", "-i", str(idea)])
    assert result.exit_code == 1 and "rci feat submit" in result.output


def test_init_writes_a_start_note_and_an_example_the_engine_refuses_to_run(tmp_path: Path) -> None:
    idea = tmp_path / "tox"
    printed = runner.invoke(app, ["init", str(idea)]).stdout
    assert f"read {idea / 'START.md'} first" in printed and f"{idea / 'research' / 'example.py'}" in printed
    start = (idea / "START.md").read_text()
    for needed in (
        f"rci next -i {idea}",
        "rci plugins",
        "rci methods",
        "research/example.py",
        "rci feat explore <file>",
        "gets polars DataFrames",
        "wide.lazy() ... .collect()",
        "`age_<source>`",
        "pandas is fine for dates",
        "time and peak memory",
        '--relayed "<their words>" --card <id>',
        "list at least one lab eval in `EVALS` in `data.py`",
        "list at least one lab eval in `EVALS` in `feat.py`",
        "every layer with `EVALS` needs at least one, and each can veto",
        "`evals/<name>.py` (copy `evals/example.py`)",
        'EVALS = [Eval(<eval>, h0="<the H0>")]',
        "must reach z ≥ 2 at every",
        "it never passes what the decisive test failed",
        "`block` (any row it flags, after the",
        "null world in `nulls/feat.py`",
        "CASES = [Case(",
        "rci eval check evals/<name>.py",
    ):
        assert needed in start, needed
    assert len(start.splitlines()) <= 40
    example = (idea / "research" / "example.py").read_text()
    assert example.startswith("# rci:example") and "TODO" in example and 'on=["t", "inst"]' in example
    assert "wide: pl.DataFrame" in example and "wide.lazy()" in example and ".collect()" in example
    evals = (idea / "evals" / "example.py").read_text()
    assert evals.startswith("# rci:example") and "CASES = [" in evals and "h0=" in evals and "TODO" in evals
    assert f"lab evals start from {idea / 'evals' / 'example.py'}" in printed


def test_plugins_lists_every_installed_plugin_with_options_and_declarations() -> None:
    text = runner.invoke(app, ["plugins"]).stdout
    for needed in (
        "data: Contract(data=Use(name, {options}))",
        "  toy.data(hours: int = 600, seed: int = 0)",
        "    sources: candles (close) · funding (funding_rate, funding_interval_hours)",
        "    levels: close · factor 5 · window 24 · mask 24h",
        "universe: Universe(name, clusters=<column>, groups=(<column>, ...))",
        "  ref.carry(price: str = 'close', flow: str | None = None)",
        "    per: entry · rolls: yes",
        "    from an entry point of research-ci",
    ):
        assert needed in text, needed
    listed = json.loads(runner.invoke(app, ["plugins", "--json"]).stdout)
    assert {p["name"]: p["kind"] for p in listed}["toy.carry"] == "consumer"


def test_methods_lists_what_research_code_can_call_and_what_backs_it() -> None:
    text = runner.invoke(app, ["methods"]).stdout
    assert text.startswith(
        "feat eval: feat.py defines EVALS = [Eval(fn, h0='<the H0 it rejects>', label='y', horizons=("
    )
    assert (
        "  from research_ci.eval.statistics.ic import rank_ic\n    rank_ic(table: pl.DataFrame, label: str = 'y')"
        in text
    )
    assert "backed by 3 worked examples · 3 properties" in text and "backed by 1 worked example · 0 properties" in text
    assert "data eval: data.py defines EVALS = [Eval(fn, h0='<the H0 it rejects>', on_fail=..., covers=" in text
    assert "pit: a data eval's module: PITS = [Pit(category, plant)] next to its CASES" in text
    for needed in (
        "lab eval: a function of the idea's own, in a module such as `evals/<name>.py`",
        "every layer with EVALS needs at least one, its h0 is required",
        "it and each null world must reach z ≥ 2.00 at every horizon it covers",
        "  from research_ci.api import Eval, Case, Size, Power, Recovers, Market, Source, Pit",
        "    Case(name, run, expect, tol=1e-09, given=None)",
        "    Eval(of, name='', h0='', on_fail=None, covers=(), label=None, horizons=None, combine=None, note='')",
        '  for example: CASES = [Case("the top two of six less the bottom two", thirds, 0.4, given=SIX)]',
        "null world: nulls/feat.py: NULLS = [Null(explains, world, evals=(...))]; every alternative of the contract",
        "  from research_ci.api import Null, PermuteWithin, ShiftInTime\n"
        "    Null(explains: str, world: PermuteWithin | ShiftInTime, keeps: ",
        "      explains  the id of the contract's alternative this world answers",
        "    PermuteWithin(bucket_by: str, clusters: str | None = None, block: str | None = None)",
        '  for example: NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", block="horizon"), '
        'evals=("<eval>",))]',
    ):
        assert needed in text, needed
    listed = json.loads(runner.invoke(app, ["methods", "--json"]).stdout)
    assert {"use", "import", "signature", "doc", "examples", "properties"} <= set(listed[0])
    assert [m["name"] for m in listed if m["use"] == "null world"] == ["Null", "PermuteWithin", "ShiftInTime"]
    assert [m["name"] for m in listed if m["use"] == "lab eval"][:2] == ["Eval", "Case"]


def test_an_idea_named_from_inside_itself_says_to_drop_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idea = tmp_path / "tox"
    runner.invoke(app, ["init", str(idea)])
    monkeypatch.chdir(idea)
    inside = runner.invoke(app, ["next", "-i", "tox"])
    assert inside.exit_code == 1 and "you are already in the idea tox; drop -i, or run from its parent" in inside.output
    monkeypatch.chdir(tmp_path)
    below = runner.invoke(app, ["status", "-i", "tox/research"])
    assert below.exit_code == 1 and "tox is the idea; pass -i tox" in below.output
    elsewhere = runner.invoke(app, ["status", "-i", "nothing"])
    assert "is not a Research CI idea; run `rci init` first" in elsewhere.output


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (
            ("data='toy.data'", "data='toy.dta'"),
            ["no plugin named 'toy.dta'", "did you mean 'toy.data'?", "rci plugins"],
        ),
        (
            ("{'flow': 'funding_rate'}", "{'flw': 'funding_rate'}"),
            [
                "plugin 'ref.carry' rejects its options",
                "it takes ref.carry(price: str = 'close', flow: str | None = None)",
            ],
        ),
        (
            ("'top': 0.34", "'top': 3.4"),
            ["plugin 'ref.mean' rejects its options", "top is a share", "it takes ref.mean(cluster: str = 'inst'"],
        ),
    ],
)
def test_a_wrong_plugin_name_or_option_says_how_to_fix_it(
    tmp_path: Path, change: tuple[str, str], said: list[str]
) -> None:
    idea = tmp_path / "tox"
    runner.invoke(app, ["init", str(idea)])
    (idea / "contract.py").write_text(CONTRACT.replace(*change))
    result = runner.invoke(app, ["contract", "derive", "-i", str(idea)])
    assert result.exit_code == 1 and "Traceback" not in result.output
    for needed in said:
        assert needed in result.output, (needed, result.output)


def test_next_names_the_file_to_write_and_the_helpers_at_every_step(tmp_path: Path) -> None:
    idea = tmp_path / "tox"
    runner.invoke(app, ["init", str(idea)])

    def reason() -> str:
        return json.loads(runner.invoke(app, ["next", "-i", str(idea), "--json"]).stdout)[0]["reason"]

    assert all(x in reason() for x in ("contract.py", "START.md", "`rci plugins`"))
    (idea / "contract.py").write_text(CONTRACT)
    assert runner.invoke(app, ["contract", "derive", "-i", str(idea)]).exit_code == 0
    assert "`rci contract show` prints it with the card id that a relayed approval quotes" in reason()
    shown = runner.invoke(app, ["contract", "show", "-i", str(idea)]).stdout
    assert shown.startswith("ContractCard") and shown.rstrip().endswith(f"card {_card(idea, 'contract')}")
    assert runner.invoke(app, ["approve", "--stage", "contract", "-i", str(idea), "--auto"]).exit_code == 0
    assert all(x in reason() for x in ("data.py", "`rci methods`", "lab eval", "h0=", "evals/example.py"))


SPREAD = """
import polars as pl


def features(wide, candidates):
    return candidates.select("t", "inst", pl.col("spread_h").alias("score"))
"""

ROWS = """
import polars as pl


def features(wide, candidates):
    rows = [(t, e, s) for t, e, s in candidates.select("t", "inst", "spread_h").iter_rows()]
    return pl.DataFrame(rows, schema=["t", "inst", "score"], orient="row")
"""


def test_feat_from_the_command_line_prints_the_card_id_approval_checks(tmp_path: Path) -> None:
    idea = _derived(tmp_path)
    for stage in ("contract", "data"):
        if stage == "data":
            assert runner.invoke(app, ["data", "check", "-i", str(idea)]).exit_code == 0
        relay = ["--relayed", "ok", "--card", _card(idea, stage)]
        assert runner.invoke(app, ["approve", "--stage", stage, "-i", str(idea), *relay]).exit_code == 0
    reason = json.loads(runner.invoke(app, ["next", "-i", str(idea), "--json"]).stdout)[0]["reason"]
    assert all(
        x in reason for x in ("research/example.py", "`rci feat explore <file>`", "row loops run with a warning")
    )

    example = runner.invoke(app, ["feat", "submit", str(idea / "research" / "example.py"), "-i", str(idea)])
    assert example.exit_code == 1 and "is still the example; write your own" in example.output
    (idea / "research" / "rows.py").write_text(ROWS)
    rows = runner.invoke(app, ["feat", "explore", str(idea / "research" / "rows.py"), "-i", str(idea)])
    warned = "warnings   [!] row-wise code: research/rows.py:6 iter_rows — polars expressions are usually much faster"
    assert rows.exit_code == 0 and rows.output.startswith("ExploreCard") and warned in rows.output, rows.output
    assert "\ncost       full call " in rows.output and ", peak memory " in rows.output
    assert Idea(idea).status().stage("feat").trials == 0

    (idea / "research" / "spread.py").write_text(SPREAD)
    submitted = runner.invoke(app, ["feat", "submit", str(idea / "research" / "spread.py"), "-i", str(idea)])
    assert submitted.exit_code == 0, submitted.output
    card = _card(idea, "feat")
    assert "verdict    PASSED" in submitted.output and f"\ncard {card}\nup for review" in submitted.output
    shown = runner.invoke(app, ["feat", "show", "-i", str(idea)]).stdout
    assert shown.rstrip().endswith(f"card {card}\nup for review; a person approves it with `rci approve --stage feat`")
    reason = json.loads(runner.invoke(app, ["next", "-i", str(idea), "--json"]).stdout)[0]["reason"]
    assert "`rci feat show` prints it with the card id that a relayed approval quotes" in reason
    relay = ["--relayed", "ok", "--card", card]
    assert runner.invoke(app, ["approve", "--stage", "feat", "-i", str(idea), *relay]).exit_code == 0
