"""The feat layer on the toy: the decisive effect alone decides. A real spread passes and goes up for review, noise
fails, a peek is tainted; diagnostics and alternatives without a world are shown, never deciding; the data check
calibrates the gate, and a decisive target that cannot see a planted signal blocks the data, or, when anything the
calibration rests on changed since, refuses the submission without a trial."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from research_ci.render.common import card_id
from research_ci.runner import plugins
from research_ci.runner.meter import HIGH_WATER, SAMPLED, Cost, measured
from research_ci.runner.research import ResearchError
from research_ci.stages import feat as fs
from research_ci.stages.contract import Settings
from research_ci.stages.feat import FeatError, FeatSettings
from research_ci.workflow.base import Approval
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import State, WorkflowError
from toy import REGISTRY, Backwards, NeverRun, label_contract, strategy_contract

FAST = Settings(workers=2, universe_cuts=2, label_cuts=1)
QUICK = FeatSettings(workers=2, calibration_reps=40, calibration_nulls=20, floor_reps=60, cuts=2)

CRITERIA = """
import polars as pl

from research_ci.api import Criterion
from research_ci.eval.statistics.ic import rank_ic


def backwards(table):
    return rank_ic(table.with_columns(-pl.col("score")))


CRITERIA = [Criterion("ic of the reversed score", backwards)]
"""

NULLS = """
from research_ci.api import Null, PermuteWithin

NULLS = [
    Null("tail", PermuteWithin("time", clusters="symbol", block="horizon"),
         criteria=("rank ic without top symbol",), reps=60),
]
"""

SPREAD = """
import polars as pl


def features(fields, candidates):
    return candidates.select("t", "entity", pl.col("spread_h").alias("score"))
"""

NOISE = """
import polars as pl


def features(fields, candidates):
    return candidates.select("t", "entity", (pl.struct("t", "entity").hash(7) / 2.0**64).alias("score"))
"""

PEEK = """
import polars as pl


def features(fields, candidates):
    rate = fields.filter(pl.col("field") == "funding_rate").group_by("entity").agg(pl.col("value").mean())
    short = rate.rename({"entity": "short", "value": "s"})
    long = rate.rename({"entity": "long", "value": "l"})
    rows = candidates.join(short, on="short").join(long, on="long")
    return rows.select("t", "entity", (pl.col("s") - pl.col("l")).alias("score"))
"""


def _idea(tmp_path: Path, contract: str | None = None, nulls: str = NULLS, criteria: str | None = CRITERIA) -> Idea:
    idea = Idea.init(tmp_path / "idea", settings=FAST, feat_settings=QUICK)
    (idea.root / "contract.py").write_text(contract or label_contract())
    for name, text in (("strategy.toml", "[s]\nk = 2\n"), ("live.toml", "[s]\nk = 2\n"), ("strategy.py", "K = 2\n")):
        (idea.root / name).write_text(text)
    idea.derive_contract()
    idea.approve("contract")
    idea.check_data()
    idea.approve("data", approval=_relayed(idea, "data"))
    assert idea.status().stage("data").state is State.PASSED
    if criteria is not None:
        (idea.root / "feat.py").write_text(criteria)
    (idea.root / "nulls").mkdir()
    (idea.root / "nulls" / "feat.py").write_text(nulls)
    return idea


def _relayed(idea: Idea, stage: str) -> Approval:
    return Approval(by="human", channel="relayed", words="ok", card=card_id(idea.card(stage)))


def _file(idea: Idea, name: str, text: str) -> Path:
    path = idea.root / "research" / f"{name}.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    return path


def _calibrations(idea: Idea) -> list[dict]:
    return [r["payload"] for r in idea.ledger.records() if r["kind"] == "calibrate"]


def _checked(idea: Idea) -> dict:
    return [r["payload"]["spec"] for r in idea.ledger.records() if r["kind"] == "check"][-1]


def test_the_decisive_effect_alone_decides_a_spread_passes_noise_fails_and_a_peek_is_tainted(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    explored, _ = idea.explore_feat(_file(idea, "peek", PEEK))
    assert [f["status"] for f in explored["floors"]] == ["pass", "fail"]
    assert idea.status().stage("feat").trials == 0 and _calibrations(idea) == []

    result, text, state = idea.submit_feat(_file(idea, "noise", NOISE))
    assert state is State.OPEN and result["verdict"]["status"] == "FAILED", text
    assert result["decisive"][0]["z"] < result["need"]
    assert "ResultCard" in text and "submission 1 of 3" in text
    calibration = _checked(idea)["calibration"]
    assert [(r["effect"], r["world"], r["ok"]) for r in calibration["rows"]] == [("carry", "floor", True)]
    assert "calibrate  ok   carry at 8h vs floor: stand-ins die" in idea.card("data")
    assert result["calibration"]["by"] == "check" and not result["calibration"]["ran"] and "calibrate" not in text
    assert _calibrations(idea) == []
    assert idea.next_actions()[0].reason.startswith("submission 1 FAILED: carry: does not beat random")
    with pytest.raises(FeatError, match="same scores as submission 1"):
        idea.submit_feat(_file(idea, "noise again", NOISE))

    result, text, state = idea.submit_feat(_file(idea, "peek", PEEK))
    assert state is State.OPEN and result["verdict"]["status"] == "TAINTED", text
    assert "criteria" not in result and "calibrate" not in text and _calibrations(idea) == []

    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.REVIEW and result["verdict"]["status"] == "PASSED", text
    decisive = result["decisive"][0]
    assert result["need"] > 2.8 and decisive["z"] >= result["need"] and decisive["outcome"] == "passed"
    assert decisive["effect"] > 0 and "gain over random" in text and "bp/entry" in text and "%/yr if rolled" in text
    assert [(a.who, a.command) for a in idea.next_actions()] == [("human", "rci approve --stage feat")]
    assert idea.card("feat") == text
    idea.approve("feat", approval=_relayed(idea, "feat"))
    assert idea.status().stage("feat").state is State.PASSED
    (after,) = idea.next_actions()
    assert (after.stage, after.command) == (
        "signal",
        "",
    ) and "not built yet, so this idea is done for now" in after.reason
    shown = idea.results()
    assert len(shown) == 3 and all(card.startswith("ResultCard") for card in shown)


def test_diagnostics_are_shown_and_never_decide(tmp_path: Path) -> None:
    """The engine's defaults run next to the idea's criteria and judge the null worlds; one that fails every world
    does not fail a submission whose decisive effect beats random."""
    idea = _idea(tmp_path)
    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.REVIEW, text
    rows = {(r["criterion"], r["by"], r["world"]): r["passed"] for r in result["criteria"]}
    assert {("rank ic", "engine", "floor"), ("rank ic without top symbol", "engine", "null 1")} <= set(rows)
    assert ("top minus bottom fifth", "engine", "null 1") not in rows
    assert rows[("ic of the reversed score", "idea", "floor")] is False
    assert "[!]  ic of the reversed score (idea) vs floor" in text and "not deciding" in text
    assert "ok   alternatives covered: chance: floor · timing: floor · tail: null 1" in text


def test_without_feat_py_or_a_world_for_an_alternative_a_submission_is_still_judged(tmp_path: Path) -> None:
    idea = _idea(tmp_path, nulls="NULLS = []\n", criteria=None)
    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.REVIEW, text
    assert result["coverage"]["status"] == "warn" and "no world answers tail" in text
    assert {r["by"] for r in result["criteria"]} == {"engine"}
    assert result["calibration"]["by"] == "check"


def test_a_null_world_that_cannot_run_is_refused_without_a_trial(tmp_path: Path) -> None:
    idea = _idea(tmp_path, nulls=NULLS.replace('"tail"', '"nowhere"'))
    with pytest.raises(FeatError, match="not an alternative of the contract"):
        idea.submit_feat(_file(idea, "spread", SPREAD))
    assert idea.status().stage("feat").trials == 0


def test_a_gate_that_cannot_be_calibrated_blocks_the_data_check(tmp_path: Path, monkeypatch) -> None:
    """A decisive target that cannot see a planted signal is found at the data check, before any features exist: the
    data stays OPEN with the reason."""
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.backwards": Backwards})
    idea = Idea.init(tmp_path / "idea", settings=FAST, feat_settings=QUICK)
    (idea.root / "contract.py").write_text(label_contract().replace("'ref.mean'", "'toy.backwards'"))
    idea.derive_contract()
    idea.approve("contract")
    spec, text = idea.check_data()
    assert spec["verdict"]["status"] == "blocked" and "FAIL carry at 8h vs floor" in text
    assert spec["verdict"]["notes"][0].startswith(
        "the gate cannot be calibrated on this data: carry in the floor world"
    )
    (row,) = spec["calibration"]["rows"]
    assert (row["effect"], row["world"], row["ok"]) == ("carry", "floor", False) and row["planted_lives"] < 0.8
    assert idea.status().stage("data").state is State.OPEN
    reason = idea.next_actions()[0].reason
    assert reason.startswith("blocked: the gate cannot be calibrated") and "`rci contract derive`" in reason


def test_feat_calibrates_again_when_what_the_data_check_calibrated_changed(tmp_path: Path, monkeypatch) -> None:
    """The settings or a plugin's code changed after the data check: the first submission calibrates again and shows
    it; a failed calibration refuses the submission without a trial, and the failure is kept."""
    idea = _idea(tmp_path)
    other = Idea(idea.root, settings=FAST, feat_settings=replace(QUICK, calibration_reps=41))
    result, text, _ = other.submit_feat(_file(idea, "noise", NOISE))
    assert result["calibration"]["ran"] and "calibrate  ok   carry at 8h vs floor" in text
    assert len(_calibrations(idea)) == 1
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "ref.mean": Backwards})
    with pytest.raises(WorkflowError, match="failed its calibration.*a planted signal lived.*no submission was spent"):
        idea.submit_feat(_file(idea, "spread", SPREAD))
    assert idea.next_actions()[0].reason.startswith("calibration failed: carry in the floor world at 8h")
    with pytest.raises(WorkflowError, match="no submission was spent"):
        idea.submit_feat(_file(idea, "spread", SPREAD))
    assert len(_calibrations(idea)) == 2 and idea.ledger.n_trials("feat") == 1
    assert "FAIL carry at 8h vs floor" in idea.results()[0]


def test_what_a_submission_used_is_recorded_and_checked_again_when_a_person_accepts_it(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    idea.submit_feat(_file(idea, "spread", SPREAD))
    deps = [r for r in idea.ledger.records() if r["kind"] == "submit"][-1]["payload"]["deps"]
    assert {"features", "feat_file", "nulls_file", "engine", "calibration", "plugin:ref.mean"} <= set(deps)
    (idea.root / "feat.py").write_text(CRITERIA + "\n# changed\n")
    with pytest.raises(WorkflowError, match="changed since submission 1: feat_file; submit again"):
        idea.approve("feat", approval=Approval(by="agent", channel="auto"))
    (idea.root / "feat.py").write_text(CRITERIA)
    idea.approve("feat", approval=Approval(by="agent", channel="auto"))
    approval = list(idea.ledger.records())[-1]["payload"]
    assert approval["deps"]["features"] == deps["features"] and "feat_submission" in approval["deps"]


def test_a_strategy_contract_judges_its_label_effect_and_never_runs_the_backtest(tmp_path: Path, monkeypatch) -> None:
    """Improving an existing strategy: its backtester gives the decision times, and only the algo layer runs it."""
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.carry": NeverRun})
    idea = _idea(tmp_path, contract=strategy_contract(), nulls=NULLS.replace('"tail"', '"level"'))
    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    decisive = result["decisive"][0]
    assert (decisive["kind"], decisive["target"]) == ("label", "ref.mean"), text
    assert decisive["outcome"] == "passed" and state is State.REVIEW, text


EAGER = """
import polars as pl


def features(fields, candidates):
    assert isinstance(fields, pl.DataFrame) and isinstance(candidates, pl.DataFrame)
    rate = fields.filter(pl.col("field") == "funding_rate").select(
        "entity", pl.col("available_at").alias("at"), "value"
    ).sort("at")
    rows = candidates.sort("t")
    for leg in ("long", "short"):
        table = rate.rename({"entity": leg, "at": f"{leg}_at", "value": leg + "_rate"})
        rows = rows.join_asof(table, left_on="t", right_on=f"{leg}_at", by=leg, check_sortedness=False)
    return rows.select("t", "entity", (pl.col("short_rate") - pl.col("long_rate")).alias("score"))
"""

LAZY = EAGER.replace("fields.filter(", "fields.lazy().filter(").replace("candidates.sort(", "candidates.lazy().sort(")

PANDAS = """
def features(fields, candidates):
    frame = candidates.to_pandas()
    frame["score"] = frame.groupby("t")["spread_h"].rank()
    return frame[["t", "entity", "score"]]
"""


def test_features_get_dataframes_and_may_return_polars_lazy_or_pandas(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    eager, _ = idea.explore_feat(_file(idea, "eager", EAGER))
    assert [f["status"] for f in eager["floors"]] == ["pass", "pass"] and eager["scores"]["share"] > 0.9
    assert "fields.lazy()" in LAZY and "candidates.lazy()" in LAZY
    lazy, _ = idea.explore_feat(_file(idea, "lazy", LAZY))
    assert [f["status"] for f in lazy["floors"]] == ["pass", "pass"]
    assert lazy["scores"]["hash"] == eager["scores"]["hash"]
    pandas, _ = idea.explore_feat(_file(idea, "pandas", PANDAS))
    assert [f["status"] for f in pandas["floors"]] == ["pass", "pass"]
    example = (idea.root / "research" / "example.py").read_text()
    mine = example.replace('FIELD = "TODO"', 'FIELD = "funding_rate"').splitlines()[2:]
    explored, _ = idea.explore_feat(_file(idea, "mine", "\n".join(mine)))
    assert explored["scores"]["rows"] == 0, "the example scores entities; the toy's candidates are pairs"


def _costly(call, memory=True):
    out, _ = measured(call, memory)
    return out, Cost(4321.0, 7 * 10**9, 10**9, SAMPLED) if memory else Cost(1234.5)


def test_what_features_cost_is_shown_and_never_judged(tmp_path: Path, monkeypatch) -> None:
    """The wall time of the full call and of the calls on cut data, and the full call's peak memory, are on the
    ExploreCard and the ResultCard and in the record; the same features cost a thousand times more, and the result is
    the same."""
    idea = _idea(tmp_path)
    path = _file(idea, "spread", SPREAD)
    explored, card = idea.explore_feat(path)
    cost = explored["cost"]
    assert cost["seconds"] >= 0 and len(cost["cut_seconds"]) == QUICK.cuts and cost["memory"] in (SAMPLED, HIGH_WATER)
    assert cost["peak_bytes"] > 0 and "cost       full call " in card and ", peak memory " in card
    frozen = idea._frozen_feat(idea._approved_contract())
    plain = fs.submit(idea.root, frozen, path, 1, {}, QUICK)
    monkeypatch.setattr(fs, "measured", _costly)
    costly = fs.submit(idea.root, frozen, path, 1, {}, QUICK)
    assert costly["cost"] == {
        "seconds": 4321.0,
        "peak_bytes": 7 * 10**9,
        "start_bytes": 10**9,
        "memory": SAMPLED,
        "cut_seconds": [1234.5] * QUICK.cuts,
    }
    same = ("cost", "seconds")
    assert {k: v for k, v in costly.items() if k not in same} == {k: v for k, v in plain.items() if k not in same}
    result, text, state = idea.submit_feat(path)
    assert state is State.REVIEW and result["verdict"]["status"] == "PASSED" and result["cost"] == costly["cost"]
    line = "cost       full call 4321.0 s, peak memory +6.00 GB (process 7.00 GB) · on cut data 1234.5 s, 1234.5 s"
    assert line in text and line in idea.results()[-1]
    stored = json.loads((idea.root / ".rci" / "feat-result-1.json").read_text())
    assert stored["cost"] == costly["cost"]


ROWS = """
import polars as pl


def features(fields, candidates):
    frame = candidates.sort("t", "entity")
    scores = [s for _, _, s in frame.select("t", "entity", "spread_h").iter_rows()]
    return frame.select("t", "entity", pl.Series("score", scores))
"""


def test_row_wise_code_runs_with_a_warning_on_the_explore_and_result_cards(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    row_wise = 'def backwards(table):\n    assert all(v == v for v in table["score"].to_list())\n'
    (idea.root / "feat.py").write_text(CRITERIA.replace("def backwards(table):\n", row_wise))
    warning = (
        "row-wise code: research/rows.py:7 iter_rows · feat.py:9 comprehension over table['score'].to_list() — "
        "polars expressions are usually much faster"
    )
    explored, card = idea.explore_feat(_file(idea, "rows", ROWS))
    assert explored["warnings"] == [warning] and f"warnings   [!] {warning}" in card
    result, text, state = idea.submit_feat(_file(idea, "rows", ROWS))
    assert state is State.REVIEW and result["warnings"] == [warning] and f"warnings   [!] {warning}" in text
    assert idea.status().stage("feat").trials == 1 and f"[!] {warning}" in idea.results()[-1]
    clean, card = idea.explore_feat(_file(idea, "spread", SPREAD))
    assert clean["warnings"] == [warning.replace("research/rows.py:7 iter_rows · ", "")]


def test_what_is_refused_before_anything_runs_says_how_to_fix_it(tmp_path: Path) -> None:
    """None of these spends a trial or runs the calibration."""
    idea = _idea(tmp_path)
    spread = _file(idea, "spread", SPREAD)
    refusals = [
        (idea.root / "research" / "example.py", "is still the example; write your own"),
        (idea.root / "research" / "missing.py", "does not exist; a features file defines features(fields, candidates)"),
        (_file(idea, "columns", SPREAD.replace('alias("score")', 'alias("s")')), "features returned t, entity, s"),
        (_file(idea, "fails", SPREAD.replace("spread_h", "spread_x")), "features(fields, candidates) failed"),
        (_file(idea, "nothing", "import polars as pl\n"), "defines no function features"),
    ]
    for path, said in refusals:
        with pytest.raises(FeatError, match="no submission was spent") as refused:
            idea.submit_feat(path)
        assert said in str(refused.value), str(refused.value)
    columns = refusals[2][1]
    with pytest.raises(FeatError, match=f"{columns}, without score; .*research/example.py shows one"):
        idea.submit_feat(refusals[2][0])
    (idea.root / "feat.py").write_text("from research_ci.api import Criterion\n")
    with pytest.raises(FeatError, match=r"feat.py: it defines no CRITERIA; feat.py defines CRITERIA = \[Criterion"):
        idea.submit_feat(spread)
    (idea.root / "feat.py").write_text("CRITERIA = [\n")
    with pytest.raises(ResearchError, match=r"feat.py:1: .*fix the file"):
        idea.submit_feat(spread)
    (idea.root / "feat.py").write_text(CRITERIA)
    (idea.root / "nulls" / "feat.py").write_text(NULLS.replace('"time", clusters', '"venue", clusters'))
    with pytest.raises(FeatError, match=r"no column venue; its columns are .*symbol, t, venue_pair, y, y_flow"):
        idea.submit_feat(spread)
    assert idea.status().stage("feat").trials == 0 and _calibrations(idea) == []
