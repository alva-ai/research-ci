"""The feat layer on the toy: the decisive effect alone decides. A real spread passes and goes up for review, noise
fails, a peek is tainted; diagnostics and alternatives without a world are shown, never deciding; a decisive target
that cannot see a planted signal fails calibration and the submission is refused without a trial."""

from pathlib import Path

import pytest

from research_ci.render.common import card_id
from research_ci.runner import plugins
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


def test_the_decisive_effect_alone_decides_a_spread_passes_noise_fails_and_a_peek_is_tainted(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    explored, _ = idea.explore_feat(_file(idea, "peek", PEEK))
    assert [f["status"] for f in explored["floors"]] == ["pass", "fail"]
    assert idea.status().stage("feat").trials == 0 and _calibrations(idea) == []

    result, text, state = idea.submit_feat(_file(idea, "noise", NOISE))
    assert state is State.OPEN and result["verdict"]["status"] == "FAILED", text
    assert result["decisive"][0]["z"] < result["need"]
    assert "ResultCard" in text and "submission 1 of 3" in text
    (calibration,) = _calibrations(idea)
    assert [(r["effect"], r["world"], r["ok"]) for r in calibration["rows"]] == [("carry", "floor", True)]
    assert result["calibration"]["ran"] and "calibrate  ok   carry at 8h vs floor: stand-ins die" in text
    assert idea.next_actions()[0].reason.startswith("submission 1 FAILED: carry: does not beat random")
    with pytest.raises(FeatError, match="same scores as submission 1"):
        idea.submit_feat(_file(idea, "noise again", NOISE))

    result, text, state = idea.submit_feat(_file(idea, "peek", PEEK))
    assert state is State.OPEN and result["verdict"]["status"] == "TAINTED", text
    assert "criteria" not in result and "calibrate" not in text and len(_calibrations(idea)) == 1

    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.REVIEW and result["verdict"]["status"] == "PASSED", text
    decisive = result["decisive"][0]
    assert result["need"] > 2.8 and decisive["z"] >= result["need"] and decisive["outcome"] == "passed"
    assert decisive["effect"] > 0 and "gain over random" in text and "bp/entry" in text and "%/yr if rolled" in text
    assert [(a.who, a.command) for a in idea.next_actions()] == [("human", "rci approve --stage feat")]
    assert idea.card("feat") == text
    idea.approve("feat", approval=_relayed(idea, "feat"))
    assert idea.status().stage("feat").state is State.PASSED and idea.next_actions()[0].stage == "signal"
    shown = idea.results()
    assert len(shown) == 3 and shown[0].startswith("ResultCard") and "calibrate  ok   carry at 8h" in shown[0]


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
    assert result["calibration"]["ran"]


def test_a_null_world_that_cannot_run_is_refused_without_a_trial(tmp_path: Path) -> None:
    idea = _idea(tmp_path, nulls=NULLS.replace('"tail"', '"nowhere"'))
    with pytest.raises(FeatError, match="not an alternative of the contract"):
        idea.submit_feat(_file(idea, "spread", SPREAD))
    assert idea.status().stage("feat").trials == 0


def test_a_failed_calibration_refuses_the_submission_and_costs_no_trial(tmp_path: Path, monkeypatch) -> None:
    """A decisive target that cannot see a planted signal is returned before anything is judged; the failure is kept,
    so the next submission is refused at once, until the target or the contract changes."""
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.backwards": Backwards})
    idea = _idea(tmp_path, contract=label_contract().replace("'ref.mean'", "'toy.backwards'"))
    with pytest.raises(WorkflowError, match="failed its calibration.*a planted signal lived"):
        idea.submit_feat(_file(idea, "spread", SPREAD))
    feat = idea.status().stage("feat")
    assert (feat.state, feat.trials) == (State.OPEN, 0)
    (row,) = _calibrations(idea)[0]["rows"]
    assert (row["effect"], row["world"], row["ok"]) == ("carry", "floor", False) and row["planted_lives"] < 0.8
    assert idea.next_actions()[0].reason.startswith("calibration failed: carry in the floor world at 8h")
    with pytest.raises(WorkflowError, match="no submission was spent"):
        idea.submit_feat(_file(idea, "noise", NOISE))
    assert len(_calibrations(idea)) == 1 and idea.ledger.n_trials("feat") == 0
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
