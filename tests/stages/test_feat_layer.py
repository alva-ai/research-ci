"""The feat layer on the toy: the decisive test alone can pass a claim, and the lab's evals and null worlds can only
take horizons away. A real spread passes and goes up for review, noise fails, a peek is tainted; the default
diagnostics are shown, never deciding; a lab eval or a world below z 2 rejects the horizons it covers, and a
``feat.py`` without a valid lab eval, or an alternative without a world, is refused before a trial is spent. The data
check calibrates the decisive test, and a decisive target that cannot see a planted signal blocks the data, or, when
anything the calibration rests on changed since, refuses the submission without a trial."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from research_ci.kernel.verdict.power import Z_BETA
from research_ci.kernel.verdict.rule import VETO_Z
from research_ci.render.common import card_id, num
from research_ci.runner import plugins
from research_ci.runner.meter import HIGH_WATER, SAMPLED, Cost, measured
from research_ci.runner.research import ResearchError
from research_ci.stages.contract.run import Settings
from research_ci.stages.feat import floor as feat_floor
from research_ci.stages.feat import run as fs
from research_ci.stages.feat.run import FeatError, FeatSettings
from research_ci.validation import load as ev
from research_ci.workflow.base import Approval
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import State, WorkflowError
from toy import FEAT_EVALS, REGISTRY, TOY_NULLS, Backwards, NeverRun, lab, label_contract, strategy_contract

FAST = Settings(workers=2, universe_cuts=2, label_cuts=1)
QUICK = FeatSettings(workers=2, calibration_reps=40, calibration_nulls=20, floor_reps=60, cuts=2)

EVALS = FEAT_EVALS
BACKWARDS = """
import polars as pl

from research_ci.api import Case, Eval
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.synthetic import hand
from evals.toy import funding_ic


def backwards(table):
    return rank_ic(table.with_columns(-pl.col("score")))


CASES = [Case("the reverse order", backwards, -1.0, given=hand({"score": [1.0, 2.0, 3.0], "y": [0.1, 0.2, 0.3]}))]
EVALS = [
    Eval(funding_ic, h0="the scores do not rank the funding part of the carry"),
    Eval(backwards, name="ic of the reversed score", h0="the scores do not rank the carry in reverse"),
]
"""

NULLS = TOY_NULLS.replace("{alternative}", "tail")

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


def _idea(tmp_path: Path, contract: str | None = None, nulls: str = NULLS, evals: str | None = EVALS) -> Idea:
    idea = Idea.init(tmp_path / "idea", settings=FAST, feat_settings=QUICK)
    (idea.root / "contract.py").write_text(contract or label_contract())
    for name, text in (("strategy.toml", "[s]\nk = 2\n"), ("live.toml", "[s]\nk = 2\n"), ("strategy.py", "K = 2\n")):
        (idea.root / name).write_text(text)
    idea.derive_contract()
    idea.approve("contract")
    lab(idea.root)
    idea.check_data()
    idea.approve("data", approval=_relayed(idea, "data"))
    assert idea.status().stage("data").state is State.PASSED
    (idea.root / "feat.py").unlink()
    if evals is not None:
        (idea.root / "feat.py").write_text(evals)
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


def test_a_spread_passes_noise_fails_and_a_peek_is_tainted(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    explored, _ = idea.explore_feat(_file(idea, "peek", PEEK))
    assert [f["status"] for f in explored["floors"]] == ["pass", "fail"]
    assert idea.status().stage("feat").trials == 0 and _calibrations(idea) == []

    result, text, state = idea.submit_feat(_file(idea, "noise", NOISE))
    assert state is State.OPEN and result["verdict"]["status"] == "FAILED", text
    assert result["decisive"][0]["z"] < result["need"] and result["verdict"]["slots"] == [1]
    assert result["decisive"][0]["passing"] == [], "whatever the lab evals say, they pass no horizon"
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
    assert "vetoes" not in result and "calibrate" not in text and _calibrations(idea) == []

    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.REVIEW and result["verdict"]["status"] == "PASSED", text
    decisive = result["decisive"][0]
    assert result["need"] > 2.8 and decisive["z"] >= result["need"] and decisive["outcome"] == "passed"
    assert decisive["effect"] > 0 and "gain over random" in text and "bp/entry" in text and "%/yr if rolled" in text
    assert result["verdict"]["horizons"] == {"carry": decisive["passing"]} and decisive["passing"]
    assert f"verdict    PASSED at carry {', '.join(decisive['passing'])}" in text
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
    """The engine's defaults run in the floor world in slot 2 and are shown: whether they reach the bar does not decide
    a submission whose decisive effect beats random and that no lab eval or world rejects. A world that names a default
    diagnostic judges it in slot 3."""
    idea = _idea(tmp_path)
    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.REVIEW, text
    shown = {(r["name"], r["by"], r["world"]): r for r in result["shown"]}
    assert set(shown) == {
        (n, "default", "floor") for n in ("rank ic", "rank ic without top symbol", "top minus bottom fifth")
    }
    assert all(r["shuffles"] == QUICK.shown_reps for r in shown.values())
    vetoes = {(r["name"], r["by"], r["world"]): r for r in result["vetoes"]}
    assert set(vetoes) == {("funding_ic", "lab", "floor"), ("rank ic without top symbol", "default", "null 1")}
    assert "2 engine lists · the default diagnostics" in text and "top minus bottom fifth (default) vs floor" in text
    assert "ok   alternatives covered: chance: floor · timing: floor · tail: null 1" in text


def test_a_world_that_shuffles_like_the_floor_is_judged_once_and_the_card_says_how_many_shuffles(
    tmp_path: Path, monkeypatch
) -> None:
    """A world that names the lab eval and shuffles like the floor shares the floor's shuffles; one that names no eval
    judges the decisive effect itself, horizon by horizon."""
    worlds = NULLS.replace('evals=("rank ic without top symbol",)', 'evals=("funding_ic",)')
    worlds = worlds.replace(
        "\n]", '\n    Null("tail", PermuteWithin("venue_pair", clusters="symbol", block="horizon"), reps=60),\n]'
    )
    idea = _idea(tmp_path, nulls=worlds)
    asked: list[tuple[int, int]] = []
    judge_all = fs.judge_all

    def counted(requests: list, workers: int) -> list:
        asked.append((len(requests), len({id(r) for r in requests})))
        return judge_all(requests, workers)

    monkeypatch.setattr(fs, "judge_all", counted)
    result, text, _ = idea.submit_feat(_file(idea, "spread", SPREAD))
    rows = {(r["name"], r["world"]): r for r in result["vetoes"]}
    floor, null = rows[("funding_ic", "floor")], rows[("funding_ic", "null 1")]
    assert floor["z"] == null["z"] and floor["shuffles"] == null["shuffles"] <= QUICK.floor_reps
    ((asked_for, judged),) = asked
    assert judged == asked_for - 1
    decisive = result["decisive"][0]
    assert decisive["shuffles"] in (QUICK.floor_batch, QUICK.floor_reps)
    assert f"z {decisive['z']:.4g} ≥ {result['need']:.2f} · {decisive['shuffles']} shuffles · PASSED" in text
    world = rows[("carry", "null 2")]
    assert world["by"] == "decisive" and world["combine"] == "each" and set(world["votes"]) == {"8h", "24h"}
    assert "✓ null 2 (tail: a few symbols carry the result) judges the decisive effect" in text


def test_without_a_valid_lab_eval_or_a_world_for_an_alternative_a_submission_is_refused_before_it_counts(
    tmp_path: Path, monkeypatch
) -> None:
    """Slot 3 is required: feat.py must list a lab eval, every lab eval must pass its tests (slot 4), and every
    alternative of the contract needs a null world; each refusal names the slot and how to fix it, before the
    evaluation tables are built or the features run, and spends no trial."""
    idea = _idea(tmp_path, evals=None)

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("the evaluation tables were built, or the features ran")

    monkeypatch.setattr(fs, "frozen", never)
    monkeypatch.setattr(feat_floor, "computed", never)
    spread = _file(idea, "spread", SPREAD)
    with pytest.raises(
        FeatError, match="slot 3, lab evals: there is no feat.py, and the feat layer needs at least one"
    ):
        idea.submit_feat(spread)
    (idea.root / "feat.py").write_text("from research_ci.eval.statistics.ic import rank_ic\nEVALS = [rank_ic]\n")
    with pytest.raises(FeatError, match="slot 3, lab evals: feat.py lists no lab eval") as refused:
        idea.submit_feat(spread)
    assert "Eval(fn, h0='<the H0 it rejects>'" in str(refused.value)
    (idea.root / "feat.py").write_text(EVALS)
    toy = idea.root / "evals" / "toy.py"
    toy.write_text(toy.read_text().replace("funding_ic, 0.8, given", "funding_ic, 0.7, given"))
    with pytest.raises(FeatError) as refused:
        idea.submit_feat(spread)
    said = str(refused.value)
    assert said.startswith("slot 4, tests of lab evals: the lab eval 'funding_ic' (evals/toy.py) failed its tests")
    assert "case 'one swap in four: 1 - 6·2/(4·15)': expected 0.7, got 0.8" in said
    assert "fix the eval or its CASES in evals/toy.py; `rci eval check evals/toy.py` runs them for free" in said
    lab(idea.root)
    (idea.root / "nulls" / "feat.py").write_text("NULLS = []\n")
    with pytest.raises(
        FeatError, match=r"slot 3, null worlds: no world answers tail \(a few symbols carry the result\)"
    ):
        idea.submit_feat(spread)
    assert idea.status().stage("feat").trials == 0 and _calibrations(idea) == []


def test_a_lab_eval_below_two_fails_the_claim_where_it_covers_and_the_card_says_why(tmp_path: Path) -> None:
    """The spread beats random at every horizon, but a lab eval that tests the reverse order is below z 2 at all of
    them: the submission FAILED in slot 3, naming the eval, its H0, its z against the bar, and that its evidence
    points the other way; each horizon shows its MDE from the shuffles drawn."""
    idea = _idea(tmp_path, evals=BACKWARDS)
    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.OPEN and result["decisive"][0]["passing"] == ["8h", "24h"], text
    v = result["verdict"]
    assert v["status"] == "FAILED" and v["slots"] == [3] and v["horizons"] == {"carry": []}
    row = next(r for r in result["vetoes"] if r["name"] == "ic of the reversed score")
    assert not row["passed"] and row["votes"] == {"8h": False, "24h": False}
    assert row["shuffles"] == QUICK.floor_batch < QUICK.floor_reps, "far below the bar, the veto settles early"
    for h in row["horizons"]:
        assert h["mde"] == pytest.approx((VETO_Z + Z_BETA) * h["null_sd"]) and h["gain"] < 0 < h["mde"]
    (note,) = v["notes"]
    assert (
        "lab eval 'ic of the reversed score' (H0: the scores do not rank the carry in reverse) vs the floor world: "
        f"z {num(row['z'])} < 2.00 (its family z, one vote for all its horizons); not rejected: it points the other "
        f"way (gain over the null {num(row['horizons'][0]['gain'])} at 8h)"
    ) in note
    assert "✗ ic of the reversed score (feat.py) · H0: the scores do not rank the carry in reverse" in text
    first = row["horizons"][0]
    assert f"MDE {num(first['mde']):<10} z {num(first['z'])} ✗" in text and note in text


def test_a_horizon_no_lab_eval_covers_does_not_pass(tmp_path: Path) -> None:
    feat = FEAT_EVALS.replace('carry")', 'carry", horizons=("24h",))')
    idea = _idea(tmp_path, evals=feat)
    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.REVIEW and result["decisive"][0]["passing"] == ["8h", "24h"], text
    assert result["verdict"]["horizons"] == {"carry": ["24h"]}
    assert "carry 8h: beats random, but slot 3 rejects it: no lab eval covers 8h" in result["verdict"]["notes"]
    assert "verdict    PASSED at carry 24h" in text


def test_a_null_world_that_cannot_run_is_refused_without_a_trial(tmp_path: Path) -> None:
    idea = _idea(tmp_path, nulls=NULLS.replace('"tail"', '"nowhere"'))
    with pytest.raises(FeatError, match="not an alternative of the contract"):
        idea.submit_feat(_file(idea, "spread", SPREAD))
    assert idea.status().stage("feat").trials == 0


def test_a_decisive_test_that_cannot_be_calibrated_blocks_the_data_check(tmp_path: Path, monkeypatch) -> None:
    """A decisive target that cannot see a planted signal is found at the data check, before any features exist: the
    data stays OPEN with the reason."""
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.backwards": Backwards})
    idea = Idea.init(tmp_path / "idea", settings=FAST, feat_settings=QUICK)
    (idea.root / "contract.py").write_text(label_contract().replace("'ref.mean'", "'toy.backwards'"))
    idea.derive_contract()
    idea.approve("contract")
    lab(idea.root)
    spec, text = idea.check_data()
    assert spec["verdict"]["status"] == "blocked" and "FAIL carry at 8h vs floor" in text
    assert spec["verdict"]["notes"][0].startswith(
        "the decisive test cannot be calibrated on this data (decisive test calibration, floor): carry in the floor"
    )
    (row,) = spec["calibration"]["rows"]
    assert (row["effect"], row["world"], row["ok"]) == ("carry", "floor", False) and row["planted_lives"] < 0.8
    assert idea.status().stage("data").state is State.OPEN
    reason = idea.next_actions()[0].reason
    assert reason.startswith("blocked: the decisive test cannot be calibrated") and "`rci contract derive`" in reason


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
    (idea.root / "feat.py").write_text(EVALS + "\n# changed\n")
    with pytest.raises(WorkflowError, match="changed since submission 1: feat_file; submit again"):
        idea.approve("feat", approval=Approval(by="agent", channel="auto"))
    (idea.root / "feat.py").write_text(EVALS)
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
    monkeypatch.setattr(feat_floor, "measured", _costly)
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


ROW_EVALS = """
from research_ci.api import Case, Eval
from evals.toy import FOUR, funding_ic


def checked_ic(table):
    assert all(v == v for v in table["score"].to_list())
    return funding_ic(table)


CASES = [Case("as funding_ic", checked_ic, 0.8, given=FOUR)]
EVALS = [Eval(checked_ic, h0="the scores do not rank the funding part of the carry")]
"""


def test_row_wise_code_runs_with_a_warning_on_the_explore_and_result_cards(tmp_path: Path) -> None:
    idea = _idea(tmp_path, evals=ROW_EVALS)
    line = ROW_EVALS.splitlines().index('    assert all(v == v for v in table["score"].to_list())') + 1
    warning = (
        f"row-wise code: research/rows.py:7 iter_rows · feat.py:{line} comprehension over table['score'].to_list() — "
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
    (idea.root / "feat.py").write_text("from research_ci.api import Eval\n")
    with pytest.raises(FeatError, match=r"feat.py: it defines no EVALS; feat.py defines EVALS = \[Eval\(fn, h0="):
        idea.submit_feat(spread)
    (idea.root / "feat.py").write_text("EVALS = [\n")
    with pytest.raises(ResearchError, match=r"feat.py:1: .*fix the file"):
        idea.submit_feat(spread)
    (idea.root / "feat.py").write_text(EVALS)
    (idea.root / "nulls" / "feat.py").write_text(NULLS.replace('"time", clusters', '"venue", clusters'))
    with pytest.raises(FeatError, match=r"no column venue; its columns are .*symbol, t, venue_pair, y, y_flow"):
        idea.submit_feat(spread)
    assert idea.status().stage("feat").trials == 0 and _calibrations(idea) == []


SPREADS = """
import polars as pl

from research_ci.api import Case
from research_ci.eval.statistics.spread import quantile_spread
from research_ci.eval.synthetic import hand

SIX = hand({"score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "y_flow": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]})


def thirds(table):
    return quantile_spread(table.with_columns(y=pl.col("y_flow")), share=1 / 3)


def halves(table):
    return quantile_spread(table.with_columns(y=pl.col("y_flow")), share=0.5)


def untested(table):
    return quantile_spread(table, share=0.25)


CASES = [
    Case("the top two of six less the bottom two", thirds, 0.4, given=SIX),
    Case("reversed scores turn it around", thirds, -0.4, given=SIX.with_columns(-pl.col("score"))),
    Case("the top half less the bottom half", halves, 0.4, given=SIX),
]
"""
CASE_EVALS = """
from research_ci.api import Eval
from evals.spreads import halves, thirds

EVALS = [
    Eval(thirds, h0="the top third by score earns no more funding than the bottom third"),
    Eval(halves, h0="the top half by score earns no more funding than the bottom half", horizons=("24h",)),
]
"""


def _case_evals(idea: Idea, feat: str = CASE_EVALS, module: str = SPREADS) -> None:
    (idea.root / "evals").mkdir(exist_ok=True)
    (idea.root / "evals" / "spreads.py").write_text(module)
    (idea.root / "feat.py").write_text(feat)


def test_a_lab_eval_judges_only_when_its_tests_pass_and_one_that_fails_them_refuses_the_submission(
    tmp_path: Path, monkeypatch
) -> None:
    """A lab eval listed with a failing test refuses the submission before it is counted, with the test, what it
    expected and got, and how to fix it: breaking an eval's tests cannot set its veto aside. Fixed, it is judged like
    any lab eval and says it is valid; the validation is kept for the next submission, the module is recorded, and
    changing it refuses the approval."""
    idea = _idea(tmp_path, evals=None)
    _case_evals(idea)
    ran: list[str] = []
    real = ev.run_tests
    monkeypatch.setattr(ev, "run_tests", lambda name, cases, props: ran.append(name) or real(name, cases, props))
    with pytest.raises(FeatError) as refused:
        idea.submit_feat(_file(idea, "noise", NOISE))
    assert str(refused.value) == (
        "slot 4, tests of lab evals: the lab eval 'halves' (evals/spreads.py) failed its tests, so it cannot judge: "
        "invalid 0/1: case 'the top half less the bottom half': expected 0.4, got 0.3; fix the eval or its CASES in "
        "evals/spreads.py; `rci eval check evals/spreads.py` runs them for free; no submission was spent"
    )
    assert idea.status().stage("feat").trials == 0 and sorted(ran) == ["halves", "thirds"]
    _case_evals(idea, module=SPREADS.replace("halves, 0.4, given", "halves, 0.3, given"))
    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.REVIEW, text
    rows = {(r["name"], r["world"]): r for r in result["vetoes"]}
    thirds, halves = rows[("thirds", "floor")], rows[("halves", "floor")]
    assert thirds["by"] == "lab" and thirds["validation"]["summary"] == "valid 2/2" and thirds["passed"]
    assert [h["h"] for h in halves["horizons"]] == ["24h"] and halves["validation"]["summary"] == "valid 1/1"
    assert (
        "✓ thirds (evals/spreads.py) · H0: the top third by score earns no more funding than the bottom third" in text
    )
    assert "ok   halves (evals/spreads.py): valid 1/1" in text
    assert sorted(ran) == ["halves", "halves", "thirds", "thirds"], "the module changed, so both ran again"
    doubled = SPREAD.replace('pl.col("spread_h")', '(2 * pl.col("spread_h"))')
    _, text, state = idea.submit_feat(_file(idea, "doubled", doubled))
    assert sorted(ran) == ["halves", "halves", "thirds", "thirds"], "kept, not run again"
    deps = [r for r in idea.ledger.records() if r["kind"] == "submit"][-1]["payload"]["deps"]
    assert "eval:evals/spreads.py" in deps and state is State.REVIEW, text
    (idea.root / "evals" / "spreads.py").write_text(SPREADS + "\n# changed\n")
    with pytest.raises(WorkflowError, match="changed since submission 2: eval:evals/spreads.py"):
        idea.approve("feat", approval=Approval(by="agent", channel="auto"))


def test_a_case_eval_without_tests_is_refused_without_a_trial(tmp_path: Path) -> None:
    idea = _idea(tmp_path, evals=None)
    _case_evals(
        idea, "from research_ci.api import Eval\nfrom evals.spreads import untested\nEVALS = [Eval(untested, h0='x')]\n"
    )
    with pytest.raises(FeatError) as refused:
        idea.submit_feat(_file(idea, "spread", SPREAD))
    said = str(refused.value)
    assert "feat.py: the lab eval 'untested' has no tests, so it cannot run: write CASES in evals/spreads.py" in said
    assert said.endswith("no submission was spent") and idea.status().stage("feat").trials == 0


PARTIAL = """
import polars as pl


def features(fields, candidates):
    score = pl.when(pl.col("symbol") != "Z").then(pl.col("spread_h"))
    return candidates.select("t", "entity", score.alias("score"))
"""
SCORED_Y = """
import polars as pl

from research_ci.api import Case, Eval
from research_ci.eval.synthetic import hand


def scored_y(table):
    return table.filter(pl.col("score").is_not_null()).group_by("t").agg(pl.col("y").mean().alias("value"))


CASES = [Case("the mean label of the scored rows", scored_y, 0.2, given=hand({"score": [1.0, None, 2.0],
                                                                                "y": [0.1, 0.9, 0.3]}))]
EVALS = [Eval(scored_y, h0="the scored rows earn what every row earns")]
"""


def test_candidates_without_a_score_are_left_out_of_every_world_alike(tmp_path: Path) -> None:
    """Features that score only some candidates: the rows without a score are left out of the real world and of
    every null world, so an eval that reads only which rows hold a score gives the same number in every world."""
    idea = _idea(tmp_path, evals=SCORED_Y)
    result, text, _ = idea.submit_feat(_file(idea, "partial", PARTIAL))
    assert 0.5 < result["scores"]["share"] < 0.9, text
    rows = [r for r in result["vetoes"] if r["name"] == "scored_y"]
    assert [r["world"] for r in rows] == ["floor"]
    for r in rows:
        for h in r["horizons"]:
            assert h["null_mean"] == pytest.approx(h["real"], abs=1e-12) and h["null_sd"] == pytest.approx(0, abs=1e-12)


HALF_NOISE = """
import polars as pl


def features(fields, candidates):
    noise = pl.struct("t", "entity").hash(7) / 2.0**64
    half = pl.struct("t", "entity").hash(3) % 2 == 0
    return candidates.select("t", "entity", pl.when(half).then(noise).alias("score"))
"""


def test_random_scores_on_half_the_candidates_do_not_beat_random(tmp_path: Path) -> None:
    """When a candidate without a score took a NaN in the null worlds, the top third of a null world filled with them
    and random scores on half the candidates beat random with z near 4; judged on the scored candidates alone, they
    do not."""
    idea = _idea(tmp_path)
    result, text, _ = idea.submit_feat(_file(idea, "half", HALF_NOISE))
    assert 0.4 < result["scores"]["share"] < 0.6
    assert result["verdict"]["status"] == "FAILED" and result["decisive"][0]["z"] < 2.0, text
