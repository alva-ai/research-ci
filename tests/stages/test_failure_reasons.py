"""Every failure on every card says why. One test per way a DataCard, a ResultCard, or an ExploreCard can fail: the
card names what failed, says why, and what to do. (The ContractCard's are in ``test_derive.py``; an invalid data eval
by its pits, a data veto, and the refusals before the data is fetched in ``test_data_stage.py``; an invalid feat eval,
a veto in slot 3, and the refusals before a trial in ``test_feat_layer.py``.)"""

from pathlib import Path

import pytest

from research_ci.kernel.verdict.check import Check
from research_ci.render.common import card_id
from research_ci.runner import plugins
from research_ci.stages.contract.run import Settings
from research_ci.stages.data import defaults as data_defaults
from research_ci.stages.data.run import DataError
from research_ci.stages.feat.run import FeatSettings
from research_ci.stages.skeleton import Found, Step
from research_ci.workflow.base import Approval
from research_ci.workflow.idea import Idea
from toy import REGISTRY, Backwards, ToyHindsight, carry_contract, lab

FAST = Settings(workers=2, universe_cuts=2, outcome_cuts=1)
QUICK = FeatSettings(workers=2, calibration_reps=40, calibration_nulls=20, floor_reps=60, cuts=2)
FLAGS = """
import polars as pl

from research_ci.api import Case

ROWS = {"candles": pl.DataFrame({"inst": ["a", "a"], "event_time": [1, 2], "close": [1.0, 0.0]})}


def zeros(sources):
    found = sources["candles"].filter(pl.col("close") == 0)
    return found.select(pl.lit("candles").alias("source"), "inst", "event_time")


def broken(sources):
    return sources["candles"].select(pl.lit("candles").alias("source"), "inst", "event_time").head(1)


def wrong_shape(sources):
    return sources["candles"].select("close")


def fails(sources):
    if sources["candles"].height:
        raise RuntimeError("cannot read these sources")
    return sources["candles"].select(pl.lit("candles").alias("source"), "inst", "event_time")


CASES = [
    Case("the zero is flagged", zeros, 1, given=ROWS),
    Case("only the zero is flagged", broken, 0, given=ROWS),
    Case("no rows, no flags", wrong_shape, 0, given={"candles": ROWS["candles"].head(0)}),
    Case("an empty table", fails, 0, given={"candles": ROWS["candles"].head(0)}),
]
H0 = "no value is zero"
"""


def _idea(tmp_path: Path, contract: str | None = None) -> Idea:
    idea = Idea.init(tmp_path / "idea", settings=FAST, feat_settings=QUICK)
    (idea.root / "contract.py").write_text(contract or carry_contract())
    idea.derive_contract()
    idea.approve("contract")
    lab(idea.root)
    return idea


def _data(idea: Idea, evals: str) -> None:
    (idea.root / "evals").mkdir(exist_ok=True)
    (idea.root / "evals" / "flags.py").write_text(FLAGS)
    (idea.root / "data.py").write_text(
        "from research_ci.api import Eval\nfrom evals.flags import H0, broken, fails, wrong_shape, zeros\n\n"
        f"EVALS = [Eval(fn, h0=H0) for fn in {evals}]\n"
    )


def _passed_data(idea: Idea) -> Idea:
    idea.check_data()
    idea.approve("data", approval=Approval(by="human", channel="relayed", words="ok", card=card_id(idea.card("data"))))
    return idea


def _features(idea: Idea, name: str, body: str) -> Path:
    path = idea.root / "research" / f"{name}.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"import polars as pl\n\n\ndef features(wide, candidates):\n    return {body}\n")
    return path


def test_datacard_a_source_that_failed_admission_says_which_checks_and_what_to_do(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyHindsight})
    _, card = _idea(tmp_path).check_data()
    assert "FAIL replay as of past times: " in card
    assert (
        "source toy.data not admitted: its admission failed replay as of past times, independent source agrees (the "
        "source lines above say why); fix the data plugin, or its reference source, and check again"
    ) in card


def test_an_invalid_lab_eval_refuses_the_data_check_saying_which_test_what_it_expected_and_got_and_how_to_fix_it(
    tmp_path: Path,
) -> None:
    idea = _idea(tmp_path)
    _data(idea, "[zeros, broken]")
    with pytest.raises(DataError) as refused:
        idea.check_data()
    assert str(refused.value) == (
        "slot 4, tests of lab evals: the lab eval 'broken' (evals/flags.py) failed its tests, so it cannot judge: "
        "invalid 0/1: case 'only the zero is flagged': expected 0, got 1; fix the eval or its CASES in evals/flags.py; "
        "`rci eval check evals/flags.py` runs them for free; nothing was recorded"
    )
    assert not [r for r in idea.ledger.records() if r["kind"] == "check"]


def test_datacard_a_failed_step_says_whose_it_is_why_and_what_to_do(tmp_path: Path, monkeypatch) -> None:
    def failing(seen: object) -> Found:
        """Fails, as a check of the default list may."""
        return Found([Check("closes are sane", "fail", "3 closes are negative")])

    monkeypatch.setattr(data_defaults, "CHECKS", [Step("closes are sane", "blocks", failing, "Fails")])
    spec, card = _idea(tmp_path).check_data()
    assert spec["verdict"]["status"] == "blocked" and "default  FAIL closes are sane: 3 closes are negative" in card
    assert "closes are sane (default) failed: 3 closes are negative; fix what it names and check again" in card


def test_datacard_a_decisive_test_that_cannot_be_calibrated_says_why(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.backwards": Backwards})
    idea = _idea(tmp_path, carry_contract().replace("'ref.mean'", "'toy.backwards'"))
    _, card = idea.check_data()
    assert "calibrate  FAIL toy.backwards at 8h vs floor: stand-ins die " in card and "needs ≥ 80%" in card
    assert (
        "the decisive test cannot be calibrated on this data (decisive test calibration, floor): toy.backwards in the "
        "floor world at 8h: a planted signal lived"
    ) in card and "change the target or the contract" in card


def test_a_data_eval_that_breaks_on_the_data_or_returns_no_rows_is_refused_saying_what_to_fix(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    _data(idea, "[fails]")
    with pytest.raises(DataError, match="eval 'fails' failed on the explore data: RuntimeError: cannot read") as e:
        idea.check_data()
    assert "it passed its tests, so add a case like this data to them and fix it" in str(e.value)
    _data(idea, "[wrong_shape]")
    with pytest.raises(DataError, match="eval 'wrong_shape' returned close, not the rows it flags; data.py defines"):
        idea.check_data()
    assert not [r for r in idea.ledger.records() if r["kind"] == "check"]


def test_resultcard_failed_says_the_z_against_the_bar_and_the_effect(tmp_path: Path) -> None:
    idea = _passed_data(_idea(tmp_path))
    noise = '(candidates.select("t", "inst", (pl.struct("t", "inst").hash(7) / 2.0**64).alias("score")))'
    result, text, _ = idea.submit_feat(_features(idea, "noise", noise))
    assert result["verdict"]["status"] == "FAILED"
    (note,) = result["verdict"]["notes"]
    assert (
        note.startswith("ref.mean: does not beat random, z ")
        and " < 2.00 (decisive test, slot 1): gain over random " in note
    )
    assert "bp per entry ± " in note and note in text
    assert "decisive   ref.mean: gain over random " in text and " < 2.00 · " in text and "· FAILED" in text
    assert result["verdict"]["slots"] == [1] and "verdict    FAILED\n" in text
    shown = [line for line in text.splitlines() if "[!]" in line and " vs floor" in line]
    assert shown and all(" < 2.00" in line for line in shown), "a diagnostic below the bar says so"


def test_resultcard_scores_that_do_not_rank_say_how_often_they_do_and_what_is_needed(tmp_path: Path) -> None:
    idea = _passed_data(_idea(tmp_path))
    result, text, _ = idea.submit_feat(_features(idea, "flat", 'candidates.select("t", "inst", score=pl.lit(1.0))'))
    assert result["verdict"]["status"] == "FAILED"
    assert (
        "the scores do not rank the candidates (scores rank candidates, floor): scores 100% of candidate rows; ranks "
        "candidates at 0% of decision times, needs ≥ 50%; give two or more candidates distinct scores at each time"
    ) in text
    assert "floor    FAIL scores rank candidates: " in text


def test_resultcard_tainted_says_where_the_cut_was_and_what_changed(tmp_path: Path) -> None:
    idea = _passed_data(_idea(tmp_path))
    peek = (
        'candidates.join(wide.group_by("inst").agg(pl.col("funding_rate").mean()).rename({"inst": "short"}), '
        'on="short").select("t", "inst", score="funding_rate")'
    )
    result, text, _ = idea.submit_feat(_features(idea, "peek", peek))
    assert result["verdict"]["status"] == "TAINTED"
    (note,) = result["verdict"]["notes"]
    assert note.startswith(
        "the scores read data from after their decision time (scores read nothing ahead, floor): 2026-01-"
    )
    assert "values of score differ" in note and note.endswith("compute each score from rows with available_at <= t")
    assert note in text and "floor    FAIL scores read nothing ahead: " in text


def test_explorecard_a_failed_floor_says_why(tmp_path: Path) -> None:
    idea = _passed_data(_idea(tmp_path))
    _, card = idea.explore_feat(_features(idea, "flat", 'candidates.select("t", "inst", score=pl.lit(1.0))'))
    assert "floor    FAIL scores rank candidates: scores 100% of candidate rows; ranks candidates at 0%" in card
