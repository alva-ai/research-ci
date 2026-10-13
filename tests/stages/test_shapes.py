"""A universe's shape on the timed toy market: one asset timed (``series``) and events among three assets
(``events``). Their floor world shifts each inst's scores in time; the data check calibrates the decisive test in it,
a planted signal passes, noise fails, a peek is tainted, and a lab eval or a ``ShiftInTime`` world of the idea can
veto."""

from pathlib import Path

import pytest

from research_ci.render.common import card_id
from research_ci.stages.contract.run import Settings
from research_ci.stages.feat.run import FeatSettings
from research_ci.workflow.base import Approval
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import State
from toy import SHAPED_FEAT, SHAPED_H0, shaped_contract, shaped_lab

FAST = Settings(workers=2, universe_cuts=2, outcome_cuts=1)
QUICK = FeatSettings(workers=2, calibration_reps=40, calibration_nulls=20, floor_reps=60, cuts=2)
REGIME = """
import polars as pl


def regime(wide, candidates):
    return candidates.join(wide.select("t", "inst", pl.col("regime").alias("value")), on=["t", "inst"])
"""
FUTURE = """
import polars as pl


def future(wide, candidates):
    later = pl.col("close").shift(-24).over("inst") / pl.col("close") - 1
    return candidates.join(wide.select("t", "inst", later.alias("value")), on=["t", "inst"])
"""
FEATURES = {
    "series": {
        "planted": REGIME + "\n\ndef features(wide, candidates):\n"
        "    return regime(wide, candidates).select('t', 'inst', (pl.col('value') * SCALE).alias('score'))\n",
        "noise": "import polars as pl\n\n\ndef features(wide, candidates):\n"
        "    noise = pl.struct('t', 'inst').hash(7) / 2.0**64 - 0.5\n"
        "    return candidates.select('t', 'inst', noise.alias('score'))\n",
        "peek": FUTURE + "\n\ndef features(wide, candidates):\n"
        "    return future(wide, candidates).select('t', 'inst', pl.col('value').alias('score'))\n",
    },
    "events": {
        "planted": REGIME + "\n\ndef features(wide, candidates):\n"
        "    rows = regime(wide, candidates).filter((pl.col('value') > 0) & (pl.col('t').dt.hour() % 8 == 0))\n"
        "    return rows.select('t', 'inst', pl.lit(SCALE).alias('score'))\n",
        "noise": "import polars as pl\n\n\ndef features(wide, candidates):\n"
        "    rows = candidates.filter(pl.struct('t', 'inst').hash(7) / 2.0**64 < 0.06)\n"
        "    return rows.select('t', 'inst', pl.lit(1.0).alias('score'))\n",
        "peek": FUTURE + "\n\ndef features(wide, candidates):\n"
        "    rows = future(wide, candidates).filter((pl.col('value') > 0) & (pl.col('t').dt.hour() % 8 == 0))\n"
        "    return rows.select('t', 'inst', pl.lit(1.0).alias('score'))\n",
    },
}
DIAGNOSTIC = {"series": "ts ic", "events": "mean at events"}


def _idea(tmp_path: Path, shape: str) -> Idea:
    idea = Idea.init(tmp_path / "idea", settings=FAST, feat_settings=QUICK)
    (idea.root / "contract.py").write_text(shaped_contract(shape))
    idea.derive_contract()
    idea.approve("contract")
    shaped_lab(idea.root, shape)
    idea.check_data()
    idea.approve("data", approval=_relayed(idea, "data"))
    assert idea.status().stage("data").state is State.PASSED, idea.card("data")
    return idea


def _relayed(idea: Idea, stage: str) -> Approval:
    return Approval(by="human", channel="relayed", words="ok", card=card_id(idea.card(stage)))


def _file(idea: Idea, shape: str, kind: str, scale: float = 1.0) -> Path:
    path = idea.root / "research" / f"{kind}{int(scale)}.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(FEATURES[shape][kind].replace("SCALE", repr(scale)))
    return path


@pytest.mark.parametrize("shape", ["series", "events"])
def test_a_planted_signal_passes_noise_fails_a_peek_is_tainted_and_lab_evals_and_shifts_in_time_veto(
    tmp_path: Path, shape: str
) -> None:
    idea = _idea(tmp_path, shape)
    contract = idea.card("contract")
    assert f"shape      {shape}: " in contract and "floor world ShiftInTime(min_shift=horizon)" in contract
    assert "(engine floor: chance)" in contract
    target = "ref.timing" if shape == "series" else "ref.mean"
    assert f"calibrate  ok   {target} at 8h vs floor: stand-ins die" in idea.card("data")

    result, text, state = idea.submit_feat(_file(idea, shape, "noise"))
    assert state is State.OPEN and result["verdict"]["status"] == "FAILED", text
    assert result["verdict"]["slots"] == [1] and result["decisive"][0]["passing"] == []

    result, text, _ = idea.submit_feat(_file(idea, shape, "peek"))
    assert result["verdict"]["status"] == "TAINTED", text

    result, text, state = idea.submit_feat(_file(idea, shape, "planted"))
    assert state is State.REVIEW and result["verdict"]["status"] == "PASSED", text
    decisive = result["decisive"][0]
    assert decisive["passing"] and decisive["effect"] > 0 and decisive["z"] >= result["need"]
    assert {(r["name"], r["world"]) for r in result["shown"]} == {(DIAGNOSTIC[shape], "floor")}
    assert {(r["name"], r["world"]) for r in result["vetoes"]} == {
        (SHAPED_H0[shape][0], w) for w in ("floor", "null 1")
    }
    assert all(r["passed"] for r in result["vetoes"]) and result["worlds"][0]["shift"] == "horizon"
    assert "alternatives covered: chance: floor · trend: null 1" in text and "vs floor (chance)" in text

    name, h0 = SHAPED_H0[shape]
    feat = SHAPED_FEAT.format(name=name, h0=h0).replace(f"import {name}", f"import {name}, backwards")
    feat = feat.replace(")]", '), Eval(backwards, h0="the score read the wrong way round is no better than chance")]')
    (idea.root / "feat.py").write_text(feat)
    nulls = (idea.root / "nulls" / "feat.py").read_text().replace(f'evals=("{name}",)', 'evals=("backwards",)')
    (idea.root / "nulls" / "feat.py").write_text(nulls)
    result, text, state = idea.submit_feat(_file(idea, shape, "planted", scale=2.0))
    assert state is State.OPEN and result["decisive"][0]["passing"], text
    assert result["verdict"]["status"] == "FAILED" and result["verdict"]["slots"] == [3]
    rows = {(r["name"], r["world"]): r for r in result["vetoes"]}
    assert not rows[("backwards", "floor")]["passed"] and not rows[("backwards", "null 1")]["passed"]
    (note,) = result["verdict"]["notes"]
    assert "lab eval 'backwards' (H0: the score read the wrong way round is no better than chance) vs the floor" in note
    assert "null 1 (trend: the score only follows the last days) judging lab eval 'backwards'" in note
