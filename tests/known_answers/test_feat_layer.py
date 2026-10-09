"""The feat layer on the toy: a real spread passes, noise fails, a peek is tainted, and a null world that does not
keep what it claims is returned before anything is judged."""

from pathlib import Path

import pytest

from research_ci.render.card import card_id
from research_ci.stages.contract import Settings
from research_ci.stages.feat import FeatError, FeatSettings
from research_ci.workflow.commands import Approval, Idea
from research_ci.workflow.machine import State, WorkflowError
from toy import label_contract

FAST = Settings(workers=2, universe_cuts=2, label_cuts=1, consumer_cuts=1, pilot=((1, 3), (2, 2)))
QUICK = FeatSettings(workers=2, calibration_reps=40, calibration_nulls=20, floor_reps=60, cuts=2)

CRITERIA = """
from research_ci.api import Criterion
from research_ci.reference_plugins.stats import rank_ic, without_top


def rest(table):
    return rank_ic(without_top(table, "symbol", 0.34))


CRITERIA = [
    Criterion("ic", rank_ic),
    Criterion("ic without the top symbol", rest),
]
"""

NULLS = """
from research_ci.api import Null, PermuteWithin

NULLS = [
    Null("tail", PermuteWithin("time", clusters="symbol", block="horizon"), {keeps}
         criteria=("ic without the top symbol",), reps=60),
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


def _contract() -> str:
    effect = "unit='per entry after the round trip')"
    target = "unit='per entry after the round trip', target=Use('ref.mean', {'cluster': 'symbol', 'top': 0.34}))"
    text = label_contract().replace(effect, target).replace("at_least=0.001", "at_least=0.0003")
    assert target in text
    return text


def _idea(tmp_path: Path, keeps: str = "") -> Idea:
    idea = Idea.init(tmp_path / "idea", "factor", settings=FAST, feat_settings=QUICK)
    (idea.root / "contract.py").write_text(_contract())
    idea.derive_contract()
    idea.approve("contract")
    idea.check_data()
    idea.approve_source(_relayed(idea, "source"))
    idea.check_data()
    idea.approve("data")
    assert idea.submit_data() is State.PASSED
    (idea.root / "feat.py").write_text(CRITERIA)
    (idea.root / "nulls").mkdir()
    (idea.root / "nulls" / "feat.py").write_text(NULLS.format(keeps=keeps))
    return idea


def _relayed(idea: Idea, stage: str) -> Approval:
    return Approval(by="human", channel="relayed", words="ok", card=card_id(idea.card(stage)))


def _file(idea: Idea, name: str, text: str) -> Path:
    path = idea.root / "research" / f"{name}.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    return path


def test_a_spread_passes_noise_fails_and_a_peek_is_tainted(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    spec, card = idea.register_feat()
    assert spec["verdict"]["status"] == "ready", card
    assert "MethodCard" in card and "tail: null 1" in card
    assert all(r["ok"] for r in spec["calibration"]), card
    assert idea.status().stage("feat").state is State.CHECKED
    idea.approve("feat", approval=_relayed(idea, "feat"))

    explored, _ = idea.explore_feat(_file(idea, "peek", PEEK))
    assert [f["status"] for f in explored["floors"]] == ["pass", "fail"]
    assert idea.status().stage("feat").trials == 0

    result, text, state = idea.submit_feat(_file(idea, "noise", NOISE))
    assert state is State.FAILED, text
    assert "ResultCard" in text and "submission 1 of 3" in text
    with pytest.raises(FeatError, match="same scores as submission 1"):
        idea.submit_feat(_file(idea, "noise again", NOISE))

    result, text, state = idea.submit_feat(_file(idea, "peek", PEEK))
    assert state is State.TAINTED, text
    assert "criteria" not in result

    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    assert state is State.PASSED, text
    assert result["need"] > 2.8
    assert all(r["passed"] for r in result["criteria"])
    assert result["decisive"][0]["outcome"] == "passed"
    assert idea.next_actions()[0].stage == "signal"


def test_a_null_world_that_does_not_keep_what_it_claims_blocks_the_method(tmp_path: Path) -> None:
    idea = _idea(tmp_path, keeps='keeps="venue_pair",')
    spec, card = idea.register_feat()
    assert spec["verdict"]["status"] == "blocked", card
    returned = [r for r in spec["calibration"] if not r["ok"]]
    assert [(r["criterion"], r["world"]) for r in returned] == [("ic without the top symbol", "null 1")]
    assert idea.status().stage("feat").state is State.CHALLENGED
    with pytest.raises(WorkflowError, match="blocked"):
        idea.approve("feat", approval=_relayed(idea, "feat"))


def test_every_alternative_needs_a_world(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    (idea.root / "nulls" / "feat.py").write_text("NULLS = []\n")
    spec, card = idea.register_feat()
    assert spec["verdict"]["status"] == "blocked"
    assert "keeps tail" in card


def test_the_method_is_pinned_once_approved(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    idea.register_feat()
    idea.approve("feat", approval=_relayed(idea, "feat"))
    (idea.root / "feat.py").write_text(CRITERIA + "\n# changed\n")
    with pytest.raises(WorkflowError, match="feat_file"):
        idea.submit_feat(_file(idea, "spread", SPREAD))
    assert idea.status().stage("feat").trials == 0
