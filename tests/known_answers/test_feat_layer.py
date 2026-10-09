"""The feat layer on the toy: a real spread passes, noise fails, a peek is tainted, and a null world that does not
keep what it claims is returned before anything is judged."""

from pathlib import Path

import pytest

from research_ci.render.card import card_id
from research_ci.stages.contract import Settings
from research_ci.stages.feat import FeatError, FeatSettings
from research_ci.workflow.commands import Approval, Idea
from research_ci.workflow.machine import State, WorkflowError
from toy import consumer_contract, label_contract

FAST = Settings(workers=2, universe_cuts=2, label_cuts=1, consumer_cuts=1, pilot=((1, 3), (2, 2)))
QUICK = FeatSettings(workers=2, calibration_reps=40, calibration_nulls=20, floor_reps=60, consumer_reps=6, cuts=2)

CRITERIA = """
from research_ci.api import Criterion
from research_ci.eval.statistics import rank_ic
from research_ci.eval.structure import without_top


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


def _random_scores_contract() -> str:
    """The consumer judges the claim against random scores of the same candidates; there is no baseline."""
    decisive = (
        "decisive=(Effect(on='consumer', at_least=0.05, unit='per year over random scores', "
        "pilot=Perturb('shuffle', 1.0, block='24h'), design=FullPath()),),"
    )
    text = label_contract().replace("execution=Execution(every='1h')", "consumer='toy.carry'")
    old = "decisive=(Effect(on='carry', at_least=0.001, unit='per entry after the round trip'),),"
    assert old in text and "consumer='toy.carry'" in text
    return text.replace(old, decisive)


def _idea(tmp_path: Path, keeps: str = "", contract: str | None = None, nulls: str | None = None) -> Idea:
    idea = Idea.init(tmp_path / "idea", "factor", settings=FAST, feat_settings=QUICK)
    (idea.root / "contract.py").write_text(contract or _contract())
    for name, text in (("strategy.toml", "[s]\nk = 2\n"), ("live.toml", "[s]\nk = 2\n"), ("strategy.py", "K = 2\n")):
        (idea.root / name).write_text(text)
    idea.derive_contract()
    idea.approve("contract")
    idea.check_data()
    idea.approve_source(_relayed(idea, "source"))
    idea.check_data()
    idea.approve("data")
    assert idea.submit_data() is State.PASSED
    (idea.root / "feat.py").write_text(CRITERIA)
    (idea.root / "nulls").mkdir()
    (idea.root / "nulls" / "feat.py").write_text(nulls or NULLS.format(keeps=keeps))
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


def test_through_the_consumer_a_spread_beats_random_scores_and_noise_does_not(tmp_path: Path) -> None:
    idea = _idea(tmp_path, contract=_random_scores_contract())
    spec, card = idea.register_feat()
    assert spec["verdict"]["status"] == "ready", card
    assert "over random changes to the submitted scores" in card
    idea.approve("feat", approval=_relayed(idea, "feat"))

    result, text, _ = idea.submit_feat(_file(idea, "noise", NOISE))
    decisive = result["decisive"][0]
    assert (decisive["kind"], decisive["design"], decisive["runs"]) == ("consumer", "full_path", 6), text
    assert decisive["outcome"] != "passed", text

    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    decisive = result["decisive"][0]
    assert decisive["outcome"] == "passed" and decisive["z"] > result["need"], text
    assert decisive["over"].startswith("the mean of runs") and decisive["run"]["trades"] > 0
    assert state is State.PASSED, text


def test_branching_from_the_baseline_gains_nothing_with_the_baseline_own_scores(tmp_path: Path) -> None:
    """The toy baseline already ranks by the spread: submitting the same scores changes no branch. Fourteen days
    of toy branches are too noisy to rule out 5% a year, so the data cannot decide."""
    nulls = NULLS.format(keeps="").replace('"tail"', '"level"').replace('criteria=("ic without the top symbol",), ', "")
    idea = _idea(tmp_path, contract=consumer_contract(), nulls=nulls)
    spec, card = idea.register_feat()
    assert spec["verdict"]["status"] == "ready", card
    idea.approve("feat", approval=_relayed(idea, "feat"))
    result, text, state = idea.submit_feat(_file(idea, "spread", SPREAD))
    decisive = result["decisive"][0]
    assert (decisive["design"], decisive["over"]) == ("branch", "the baseline"), text
    assert decisive["branches"] > 3 and decisive["effect"] == 0.0
    assert decisive["outcome"] == "insufficient" and state is State.INSUFFICIENT, text
