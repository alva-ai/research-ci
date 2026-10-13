"""Every protocol test must pass on a correct plugin and fail on a plugin broken in the way it claims to catch."""

import pandas as pd
import polars as pl
import pytest

from research_ci.eval import protocol as pr
from research_ci.kernel.time.asof import wide
from research_ci.reference_plugins.costs import FlatCosts
from research_ci.reference_plugins.outcomes import HoldingReturn, PairCarry
from research_ci.reference_plugins.targets import ClusterMean
from toy import (
    EarlyCredit,
    FlippedFlow,
    LaggedScores,
    LateBooking,
    LazyOutcome,
    NoFees,
    Noisy,
    Peeking,
    PeekingOutcome,
    SameInstantCredit,
    Sources,
    ToyCarry,
    ToyNeedsNextBar,
    ToyPairs,
    ToyPeekPairs,
    Unbalanced,
    toy_sources,
)

HOUR_NS = 3_600 * 10**9
COSTS = FlatCosts(1, 1, legs=2)


@pytest.fixture(scope="module")
def sources() -> dict[str, pl.DataFrame]:
    return toy_sources(hours=24 * 12)


def _times(sources: Sources) -> list[int]:
    t = sources["candles"].select(pl.col("available_at").dt.epoch("ns")).unique()
    return sorted(int(x) for x in t["available_at"])


def _cands(sources: Sources) -> pl.DataFrame:
    return ToyPairs().candidates(wide(sources, _times(sources)), sources)


@pytest.mark.parametrize(
    ("plugin", "cut", "causal"),
    [(ToyPairs(), "pass", "pass"), (ToyNeedsNextBar(), "pass", "pass"), (ToyPeekPairs(), "fail", "fail")],
)
def test_universe_checks_catch_look_ahead_but_spare_plugins_that_need_the_next_bar(
    sources: Sources, plugin, cut, causal
) -> None:
    times = _times(sources)[24::24]

    def candidates(f: Sources, ts: list[int]) -> pl.DataFrame:
        return plugin.candidates(wide(f, ts), f)

    full = candidates(sources, times)
    cuts = [t + HOUR_NS for t in times[2:6]]
    assert pr.universe_pit(candidates, sources, full, times, cuts, HOUR_NS).status == cut
    assert pr.universe_causal(candidates, sources, full, times, times[3]).status == causal


def test_reference_outcomes_pass_their_own_worked_examples() -> None:
    for name, plugin in (("return", HoldingReturn()), ("carry", PairCarry(flow="funding"))):
        assert [c.status for c in pr.outcome_cases(name, plugin.values, plugin.cases())] == ["pass", "pass"]


def test_examples_that_exercise_nothing_do_not_count() -> None:
    plugin = LazyOutcome()
    checks = {c.name: c for c in pr.outcome_cases("lazy", plugin.values, plugin.cases())}
    assert checks["outcome lazy: worked examples"].status == "pass"
    assert checks["outcome lazy: examples bite"].status == "fail"
    assert "shifting the data one step later did not change" in checks["outcome lazy: examples bite"].detail


@pytest.mark.parametrize(
    ("plugin", "status"), [(PairCarry(flow="funding_rate"), "pass"), (PeekingOutcome(flow="funding_rate"), "fail")]
)
def test_truncation_catches_an_outcome_that_reads_past_its_window(sources: Sources, plugin, status) -> None:
    cuts = _times(sources)[100:200:50]
    check = pr.outcome_truncation("carry", plugin.values, sources, _cands(sources), pd.Timedelta("8h"), cuts)
    assert check.status == status, check.detail


def test_an_outcome_that_says_it_reads_past_its_window_is_caught(sources: Sources) -> None:
    def late(f: Sources, cands: pl.DataFrame, h: pd.Timedelta) -> pl.DataFrame:
        found = PairCarry(flow="funding_rate").values(f, cands, h)
        return found.with_columns(pl.col("y_end") + pl.duration(hours=1, time_unit="ns"))

    check = pr.outcome_truncation("late", late, sources, _cands(sources), pd.Timedelta("8h"), _times(sources)[150:151])
    assert check.status == "fail" and "rows have y_end after t + h" in check.detail


def test_a_correct_outcome_passes_truncation_when_decisions_fall_between_prices(sources: Sources) -> None:
    """Decisions a minute after each close: the sources cut at a decision end a minute before it, at the close."""
    minute = 60 * 10**9
    cands = _cands(sources).with_columns(pl.col("t") + pl.duration(nanoseconds=minute))
    cuts = [t + minute for t in _times(sources)[100:200:50]]
    outcome = PairCarry(flow="funding_rate")
    check = pr.outcome_truncation("carry", outcome.values, sources, cands, pd.Timedelta("8h"), cuts)
    assert check.status == "pass", check.detail


@pytest.mark.parametrize(
    ("target", "status"), [(ClusterMean("symbol", top=0.34), "pass"), (ClusterMean("symbol"), "fail")]
)
def test_a_target_that_gives_the_same_number_for_any_scores_is_caught(sources: Sources, target, status) -> None:
    h = pd.Timedelta("8h")
    cands = _cands(sources)
    rows = PairCarry(flow="funding_rate").values(sources, cands, h).join(cands, on=["t", "inst"], suffix="_candidate")
    check = pr.target_reads_score("ref.mean", lambda t: target.reduce(t, h, COSTS), rows)
    assert check.status == status, check.detail


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (Peeking(), "fail")])
def test_consumer_truncation_catches_a_consumer_that_trades_on_the_future(sources: Sources, plugin, status) -> None:
    base = plugin.run(sources)
    cuts = _times(sources)[100:200:50]
    assert pr.consumer_pit(plugin.run, sources, base, cuts, 2 * HOUR_NS).status == status


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (EarlyCredit(), "fail")])
def test_causality_catches_consumers_that_use_data_before_it_exists(sources: Sources, plugin, status) -> None:
    at = pr.busiest(sources, _times(sources), HOUR_NS)
    check = pr.causal(plugin.run, sources, plugin.run(sources), at, HOUR_NS)
    assert check.status == status, check.detail


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (Noisy(), "fail")])
def test_two_identical_runs_must_agree(sources: Sources, plugin, status) -> None:
    assert pr.deterministic(plugin.run(sources), plugin.run(sources)).status == status


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (NoFees(), "fail")])
def test_fees_must_match_the_cost_model(sources: Sources, plugin, status) -> None:
    result = plugin.run(sources)
    assert [c.status for c in pr.adds_up(result)] == ["pass", "pass"]
    assert pr.fees_match(result, COSTS).status == status


@pytest.mark.parametrize(
    ("plugin", "status"),
    [
        (ToyCarry(), "pass"),
        (FlippedFlow(), "fail"),
        (EarlyCredit(), "fail"),
        (SameInstantCredit(), "fail"),
        (LateBooking(), "fail"),
    ],
)
def test_the_consumer_must_earn_what_the_outcome_says_its_positions_earned(sources, plugin, status) -> None:
    result = plugin.run(sources)
    held = result.positions.join(_cands(sources), on=["t", "inst"], how="inner")
    outcome = PairCarry(flow="funding_rate").values(sources, held, pd.Timedelta(hours=1))
    check = pr.matches_outcome(result, outcome, HOUR_NS)
    assert check.status == status, check.detail


def test_own_scores_pass_through_and_perturbations_keep_their_share(sources: Sources) -> None:
    plugin = ToyCarry()
    base = plugin.run(sources)
    scores = _cands(sources).select("t", "inst", "score")
    assert pr.passthrough(plugin.run, sources, base, scores).status == "pass"
    block = int(pd.Timedelta("24h").value)
    kept = pr.perturb(scores, "drop", 0.2, block, seed=1)
    assert 0.7 < kept.height / scores.height < 0.9
    per_block = kept.with_columns(b=pl.col("t").dt.epoch("ns") // block).group_by("b", "inst").len()
    whole = scores.with_columns(b=pl.col("t").dt.epoch("ns") // block).group_by("b", "inst").len()
    partial = per_block.join(whole, on=["b", "inst"], suffix="_all").filter(pl.col("len") != pl.col("len_all"))
    assert partial.is_empty()
    shuffled = pr.perturb(scores, "shuffle", 0.5, block, seed=1)
    assert shuffled.height == scores.height
    assert sorted(shuffled["score"].to_list()) == sorted(scores["score"].to_list())
    assert (shuffled.sort("t", "inst")["score"] != scores.sort("t", "inst")["score"]).any()


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (Unbalanced(), "fail")])
def test_components_must_add_up_to_net(sources: Sources, plugin, status) -> None:
    assert pr.adds_up(plugin.run(sources))[0].status == status


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (LaggedScores(), "fail")])
def test_the_consumer_must_reproduce_itself_from_its_own_scores(sources: Sources, plugin, status) -> None:
    scores = _cands(sources).select("t", "inst", "score")
    assert pr.passthrough(plugin.run, sources, plugin.run(sources), scores).status == status
