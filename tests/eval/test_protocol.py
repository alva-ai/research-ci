"""Every protocol test must pass on a correct plugin and fail on a plugin broken in the way it claims to catch."""

import pandas as pd
import polars as pl
import pytest

from research_ci.eval import protocol as pr
from research_ci.kernel.time.clock import from_ns
from research_ci.reference_plugins.costs import FlatCosts
from research_ci.reference_plugins.labels import HoldingReturn, PairCarry
from research_ci.reference_plugins.targets import ClusterMean
from toy import (
    EarlyCredit,
    FlippedFlow,
    LaggedScores,
    LateBooking,
    LazyLabel,
    NoFees,
    Noisy,
    Peeking,
    PeekingLabel,
    SameInstantCredit,
    ToyCarry,
    ToyNeedsNextBar,
    ToyPairs,
    ToyPeekPairs,
    Unbalanced,
    toy_fields,
)

HOUR_NS = 3_600 * 10**9
COSTS = FlatCosts(1, 1, legs=2)


@pytest.fixture(scope="module")
def fields() -> pl.DataFrame:
    return toy_fields(hours=24 * 12)


def _times(fields: pl.DataFrame) -> list[int]:
    t = fields.filter(pl.col("field") == "close").select(pl.col("available_at").dt.epoch("ns")).unique()
    return sorted(int(x) for x in t["available_at"])


def _cands(fields: pl.DataFrame) -> pl.DataFrame:
    return ToyPairs().candidates(fields, [from_ns(t) for t in _times(fields)])


@pytest.mark.parametrize(
    ("plugin", "cut", "causal"),
    [(ToyPairs(), "pass", "pass"), (ToyNeedsNextBar(), "pass", "pass"), (ToyPeekPairs(), "fail", "fail")],
)
def test_universe_checks_catch_look_ahead_but_spare_plugins_that_need_the_next_bar(
    fields: pl.DataFrame, plugin, cut, causal
) -> None:
    times = _times(fields)[24::24]

    def candidates(f: pl.DataFrame, ts: list[int]) -> pl.DataFrame:
        return plugin.candidates(f, [from_ns(t) for t in ts])

    full = candidates(fields, times)
    cuts = [t + HOUR_NS for t in times[2:6]]
    assert pr.universe_pit(candidates, fields, full, cuts, HOUR_NS).status == cut
    assert pr.universe_causal(candidates, fields, full, times[3]).status == causal


def test_reference_labels_pass_their_own_worked_examples() -> None:
    for name, plugin in (("return", HoldingReturn()), ("carry", PairCarry(flow="funding"))):
        assert [c.status for c in pr.label_cases(name, plugin.values, plugin.cases())] == ["pass", "pass"]


def test_examples_that_exercise_nothing_do_not_count() -> None:
    plugin = LazyLabel()
    checks = {c.name: c for c in pr.label_cases("lazy", plugin.values, plugin.cases())}
    assert checks["label lazy: worked examples"].status == "pass"
    assert checks["label lazy: examples bite"].status == "fail"
    assert "shifting the data one step later did not change" in checks["label lazy: examples bite"].detail


@pytest.mark.parametrize(
    ("plugin", "status"), [(PairCarry(flow="funding_rate"), "pass"), (PeekingLabel(flow="funding_rate"), "fail")]
)
def test_label_truncation_catches_a_label_that_reads_past_its_window(fields: pl.DataFrame, plugin, status) -> None:
    cuts = _times(fields)[100:200:50]
    check = pr.label_truncation("carry", plugin.values, fields, _cands(fields), pd.Timedelta("8h"), cuts)
    assert check.status == status, check.detail


def test_a_correct_label_passes_truncation_when_decisions_fall_between_prices(fields: pl.DataFrame) -> None:
    """Decisions a minute after each close: the data cut at a decision ends a minute before it, at the close."""
    minute = 60 * 10**9
    cands = _cands(fields).with_columns(pl.col("t") + pl.duration(nanoseconds=minute))
    cuts = [t + minute for t in _times(fields)[100:200:50]]
    label = PairCarry(flow="funding_rate")
    check = pr.label_truncation("carry", label.values, fields, cands, pd.Timedelta("8h"), cuts)
    assert check.status == "pass", check.detail


@pytest.mark.parametrize(
    ("target", "status"), [(ClusterMean("symbol", top=0.34), "pass"), (ClusterMean("symbol"), "fail")]
)
def test_a_target_that_gives_the_same_number_for_any_scores_is_caught(fields: pl.DataFrame, target, status) -> None:
    h = pd.Timedelta("8h")
    cands = _cands(fields)
    rows = PairCarry(flow="funding_rate").values(fields, cands, h).join(cands, on=["t", "entity"], suffix="_candidate")
    check = pr.target_reads_score("ref.mean", lambda t: target.reduce(t, h, COSTS), rows)
    assert check.status == status, check.detail


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (Peeking(), "fail")])
def test_consumer_truncation_catches_a_consumer_that_trades_on_the_future(fields: pl.DataFrame, plugin, status) -> None:
    base = plugin.run(fields)
    cuts = _times(fields)[100:200:50]
    assert pr.consumer_pit(plugin.run, fields, base, cuts, 2 * HOUR_NS).status == status


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (EarlyCredit(), "fail")])
def test_causality_catches_consumers_that_use_data_before_it_exists(fields: pl.DataFrame, plugin, status) -> None:
    at = pr.busiest(fields, _times(fields), HOUR_NS)
    check = pr.causal(plugin.run, fields, plugin.run(fields), at, HOUR_NS)
    assert check.status == status, check.detail


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (Noisy(), "fail")])
def test_two_identical_runs_must_agree(fields: pl.DataFrame, plugin, status) -> None:
    assert pr.deterministic(plugin.run(fields), plugin.run(fields)).status == status


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (NoFees(), "fail")])
def test_fees_must_match_the_cost_model(fields: pl.DataFrame, plugin, status) -> None:
    result = plugin.run(fields)
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
def test_the_consumer_must_earn_what_the_labels_say_its_positions_earned(fields, plugin, status) -> None:
    result = plugin.run(fields)
    held = result.positions.join(_cands(fields), on=["t", "entity"], how="inner")
    labels = PairCarry(flow="funding_rate").values(fields, held, pd.Timedelta(hours=1))
    check = pr.matches_labels(result, labels, HOUR_NS)
    assert check.status == status, check.detail


def test_own_scores_pass_through_and_perturbations_keep_their_share(fields: pl.DataFrame) -> None:
    plugin = ToyCarry()
    base = plugin.run(fields)
    scores = _cands(fields).select("t", "entity", "score")
    assert pr.passthrough(plugin.run, fields, base, scores).status == "pass"
    block = int(pd.Timedelta("24h").value)
    kept = pr.perturb(scores, "drop", 0.2, block, seed=1)
    assert 0.7 < kept.height / scores.height < 0.9
    per_block = kept.with_columns(b=pl.col("t").dt.epoch("ns") // block).group_by("b", "entity").len()
    whole = scores.with_columns(b=pl.col("t").dt.epoch("ns") // block).group_by("b", "entity").len()
    partial = per_block.join(whole, on=["b", "entity"], suffix="_all").filter(pl.col("len") != pl.col("len_all"))
    assert partial.is_empty()
    shuffled = pr.perturb(scores, "shuffle", 0.5, block, seed=1)
    assert shuffled.height == scores.height
    assert sorted(shuffled["score"].to_list()) == sorted(scores["score"].to_list())
    assert (shuffled.sort("t", "entity")["score"] != scores.sort("t", "entity")["score"]).any()


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (Unbalanced(), "fail")])
def test_components_must_add_up_to_net(fields: pl.DataFrame, plugin, status) -> None:
    assert pr.adds_up(plugin.run(fields))[0].status == status


@pytest.mark.parametrize(("plugin", "status"), [(ToyCarry(), "pass"), (LaggedScores(), "fail")])
def test_the_consumer_must_reproduce_itself_from_its_own_scores(fields: pl.DataFrame, plugin, status) -> None:
    scores = _cands(fields).select("t", "entity", "score")
    assert pr.passthrough(plugin.run, fields, plugin.run(fields), scores).status == status
