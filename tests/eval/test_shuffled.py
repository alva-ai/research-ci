"""The batched forms give their statistics' numbers on every shuffle, to 1e-12, on tables with ties, missing and NaN
values, rows out of time order, and missing clusters; the judge draws each shuffle of a lane once for every request
that reads it, and uses a form only when it gives its statistic's numbers."""

import math
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import polars as pl
import pytest

from research_ci.api.evals import Market
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.statistics.shuffled import QuantileSpread, RankCorrelation
from research_ci.eval.statistics.spread import quantile_spread
from research_ci.eval.structure.filters import kept_without_top, without_top
from research_ci.eval.synthetic import HOUR_NS, market
from research_ci.kernel.verdict.judge import Index, Lane, Request, Statistic, judge_all
from research_ci.kernel.verdict.permute import Permutation
from research_ci.reference_plugins.costs import FlatCosts
from research_ci.reference_plugins.targets import ClusterMean

COSTS = FlatCosts(1, 1, 2)
HOUR = pd.Timedelta(hours=1)


def _target(top: float, net: bool) -> Statistic:
    plugin = ClusterMean("cluster", net, top)
    form = plugin.batched(HOUR, COSTS)
    assert form is not None
    return Statistic(lambda table: plugin.reduce(table, HOUR, COSTS), form)


STATISTICS = {
    "rank ic": Statistic(rank_ic, RankCorrelation()),
    "rank ic without top": Statistic(
        lambda t: rank_ic(without_top(t, "cluster", 0.2)),
        RankCorrelation(keep=lambda t: kept_without_top(t, "cluster", 0.2)),
    ),
    "top minus bottom fifth": Statistic(lambda t: quantile_spread(t, 0.2), QuantileSpread(0.2)),
    "top minus bottom third": Statistic(lambda t: quantile_spread(t, 0.34), QuantileSpread(0.34)),
    "ref.mean, top 0.3, net": _target(0.3, True),
    "ref.mean, top 0.5, gross": _target(0.5, False),
}


def _tables(seed: int) -> Iterator[tuple[str, pl.DataFrame]]:
    rng = np.random.default_rng(seed)
    table = market(Market(times=30, insts=25, clusters=7, ic=0.2), seed)
    flip = pl.Series(rng.random(table.height))
    yield "plain", table
    yield "ties", table.with_columns(pl.col("score").round(0), pl.col("y").round(1))
    yield "heavy ties", table.with_columns((pl.col("score") > 0).cast(pl.Float64), pl.col("y").round(0))
    score = pl.when(flip < 0.1).then(None).otherwise(pl.col("score"))
    yield "missing", table.with_columns(score, y=pl.when(flip > 0.93).then(None).otherwise(pl.col("y")))
    yield (
        "NaN scores",
        table.with_columns(score=pl.when(flip < 0.05).then(float("nan")).otherwise(pl.col("score").round(1))),
    )
    yield "NaN outcomes", table.with_columns(y=pl.when(flip < 0.02).then(float("nan")).otherwise(pl.col("y")))
    yield "rows out of order", table.filter(flip < 0.5).sample(fraction=1.0, shuffle=True, seed=seed)
    yield "missing clusters", table.with_columns(cluster=pl.when(flip < 0.1).then(None).otherwise(pl.col("cluster")))
    yield (
        "a time without scores",
        table.with_columns(score=pl.when(pl.col("t") == pl.col("t").min()).then(None).otherwise(pl.col("score"))),
    )


def _lanes(table: pl.DataFrame, bucket: str | None) -> dict[str, Lane]:
    return {
        h: Lane(table, h_ns, (bucket, h), _shuffle(table, bucket, h_ns))
        for h, h_ns in (("1h", HOUR_NS), ("4h", 4 * HOUR_NS))
    }


def _shuffle(table: pl.DataFrame, bucket: str | None, h_ns: int) -> Callable[[], Index]:
    return lambda: Permutation(table, bucket, "cluster", h_ns).index


def _close(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= 1e-12 * max(1.0, abs(b))


@pytest.mark.parametrize("seed", [0, 1])
def test_judging_with_the_forms_gives_the_numbers_of_judging_shuffle_by_shuffle(seed: int) -> None:
    for name, table in _tables(seed):
        for bucket in (None, "group"):
            requests = [Request(f, _lanes(table, bucket), 25, 3, "max") for f in STATISTICS.values()]
            batched = judge_all(requests, workers=2)
            one_by_one = judge_all(requests, workers=2, batched=False)
            for statistic, a, b in zip(STATISTICS, batched, one_by_one, strict=True):
                assert isinstance(a, dict) and isinstance(b, dict), (name, statistic)
                for x, y in zip(a["horizons"], b["horizons"], strict=True):
                    for key in ("real", "null_mean", "null_sd", "z"):
                        assert _close(x[key], y[key]), (name, statistic, bucket, x["h"], key, x[key], y[key])


def test_every_form_gives_its_statistic_at_every_time_of_a_shuffle() -> None:
    table = market(Market(times=20, insts=25, clusters=7, ic=0.2), 9).with_columns(pl.col("score").round(0))
    perm = Permutation(table, None, "cluster", HOUR_NS)
    index = np.stack([perm.index(s) for s in range(5)])
    score = table["score"].to_numpy()
    for name, statistic in STATISTICS.items():
        assert statistic.batched is not None
        prepared = statistic.batched(table)
        found = prepared(index)
        for row, shuffle in zip(found, index, strict=True):
            shuffled = statistic(table.with_columns(pl.Series("score", score[shuffle])))
            expected = dict(zip(shuffled["t"].dt.epoch("ns").to_list(), shuffled["value"].to_list(), strict=True))
            for t, value in zip(prepared.times.tolist(), row.tolist(), strict=True):
                assert _close(value, expected.get(t, math.nan)), (name, t, value, expected.get(t))


def test_each_shuffle_of_a_lane_is_drawn_once_for_every_request_that_reads_it() -> None:
    table = market(Market(times=30, insts=25, clusters=7), 0)
    drawn: list[int] = []
    perm = Permutation(table, None, "cluster", HOUR_NS)

    def counted(seed: int) -> np.ndarray:
        drawn.append(seed)
        return perm.index(seed)

    lane = Lane(table, HOUR_NS, "floor", lambda: counted)
    requests = [Request(f, {"1h": lane}, 40, 3, "max") for f in STATISTICS.values()]
    results = judge_all(requests, workers=2)
    assert all(isinstance(r, dict) and r["shuffles"] == 40 for r in results)
    assert sorted(drawn) == list(range(40))


def test_a_form_that_does_not_give_its_statistic_is_not_used() -> None:
    table = market(Market(times=30, insts=25, clusters=7, ic=0.2), 0)
    wrong = Statistic(rank_ic, QuantileSpread(0.2))
    lanes = _lanes(table, None)
    found = judge_all([Request(wrong, lanes, 25, 3, "max")])[0]
    expected = judge_all([Request(rank_ic, lanes, 25, 3, "max")])[0]
    assert isinstance(found, dict) and isinstance(expected, dict)
    assert [h["null_mean"] for h in found["horizons"]] == [h["null_mean"] for h in expected["horizons"]]
