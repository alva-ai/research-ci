"""``ShiftInTime``: one offset per shuffle, the same for every inst, never closer than ``min_shift`` either way round;
a request that asks for more shuffles than there are offsets draws each once and says so; events move with their
gaps, and a target judges only the rows that hold a score after the shift."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from research_ci.api import Null, ShiftInTime, Universe
from research_ci.api.contract import ContractError
from research_ci.eval.statistics.ic import ts_ic
from research_ci.eval.synthetic import HOUR_NS
from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.judge import Request, judge_all, with_scores
from research_ci.kernel.verdict.shift import Shift
from research_ci.kernel.verdict.worlds import floor_world, lanes
from research_ci.reference_plugins.costs import FlatCosts
from research_ci.reference_plugins.targets import ClusterMean


def _series(hours: int = 50, insts: int = 3, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.tile(np.arange(hours, dtype=np.int64) * HOUR_NS, insts)
    frame = pl.DataFrame(
        {
            "t": t,
            "inst": np.repeat([f"e{i}" for i in range(insts)], hours),
            "score": rng.standard_normal(hours * insts),
            "y": rng.standard_normal(hours * insts),
        }
    )
    frame = frame.with_columns(pl.col("t").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC").cast(TIME))
    return frame.sample(fraction=1.0, shuffle=True, seed=seed)


def test_each_shuffle_shifts_every_inst_by_one_offset_of_at_least_min_shift() -> None:
    table = _series()
    shift = Shift(table, 5 * HOUR_NS)
    hour = (table["t"].dt.epoch("ns") // HOUR_NS).to_numpy()
    inst = table["inst"].to_numpy()
    offsets = set()
    for seed in range(shift.distinct):
        index = shift.index(seed)
        assert (inst[index] == inst).all()
        moved = set(((hour - hour[index]) % 50).tolist())
        assert len(moved) == 1, "one offset for every row and inst"
        (k,) = moved
        assert 5 <= k <= 45
        offsets.add(k)
    assert shift.distinct == 41 and offsets == set(range(5, 46))
    assert np.array_equal(np.sort(shift.index(0)), np.arange(table.height)), "every score lands once"


def test_a_short_inst_bounds_the_offsets_and_too_short_says_why() -> None:
    table = _series().filter((pl.col("inst") != "e1") | (pl.col("t") < pl.col("t").min() + pl.duration(hours=40)))
    assert Shift(table, 5 * HOUR_NS).distinct == 40 - 2 * 5 + 1
    with pytest.raises(ValueError, match="needs every inst to have 60 or more, and the shortest has 50"):
        Shift(_series(), 30 * HOUR_NS).index(0)


def _mean(table: pl.DataFrame) -> pl.DataFrame:
    return table.group_by("t").agg((pl.col("score") * pl.col("y")).mean().alias("value"))


def test_fewer_offsets_than_shuffles_are_each_drawn_once_and_reported() -> None:
    table = _series(hours=30)
    world = floor_world("inst", 200, "series")
    request = Request(_mean, lanes({"4h": (table, 4 * HOUR_NS)}, world), 200, 2, "max")
    (result,) = judge_all([request])
    assert isinstance(result, dict) and result["shuffles"] == result["distinct"] == 30 - 2 * 4 + 1


def test_events_move_with_their_gaps_and_only_the_scored_rows_count() -> None:
    table = _series(hours=40, insts=2).sort("inst", "t")
    events = table.with_columns(score=pl.when(pl.col("t").dt.hour() % 7 == 0).then(1.0).otherwise(float("nan")))
    index = Shift(events, 3 * HOUR_NS).index(1)
    shifted = with_scores(events, events["score"].to_numpy()[index])
    assert shifted["score"].is_nan().sum() == events["score"].is_nan().sum()
    got = ClusterMean(net=False).reduce(shifted, pd.Timedelta(hours=3), FlatCosts(0, 0))
    want = shifted.filter(pl.col("score").is_finite()).group_by("t").agg(pl.col("y").mean().alias("value")).sort("t")
    assert got.equals(want.select("t", "value"))


def test_ts_ic_over_time_is_spearman() -> None:
    table = _series(hours=60, insts=1, seed=4)
    rho = table.select(pl.corr("score", "y", method="spearman")).item()
    assert ts_ic(table)["value"].mean() == pytest.approx(rho, abs=1e-12)


def test_shapes_and_worlds_are_declared_and_checked() -> None:
    assert Universe("u", clusters="inst").shape == "cross"
    with pytest.raises(ContractError, match="shape is one of cross, series, events"):
        Universe("u", clusters="inst", shape="panel")  # type: ignore[arg-type]
    assert Null("trend", ShiftInTime()).keeps == "inst"
    with pytest.raises(ValueError, match='min_shift is "horizon" or a positive duration'):
        ShiftInTime(min_shift="0h")
