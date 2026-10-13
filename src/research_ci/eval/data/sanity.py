"""Sanity of the value columns a plugin declares as levels, such as prices: a level is positive, it changes now and
then, and it does not jump to another level for a moment and back. Each check reads the sources and returns the cells
it flags (``source``, ``column``, ``inst``, ``event_time``), at the source's own cadence. A lasting jump is not flagged
here: the baseline calls it a level shift, and admission asks the independent source about it."""

from collections.abc import Mapping

import numpy as np
import polars as pl

from research_ci.api.evals import Case, Source
from research_ci.eval import synthetic
from research_ci.eval.data.baseline import CELL, cells, events
from research_ci.eval.synthetic import HOUR_NS, START_NS

FLAGS = ["source", "column", "inst", "event_time"]


def nonpositive(sources: Mapping[str, pl.DataFrame], levels: tuple[str, ...]) -> pl.DataFrame:
    """Events of a declared level at zero or below."""
    return cells(sources, levels).filter(pl.col("value") <= 0).select(FLAGS)


def stale(sources: Mapping[str, pl.DataFrame], levels: tuple[str, ...], runs: float) -> pl.DataFrame:
    """Events repeating the value before them in a long run. The run of one value lasts longer than ``runs`` of the
    column's usual spacings for the inst; the first event of the run, the last one that changed, is kept."""
    e = events(cells(sources, levels), CELL)
    new = (pl.col("value") != pl.col("value").shift(1)).fill_null(True)
    e = e.with_columns(run=new.cum_sum().over(CELL))
    first = pl.col("event_time").min().over(*CELL, "run")
    long = (pl.col("event_time").max().over(*CELL, "run") - first).dt.total_nanoseconds() > runs * pl.col("spacing")
    return e.filter(long.fill_null(False) & (pl.col("event_time") > first)).select(FLAGS)


def spikes(sources: Mapping[str, pl.DataFrame], levels: tuple[str, ...], factor: float, window: int) -> pl.DataFrame:
    """Events of a declared level that jump and come back. An event is ``factor`` or more away from the median of the
    last ``window`` events, and its move does not last: the median of the ``window`` events from this one is back within
    ``factor`` of the one before."""
    level = pl.col("value")
    e = events(cells(sources, levels), CELL).with_columns(
        before=level.rolling_median(window, min_samples=1).shift(1).over(CELL),
        after=level.reverse().rolling_median(window, min_samples=1).reverse().over(CELL),
    )
    away = (pl.col("before") > 0) & (level > 0) & _apart(level / pl.col("before"), factor)
    lasting = _apart(pl.col("after") / pl.col("before"), factor)
    return e.filter((away & ~lasting).fill_null(False)).select(FLAGS)


def _apart(ratio: pl.Expr, k: float) -> pl.Expr:
    return (ratio >= k) | (ratio <= 1 / k)


def _hours(flagged: pl.DataFrame) -> list[tuple[str, int]]:
    t = (pl.col("event_time").dt.epoch("ns") - START_NS) // HOUR_NS
    return [(str(e), int(h)) for e, h in flagged.select("inst", t).sort("inst", "event_time").rows()]


def _set(inst: str, hours: range, value: float | None = None, k: float = 1.0) -> dict[str, pl.DataFrame]:
    """Synthetic sources with ``inst``'s prices in ``hours`` set to ``value``, or times ``k``."""
    tables = synthetic.sources(Source(), 0)
    t = (pl.col("event_time").dt.epoch("ns") - START_NS) // HOUR_NS
    hit = (pl.col("inst") == inst) & t.is_between(hours.start, hours.stop - 1)
    new = pl.lit(value) if value is not None else pl.col("price") * k
    return {**tables, "prices": tables["prices"].with_columns(price=pl.when(hit).then(new).otherwise("price"))}


def _held(hours: range) -> dict[str, pl.DataFrame]:
    """e0's price held at its value of the hour before ``hours`` throughout them."""
    tables = synthetic.sources(Source(), 0)
    prices = tables["prices"]
    t = (pl.col("event_time").dt.epoch("ns") - START_NS) // HOUR_NS
    e0 = pl.col("inst") == "e0"
    held = prices.filter(e0 & (t == hours.start - 1))["price"][0]
    kept = pl.when(e0 & t.is_between(hours.start, hours.stop - 1)).then(pl.lit(held)).otherwise("price")
    return {**tables, "prices": prices.with_columns(price=kept)}


def _flags(f: Mapping[str, pl.DataFrame]) -> int:
    return nonpositive(f, ("price",)).height + stale(f, ("price",), 24).height + spikes(f, ("price",), 5.0, 24).height


def _clean(seeds: int = 10) -> int:
    return sum(_flags(synthetic.sources(Source(lag=2, revised=0.3), s)) for s in range(seeds))


def _planted(pit: str, seeds: int = 10) -> int:
    """On how many sources a built-in pit planted in prices is flagged as a spike."""
    from research_ci.eval.data.pits import BUILTIN

    planted = (BUILTIN[pit](synthetic.sources(Source(), s), "price", np.random.default_rng(s)) for s in range(seeds))
    return sum(spikes(f, ("price",), 5.0, 24).height == 1 for f in planted)


CASES = [
    Case("clean sources with lags and revisions flag nothing", _clean, 0),
    Case(
        "prices at zero and below are flagged",
        lambda: _hours(nonpositive(_set("e0", range(5, 8), 0.0), ("price",))),
        [("e0", 5), ("e0", 6), ("e0", 7)],
    ),
    Case(
        "a level that is not declared is not checked", lambda: nonpositive(_set("e0", range(5, 8), 0.0), ()).height, 0
    ),
    Case(
        "a price held for 30 hours: its repeats are stale",
        lambda: len(_hours(stale(_held(range(100, 130)), ("price",), 24))),
        30,
    ),
    Case("held for 20 hours, it is not", lambda: stale(_held(range(100, 120)), ("price",), 24).height, 0),
    Case(
        "a one-hour tenfold spike is flagged, the hour it is back is not",
        lambda: _hours(spikes(_set("e0", range(100, 101), k=10.0), ("price",), 5.0, 24)),
        [("e0", 100)],
    ),
    Case(
        "a three-hour spike is flagged in every hour",
        lambda: len(_hours(spikes(_set("e0", range(100, 103), k=0.1), ("price",), 5.0, 24))),
        3,
    ),
    Case(
        "a lasting tenfold move is a shift, not a spike",
        lambda: spikes(_set("e0", range(100, 337), k=10.0), ("price",), 5.0, 24).height,
        0,
    ),
    Case("the built-in spike pit is flagged, on every source", lambda: _planted("spike"), 10),
    Case(
        "a flagged cell names its source and column",
        lambda: nonpositive(_set("e0", range(5, 6), 0.0), ("price",)).select("source", "column").rows(),
        [("prices", "price")],
    ),
]
