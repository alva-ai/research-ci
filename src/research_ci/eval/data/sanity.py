"""Sanity of the fields a source declares as levels, such as prices: a level is positive, it changes now and then, and
it does not jump to another level for a moment and back. Each check returns the rows it flags (``entity``,
``field``, ``event_time``). A lasting jump is not flagged here: the baseline calls it a level shift, and admission asks
the independent source about it."""

import numpy as np
import polars as pl

from research_ci.api.methods import Case, Source
from research_ci.eval.data.baseline import events
from research_ci.eval.synthetic import HOUR_NS, START_NS, source

KEYS = ["entity", "field", "event_time"]
BY = ["entity", "field"]


def nonpositive(fields: pl.DataFrame, levels: tuple[str, ...]) -> pl.DataFrame:
    """Events of a declared level at zero or below."""
    return _levels(fields, levels).filter(pl.col("value") <= 0).select(KEYS)


def stale(fields: pl.DataFrame, levels: tuple[str, ...], runs: float) -> pl.DataFrame:
    """Events repeating the value before them in a long run. The run of one value lasts longer than ``runs`` of the
    entity's usual spacings; the first event of the run, the last one that changed, is kept."""
    e = events(_levels(fields, levels))
    new = (pl.col("value") != pl.col("value").shift(1)).fill_null(True)
    e = e.with_columns(run=new.cum_sum().over(BY))
    first = pl.col("event_time").min().over(*BY, "run")
    long = (pl.col("event_time").max().over(*BY, "run") - first).dt.total_nanoseconds() > runs * pl.col("spacing")
    return e.filter(long.fill_null(False) & (pl.col("event_time") > first)).select(KEYS)


def spikes(fields: pl.DataFrame, levels: tuple[str, ...], factor: float, window: int) -> pl.DataFrame:
    """Events of a declared level that jump and come back. An event is ``factor`` or more away from the median of the
    last ``window`` events, and its move does not last: the median of the ``window`` events from this one is back within
    ``factor`` of the one before."""
    level = pl.col("value")
    e = events(_levels(fields, levels)).with_columns(
        before=level.rolling_median(window, min_samples=1).shift(1).over(BY),
        after=level.reverse().rolling_median(window, min_samples=1).reverse().over(BY),
    )
    away = (pl.col("before") > 0) & (level > 0) & _apart(level / pl.col("before"), factor)
    lasting = _apart(pl.col("after") / pl.col("before"), factor)
    return e.filter((away & ~lasting).fill_null(False)).select(KEYS)


def _levels(fields: pl.DataFrame, levels: tuple[str, ...]) -> pl.DataFrame:
    return fields.filter(pl.col("field").is_in(list(levels)))


def _apart(ratio: pl.Expr, k: float) -> pl.Expr:
    return (ratio >= k) | (ratio <= 1 / k)


def _hours(flagged: pl.DataFrame) -> list[tuple[str, int]]:
    t = (pl.col("event_time").dt.epoch("ns") - START_NS) // HOUR_NS
    return [(str(e), int(h)) for e, h in flagged.select("entity", t).sort("entity", "event_time").rows()]


def _set(entity: str, hours: range, value: float | None = None, k: float = 1.0) -> pl.DataFrame:
    """A synthetic source with ``entity``'s prices in ``hours`` set to ``value``, or times ``k``."""
    f = source(Source(), 0)
    t = (pl.col("event_time").dt.epoch("ns") - START_NS) // HOUR_NS
    hit = (pl.col("entity") == entity) & (pl.col("field") == "price") & t.is_between(hours.start, hours.stop - 1)
    new = pl.lit(value) if value is not None else pl.col("value") * k
    return f.with_columns(value=pl.when(hit).then(new).otherwise("value"))


def _held(hours: range) -> pl.DataFrame:
    """e0's price held at its value of the hour before ``hours`` throughout them."""
    f = source(Source(), 0)
    t = (pl.col("event_time").dt.epoch("ns") - START_NS) // HOUR_NS
    price = (pl.col("entity") == "e0") & (pl.col("field") == "price")
    held = f.filter(price & (t == hours.start - 1))["value"][0]
    return f.with_columns(
        value=pl.when(price & t.is_between(hours.start, hours.stop - 1)).then(pl.lit(held)).otherwise("value")
    )


def _flags(f: pl.DataFrame) -> int:
    return nonpositive(f, ("price",)).height + stale(f, ("price",), 24).height + spikes(f, ("price",), 5.0, 24).height


def _clean(seeds: int = 10) -> int:
    return sum(_flags(source(Source(lag=2, revised=0.3), s)) for s in range(seeds))


def _planted(pit: str, seeds: int = 10) -> int:
    """On how many sources a built-in pit planted in prices is flagged as a spike."""
    from research_ci.eval.data.pits import BUILTIN

    planted = (BUILTIN[pit](source(Source(), s), "price", np.random.default_rng(s)) for s in range(seeds))
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
]
