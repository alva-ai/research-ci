"""Identity breaks: a declared level that comes back after a long gap at a level that does not match the one before it
is another asset under the same name, such as a delisted ticker relaunched for something else. The venue reports
both, so admission finds the rows faithful; they are still two series, and nothing measured across the break means
anything. The treatment splits the entity there: its rows from the break on become ``<entity>#2`` (``#3`` after a
second break), and a label of the old series stops where the old series ends."""

from collections.abc import Collection

import numpy as np
import polars as pl

from research_ci.api.methods import Case, Source
from research_ci.eval.data.baseline import events
from research_ci.eval.synthetic import HOUR_NS, START_NS, source
from research_ci.kernel.time.clock import TIME

BREAK_SCHEMA = {"entity": pl.String, "last": TIME, "at": TIME, "ratio": pl.Float64, "fields": pl.List(pl.String)}


def breaks(
    fields: pl.DataFrame, levels: tuple[str, ...], factor: float, window: int, gap_factor: float
) -> pl.DataFrame:
    """Where a declared level comes back, after a gap longer than ``gap_factor`` of the entity's usual spacings (the
    baseline's long gap), at ``factor`` or more away from where it was: the median of the ``window`` events from the
    return against the median of the ``window`` events before the gap. One row per entity and gap, however many
    fields show it: ``entity``, ``last`` (the last event before the gap), ``at`` (the first after it), ``ratio`` (the
    level after over the level before, the largest move among the fields) and ``fields``."""
    if not levels:
        return pl.DataFrame(schema=BREAK_SCHEMA)
    level, by = pl.col("value"), ["entity", "field"]
    e = events(fields.filter(pl.col("field").is_in(list(levels)))).with_columns(
        before=level.rolling_median(window, min_samples=1).shift(1).over(by),
        after=level.reverse().rolling_median(window, min_samples=1).reverse().over(by),
    )
    long = (pl.col("n") >= 3) & (pl.col("gap") > gap_factor * pl.col("spacing"))
    ratio = pl.col("after") / pl.col("before")
    far = (pl.col("before") > 0) & (pl.col("after") > 0) & ((ratio >= factor) | (ratio <= 1 / factor))
    hits = e.filter((long & far).fill_null(False)).select(
        "entity", "field", pl.col("prev").alias("last"), pl.col("event_time").alias("at"), ratio.alias("ratio")
    )
    hits = hits.sort("entity", "last").with_columns(reach=pl.col("at").cum_max().shift(1).over("entity"))
    apart = pl.col("reach").is_null() | (pl.col("last") > pl.col("reach"))
    hits = hits.with_columns(apart=apart).with_columns(gap=pl.col("apart").cum_sum().over("entity"))
    widest = pl.col("ratio").sort_by(pl.col("ratio").log().abs(), descending=True).first()
    merged = hits.group_by("entity", "gap").agg(
        pl.col("last").max(), pl.col("at").min(), widest.alias("ratio"), pl.col("field").unique().sort().alias("fields")
    )
    return merged.select(list(BREAK_SCHEMA)).sort("entity", "at").cast(BREAK_SCHEMA)  # type: ignore[arg-type]


def renames(found: pl.DataFrame, taken: pl.Series | Collection[str]) -> pl.DataFrame:
    """What the split calls each new series: ``entity``, ``at`` and ``to``. The k-th break of ``E`` starts
    ``E#{k+1}``; an entity already named ``E#n`` continues at ``E#{n+1}``; names already in use are skipped."""
    schema = {"entity": pl.String, "at": TIME, "to": pl.String}
    if found.is_empty():
        return pl.DataFrame(schema=schema)
    names = pl.DataFrame({"to": pl.Series(list(taken) if not isinstance(taken, pl.Series) else taken, dtype=pl.String)})
    parts = pl.col("entity").str.extract_groups(r"^(.*)#(\d+)$")
    breaks = (
        found.select("entity", "at")
        .sort("entity", "at")
        .with_columns(
            base=pl.coalesce(parts.struct.field("1"), pl.col("entity")),
            n=pl.coalesce(parts.struct.field("2").cast(pl.Int64), pl.lit(1)),
            k=pl.int_range(1, pl.len() + 1).over("entity"),
        )
    )
    used = names.select(base=pl.col("to").str.extract(r"^(.*)#\d+$")).group_by("base").len("used")
    per = breaks.group_by("entity", "base", "n").agg(pl.len().alias("breaks"))
    per = per.join(used, on="base", how="left").with_columns(pl.col("used").fill_null(0))
    free = per.select(
        "entity",
        "base",
        pl.int_ranges(pl.col("n") + 1, pl.col("n") + pl.col("breaks") + pl.col("used") + 1).alias("m"),
    ).explode("m", empty_as_null=False)
    free = free.with_columns(to=pl.format("{}#{}", "base", "m")).join(names, on="to", how="anti")
    free = free.sort("entity", "m").with_columns(k=pl.int_range(1, pl.len() + 1).over("entity"))
    out = breaks.join(free.select("entity", "k", "to"), on=["entity", "k"])
    return out.select("entity", "at", "to").sort("entity", "at").cast(schema)  # type: ignore[arg-type]


def _relaunched(seed: int, k: float = 1000.0, gap: int = 12, at: int = 150) -> pl.DataFrame:
    """e0's price stops after hour ``at`` and comes back ``gap`` hours later times ``k``: another asset."""
    f = source(Source(), seed)
    t = pl.col("event_time").dt.epoch("ns")
    e0 = (pl.col("entity") == "e0") & (pl.col("field") == "price")
    hole = e0 & t.is_between(START_NS + (at + 1) * HOUR_NS, START_NS + (at + gap - 1) * HOUR_NS)
    later = e0 & (t >= START_NS + (at + gap) * HOUR_NS)
    return f.filter(~hole).with_columns(value=pl.when(later).then(pl.col("value") * k).otherwise("value"))


def _found(fields: pl.DataFrame) -> list[tuple[str, int]]:
    """Each break as (entity, hours from the start of the source to the first event after it)."""
    b = breaks(fields, ("price",), 5.0, 24, 3.0)
    return [(str(e), (int(t) - START_NS) // HOUR_NS) for e, t in b.select("entity", pl.col("at").dt.epoch("ns")).rows()]


def _caught(seeds: int = 10, k: float = 1000.0, gap: int = 12) -> int:
    """On how many sources exactly the planted break is found."""
    return sum(_found(_relaunched(s, k, gap)) == [("e0", 150 + gap)] for s in range(seeds))


def _any(seeds: int = 10, k: float = 1000.0, gap: int = 12) -> int:
    """On how many sources any break is found."""
    return sum(bool(_found(_relaunched(s, k, gap))) for s in range(seeds))


def _gapped(seeds: int = 10) -> int:
    """Gaps of the same asset, the built-in gap pit planted in prices: never a break."""
    from research_ci.eval.data.pits import gap

    return sum(bool(_found(gap(source(Source(), s), "price", np.random.default_rng(s)))) for s in range(seeds))


def _named() -> list[tuple[str, str]]:
    found = pl.DataFrame(
        {"entity": ["a", "a", "b#2"], "at": [2 * HOUR_NS, HOUR_NS, HOUR_NS]},
        schema={"entity": pl.String, "at": pl.Int64},
    ).with_columns(pl.col("at").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC"))
    return [(e, to) for e, _, to in renames(found, {"a#3"}).rows()]


CASES = [
    Case("clean sources have no identity break", lambda: sum(len(_found(source(Source(), s))) for s in range(10)), 0),
    Case("a level that comes back a thousandfold after a gap is another asset, on every source", _caught, 10),
    Case("one that comes back a thousand times lower too", lambda: _caught(k=1e-3), 10),
    Case("a gap after which the level carries on is the same asset, on every source", _gapped, 0),
    Case("a level that moves tenfold without a gap is a shift, not a break", lambda: _any(gap=1, k=10.0), 0),
    Case("a gap of two spacings is not long", lambda: _any(gap=2), 0),
    Case(
        "new series are numbered from 2, after a name's own number, past names in use",
        _named,
        [("a", "a#2"), ("a", "a#4"), ("b#2", "b#3")],
    ),
]
