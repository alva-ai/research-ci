"""Built-in pits: generic ways to break one value column of one inst in copies of the sources. Each picks, at random,
an inst whose column actually varies, so a pit never lands where it changes nothing. A pit in a column changes the
cells of that column, or, for a gap, a duplicate or a shift in time, the whole rows of its source."""

from collections.abc import Callable, Mapping

import numpy as np
import polars as pl

from research_ci.api.evals import Case, Source
from research_ci.eval.data.baseline import events
from research_ci.eval.synthetic import sources
from research_ci.kernel.tables import columns_of

Sources = Mapping[str, pl.DataFrame]
Plant = Callable[[Sources, str, np.random.Generator], dict[str, pl.DataFrame]]


def holder(tables: Sources, column: str) -> str:
    """The source that holds ``column``."""
    found = columns_of(tables).get(column)
    if found is None:
        raise ValueError(f"no source holds a column {column}; the columns are {', '.join(columns_of(tables))}")
    return found


def target(tables: Sources, column: str, rng: np.random.Generator, exclude: str | None = None) -> str:
    """An inst whose ``column`` has enough values for a pit. It has at least 20, and more than two distinct ones."""
    table = tables[holder(tables, column)]
    stats = table.filter(pl.col(column).is_not_null()).group_by("inst").agg(pl.len(), pl.col(column).n_unique())
    other = pl.col("inst").ne_missing(pl.lit(exclude, dtype=pl.String))
    names = stats.filter((pl.col("len") >= 20) & (pl.col(column) > 2) & other)["inst"].sort()
    if names.is_empty():
        raise ValueError(f"no inst where {column} varies enough to plant a pit")
    return str(names[int(rng.integers(names.len()))])


def _replaced(tables: Sources, column: str, table: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {**tables, holder(tables, column): table}


def _scale(tables: Sources, column: str, rng: np.random.Generator, k: float, after: bool) -> dict[str, pl.DataFrame]:
    table = tables[holder(tables, column)]
    hit = pl.col("inst") == target(tables, column, rng)
    if after:
        times = table.filter(hit)["event_time"].sort()
        hit = hit & (pl.col("event_time") >= times[int(len(times) * rng.uniform(0.3, 0.7))])
    return _replaced(tables, column, table.with_columns(pl.when(hit).then(pl.col(column) * k).otherwise(column)))


def unit(tables: Sources, column: str, rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    """A unit error: one inst's column a thousand times larger. It holds throughout."""
    return _scale(tables, column, rng, 1000.0, after=False)


def level_break(tables: Sources, column: str, rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    """A level break: one inst's column half again as large. It holds from a random time on."""
    return _scale(tables, column, rng, 1.5, after=True)


def gap(tables: Sources, column: str, rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    """A gap: ten usual spacings of one inst's rows missing. They start at a random time, in the column's source."""
    name = holder(tables, column)
    hit = pl.col("inst") == target(tables, column, rng)
    ev = events(tables[name].filter(hit).with_columns(source=pl.lit(name)))
    spacing = int(ev["spacing"].drop_nulls()[0])
    times = ev["event_time"]
    start = times[int(len(times) * rng.uniform(0.3, 0.7))]
    window = pl.col("event_time").is_between(
        start, start + pl.duration(nanoseconds=10 * spacing, time_unit="ns"), closed="left"
    )
    return _replaced(tables, column, tables[name].filter(~(hit & window)))


def duplicate(tables: Sources, column: str, rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    """A duplicate: one row of an inst repeated a millisecond later. The row is one of the column's source."""
    table = tables[holder(tables, column)]
    rows = table.filter(pl.col("inst") == target(tables, column, rng))
    copy = rows.sample(1, seed=int(rng.integers(2**31))).with_columns(
        (pl.col(c) + pl.duration(milliseconds=1, time_unit="ns")).alias(c) for c in ("event_time", "available_at")
    )
    return _replaced(tables, column, pl.concat([table, copy]))


def shift(tables: Sources, column: str, rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    """A shift: one inst's rows stamped one usual spacing late. The rows are those of the column's source."""
    name = holder(tables, column)
    hit = pl.col("inst") == target(tables, column, rng)
    spacing = int(events(tables[name].filter(hit).with_columns(source=pl.lit(name)))["spacing"].drop_nulls()[0])
    move = pl.duration(nanoseconds=spacing, time_unit="ns")
    late = [pl.when(hit).then(pl.col(c) + move).otherwise(c).alias(c) for c in ("event_time", "available_at")]
    return _replaced(tables, column, tables[name].with_columns(late))


def leak(tables: Sources, column: str, rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    """A leak: each event of one inst carries the next one's value. Only in the column; the last carries none."""
    table = tables[holder(tables, column)].sort("inst", "event_time")
    hit = pl.col("inst") == target(tables, column, rng)
    ahead = pl.when(hit).then(pl.col(column).shift(-1).over("inst")).otherwise(column)
    return _replaced(tables, column, table.with_columns(ahead.alias(column)))


def spike(tables: Sources, column: str, rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    """A spike: one event of one inst ten times larger. Only in the column."""
    table = tables[holder(tables, column)]
    hit = pl.col("inst") == target(tables, column, rng)
    times = table.filter(hit)["event_time"].sort()
    at = times[int(len(times) * rng.uniform(0.3, 0.7))]
    spiked = pl.when(hit & (pl.col("event_time") == at)).then(pl.col(column) * 10).otherwise(column)
    return _replaced(tables, column, table.with_columns(spiked))


def swap(tables: Sources, column: str, rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    """A swap: one inst's column replaced by another inst's. The other's values are taken at the same times, rescaled
    to start at the same level: a ticker reused for a different asset, which a check on levels alone will not see."""
    table = tables[holder(tables, column)]
    a = target(tables, column, rng)
    b = target(tables, column, rng, exclude=a)
    theirs = table.filter(pl.col("inst") == b).sort("vintage").unique("event_time", keep="first")
    joined = table.filter(pl.col("inst") == a).join(
        theirs.select("event_time", pl.col(column).alias("_other")), on="event_time", how="left"
    )
    both = joined.drop_nulls([column, "_other"]).sort("event_time")
    if both.is_empty():
        raise ValueError(f"{a} and {b} share no times of {column}")
    k = both[column][0] / both["_other"][0] if both["_other"][0] else 1.0
    moved = joined.with_columns((pl.col("_other") * k).alias(column)).drop("_other")
    return _replaced(tables, column, pl.concat([table.filter(pl.col("inst") != a), moved.select(table.columns)]))


BUILTIN: dict[str, Plant] = {
    "unit": unit,
    "level_break": level_break,
    "gap": gap,
    "duplicate": duplicate,
    "shift": shift,
    "leak": leak,
    "spike": spike,
    "swap": swap,
}


def _touched(name: str, seed: int) -> list[str]:
    """The insts whose rows a pit changed, planted in prices of synthetic sources."""
    clean = sources(Source(), seed)
    planted = BUILTIN[name](clean, "price", np.random.default_rng(seed))
    pairs = [(before, planted[source]) for source, before in clean.items()]
    changed = [a.join(b, on=a.columns, how="anti") for x, y in pairs for a, b in ((x, y), (y, x))]
    return sorted({e for frame in changed for e in frame["inst"]})


CASES = [
    Case(
        "every pit changes exactly one inst, on every source",
        lambda: sorted({name for name in BUILTIN for seed in range(5) if len(_touched(name, seed)) != 1}),
        [],
    ),
]
