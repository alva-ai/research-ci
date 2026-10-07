"""Built-in pits: generic ways to break one field of one entity in a copy of a field table. Each picks, at random, an
entity whose field actually varies, so a pit never lands where it changes nothing."""

from collections.abc import Callable

import numpy as np
import polars as pl

from research_ci.kernel.checks.baseline import events

Plant = Callable[[pl.DataFrame, str, np.random.Generator], pl.DataFrame]
NS = pl.Duration("ns")


def target(fields: pl.DataFrame, field: str, rng: np.random.Generator, exclude: str | None = None) -> str:
    """An entity with at least 20 rows of ``field`` and more than two distinct values."""
    stats = fields.filter(pl.col("field") == field).group_by("entity").agg(pl.len(), pl.col("value").n_unique())
    names = stats.filter((pl.col("len") >= 20) & (pl.col("value") > 2))["entity"].sort().to_list()
    names = [n for n in names if n != exclude]
    if not names:
        raise ValueError(f"no entity where {field} varies enough to plant a pit")
    return str(names[int(rng.integers(len(names)))])


def _rows(fields: pl.DataFrame, entity: str, field: str) -> pl.Expr:
    return (pl.col("entity") == entity) & (pl.col("field") == field)


def _scale(fields: pl.DataFrame, field: str, rng: np.random.Generator, k: float, after: bool) -> pl.DataFrame:
    e = target(fields, field, rng)
    hit = _rows(fields, e, field)
    if after:
        times = fields.filter(hit)["event_time"].sort()
        hit = hit & (pl.col("event_time") >= times[int(len(times) * rng.uniform(0.3, 0.7))])
    return fields.with_columns(value=pl.when(hit).then(pl.col("value") * k).otherwise("value"))


def unit(fields: pl.DataFrame, field: str, rng: np.random.Generator) -> pl.DataFrame:
    return _scale(fields, field, rng, 1000.0, after=False)


def level_break(fields: pl.DataFrame, field: str, rng: np.random.Generator) -> pl.DataFrame:
    return _scale(fields, field, rng, 1.5, after=True)


def gap(fields: pl.DataFrame, field: str, rng: np.random.Generator) -> pl.DataFrame:
    e = target(fields, field, rng)
    ev = events(fields.filter(_rows(fields, e, field)))
    spacing = int(ev["spacing"].drop_nulls()[0])
    times = ev["event_time"]
    start = times[int(len(times) * rng.uniform(0.3, 0.7))]
    window = pl.col("event_time").is_between(
        start, start + pl.duration(nanoseconds=10 * spacing, time_unit="ns"), closed="left"
    )
    return fields.filter(~(_rows(fields, e, field) & window))


def duplicate(fields: pl.DataFrame, field: str, rng: np.random.Generator) -> pl.DataFrame:
    e = target(fields, field, rng)
    rows = fields.filter(_rows(fields, e, field))
    copy = rows.sample(1, seed=int(rng.integers(2**31))).with_columns(
        (pl.col(c) + pl.duration(milliseconds=1, time_unit="ns")).alias(c) for c in ("event_time", "available_at")
    )
    return pl.concat([fields, copy])


def shift(fields: pl.DataFrame, field: str, rng: np.random.Generator) -> pl.DataFrame:
    e = target(fields, field, rng)
    spacing = int(events(fields.filter(_rows(fields, e, field)))["spacing"].drop_nulls()[0])
    hit = _rows(fields, e, field)
    move = pl.duration(nanoseconds=spacing, time_unit="ns")
    return fields.with_columns(
        event_time=pl.when(hit).then(pl.col("event_time") + move).otherwise("event_time"),
        available_at=pl.when(hit).then(pl.col("available_at") + move).otherwise("available_at"),
    )


def leak(fields: pl.DataFrame, field: str, rng: np.random.Generator) -> pl.DataFrame:
    e = target(fields, field, rng)
    hit = _rows(fields, e, field)
    rows = fields.filter(hit).sort("event_time").with_columns(pl.col("value").shift(-1)).drop_nulls("value")
    return pl.concat([fields.filter(~hit), rows])


def spike(fields: pl.DataFrame, field: str, rng: np.random.Generator) -> pl.DataFrame:
    e = target(fields, field, rng)
    hit = _rows(fields, e, field)
    times = fields.filter(hit)["event_time"].sort()
    at = times[int(len(times) * rng.uniform(0.3, 0.7))]
    return fields.with_columns(
        value=pl.when(hit & (pl.col("event_time") == at)).then(pl.col("value") * 10).otherwise("value")
    )


def swap(fields: pl.DataFrame, field: str, rng: np.random.Generator) -> pl.DataFrame:
    """One entity's series replaced by another entity's at the same times, rescaled to start at the same level: a
    ticker reused for a different asset, which a check on levels alone will not see."""
    a = target(fields, field, rng)
    b = target(fields, field, rng, exclude=a)
    mine = fields.filter(_rows(fields, a, field))
    theirs = fields.filter(_rows(fields, b, field)).select("event_time", pl.col("value").alias("other"))
    joined = mine.join(theirs, on="event_time", how="inner").sort("event_time")
    if joined.is_empty():
        raise ValueError(f"{a} and {b} share no times of {field}")
    k = joined["value"][0] / joined["other"][0] if joined["other"][0] else 1.0
    moved = joined.with_columns(value=pl.col("other") * k).drop("other")
    return pl.concat([fields.filter(~_rows(fields, a, field)), moved.select(fields.columns)])


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
