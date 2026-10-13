"""Source tables: what a data plugin gives, one wide table per source at the source's own cadence. A row is one event
of one inst: ``inst``, ``event_time``, ``available_at`` and ``vintage`` (``KEYS``), then the source's value columns,
which share the row's times. A value column's name is unique across the sources an idea reads."""

import re
from collections.abc import Mapping

import pandas as pd
import polars as pl

from research_ci.kernel.time.clock import TIME

KEYS = ("inst", "event_time", "available_at", "vintage")
NAME = re.compile(r"[A-Za-z0-9_]+")


class SchemaError(ValueError):
    pass


def source_tables(tables: Mapping[str, pl.DataFrame | pd.DataFrame]) -> dict[str, pl.DataFrame]:
    """Every source checked and normalized (``source_table``), by name; a value column that two sources hold is an
    error."""
    if not isinstance(tables, Mapping) or not tables:
        raise SchemaError("a data plugin's fetch returns a dict of source name to its table, with one source or more")
    out = {name: source_table(name, tables[name]) for name in sorted(tables)}
    columns = [c for table in out.values() for c in values(table)]
    twice = sorted({c for c in columns if columns.count(c) > 1})
    if twice:
        raise SchemaError(f"value columns must be unique across the sources; {', '.join(twice)} is in more than one")
    return out


def source_table(name: str, table: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """One source normalized: its keys present and not null, its times timezone-aware and in UTC ns, its value columns
    numbers, as floats, no two rows with one inst, event time and vintage, and no revision available before the vintage
    it revises; sorted by ``available_at``."""
    if not NAME.fullmatch(name):
        raise SchemaError(f"a source's name is letters, digits and _, not {name!r}")
    frame = pl.from_pandas(table) if isinstance(table, pd.DataFrame) else table
    if not isinstance(frame, pl.DataFrame):
        raise SchemaError(f"source {name}: a table is a polars or pandas DataFrame, not {type(table).__name__}")
    missing = [c for c in KEYS if c not in frame.columns]
    if missing:
        raise SchemaError(f"source {name}: missing columns {missing}")
    own = values(frame)
    if not own:
        raise SchemaError(f"source {name}: no value columns besides {', '.join(KEYS)}")
    reserved = [c for c in own if c == "t" or c.startswith("age_")]
    if reserved:
        raise SchemaError(f"source {name}: t and age_<source> name columns of wide; rename {', '.join(reserved)}")
    text = [c for c in own if not frame.schema[c].is_numeric()]
    if text:
        raise SchemaError(f"source {name}: value columns are numbers; {', '.join(text)} is not")
    for col in ("event_time", "available_at"):
        dtype = frame.schema[col]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
            raise SchemaError(f"source {name}: {col} must be a timezone-aware datetime column, got {dtype}")
    if frame.select(pl.any_horizontal(pl.col(KEYS).is_null()).any()).item():
        raise SchemaError(f"source {name}: inst, event_time, available_at and vintage must not be null")
    try:
        frame = frame.select(
            pl.col("inst").cast(pl.String),
            pl.col("event_time").dt.convert_time_zone("UTC").cast(TIME),
            pl.col("available_at").dt.convert_time_zone("UTC").cast(TIME),
            pl.col("vintage").cast(pl.Int64),
            *(pl.col(c).cast(pl.Float64) for c in own),
        )
    except pl.exceptions.PolarsError as exc:
        raise SchemaError(f"source {name}: {exc}") from exc
    duplicated, revised_early = _key_checks(frame)
    if duplicated:
        raise SchemaError(f"source {name}: {duplicated} duplicate rows on inst, event_time, vintage")
    if revised_early:
        raise SchemaError(f"source {name}: a revision must not be available before the vintage it revises")
    return frame.sort("available_at", "inst", "event_time", "vintage").with_columns(pl.col("available_at").set_sorted())


def values(table: pl.DataFrame) -> list[str]:
    """A source table's value columns, in its order."""
    return [c for c in table.columns if c not in KEYS]


def columns_of(sources: Mapping[str, pl.DataFrame]) -> dict[str, str]:
    """Each value column with the source that holds it."""
    return {c: name for name, table in sources.items() for c in values(table)}


def _key_checks(frame: pl.DataFrame) -> tuple[int, bool]:
    """How many rows share their key with another, and whether a revision is available before the vintage it revises:
    both from one sort of the key, with inst as codes."""
    keyed = frame.select(
        pl.col("inst").cast(pl.Categorical).to_physical().alias("e"),
        pl.col("event_time").dt.epoch("ns").alias("t"),
        pl.col("vintage").alias("v"),
        pl.col("available_at").dt.epoch("ns").alias("a"),
    ).sort("e", "t", "v")

    def same(columns: str, k: int) -> pl.Expr:
        return pl.all_horizontal(pl.col(c) == pl.col(c).shift(k) for c in columns).fill_null(False)

    found = keyed.select(
        (same("etv", 1) | same("etv", -1)).sum().alias("duplicated"),
        (same("et", 1) & (pl.col("a") < pl.col("a").shift(1))).any().alias("early"),
    )
    return int(found["duplicated"].item()), bool(found["early"].item())
