import pandas as pd
import polars as pl

from research_ci.kernel.time.clock import TIME

KEY = ["inst", "field", "event_time", "vintage"]
REQUIRED = ["inst", "field", "event_time", "available_at", "value"]
COLUMNS = ["inst", "field", "event_time", "available_at", "value", "vintage", "source"]
PANEL_KEYS = ["time", "inst"]


class SchemaError(ValueError):
    pass


def field_table(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    frame = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if "vintage" not in frame.columns:
        frame = frame.with_columns(vintage=pl.lit(0, dtype=pl.Int64))
    if "source" not in frame.columns:
        frame = frame.with_columns(source=pl.lit(""))
    for col in ("event_time", "available_at"):
        dtype = frame.schema[col]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
            raise SchemaError(f"{col} must be a timezone-aware datetime column, got {dtype}")
    if frame.select(pl.any_horizontal(pl.col(KEY + ["available_at"]).is_null()).any()).item():
        raise SchemaError("inst, field, event_time, available_at and vintage must not be null")
    try:
        frame = frame.select(
            pl.col("inst").cast(pl.String),
            pl.col("field").cast(pl.String),
            pl.col("event_time").dt.convert_time_zone("UTC").cast(TIME),
            pl.col("available_at").dt.convert_time_zone("UTC").cast(TIME),
            pl.col("value").cast(pl.Float64),
            pl.col("vintage").cast(pl.Int64),
            pl.col("source").cast(pl.String),
        )
    except pl.exceptions.PolarsError as exc:
        raise SchemaError(str(exc)) from exc

    duplicated, revised_early = _key_checks(frame)
    if duplicated:
        raise SchemaError(f"{duplicated} duplicate rows on {KEY}")
    if revised_early:
        raise SchemaError("a revision must not be available before the vintage it revises")
    return frame.sort(["available_at", *KEY]).with_columns(pl.col("available_at").set_sorted())


def _key_checks(frame: pl.DataFrame) -> tuple[int, bool]:
    """How many rows share their key with another, and whether a revision is available before the vintage it revises:
    both from one sort of the key, with inst and field as codes."""
    keyed = frame.select(
        pl.col("inst").cast(pl.Categorical).to_physical().alias("e"),
        pl.col("field").cast(pl.Categorical).to_physical().alias("f"),
        pl.col("event_time").dt.epoch("ns").alias("t"),
        pl.col("vintage").alias("v"),
        pl.col("available_at").dt.epoch("ns").alias("a"),
    ).sort("e", "f", "t", "v")

    def same(columns: str, k: int) -> pl.Expr:
        return pl.all_horizontal(pl.col(c) == pl.col(c).shift(k) for c in columns).fill_null(False)

    found = keyed.select(
        (same("eftv", 1) | same("eftv", -1)).sum().alias("duplicated"),
        (same("eft", 1) & (pl.col("a") < pl.col("a").shift(1))).any().alias("early"),
    )
    return int(found["duplicated"].item()), bool(found["early"].item())
