import pandas as pd
import polars as pl

from alpha_ci.kernel.time.clock import TIME

KEY = ["entity", "field", "event_time", "vintage"]
REQUIRED = ["entity", "field", "event_time", "available_at", "value"]
COLUMNS = ["entity", "field", "event_time", "available_at", "value", "vintage", "source"]
PANEL_KEYS = ["time", "entity"]


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
        raise SchemaError("entity, field, event_time, available_at and vintage must not be null")
    try:
        frame = frame.select(
            pl.col("entity").cast(pl.String),
            pl.col("field").cast(pl.String),
            pl.col("event_time").dt.convert_time_zone("UTC").cast(TIME),
            pl.col("available_at").dt.convert_time_zone("UTC").cast(TIME),
            pl.col("value").cast(pl.Float64),
            pl.col("vintage").cast(pl.Int64),
            pl.col("source").cast(pl.String),
        )
    except pl.exceptions.PolarsError as exc:
        raise SchemaError(str(exc)) from exc

    duplicated = int(frame.select(pl.struct(KEY).is_duplicated().sum()).item())
    if duplicated:
        raise SchemaError(f"{duplicated} duplicate rows on {KEY}")
    revised_early = frame.sort(KEY).select(
        (pl.col("available_at").diff().over(["entity", "field", "event_time"]) < pl.duration(nanoseconds=0)).any()
    )
    if revised_early.item():
        raise SchemaError("a revision must not be available before the vintage it revises")
    return frame.sort(["available_at", *KEY]).with_columns(pl.col("available_at").set_sorted())
