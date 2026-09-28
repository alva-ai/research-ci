from collections.abc import Iterable
from datetime import datetime

import pandas as pd
import polars as pl

from research_ci.kernel.time.clock import TIME, span_ns, time_series, to_ns


def holding_return(
    fields: pl.DataFrame,
    price_field: str,
    decision_times: Iterable[datetime | pd.Timestamp],
    horizons: Iterable[pd.Timedelta],
) -> pl.DataFrame:
    """``p(t + h) / p(t) - 1`` for every entity, with each price taken as it was known at that moment."""
    prices = _known_prices(fields, price_field)
    starts = pl.DataFrame({"t": time_series([to_ns(t) for t in decision_times])}).join(
        prices.select("entity").unique(), how="cross"
    )
    return _returns(prices, starts, horizons).select("t", "entity", "horizon", "value").sort("t", "entity", "horizon")


def markout(
    fields: pl.DataFrame, price_field: str, events: pl.DataFrame, horizons: Iterable[pd.Timedelta]
) -> pl.DataFrame:
    """``side * (p(t + h) / p(t) - 1)`` in basis points for each event row (columns t, entity, side)."""
    starts = events.select(pl.col("t").cast(TIME), pl.col("entity").cast(pl.String), pl.col("side").cast(pl.Float64))
    out = _returns(_known_prices(fields, price_field), starts, horizons)
    return out.select("t", "entity", "horizon", (pl.col("side") * pl.col("value") * 1e4).alias("value")).sort(
        "t", "entity", "horizon"
    )


def _known_prices(fields: pl.DataFrame, price_field: str) -> pl.DataFrame:
    """Each entity's latest price by event time, as a step function of the time it became known (first vintage)."""
    rows = fields.filter(pl.col("field") == price_field).sort("vintage")
    rows = rows.unique(subset=["entity", "event_time"], keep="first", maintain_order=True)
    rows = rows.sort("available_at", "event_time")
    rows = rows.filter(pl.col("event_time") == pl.col("event_time").cum_max().over("entity"))
    prices = rows.select("entity", pl.col("available_at").alias("known_at"), pl.col("value").alias("price"))
    return prices.unique(subset=["entity", "known_at"], keep="last", maintain_order=True)


def _returns(prices: pl.DataFrame, starts: pl.DataFrame, horizons: Iterable[pd.Timedelta]) -> pl.DataFrame:
    base = _price_at(prices, starts.with_columns(at=pl.col("t")), "p0").drop_nulls("p0")
    frames = []
    for horizon in horizons:
        ns = span_ns(horizon)
        at = base.with_columns(at=pl.col("t") + pl.duration(nanoseconds=ns))
        tag = pl.lit(ns, dtype=pl.Int64).cast(pl.Duration("ns")).alias("horizon")
        frames.append(_price_at(prices, at, "p1").with_columns(tag))
    out = pl.concat(frames)
    return out.with_columns(value=pl.col("p1") / pl.col("p0") - 1.0)


def _price_at(prices: pl.DataFrame, queries: pl.DataFrame, name: str) -> pl.DataFrame:
    latest = prices.select(
        pl.col("known_at").max().over("entity").alias("_last_known"), "entity", "known_at", pl.col("price")
    )
    joined = queries.sort("at").join_asof(
        latest.sort("known_at"), left_on="at", right_on="known_at", by="entity", check_sortedness=False
    )
    covered = pl.col("at") <= pl.col("_last_known")
    return joined.with_columns(pl.when(covered).then(pl.col("price")).otherwise(None).alias(name)).drop(
        "price", "known_at", "_last_known", "at"
    )
