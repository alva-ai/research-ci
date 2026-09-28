from collections.abc import Iterable
from datetime import datetime

import pandas as pd
import polars as pl

TIME = pl.Datetime("ns", "UTC")


def to_ns(t: datetime | pd.Timestamp) -> int:
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware: {t}")
    return int(ts.tz_convert("UTC").value)


def span_ns(delta: pd.Timedelta) -> int:
    return int(pd.Timedelta(delta).value)


def lit_time(ns: int) -> pl.Expr:
    return pl.lit(ns, dtype=pl.Int64).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")


def time_series(ns: Iterable[int], name: str = "t") -> pl.Series:
    return pl.Series(name, list(ns), dtype=pl.Int64).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")


def from_ns(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, tz="UTC")
