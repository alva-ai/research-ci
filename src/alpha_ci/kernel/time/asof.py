from datetime import datetime

import pandas as pd
import polars as pl

from alpha_ci.kernel.tables import PANEL_KEYS
from alpha_ci.kernel.time.clock import TIME, lit_time, span_ns, time_series, to_ns


def asof(
    fields: pl.DataFrame,
    t: datetime | pd.Timestamp | int,
    lookback: pd.Timedelta | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Wide panel of everything known at ``t``: columns time, entity, then one column per field.

    Only rows with ``available_at <= t`` are visible, and each value is the latest vintage published by ``t``.
    """
    cutoff = t if isinstance(t, int) else to_ns(t)
    available = fields["available_at"]
    if available.flags["SORTED_ASC"]:
        visible = int(available.search_sorted(time_series([cutoff], "available_at"), side="right").item())
        known = fields.slice(0, visible)
    else:
        known = fields.filter(pl.col("available_at") <= lit_time(cutoff))
    if lookback is not None:
        known = known.filter(pl.col("event_time") > lit_time(cutoff - span_ns(lookback)))
    columns = field_names(fields) if columns is None else columns
    if known.is_empty():
        return pl.DataFrame(schema={"time": TIME, "entity": pl.String, **dict.fromkeys(columns, pl.Float64)})
    latest = known.sort("vintage").unique(subset=["entity", "field", "event_time"], keep="last", maintain_order=True)
    panel = latest.pivot(on="field", index=["event_time", "entity"], values="value").rename({"event_time": "time"})
    panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(c) for c in columns if c not in panel.columns)
    return panel.select(*PANEL_KEYS, *columns).sort(PANEL_KEYS)


def field_names(fields: pl.DataFrame) -> list[str]:
    return sorted(fields["field"].unique().to_list())
