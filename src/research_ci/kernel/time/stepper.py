from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from research_ci.kernel.tables import PANEL_KEYS, SchemaError
from research_ci.kernel.time.asof import asof, field_names
from research_ci.kernel.time.clock import TIME, from_ns, lit_time, time_series, to_ns

PanelFormat = Literal["pandas", "polars"]
Features = Callable[[Any], Any]
OUTPUT_KEYS = ["t", "entity"]


@dataclass(frozen=True)
class Mismatch:
    t: pd.Timestamp
    entity: str
    column: str
    stepped: float
    full: float


@dataclass(frozen=True)
class LeakReport:
    kind: Literal["none", "lookahead", "lookback"]
    checked: int
    mismatches: int
    first: Mismatch | None = None
    by_column: dict[str, int] = field(default_factory=dict)


def step(
    features: Features,
    fields: pl.DataFrame,
    decision_times: Iterable[datetime | pd.Timestamp],
    lookback: pd.Timedelta | None = None,
    panel_format: PanelFormat = "pandas",
) -> pl.DataFrame:
    """Call ``features`` once per decision time on the panel known at that time; keep each entity's latest row."""
    return _step_ns(features, fields, [to_ns(t) for t in decision_times], lookback, panel_format)


def check_leaks(
    features: Features,
    fields: pl.DataFrame,
    decision_times: Iterable[datetime | pd.Timestamp],
    lookback: pd.Timedelta | None = None,
    panel_format: PanelFormat = "pandas",
) -> LeakReport:
    """Compare stepping with one full-history run on an availability-timed copy of ``fields``.

    On that copy every row sits at the time it became known, so code that only reads past rows gives identical
    results either way. A remaining difference means the code read rows after ``t``, unless it disappears once the
    lookback is removed, in which case the declared lookback is just too short.
    """
    times = [to_ns(t) for t in decision_times]
    base = _availability_timed(fields)
    full = _full_run(features, base, times, panel_format)
    mismatches = _compare(_step_ns(features, base, times, lookback, panel_format), full)
    if not mismatches:
        return LeakReport(kind="none", checked=len(times), mismatches=0)
    if lookback is not None:
        suspect = sorted({int(m.t.value) for m in mismatches})
        unbounded = _step_ns(features, base, suspect, None, panel_format)
        remaining = _compare(unbounded, full.filter(pl.col("t").is_in(time_series(suspect).implode())))
        if not remaining:
            return _report("lookback", len(times), mismatches)
        mismatches = remaining
    return _report("lookahead", len(times), mismatches)


def _step_ns(
    features: Features,
    fields: pl.DataFrame,
    times: list[int],
    lookback: pd.Timedelta | None,
    panel_format: PanelFormat,
) -> pl.DataFrame:
    columns = field_names(fields)
    frames = [
        _latest_at(_normalize(features(_to_format(asof(fields, t, lookback, columns), panel_format))), t) for t in times
    ]
    return _stack(frames)


def _availability_timed(fields: pl.DataFrame) -> pl.DataFrame:
    keys = ["entity", "field", "event_time"]
    first = fields.sort("vintage").unique(subset=keys, keep="first", maintain_order=True)
    moved = first.sort("event_time").with_columns(event_time=pl.col("available_at"))
    unique = moved.unique(subset=keys, keep="last", maintain_order=True)
    return unique.sort("available_at", maintain_order=True).with_columns(pl.col("available_at").set_sorted())


def _full_run(features: Features, fields: pl.DataFrame, times: list[int], panel_format: PanelFormat) -> pl.DataFrame:
    end = int(fields.select(pl.col("available_at").max().dt.epoch("ns")).item())
    out = _normalize(features(_to_format(asof(fields, end), panel_format)))
    return _stack([_latest_at(out, t) for t in times])


def _to_format(panel: pl.DataFrame, panel_format: PanelFormat) -> pl.DataFrame | pd.DataFrame:
    if panel_format == "polars":
        return panel
    return panel.to_pandas().set_index(PANEL_KEYS)


def _normalize(out: object) -> pl.DataFrame:
    if isinstance(out, pd.Series):
        out = out.to_frame(out.name if out.name is not None else "value")
    if isinstance(out, pd.DataFrame):
        if list(out.index.names) != PANEL_KEYS:
            raise SchemaError(f"features must return rows indexed by {PANEL_KEYS}, got {list(out.index.names)}")
        out = pl.from_pandas(out.reset_index(), nan_to_null=True)
    if not isinstance(out, pl.DataFrame):
        raise SchemaError(f"features must return a pandas Series/DataFrame or a polars DataFrame, got {type(out)}")
    if not set(PANEL_KEYS) <= set(out.columns):
        raise SchemaError(f"features must return the columns {PANEL_KEYS}, got {out.columns}")
    values = [c for c in out.columns if c not in PANEL_KEYS]
    if "t" in values:
        raise SchemaError("features must not return a column named 't'")
    return out.select(
        pl.col("time").dt.convert_time_zone("UTC").cast(TIME),
        pl.col("entity").cast(pl.String),
        *(pl.col(c).cast(pl.Float64).fill_nan(None) for c in values),
    )


def _latest_at(out: pl.DataFrame, t: int) -> pl.DataFrame:
    visible = out.filter(pl.col("time") <= lit_time(t)).sort("time", maintain_order=True)
    latest = visible.group_by("entity", maintain_order=True).last().drop("time")
    return latest.select(lit_time(t).alias("t"), pl.all())


def _stack(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame(schema={"t": TIME, "entity": pl.String})
    return pl.concat(frames, how="diagonal_relaxed").sort(OUTPUT_KEYS)


def _compare(stepped: pl.DataFrame, full: pl.DataFrame) -> list[Mismatch]:
    columns = sorted((set(stepped.columns) | set(full.columns)) - set(OUTPUT_KEYS))
    a = stepped.with_columns(pl.lit(None, dtype=pl.Float64).alias(c) for c in columns if c not in stepped.columns)
    b = full.with_columns(pl.lit(None, dtype=pl.Float64).alias(c) for c in columns if c not in full.columns)
    joined = a.join(b, on=OUTPUT_KEYS, how="full", coalesce=True, suffix="__full").sort(OUTPUT_KEYS)
    ts = joined.select(pl.col("t").dt.epoch("ns")).to_series().to_list()
    entities = joined["entity"].to_list()
    mismatches: list[Mismatch] = []
    for column in columns:
        x = joined[column].to_numpy().astype("float64")
        y = joined[f"{column}__full"].to_numpy().astype("float64")
        same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-9, atol=1e-12)
        for i in np.nonzero(~same)[0].tolist():
            mismatches.append(
                Mismatch(t=from_ns(ts[i]), entity=str(entities[i]), column=column, stepped=x[i], full=y[i])
            )
    return mismatches


def _report(kind: Literal["lookahead", "lookback"], checked: int, mismatches: list[Mismatch]) -> LeakReport:
    by_column: dict[str, int] = {}
    for m in mismatches:
        by_column[m.column] = by_column.get(m.column, 0) + 1
    first = min(mismatches, key=lambda m: (m.t, m.entity, m.column))
    return LeakReport(kind=kind, checked=checked, mismatches=len(mismatches), first=first, by_column=by_column)
