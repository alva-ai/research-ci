"""Checks any field table must pass, whatever it holds. What they flag is not judged here: the anomalies are taken
to an independent source first, which says whether the data is faithful or wrong."""

import polars as pl

from research_ci.kernel.judge.protocol import Check
from research_ci.kernel.time.clock import TIME

ANOMALY_SCHEMA = {"kind": pl.String, "entity": pl.String, "field": pl.String, "start": TIME, "end": TIME}


def first_vintages(fields: pl.DataFrame) -> pl.DataFrame:
    """One row per event: the first vintage, as a decision at the time would have seen it."""
    return (
        fields.sort("vintage")
        .unique(["entity", "field", "event_time"], keep="first")
        .sort("entity", "field", "event_time")
    )


def events(fields: pl.DataFrame) -> pl.DataFrame:
    """First vintages with the spacing to the previous event of the same entity and field, and its typical size."""
    gap = pl.col("event_time").diff().over("entity", "field").dt.total_nanoseconds()
    return (
        first_vintages(fields)
        .with_columns(prev=pl.col("event_time").shift(1).over("entity", "field"), gap=gap)
        .with_columns(spacing=pl.col("gap").median().over("entity", "field"), n=pl.len().over("entity", "field"))
    )


def baseline(
    fields: pl.DataFrame,
    levels: tuple[str, ...] = (),
    shift_ratio: float = 0.0,
    window: int = 0,
    gap_factor: float = 3.0,
    near_ns: int = 60 * 10**9,
) -> tuple[list[Check], pl.DataFrame]:
    """Generic checks, and the anomalies they flag: long gaps between events and events closer than ``near_ns``; and,
    in the fields the source declares as ``levels``, shifts of ``shift_ratio`` or more that persist over the next
    ``window`` events."""
    e = events(fields)
    checks = []
    bad = fields.filter(pl.col("value").is_null() | ~pl.col("value").is_finite()).height
    checks.append(_count("values are finite", bad, "values null, NaN or infinite"))
    early = fields.filter(pl.col("available_at") < pl.col("event_time")).height
    checks.append(_count("known no earlier than the event", early, "rows known before their event time"))
    gaps = e.filter((pl.col("n") >= 3) & (pl.col("gap") > gap_factor * pl.col("spacing")))
    near = e.filter(pl.col("gap") < near_ns)
    checks.append(_count("no long gaps", gaps.height, f"gaps longer than {gap_factor:g}× the usual spacing", gaps))
    checks.append(
        _count("no near-duplicate events", near.height, f"events within {near_ns / 1e9:g}s of the last", near)
    )
    found = [_anomalies(gaps, "gap"), _anomalies(near, "near")]
    if levels:
        level = pl.col("value")
        e = e.filter(pl.col("field").is_in(list(levels))).with_columns(
            before=level.shift(1).over("entity", "field"),
            after=level.rolling_median(window).shift(-(window - 1)).over("entity", "field"),
        )
        big = _ratio("value", "before", shift_ratio) & _ratio("after", "before", shift_ratio)
        shifts = e.filter((pl.col("before") > 0) & (pl.col("value") > 0) & big)
        what = f"lasting moves of {shift_ratio:g}× or more"
        checks.append(_count("no level shifts", shifts.height, what, shifts))
        found.append(_anomalies(shifts, "shift"))
    else:
        checks.append(Check("no level shifts", "skip", "the source declares no level fields"))
    return checks, pl.concat(found)


def _ratio(a: str, b: str, k: float) -> pl.Expr:
    r = pl.col(a) / pl.col(b)
    return (r >= k) | (r <= 1 / k)


def _anomalies(rows: pl.DataFrame, kind: str) -> pl.DataFrame:
    return rows.select(
        pl.lit(kind).alias("kind"), "entity", "field", pl.col("prev").alias("start"), pl.col("event_time").alias("end")
    ).cast(ANOMALY_SCHEMA)  # type: ignore[arg-type]


def _count(name: str, n: int, what: str, rows: pl.DataFrame | None = None) -> Check:
    if n == 0:
        return Check(name, "pass", f"none: {what}")
    where = ""
    if rows is not None and rows.height:
        top = rows.group_by("field").len().sort("len", descending=True).head(3)
        where = " (" + ", ".join(f"{f} {k}" for f, k in top.iter_rows()) + ")"
    return Check(name, "warn", f"{n:,} {what}{where}")
