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
    gap_factor: float = 3.0,
    near_ns: int = 60 * 10**9,
    shift_ratio: float = 5.0,
    window: int = 24,
) -> tuple[list[Check], pl.DataFrame]:
    """Generic checks, and the anomalies they flag: long gaps between events, events closer than ``near_ns``, and
    level shifts of ``shift_ratio`` or more that persist over the next ``window`` events."""
    e = events(fields)
    checks = []
    bad = fields.filter(pl.col("value").is_null() | ~pl.col("value").is_finite()).height
    checks.append(_count("values are finite", bad, "values null, NaN or infinite"))
    early = fields.filter(pl.col("available_at") < pl.col("event_time")).height
    checks.append(_count("known no earlier than the event", early, "rows known before their event time"))
    gaps = e.filter((pl.col("n") >= 3) & (pl.col("gap") > gap_factor * pl.col("spacing")))
    near = e.filter(pl.col("gap") < near_ns)
    level = pl.col("value")
    e = e.filter(pl.col("field").is_in(levels(e))).with_columns(
        before=level.shift(1).over("entity", "field"),
        after=level.rolling_median(window).shift(-(window - 1)).over("entity", "field"),
    )
    big = _ratio("value", "before", shift_ratio) & _ratio("after", "before", shift_ratio)
    shifts = e.filter((pl.col("before") > 0) & (pl.col("value") > 0) & big)
    checks.append(_count("no long gaps", gaps.height, f"gaps longer than {gap_factor:g}× the usual spacing", gaps))
    checks.append(
        _count("no near-duplicate events", near.height, f"events within {near_ns / 1e9:g}s of the last", near)
    )
    checks.append(_count("no level shifts", shifts.height, f"lasting moves of {shift_ratio:g}× or more", shifts))
    anomalies = pl.concat(
        [
            _anomalies(gaps, "gap"),
            _anomalies(near, "near"),
            _anomalies(shifts, "shift"),
        ]
    )
    return checks, anomalies


def levels(e: pl.DataFrame, smooth: float = 0.05) -> list[str]:
    """Fields that behave like price levels: positive, never constant, and usually moving by less than ``smooth``
    (relative) from one event to the next. A lasting jump means something only in such a field; volumes, rates and
    step functions such as settlement intervals jump by design."""
    step = (pl.col("value") / pl.col("value").shift(1).over("entity", "field")).log().abs()
    moves = e.filter(pl.col("value") > 0).with_columns(step=step).group_by("field").agg(pl.col("step").median())
    return moves.filter((pl.col("step") > 0) & (pl.col("step") < smooth))["field"].sort().to_list()


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
