"""Checks any field table must pass, whatever it holds. What they flag is not judged here: the anomalies are taken
to an independent source first, which says whether the data is faithful or wrong."""

import numpy as np
import polars as pl

from research_ci.api.methods import Case, Source
from research_ci.eval.synthetic import HOUR_NS, START_NS, source
from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.check import Check

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
    in the fields the source declares as ``levels``, moves of ``shift_ratio`` or more from the median of the last
    ``window`` events that persist over the next ``window``, flagged where they start."""
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
            before=level.rolling_median(window, min_samples=1).shift(1).over("entity", "field"),
            after=level.rolling_median(window).shift(-(window - 1)).over("entity", "field"),
        )
        big = (pl.col("before") > 0) & (pl.col("value") > 0) & _ratio("value", "before", shift_ratio)
        big = (big & _ratio("after", "before", shift_ratio)).fill_null(False)
        e = e.with_columns(big=big).with_columns(prior=pl.col("big").shift(1).over("entity", "field").fill_null(False))
        shifts = e.filter(pl.col("big") & ~pl.col("prior"))
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


def _kinds(fields: pl.DataFrame, levels: tuple[str, ...] = ("price",)) -> dict[str, int]:
    _, found = baseline(fields, levels, 5.0, 24)
    return {str(k): int(n) for k, n in found.group_by("kind").len().iter_rows()}


def _clean(seeds: int = 10) -> int:
    return sum(sum(_kinds(source(Source(lag=2, revised=0.3), s)).values()) for s in range(seeds))


def _caught(pit: str, kind: str, seeds: int = 10) -> int:
    """On how many sources a built-in pit planted in prices shows up as ``kind``."""
    from research_ci.eval.data.pits import BUILTIN

    planted = (BUILTIN[pit](source(Source(), s), "price", np.random.default_rng(s)) for s in range(seeds))
    return sum(kind in _kinds(f) for f in planted)


def _moved(k: float, hours: int) -> int:
    """Shifts flagged when entity e0's price is times ``k`` for ``hours`` hours from hour 100."""
    t = pl.col("event_time").dt.epoch("ns")
    hit = (pl.col("entity") == "e0") & (pl.col("field") == "price")
    hit = hit & t.is_between(START_NS + 100 * HOUR_NS, START_NS + (99 + hours) * HOUR_NS)
    fields = source(Source(), 0)
    return _kinds(fields.with_columns(value=pl.when(hit).then(pl.col("value") * k).otherwise("value"))).get("shift", 0)


def _status(name: str, fields: pl.DataFrame, levels: tuple[str, ...] = ()) -> str:
    return next(c.status for c in baseline(fields, levels, 5.0, 24)[0] if c.name == name)


def _known_early() -> pl.DataFrame:
    flow = pl.col("field") == "flow"
    early = pl.when(flow).then(pl.col("event_time") - pl.duration(hours=1)).otherwise("available_at")
    return source(Source(lag=2), 0).with_columns(available_at=early)


CASES = [
    Case("clean sources with lags and revisions flag nothing", _clean, 0),
    Case("a missing stretch is a gap, on every source", lambda: _caught("gap", "gap"), 10),
    Case("an event recorded twice is flagged, on every source", lambda: _caught("duplicate", "near"), 10),
    Case("a lasting tenfold move in a declared level is a shift", lambda: _moved(10.0, 200) > 0, True),
    Case("a one-hour tenfold spike is not a shift", lambda: _moved(10.0, 1), 0),
    Case("no declared levels: not checked", lambda: _status("no level shifts", source(Source(), 0)), "skip"),
    Case(
        "rows known before their event are counted",
        lambda: _status("known no earlier than the event", _known_early()),
        "warn",
    ),
]
