"""Checks any source must pass, whatever it holds, at its own cadence. What they flag is not judged here: the anomalies
are taken to an independent source first, which says whether the data is faithful or wrong. Gaps and near-duplicate
events are rows of a source and an inst; level shifts are cells of a column."""

from collections.abc import Collection, Mapping

import numpy as np
import polars as pl

from research_ci.api.evals import Case, Source
from research_ci.eval.synthetic import HOUR_NS, START_NS, sources
from research_ci.kernel.tables import KEYS, values
from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.check import Check

ANOMALY_SCHEMA = {
    "kind": pl.String,
    "source": pl.String,
    "inst": pl.String,
    "column": pl.String,
    "start": TIME,
    "end": TIME,
}
ROWS = {"inst": pl.String, "event_time": TIME, "available_at": TIME, "vintage": pl.Int64}
ROW = ["source", "inst"]
CELL = ["source", "column", "inst"]
Sources = Mapping[str, pl.DataFrame]


def rows(tables: Sources) -> pl.DataFrame:
    """Every row of the sources by its keys. The columns are ``source``, ``inst``, ``event_time``, ``available_at``
    and ``vintage``."""
    found = [t.select(pl.lit(name).alias("source"), *KEYS) for name, t in tables.items()]
    return pl.concat([pl.DataFrame(schema={"source": pl.String, **ROWS}), *found], how="vertical_relaxed")


def cells(tables: Sources, columns: Collection[str] | None = None) -> pl.DataFrame:
    """The cells that hold a value, one per row and column. Of the value columns ``columns``, all of them by default:
    ``source``, ``column``, ``inst``, ``event_time``, ``available_at``, ``vintage``, ``value``."""
    schema = {"source": pl.String, "column": pl.String, **ROWS, "value": pl.Float64}
    found = [
        t.unpivot(index=list(KEYS), on=own, variable_name="column", value_name="value")
        .drop_nulls("value")
        .select(pl.lit(name).alias("source"), "column", *KEYS, "value")
        for name, t in tables.items()
        if (own := [c for c in values(t) if columns is None or c in columns])
    ]
    return pl.concat([pl.DataFrame(schema=schema), *found], how="vertical_relaxed")


def first_vintages(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """One row per event of each group ``by``: its first vintage. That is the one a decision at the time would have
    seen."""
    return frame.sort("vintage").unique([*by, "event_time"], keep="first").sort(*by, "event_time")


def events(frame: pl.DataFrame, by: list[str] = ROW) -> pl.DataFrame:
    """First vintages with each event's spacing, its group's usual spacing and size. The spacing is to the
    previous event of the same group ``by``: a source and an inst, or a column of it."""
    gap = pl.col("event_time").diff().over(by).dt.total_nanoseconds()
    return (
        first_vintages(frame, by)
        .with_columns(prev=pl.col("event_time").shift(1).over(by), gap=gap)
        .with_columns(spacing=pl.col("gap").median().over(by), n=pl.len().over(by))
    )


def baseline(
    tables: Sources,
    levels: tuple[str, ...] = (),
    shift_ratio: float = 0.0,
    window: int = 0,
    gap_factor: float = 3.0,
    near_ns: int = 60 * 10**9,
) -> tuple[list[Check], pl.DataFrame]:
    """Generic checks and their anomalies: gaps, near-duplicates, level shifts. The anomalies are long gaps between a
    source's events of an inst and events closer than ``near_ns``; and, in the columns the plugin declares as
    ``levels``, moves of ``shift_ratio`` or more from the median of the last ``window`` events that persist over the
    next ``window``, flagged where they start."""
    r = rows(tables)
    e = events(r)
    checks = []
    bad = pl.concat(
        [pl.DataFrame(schema={"where": pl.String, "n": pl.UInt32})]
        + [
            t.select((~pl.col(c).is_finite()).fill_null(True).sum().alias(f"{name}.{c}") for c in values(t))
            .unpivot(variable_name="where", value_name="n")
            .cast({"n": pl.UInt32})
            for name, t in tables.items()
        ]
    ).filter(pl.col("n") > 0)
    checks.append(_count("values are finite", int(bad["n"].sum()), "values null, NaN or infinite", bad))
    early = r.filter(pl.col("available_at") < pl.col("event_time")).height
    checks.append(_count("known no earlier than the event", early, "rows known before their event time"))
    gaps = e.filter((pl.col("n") >= 3) & (pl.col("gap") > gap_factor * pl.col("spacing")))
    near = e.filter(pl.col("gap") < near_ns)
    long = f"gaps longer than {gap_factor:g}× the usual spacing"
    checks.append(_count("no long gaps", gaps.height, long, _where(gaps)))
    what = f"events within {near_ns / 1e9:g}s of the last"
    checks.append(_count("no near-duplicate events", near.height, what, _where(near)))
    found = [_anomalies(gaps, "gap"), _anomalies(near, "near")]
    if levels:
        shifts = _shifts(events(cells(tables, levels), CELL), shift_ratio, window)
        what = f"lasting moves of {shift_ratio:g}× or more"
        checks.append(_count("no level shifts", shifts.height, what, _where(shifts)))
        found.append(_anomalies(shifts, "shift"))
    else:
        checks.append(Check("no level shifts", "skip", "the source declares no level columns"))
    return checks, pl.concat(found)


def level_shift_anomalies(
    tables: Sources, levels: tuple[str, ...] = (), shift_ratio: float = 0.0, window: int = 0
) -> pl.DataFrame:
    """The baseline's level shifts alone, from the declared levels only."""
    if not levels:
        return pl.DataFrame(schema=ANOMALY_SCHEMA)
    return _anomalies(_shifts(events(cells(tables, levels), CELL), shift_ratio, window), "shift")


def _shifts(e: pl.DataFrame, shift_ratio: float, window: int) -> pl.DataFrame:
    """Events of declared levels ``shift_ratio`` or more from the median of the last ``window`` events whose next
    ``window`` stay there, flagged where they start."""
    level = pl.col("value")
    e = e.with_columns(
        before=level.rolling_median(window, min_samples=1).shift(1).over(CELL),
        after=level.rolling_median(window).shift(-(window - 1)).over(CELL),
    )
    big = (pl.col("before") > 0) & (pl.col("value") > 0) & _ratio("value", "before", shift_ratio)
    big = (big & _ratio("after", "before", shift_ratio)).fill_null(False)
    e = e.with_columns(big=big).with_columns(prior=pl.col("big").shift(1).over(CELL).fill_null(False))
    return e.filter(pl.col("big") & ~pl.col("prior"))


def _ratio(a: str, b: str, k: float) -> pl.Expr:
    r = pl.col(a) / pl.col(b)
    return (r >= k) | (r <= 1 / k)


def _anomalies(found: pl.DataFrame, kind: str) -> pl.DataFrame:
    column = pl.col("column") if "column" in found.columns else pl.lit(None, dtype=pl.String)
    return found.select(
        pl.lit(kind).alias("kind"),
        "source",
        "inst",
        column.alias("column"),
        pl.col("prev").alias("start"),
        pl.col("event_time").alias("end"),
    ).cast(ANOMALY_SCHEMA)  # type: ignore[arg-type]


def _where(found: pl.DataFrame) -> pl.DataFrame:
    """Where each flagged row is, as the card names it: the source, or the source's column."""
    where = pl.format("{}.{}", "source", "column") if "column" in found.columns else pl.col("source")
    return found.group_by(where.alias("where")).len("n")


def _count(name: str, n: int, what: str, where: pl.DataFrame | None = None) -> Check:
    if n == 0:
        return Check(name, "pass", f"none: {what}")
    shown = ""
    if where is not None and where.height:
        top = where.sort("n", "where", descending=[True, False]).head(3)
        shown = " (" + ", ".join(f"{w} {k}" for w, k in top.iter_rows()) + ")"
    return Check(name, "warn", f"{n:,} {what}{shown}")


def _kinds(tables: Sources, levels: tuple[str, ...] = ("price",)) -> dict[str, int]:
    _, found = baseline(tables, levels, 5.0, 24)
    return {str(k): int(n) for k, n in found.group_by("kind").len().iter_rows()}


def _clean(seeds: int = 10) -> int:
    return sum(sum(_kinds(sources(Source(lag=2, revised=0.3), s)).values()) for s in range(seeds))


def _caught(pit: str, kind: str, seeds: int = 10) -> int:
    """On how many sources a built-in pit planted in prices shows up as ``kind``."""
    from research_ci.eval.data.pits import BUILTIN

    planted = (BUILTIN[pit](sources(Source(), s), "price", np.random.default_rng(s)) for s in range(seeds))
    return sum(kind in _kinds(f) for f in planted)


def _moved(k: float, hours: int) -> int:
    """Shifts flagged when inst e0's price is times ``k`` for ``hours`` hours from hour 100."""
    t = pl.col("event_time").dt.epoch("ns")
    hit = (pl.col("inst") == "e0") & t.is_between(START_NS + 100 * HOUR_NS, START_NS + (99 + hours) * HOUR_NS)
    tables = sources(Source(), 0)
    prices = tables["prices"].with_columns(price=pl.when(hit).then(pl.col("price") * k).otherwise("price"))
    return _kinds({**tables, "prices": prices}).get("shift", 0)


def _status(name: str, tables: Sources, levels: tuple[str, ...] = ()) -> str:
    return next(c.status for c in baseline(tables, levels, 5.0, 24)[0] if c.name == name)


def _known_early() -> dict[str, pl.DataFrame]:
    tables = sources(Source(lag=2), 0)
    flows = tables["flows"].with_columns(available_at=pl.col("event_time") - pl.duration(hours=1, time_unit="ns"))
    return {**tables, "flows": flows}


def _missing() -> str:
    """Two prices of e1 and a flow of e2 missing: the counts by source and column."""
    tables = sources(Source(), 0)
    prices = tables["prices"].with_columns(price=pl.when(pl.col("inst") == "e1").then(None).otherwise("price"))
    flows = tables["flows"].with_columns(flow=pl.when(pl.col("inst") == "e2").then(float("nan")).otherwise("flow"))
    return next(c.detail for c in baseline({"prices": prices.head(40), "flows": flows.head(30)})[0])


CASES = [
    Case("clean sources with lags and revisions flag nothing", _clean, 0),
    Case("a missing stretch is a gap, on every source", lambda: _caught("gap", "gap"), 10),
    Case("an event recorded twice is flagged, on every source", lambda: _caught("duplicate", "near"), 10),
    Case("a lasting tenfold move in a declared level is a shift", lambda: _moved(10.0, 200) > 0, True),
    Case("a one-hour tenfold spike is not a shift", lambda: _moved(10.0, 1), 0),
    Case("no declared levels: not checked", lambda: _status("no level shifts", sources(Source(), 0)), "skip"),
    Case(
        "rows known before their event are counted",
        lambda: _status("known no earlier than the event", _known_early()),
        "warn",
    ),
    Case(
        "values that are null or NaN are counted by source and column",
        _missing,
        "7 values null, NaN or infinite (prices.price 4, flows.flow 3)",
    ),
]
