"""Checking sources against an independent source, on cells the engine picks: a random sample of every column, and
every anomaly the generic checks flagged, each column of the anomaly's source for a gap or a near-duplicate. An
anomaly the independent source shows too is faithful data, not an error. Each request asks about one column of one
source and inst, within a span."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl

from research_ci.api.evals import Case, Source
from research_ci.api.plugins import Reference
from research_ci.eval import synthetic
from research_ci.eval.data.baseline import baseline, cells
from research_ci.eval.data.pits import gap
from research_ci.kernel.tables import values
from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.check import Check

Sources = Mapping[str, pl.DataFrame]
KEYS = ["source", "column", "inst"]
REQUEST_SCHEMA = {
    "rid": pl.Int64,
    "kind": pl.String,
    "source": pl.String,
    "inst": pl.String,
    "column": pl.String,
    "start": TIME,
    "end": TIME,
}


def requests(
    full: Sources, anomalies: pl.DataFrame, per_column: int, max_anomalies: int, tol_ns: int, seed: int
) -> pl.DataFrame:
    """Point requests for random cells, and window requests for anomalies. Point requests go around ``per_column``
    random cells of every column; window requests go inside a gap (where the source should have nothing), around
    near-duplicate events and level shifts (where it should have what we have), one per column of the anomaly's source
    when the anomaly is a whole row."""
    tol = pl.duration(nanoseconds=tol_ns, time_unit="ns")
    draw = pl.int_range(pl.len(), dtype=pl.UInt32).shuffle(seed=seed).over("source", "column")
    picked = (
        cells(full)
        .filter(draw < per_column)
        .sort("source", "column", "inst", "event_time", "vintage")
        .select(
            pl.lit("sample").alias("kind"),
            "source",
            "inst",
            "column",
            (pl.col("event_time") - tol).alias("start"),
            (pl.col("event_time") + tol).alias("end"),
        )
    )
    owned = pl.DataFrame(
        [(name, c) for name, t in full.items() for c in values(t)],
        schema={"source": pl.String, "column": pl.String},
        orient="row",
    )
    whole = anomalies.filter(pl.col("column").is_null()).drop("column").join(owned, on="source")
    a = pl.concat([anomalies.filter(pl.col("column").is_not_null()), whole.select(anomalies.columns)])
    if a.height > max_anomalies:
        order = pl.int_range(pl.len(), dtype=pl.UInt32).shuffle(seed=seed).over("kind")
        a = a.filter(order < max(1, max_anomalies // 3)).sort("kind", "source", "column", "inst", "start")
    gap = a.filter(pl.col("kind") == "gap").with_columns(start=pl.col("start") + tol, end=pl.col("end") - tol)
    rest = a.filter(pl.col("kind") != "gap").with_columns(start=pl.col("start") - tol, end=pl.col("end") + tol)
    out = pl.concat([picked, gap.select(picked.columns), rest.select(picked.columns)])
    return out.with_row_index("rid").cast(REQUEST_SCHEMA)  # type: ignore[arg-type]


def compare(reqs: pl.DataFrame, ours: Sources, ref: Reference, tol_ns: int, rtol: float = 1e-6) -> list[dict[str, Any]]:
    """One result per request: ``match``, ``mismatch`` or ``unanswered``. A sample request matches when the source
    has our value within ``tol_ns`` of our time; a gap matches when the source has nothing inside it; a window
    around other anomalies matches when every cell we have there is at the source too."""
    answered = ref.answered.select("rid") if "rid" in ref.answered.columns else pl.DataFrame(schema={"rid": pl.Int64})
    r = reqs.with_columns(
        lo=pl.col("start").dt.epoch("ns"),
        hi=pl.col("end").dt.epoch("ns"),
        answered=pl.col("rid").is_in(answered["rid"].cast(pl.Int64).implode()),
    )
    window = r.select("rid", *KEYS, "lo", "hi")
    theirs = _timed(ref.rows, "u", "w").with_row_index("_j")
    mine = _timed(ours, "t", "v").join(reqs.select(KEYS).unique(), on=KEYS, how="semi").with_row_index("_i")
    there = window.join(theirs, on=KEYS).filter(pl.col("u").is_between("lo", "hi")).sort("rid", "_j")
    inside = window.join(mine, on=KEYS).filter(pl.col("t").is_between("lo", "hi")).sort("rid", "_i")
    near = inside.join(there.select("rid", "u", "w"), on="rid").filter(
        ((pl.col("t") - pl.col("u")).abs() <= tol_ns)
        & ((pl.col("v") - pl.col("w")).abs() <= 1e-12 + rtol * pl.col("w").abs())
    )
    missing = inside.join(near.select("rid", "_i").unique(), on=["rid", "_i"], how="anti")
    first = missing.group_by("rid").agg(pl.col("t").first().alias("mt"), pl.col("v").first().alias("mv"))
    seen = there.group_by("rid").agg(pl.len().alias("n_there"), _text(pl.col("w")).head(3).str.join(", ").alias("seen"))
    counts = inside.group_by("rid").agg(pl.len().alias("n_mine"))
    out = r.join(seen, on="rid", how="left").join(counts, on="rid", how="left").join(first, on="rid", how="left")
    out = out.with_columns(pl.col("n_there", "n_mine").fill_null(0))
    gap, sample = pl.col("kind") == "gap", pl.col("kind") == "sample"
    status = (
        pl.when(~pl.col("answered"))
        .then(pl.lit("unanswered"))
        .when(gap)
        .then(pl.when(pl.col("n_there") > 0).then(pl.lit("mismatch")).otherwise(pl.lit("match")))
        .when((pl.col("n_mine") == 0) | pl.col("mt").is_not_null())
        .then(pl.lit("mismatch"))
        .otherwise(pl.lit("match"))
    )
    ours_at = pl.format(
        "ours {} at {}; the source has {}",
        _text(pl.col("mv")),
        pl.from_epoch("mt", time_unit="ns").dt.strftime("%Y-%m-%dT%H:%M:%S%.3fZ"),
        pl.col("seen").fill_null("nothing"),
    )
    detail = (
        pl.when(~pl.col("answered"))
        .then(pl.lit("the source could not check this"))
        .when(gap & (pl.col("n_there") > 0))
        .then(pl.format("the source has {} rows inside our gap", "n_there"))
        .when(gap)
        .then(pl.lit("the source has nothing either"))
        .when(pl.col("n_mine") == 0)
        .then(pl.lit("we have no row here"))
        .when(pl.col("mt").is_not_null())
        .then(ours_at)
        .when(sample)
        .then(pl.lit("the source agrees"))
        .otherwise(pl.lit("faithful: the source has it"))
    )
    columns = ["rid", "kind", "source", "inst", "column"]
    return out.sort("rid").select(*columns, status.alias("status"), detail.alias("detail")).to_dicts()


def summarize(results: list[dict[str, Any]], columns: list[str]) -> tuple[Check, dict[str, str], dict[str, Any]]:
    """The check, each column's grade, and counts. Columns are named ``<source>.<column>``; the counts are per column
    and per anomaly kind."""
    schema = {
        "kind": pl.String,
        "source": pl.String,
        "column": pl.String,
        "status": pl.String,
        "inst": pl.String,
        "detail": pl.String,
    }
    frame = pl.DataFrame(results, schema=schema) if results else pl.DataFrame(schema=schema)
    named = pl.format("{}.{}", "source", "column")
    frame = frame.with_columns(group=pl.when(pl.col("kind") == "sample").then(named).otherwise("kind"))
    counts = frame.group_by("group", "status").len()
    per = counts.pivot("status", index="group", values="len") if counts.height else pl.DataFrame({"group": []})
    per = per.with_columns(pl.lit(0, pl.UInt32).alias(c) for c in STATUSES if c not in per.columns)
    grade = (
        pl.when(pl.col("mismatch") > 0)
        .then(pl.lit("failed"))
        .when(pl.col("match") > 0)
        .then(pl.lit("verified"))
        .when(pl.col("unanswered") > 0)
        .then(pl.lit("unverified"))
        .otherwise(pl.lit("not sampled"))
    )
    named = pl.DataFrame({"group": columns}, schema={"group": pl.String}).join(per, on="group", how="left")
    graded = named.with_columns(pl.col(*STATUSES).fill_null(0)).select("group", grade.alias("grade"))
    grades = dict(graded.iter_rows())
    checked = frame.filter(pl.col("status") != "unanswered")
    bad = checked.filter(pl.col("status") == "mismatch")
    detail = f"{checked.height - bad.height} of {checked.height} checked rows agree with the source"
    unanswered = frame.height - checked.height
    if unanswered:
        detail += f" ({unanswered} it could not check)"
    if bad.height:
        first = bad.row(0, named=True)
        detail += f"; first mismatch {first['inst']} {first['source']}.{first['column']}: {first['detail']}"
    status = "fail" if bad.height else ("pass" if checked.height else "warn")
    rows = counts.sort("group", "status").group_by("group", maintain_order=True).agg("status", "len")
    table = {g: dict(zip(s, n, strict=True)) for g, s, n in rows.iter_rows()}
    return Check("independent source agrees", status, detail), grades, table  # type: ignore[arg-type]


STATUSES = ("match", "mismatch", "unanswered")


def _timed(tables: Sources, at: str, value: str) -> pl.DataFrame:
    """Every cell of the tables that holds a value: ``source``, ``column``, ``inst``, the event time in ns and the
    value. A table needs ``inst`` and ``event_time``; its other columns, but for ``available_at`` and ``vintage``, are
    value columns."""
    schema = {"source": pl.String, "column": pl.String, "inst": pl.String, at: pl.Int64, value: pl.Float64}
    when = pl.col("event_time").dt.convert_time_zone("UTC").cast(TIME).dt.epoch("ns").alias(at)
    found = [
        t.select("inst", when, *own)
        .unpivot(index=["inst", at], on=own, variable_name="column", value_name=value)
        .drop_nulls(value)
        .select(pl.lit(name).alias("source"), "column", "inst", at, pl.col(value).cast(pl.Float64))
        for name, t in tables.items()
        if not t.is_empty()
        and (own := [c for c in t.columns if c not in ("inst", "event_time", "available_at", "vintage")])
    ]
    return pl.concat([pl.DataFrame(schema=schema), *found])


def _text(x: pl.Expr) -> pl.Expr:
    return x.round_sig_figs(6).cast(pl.String)


TOL_NS = 60 * 10**9


def _statuses(ours: Sources, theirs: Sources, kind: str | None = None, answer: bool = True) -> dict[str, set[str]]:
    """Request statuses by column, checking ``ours`` against ``theirs`` on the cells and anomalies the engine picks."""
    _, anomalies = baseline(ours)
    reqs = requests(ours, anomalies, 20, 10, TOL_NS, 0)
    answered = reqs.select("rid") if answer else reqs.select("rid").head(0)
    found = pl.DataFrame(compare(reqs, ours, Reference(theirs, answered), TOL_NS))
    found = found.filter(pl.col("kind") == kind) if kind is not None else found
    return {f: set(s) for f, s in found.group_by("column").agg(pl.col("status").unique()).iter_rows()}


def _gapped(seed: int) -> dict[str, pl.DataFrame]:
    return gap(_sources(seed), "price", np.random.default_rng(seed))


def _sources(seed: int) -> dict[str, pl.DataFrame]:
    return synthetic.sources(Source(), seed)


def _off(column: str, k: float) -> dict[str, pl.DataFrame]:
    tables = _sources(0)
    return {name: t.with_columns(pl.col(c) * k for c in values(t) if c == column) for name, t in tables.items()}


CASES = [
    Case("the same data agrees", lambda: _statuses(_sources(0), _sources(0)), {"flow": {"match"}, "price": {"match"}}),
    Case(
        "values off by a tenth disagree in that column only",
        lambda: _statuses(_off("flow", 1.1), _sources(0)),
        {"flow": {"mismatch"}, "price": {"match"}},
    ),
    Case(
        "a gap the source has too is faithful", lambda: _statuses(_gapped(0), _gapped(0), "gap"), {"price": {"match"}}
    ),
    Case(
        "a gap the source does not have is a mismatch",
        lambda: _statuses(_gapped(0), _sources(0), "gap"),
        {"price": {"mismatch"}},
    ),
    Case(
        "what the source cannot answer is not agreement",
        lambda: _statuses(_sources(0), _sources(0), answer=False),
        {"flow": {"unanswered"}, "price": {"unanswered"}},
    ),
]
