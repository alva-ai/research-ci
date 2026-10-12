"""Checking a field table against an independent source, on rows the engine picks: a random sample of every field,
and every anomaly the generic checks flagged. An anomaly the source shows too is faithful data, not an error."""

from typing import Any

import numpy as np
import polars as pl

from research_ci.api.methods import Case, Source
from research_ci.api.plugins import Reference
from research_ci.eval.data.baseline import baseline
from research_ci.eval.data.pits import gap
from research_ci.eval.synthetic import source
from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.check import Check

REQUEST_SCHEMA = {
    "rid": pl.Int64,
    "kind": pl.String,
    "entity": pl.String,
    "field": pl.String,
    "start": TIME,
    "end": TIME,
}


def requests(
    full: pl.DataFrame, anomalies: pl.DataFrame, per_field: int, max_anomalies: int, tol_ns: int, seed: int
) -> pl.DataFrame:
    """Point requests for random rows, and window requests for anomalies. Point requests go around ``per_field`` random
    rows of every field; window requests go inside a gap (where the source should have nothing), around near-duplicate
    events and level shifts (where it should have what we have)."""
    tol = pl.duration(nanoseconds=tol_ns, time_unit="ns")
    draw = pl.int_range(pl.len(), dtype=pl.UInt32).shuffle(seed=seed).over("field")
    picked = (
        full.filter(draw < per_field)
        .sort("field", "entity", "event_time", "vintage")
        .select(
            pl.lit("sample").alias("kind"),
            "entity",
            "field",
            (pl.col("event_time") - tol).alias("start"),
            (pl.col("event_time") + tol).alias("end"),
        )
    )
    a = anomalies
    if a.height > max_anomalies:
        order = pl.int_range(pl.len(), dtype=pl.UInt32).shuffle(seed=seed).over("kind")
        a = a.filter(order < max(1, max_anomalies // 3)).sort("kind", "entity", "field", "start")
    gap = a.filter(pl.col("kind") == "gap").with_columns(start=pl.col("start") + tol, end=pl.col("end") - tol)
    rest = a.filter(pl.col("kind") != "gap").with_columns(start=pl.col("start") - tol, end=pl.col("end") + tol)
    out = pl.concat([picked, gap.select(picked.columns), rest.select(picked.columns)])
    return out.with_row_index("rid").cast(REQUEST_SCHEMA)  # type: ignore[arg-type]


def compare(
    reqs: pl.DataFrame, ours: pl.DataFrame, ref: Reference, tol_ns: int, rtol: float = 1e-6
) -> list[dict[str, Any]]:
    """One result per request: ``match``, ``mismatch`` or ``unanswered``. A sample request matches when the source
    has our value within ``tol_ns`` of our time; a gap matches when the source has nothing inside it; a window
    around other anomalies matches when every row we have there is at the source too."""
    keys = ["entity", "field"]
    answered = ref.answered.select("rid") if "rid" in ref.answered.columns else pl.DataFrame(schema={"rid": pl.Int64})
    r = reqs.with_columns(
        lo=pl.col("start").dt.epoch("ns"),
        hi=pl.col("end").dt.epoch("ns"),
        answered=pl.col("rid").is_in(answered["rid"].cast(pl.Int64).implode()),
    )
    window = r.select("rid", *keys, "lo", "hi")
    theirs = _timed(ref.rows, "u", "w").with_row_index("_j")
    mine = _timed(ours.join(reqs.select(keys).unique(), on=keys, how="semi"), "t", "v").with_row_index("_i")
    there = window.join(theirs, on=keys).filter(pl.col("u").is_between("lo", "hi")).sort("rid", "_j")
    inside = window.join(mine, on=keys).filter(pl.col("t").is_between("lo", "hi")).sort("rid", "_i")
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
    columns = ["rid", "kind", "entity", "field", "status", "detail"]
    return out.sort("rid").select(*columns[:4], status.alias("status"), detail.alias("detail")).to_dicts()


def summarize(results: list[dict[str, Any]], fields: list[str]) -> tuple[Check, dict[str, str], dict[str, Any]]:
    """The check, each field's grade, and counts. The counts are per field and per anomaly kind."""
    schema = {"kind": pl.String, "field": pl.String, "status": pl.String, "entity": pl.String, "detail": pl.String}
    frame = pl.DataFrame(results, schema=schema) if results else pl.DataFrame(schema=schema)
    frame = frame.with_columns(group=pl.when(pl.col("kind") == "sample").then("field").otherwise("kind"))
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
    named = pl.DataFrame({"group": fields}, schema={"group": pl.String}).join(per, on="group", how="left")
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
        detail += f"; first mismatch {first['entity']} {first['field']}: {first['detail']}"
    status = "fail" if bad.height else ("pass" if checked.height else "warn")
    rows = counts.sort("group", "status").group_by("group", maintain_order=True).agg("status", "len")
    table = {g: dict(zip(s, n, strict=True)) for g, s, n in rows.iter_rows()}
    return Check("independent source agrees", status, detail), grades, table  # type: ignore[arg-type]


STATUSES = ("match", "mismatch", "unanswered")


def _timed(rows: pl.DataFrame, at: str, value: str) -> pl.DataFrame:
    """``entity``, ``field``, the event time in ns and the value, in the order given."""
    if rows.is_empty() or "event_time" not in rows.columns:
        return pl.DataFrame(schema={"entity": pl.String, "field": pl.String, at: pl.Int64, value: pl.Float64})
    when = pl.col("event_time").dt.convert_time_zone("UTC").cast(TIME).dt.epoch("ns").alias(at)
    return rows.select("entity", "field", when, pl.col("value").cast(pl.Float64).alias(value))


def _text(x: pl.Expr) -> pl.Expr:
    return x.round_sig_figs(6).cast(pl.String)


TOL_NS = 60 * 10**9


def _statuses(
    ours: pl.DataFrame, theirs: pl.DataFrame, kind: str | None = None, answer: bool = True
) -> dict[str, set[str]]:
    """Request statuses by field, checking ``ours`` against ``theirs`` on the rows and anomalies the engine picks."""
    _, anomalies = baseline(ours)
    reqs = requests(ours, anomalies, 20, 10, TOL_NS, 0)
    answered = reqs.select("rid") if answer else reqs.select("rid").head(0)
    ref = Reference(theirs.select("entity", "field", "event_time", "value"), answered)
    found = pl.DataFrame(compare(reqs, ours, ref, TOL_NS))
    found = found.filter(pl.col("kind") == kind) if kind is not None else found
    return {f: set(s) for f, s in found.group_by("field").agg(pl.col("status").unique()).iter_rows()}


def _gapped(seed: int) -> pl.DataFrame:
    return gap(_source(seed), "price", np.random.default_rng(seed))


def _source(seed: int) -> pl.DataFrame:
    return source(Source(), seed)


def _off(field: str, k: float) -> pl.DataFrame:
    hit = pl.col("field") == field
    return _source(0).with_columns(value=pl.when(hit).then(pl.col("value") * k).otherwise("value"))


CASES = [
    Case("the same data agrees", lambda: _statuses(_source(0), _source(0)), {"flow": {"match"}, "price": {"match"}}),
    Case(
        "values off by a tenth disagree in that field only",
        lambda: _statuses(_off("flow", 1.1), _source(0)),
        {"flow": {"mismatch"}, "price": {"match"}},
    ),
    Case(
        "a gap the source has too is faithful", lambda: _statuses(_gapped(0), _gapped(0), "gap"), {"price": {"match"}}
    ),
    Case(
        "a gap the source does not have is a mismatch",
        lambda: _statuses(_gapped(0), _source(0), "gap"),
        {"price": {"mismatch"}},
    ),
    Case(
        "what the source cannot answer is not agreement",
        lambda: _statuses(_source(0), _source(0), answer=False),
        {"flow": {"unanswered"}, "price": {"unanswered"}},
    ),
]
