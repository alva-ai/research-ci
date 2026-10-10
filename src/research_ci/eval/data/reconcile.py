"""Checking a field table against an independent source, on rows the engine picks: a random sample of every field,
and every anomaly the generic checks flagged. An anomaly the source shows too is faithful data, not an error."""

from collections import Counter
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
    """Point requests around ``per_field`` random rows of every field, and window requests for anomalies: inside a
    gap (where the source should have nothing), around near-duplicate events and level shifts (where it should have
    what we have)."""
    tol = pl.duration(nanoseconds=tol_ns, time_unit="ns")
    picked = []
    for i, name in enumerate(sorted(full["field"].unique().to_list())):
        rows = full.filter(pl.col("field") == name)
        rows = rows.sample(min(per_field, rows.height), seed=seed + i) if rows.height else rows
        picked.append(
            rows.select(
                pl.lit("sample").alias("kind"),
                "entity",
                "field",
                (pl.col("event_time") - tol).alias("start"),
                (pl.col("event_time") + tol).alias("end"),
            )
        )
    a = anomalies
    if a.height > max_anomalies:
        a = pl.concat(
            [g.head(max(1, max_anomalies // 3)) for _, g in a.sample(fraction=1.0, seed=seed).group_by("kind")]
        )
    gap = a.filter(pl.col("kind") == "gap").with_columns(start=pl.col("start") + tol, end=pl.col("end") - tol)
    rest = a.filter(pl.col("kind") != "gap").with_columns(start=pl.col("start") - tol, end=pl.col("end") + tol)
    out = pl.concat([*picked, gap.select(picked[0].columns), rest.select(picked[0].columns)])
    return out.with_row_index("rid").cast(REQUEST_SCHEMA)  # type: ignore[arg-type]


def compare(
    reqs: pl.DataFrame, ours: pl.DataFrame, ref: Reference, tol_ns: int, rtol: float = 1e-6
) -> list[dict[str, Any]]:
    """One result per request: ``match``, ``mismatch`` or ``unanswered``. A sample request matches when the source
    has our value within ``tol_ns`` of our time; a gap matches when the source has nothing inside it; a window
    around other anomalies matches when every row we have there is at the source too."""
    answered = set(ref.answered["rid"].to_list()) if "rid" in ref.answered.columns else set()
    theirs = _by_key(ref.rows)
    mine_all = _by_key(ours.join(reqs.select("entity", "field").unique(), on=["entity", "field"]))
    out = []
    span = reqs.with_columns(lo=pl.col("start").dt.epoch("ns"), hi=pl.col("end").dt.epoch("ns"))
    for r in span.iter_rows(named=True):
        key = (r["entity"], r["field"])
        lo, hi = r["lo"], r["hi"]
        if r["rid"] not in answered:
            out.append(_res(r, "unanswered", "the source could not check this"))
            continue
        there = [(t, v) for t, v in theirs.get(key, []) if lo <= t <= hi]
        if r["kind"] == "gap":
            status = "mismatch" if there else "match"
            detail = f"the source has {len(there)} rows inside our gap" if there else "the source has nothing either"
            out.append(_res(r, status, detail))
            continue
        mine = [(t, v) for t, v in mine_all.get(key, []) if lo <= t <= hi]
        missing = [(t, v) for t, v in mine if not _found(t, v, there, tol_ns, rtol)]
        if not mine:
            out.append(_res(r, "mismatch", "we have no row here"))
        elif missing:
            t, v = missing[0]
            seen = ", ".join(f"{x:g}" for _, x in there[:3]) or "nothing"
            out.append(_res(r, "mismatch", f"ours {v:g} at {_iso(t)}; the source has {seen}"))
        else:
            out.append(
                _res(r, "match", "the source agrees" if r["kind"] == "sample" else "faithful: the source has it")
            )
    return out


def summarize(results: list[dict[str, Any]], fields: list[str]) -> tuple[Check, dict[str, str], dict[str, Any]]:
    """The check, each field's grade, and counts per field and per anomaly kind."""
    counts: dict[str, Counter[str]] = {}
    for r in results:
        counts.setdefault(r["field"] if r["kind"] == "sample" else r["kind"], Counter())[r["status"]] += 1
    grades = {}
    for f in fields:
        c = counts.get(f, Counter())
        if c["mismatch"]:
            grades[f] = "failed"
        elif c["match"]:
            grades[f] = "verified"
        elif c["unanswered"]:
            grades[f] = "unverified"
        else:
            grades[f] = "not sampled"
    checked = [r for r in results if r["status"] != "unanswered"]
    bad = [r for r in checked if r["status"] == "mismatch"]
    detail = f"{len(checked) - len(bad)} of {len(checked)} checked rows agree with the source"
    unanswered = len(results) - len(checked)
    if unanswered:
        detail += f" ({unanswered} it could not check)"
    if bad:
        first = bad[0]
        detail += f"; first mismatch {first['entity']} {first['field']}: {first['detail']}"
    status = "fail" if bad else ("pass" if checked else "warn")
    table = {k: dict(v) for k, v in sorted(counts.items())}
    return Check("independent source agrees", status, detail), grades, table  # type: ignore[arg-type]


def _by_key(rows: pl.DataFrame) -> dict[tuple[str, str], list[tuple[int, float]]]:
    out: dict[tuple[str, str], list[tuple[int, float]]] = {}
    if rows.is_empty():
        return out
    when = pl.col("event_time").dt.convert_time_zone("UTC").cast(TIME).dt.epoch("ns")
    frame = rows.select("entity", "field", when, "value")
    for e, f, t, v in frame.iter_rows():
        out.setdefault((e, f), []).append((int(t), float(v)))
    return out


def _found(t: int, v: float, there: list[tuple[int, float]], tol_ns: int, rtol: float) -> bool:
    return any(abs(t - u) <= tol_ns and bool(np.isclose(v, w, rtol=rtol, atol=1e-12)) for u, w in there)


def _res(r: dict[str, Any], status: str, detail: str) -> dict[str, Any]:
    return {"rid": r["rid"], "kind": r["kind"], "entity": r["entity"], "field": r["field"], "status": status} | {
        "detail": detail
    }


def _iso(ns: int) -> str:
    return str(np.datetime64(ns, "ns"))[:23] + "Z"


TOL_NS = 60 * 10**9


def _statuses(
    ours: pl.DataFrame, theirs: pl.DataFrame, kind: str | None = None, answer: bool = True
) -> dict[str, set[str]]:
    """Request statuses by field, checking ``ours`` against ``theirs`` on the rows and anomalies the engine picks."""
    _, anomalies = baseline(ours)
    reqs = requests(ours, anomalies, 20, 10, TOL_NS, 0)
    answered = reqs.select("rid") if answer else reqs.select("rid").head(0)
    ref = Reference(theirs.select("entity", "field", "event_time", "value"), answered)
    out: dict[str, set[str]] = {}
    for r in compare(reqs, ours, ref, TOL_NS):
        if kind is None or r["kind"] == kind:
            out.setdefault(r["field"], set()).add(r["status"])
    return out


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
