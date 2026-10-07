"""Admitting a data source: does the plugin give, for any past time, what was known then, and does an independent
source agree with what it gives? Run once per plugin, options and code; every idea using the source reuses it.

Two canaries check the checks on this very data: a deletion made with hindsight must be caught by the replay, and
values altered on our side must be caught by the reconciliation."""

import inspect
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import Use, span
from research_ci.api.plugins import REQUEST_COLUMNS, Reference
from research_ci.kernel.checks import reconcile as rc
from research_ci.kernel.checks.baseline import baseline, events
from research_ci.kernel.checks.replay import Cut, cut_points, replay
from research_ci.kernel.judge.protocol import Check
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import from_ns, lit_time, span_ns
from research_ci.runner.plugins import code_hash


@dataclass(frozen=True)
class SourceSettings:
    holes: int = 200
    late: int = 40
    randoms: int = 5
    random_entities: int = 20
    per_field: int = 12
    max_anomalies: int = 60
    tolerance: str = "1s"
    seed: int = 0


def can_replay(plugin: Any) -> bool:
    try:
        params = inspect.signature(plugin.fetch).parameters
    except (TypeError, ValueError):
        return False
    return "as_of" in params and "entities" in params


def admit(use: Use, plugin: Any, fields: pl.DataFrame, settings: SourceSettings | None = None) -> dict[str, Any]:
    s = settings or SourceSettings()
    started = time.monotonic()
    rng = np.random.default_rng(s.seed)
    names = sorted(fields["field"].unique().to_list())
    spec: dict[str, Any] = {
        "plugin": use.name,
        "options": dict(use.options),
        "code": code_hash(plugin),
        "rows": fields.height,
        "entities": fields["entity"].n_unique(),
        "fields": names,
        "settings": asdict(s),
    }
    checks, anomalies = baseline(fields)
    spec["anomalies"] = {k: n for k, n in anomalies.group_by("kind").len().iter_rows()}
    if can_replay(plugin):

        def fetch(at: int, entities: list[str]) -> pl.DataFrame:
            return field_table(plugin.fetch(as_of=from_ns(at), entities=entities))

        cuts = cut_points(fields, rng, s.holes, s.late, s.randoms, s.random_entities)
        check, results = replay(fetch, fields, cuts)
        checks.append(check)
        spec["replay"] = results
        checks.append(_replay_canary(fetch, fields, rng))
    else:
        checks.append(Check("replay as of past times", "fail", "the plugin cannot fetch as of a time and entities"))
    tol = span_ns(span(s.tolerance))
    grades = dict.fromkeys(names, "unverified")
    if callable(getattr(plugin, "reference", None)):
        reqs = rc.requests(fields, anomalies, s.per_field, s.max_anomalies, tol, s.seed)
        ref = _reference(plugin, reqs)
        results = rc.compare(reqs, fields, ref, tol)
        check, grades, table = rc.summarize(results, names)
        checks.append(check)
        checks.append(_reference_canary(reqs, fields, ref, results, tol))
        spec["reference"] = {"note": getattr(plugin, "reference_note", ""), "counts": table}
        spec["mismatches"] = [r for r in results if r["status"] == "mismatch"][:20]
        shifts = reqs.filter(pl.col("kind") == "shift").select("rid", "entity", "field", "end")
        ok = {r["rid"] for r in results if r["kind"] == "shift" and r["status"] == "match"}
        spec["confirmed_shifts"] = [
            [e, f, end.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")] for rid, e, f, end in shifts.iter_rows() if rid in ok
        ]
    else:
        checks.append(Check("independent source agrees", "warn", "the plugin names no independent source"))
    spec["grades"] = grades
    spec["checks"] = [asdict(c) for c in checks]
    failed = [c.name for c in checks if c.status == "fail"]
    status = "not admitted" if failed else "admissible"
    notes = [f"failed: {', '.join(failed)}"] if failed else ["a person admits it once: rci source approve"]
    spec["verdict"] = {"status": status, "notes": notes}
    spec["seconds"] = round(time.monotonic() - started, 1)
    return spec


def _reference(plugin: Any, reqs: pl.DataFrame) -> Reference:
    ref = plugin.reference(reqs.select("rid", *REQUEST_COLUMNS))
    rows = ref.rows if not ref.rows.is_empty() else pl.DataFrame(schema={"entity": pl.String, "field": pl.String})
    return Reference(rows, ref.answered)


def _replay_canary(fetch: Any, fields: pl.DataFrame, rng: np.random.Generator) -> Check:
    """Delete, with hindsight, five events of one entity from the full fetch: the replay must notice."""
    name = "canary: replay catches a hindsight deletion"
    e = events(fields).filter(pl.col("n") >= 20)
    if e.is_empty():
        return Check(name, "skip", "no entity with enough events")
    entities = e["entity"].unique().sort().to_list()
    entity = entities[int(rng.integers(len(entities)))]
    rows = e.filter(pl.col("entity") == entity)
    busiest = rows.group_by("field").len().sort(["len", "field"], descending=[True, False])["field"][0]
    times = rows.filter(pl.col("field") == busiest)["event_time"].sort()
    i = int(len(times) * rng.uniform(0.3, 0.7))
    lo, hi = times[i], times[min(i + 4, len(times) - 1)]
    cut = fields.filter(~((pl.col("entity") == entity) & pl.col("event_time").is_between(lo, hi)))
    check, _ = replay(fetch, cut, [Cut(int(pd.Timestamp(lo).value), "canary", (entity,))])
    if check.status == "fail":
        return Check(name, "pass", f"deleting {entity} {busiest} from {lo:%Y-%m-%d %H:%M} was caught")
    return Check(name, "fail", f"deleting {entity} {busiest} from {lo:%Y-%m-%d %H:%M} went unnoticed")


def _reference_canary(
    reqs: pl.DataFrame, fields: pl.DataFrame, ref: Reference, results: list[dict[str, Any]], tol: int
) -> Check:
    """Alter, on our side, three rows the source agreed with: the comparison must now disagree on each."""
    name = "canary: reconciliation catches altered values"
    agreed = [r["rid"] for r in results if r["kind"] == "sample" and r["status"] == "match"][:3]
    if not agreed:
        return Check(name, "skip", "the source agreed with no sampled row")
    picked = reqs.filter(pl.col("rid").is_in(agreed))
    hit = pl.lit(False)
    for r in picked.iter_rows(named=True):
        hit = hit | (
            (pl.col("entity") == r["entity"])
            & (pl.col("field") == r["field"])
            & (pl.col("event_time") >= lit_time(int(pd.Timestamp(r["start"]).value)))
            & (pl.col("event_time") <= lit_time(int(pd.Timestamp(r["end"]).value)))
        )
    altered = fields.with_columns(value=pl.when(hit).then(pl.col("value") * 1.01 + 1e-9).otherwise("value"))
    again = rc.compare(picked, altered, ref, tol)
    caught = sum(r["status"] == "mismatch" for r in again)
    if caught == len(agreed):
        return Check(name, "pass", f"{caught} of {len(agreed)} altered rows caught")
    return Check(name, "fail", f"only {caught} of {len(agreed)} altered rows caught")
