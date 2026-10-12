"""Admitting a data source: does the plugin give, for any past time, what was known then, and does an independent
source agree with what it gives? Run once per plugin, options and code; every idea using the source reuses it, and
the first approval of an idea's data that uses it admits it. The steps are the admission's floor list (``STEPS``), run
in order.

The steps of a data source's admission, run once per plugin, options and code: the data check runs them the first
time an idea uses the source. They check fidelity, that the rows are what the venue reported, and have no default
list: two canaries check the checks on this very data."""

import inspect
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import Use, span
from research_ci.api.data import Levels
from research_ci.api.plugins import REQUEST_COLUMNS, Reference
from research_ci.eval.data import reconcile as rc
from research_ci.eval.data.baseline import baseline, events
from research_ci.eval.data.replay import Cut, cut_points, replay
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import from_ns, lit_time, span_ns
from research_ci.kernel.verdict.check import Check
from research_ci.runner.plugins import code_hash
from research_ci.stages.skeleton import Found, Step, combined

Fetch = Callable[[int, list[str]], pl.DataFrame]


@dataclass(frozen=True)
class SourceSettings:
    holes: int = 200
    late: int = 40
    randoms: int = 5
    random_insts: int = 20
    per_field: int = 12
    max_anomalies: int = 60
    tolerance: str = "1s"
    seed: int = 0


def can_replay(plugin: Any) -> bool:
    try:
        params = inspect.signature(plugin.fetch).parameters
    except (TypeError, ValueError):
        return False
    return "as_of" in params and "insts" in params


def declared_levels(plugin: Any) -> tuple[tuple[str, ...], float, int]:
    """The plugin's ``levels`` as the baseline takes them: fields, factor and window; none when it declares none."""
    levels = getattr(plugin, "levels", None)
    if levels is None:
        return (), 0.0, 0
    if not isinstance(levels, Levels):
        raise TypeError("a data plugin's levels must be an research_ci.api.Levels")
    return levels.fields, levels.factor, levels.window


@dataclass(frozen=True)
class Seen:
    """What an admission step reads: the data plugin, its full fetch and field names, the settings, the random draws
    the steps share in order, and the fetch as of a time and insts, ``None`` when the plugin cannot replay. The
    baseline and the independent source's answers are worked out once, when a step first needs them."""

    plugin: Any
    fields: pl.DataFrame
    names: list[str]
    settings: SourceSettings
    rng: np.random.Generator
    fetch: Fetch | None

    @cached_property
    def baseline(self) -> tuple[list[Check], pl.DataFrame]:
        return baseline(self.fields, *declared_levels(self.plugin))

    @cached_property
    def tolerance(self) -> int:
        return span_ns(span(self.settings.tolerance))

    @cached_property
    def reconciled(self) -> tuple[pl.DataFrame, Reference, list[dict[str, Any]]]:
        """The requests to the independent source, its answers, and how each compares with our rows."""
        s = self.settings
        reqs = rc.requests(self.fields, self.baseline[1], s.per_field, s.max_anomalies, self.tolerance, s.seed)
        ref = self.plugin.reference(reqs.select("rid", *REQUEST_COLUMNS))
        rows = ref.rows if not ref.rows.is_empty() else pl.DataFrame(schema={"inst": pl.String, "field": pl.String})
        ref = Reference(rows, ref.answered)
        return reqs, ref, rc.compare(reqs, self.fields, ref, self.tolerance)

    @property
    def has_reference(self) -> bool:
        return callable(getattr(self.plugin, "reference", None))


def generic(seen: Seen) -> Found:
    """Generic checks of the rows, and the anomalies they flag: long gaps, near-duplicate events, and level shifts in
    the declared levels. The independent source is asked about the anomalies."""
    checks, anomalies = seen.baseline
    return Found(list(checks), shown={"anomalies": {k: n for k, n in anomalies.group_by("kind").len().iter_rows()}})


def replayed(seen: Seen) -> Found:
    """The plugin's fetch as of past times, for some insts, must equal the full fetch cut at those times."""
    if seen.fetch is None:
        return Found([Check("replay as of past times", "fail", "the plugin cannot fetch as of a time and insts")])
    s = seen.settings
    cuts = cut_points(seen.fields, seen.rng, s.holes, s.late, s.randoms, s.random_insts)
    check, results = replay(seen.fetch, seen.fields, cuts)
    return Found([check], shown={"replay": results})


def replay_canary(seen: Seen) -> Found:
    """Delete, with hindsight, five events of one inst from the full fetch: the replay must notice."""
    if seen.fetch is None:
        return Found()
    name = "canary: replay catches a hindsight deletion"
    e = events(seen.fields).filter(pl.col("n") >= 20)
    if e.is_empty():
        return Found([Check(name, "skip", "no inst with enough events")])
    insts = e["inst"].unique().sort().to_list()
    inst = insts[int(seen.rng.integers(len(insts)))]
    rows = e.filter(pl.col("inst") == inst)
    busiest = rows.group_by("field").len().sort(["len", "field"], descending=[True, False])["field"][0]
    times = rows.filter(pl.col("field") == busiest)["event_time"].sort()
    i = int(len(times) * seen.rng.uniform(0.3, 0.7))
    lo, hi = times[i], times[min(i + 4, len(times) - 1)]
    cut = seen.fields.filter(~((pl.col("inst") == inst) & pl.col("event_time").is_between(lo, hi)))
    check, _ = replay(seen.fetch, cut, [Cut(int(pd.Timestamp(lo).value), "canary", (inst,))])
    if check.status == "fail":
        return Found([Check(name, "pass", f"deleting {inst} {busiest} from {lo:%Y-%m-%d %H:%M} was caught")])
    return Found([Check(name, "fail", f"deleting {inst} {busiest} from {lo:%Y-%m-%d %H:%M} went unnoticed")])


def agrees(seen: Seen) -> Found:
    """An independent source, asked about random rows of every field and about the anomalies, must agree: it grades
    each field, and confirms the level shifts that are real."""
    if not seen.has_reference:
        check = Check("independent source agrees", "warn", "the plugin names no independent source")
        return Found([check], shown={"grades": dict.fromkeys(seen.names, "unverified")})
    reqs, _, results = seen.reconciled
    check, grades, table = rc.summarize(results, seen.names)
    shifts = reqs.filter(pl.col("kind") == "shift").select("rid", "inst", "field", "end")
    ok = {r["rid"] for r in results if r["kind"] == "shift" and r["status"] == "match"}
    shown = {
        "grades": grades,
        "reference": {"note": getattr(seen.plugin, "reference_note", ""), "counts": table},
        "mismatches": [r for r in results if r["status"] == "mismatch"][:20],
        "confirmed_shifts": [
            [e, f, end.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")] for rid, e, f, end in shifts.iter_rows() if rid in ok
        ],
    }
    return Found([check], shown=shown)


def agreement_canary(seen: Seen) -> Found:
    """Alter, on our side, three rows the source agreed with: the comparison must now disagree on each."""
    if not seen.has_reference:
        return Found()
    reqs, ref, results = seen.reconciled
    name = "canary: reconciliation catches altered values"
    agreed = [r["rid"] for r in results if r["kind"] == "sample" and r["status"] == "match"][:3]
    if not agreed:
        return Found([Check(name, "skip", "the source agreed with no sampled row")])
    picked = reqs.filter(pl.col("rid").is_in(agreed))
    hit = pl.lit(False)
    for r in picked.iter_rows(named=True):
        hit = hit | (
            (pl.col("inst") == r["inst"])
            & (pl.col("field") == r["field"])
            & (pl.col("event_time") >= lit_time(int(pd.Timestamp(r["start"]).value)))
            & (pl.col("event_time") <= lit_time(int(pd.Timestamp(r["end"]).value)))
        )
    altered = seen.fields.with_columns(value=pl.when(hit).then(pl.col("value") * 1.01 + 1e-9).otherwise("value"))
    again = rc.compare(picked, altered, ref, seen.tolerance)
    caught = sum(r["status"] == "mismatch" for r in again)
    if caught == len(agreed):
        return Found([Check(name, "pass", f"{caught} of {len(agreed)} altered rows caught")])
    return Found([Check(name, "fail", f"only {caught} of {len(agreed)} altered rows caught")])


STEPS = (
    Step("baseline", "warns", generic, "Gaps, near-duplicates, and level shifts in the rows"),
    Step("replay as of past times", "blocks", replayed, "Fetches as of past times match the full fetch"),
    Step("canary: replay", "blocks", replay_canary, "The replay catches a deletion made with hindsight"),
    Step("independent source agrees", "blocks", agrees, "Another source agrees; grades every field", "reference"),
    Step("canary: reconciliation", "blocks", agreement_canary, "The comparison catches altered values", "reference"),
)


def admit(use: Use, plugin: Any, fields: pl.DataFrame, settings: SourceSettings | None = None) -> dict[str, Any]:
    s = settings or SourceSettings()
    started = time.monotonic()
    names = sorted(fields["field"].unique().to_list())
    spec: dict[str, Any] = {
        "plugin": use.name,
        "options": dict(use.options),
        "code": code_hash(plugin),
        "rows": fields.height,
        "insts": fields["inst"].n_unique(),
        "fields": names,
        "settings": asdict(s),
    }

    def fetch(at: int, insts: list[str]) -> pl.DataFrame:
        return field_table(plugin.fetch(as_of=from_ns(at), insts=insts))

    seen = Seen(plugin, fields, names, s, np.random.default_rng(s.seed), fetch if can_replay(plugin) else None)
    found = combined(step.run(seen) for step in STEPS)
    spec |= found.shown
    spec["checks"] = [asdict(c) for c in found.checks]
    failed = [c.name for c in found.checks if c.status == "fail"]
    status = "not admitted" if failed else "admissible"
    notes = [f"failed: {', '.join(failed)}"] if failed else ["admitted once, by approving the data of an idea"]
    spec["verdict"] = {"status": status, "notes": notes}
    spec["seconds"] = round(time.monotonic() - started, 1)
    return spec
