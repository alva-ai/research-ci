"""Admitting a data source: does the plugin give, for any past time, what was known then, and does an independent
source agree with what it gives? Run once per plugin, options and code; every idea using the source reuses it, and
the first approval of an idea's data that uses it admits it. The steps are the admission's floor list (``STEPS``), run
in order.

The steps of a data source's admission, run once per plugin, options and code: the data check runs them the first
time an idea uses the source. They check fidelity, that the rows are what the venue reported, on every source the
plugin gives at its own cadence and column by column, and have no default list: two canaries check the checks on this
very data."""

import inspect
import time
from collections.abc import Callable, Mapping
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
from research_ci.eval.data.baseline import baseline, events, rows
from research_ci.eval.data.replay import Cut, cut_points, replay
from research_ci.kernel.tables import source_tables, values
from research_ci.kernel.time.clock import from_ns, lit_time, span_ns
from research_ci.kernel.verdict.check import Check
from research_ci.runner.plugins import code_hash
from research_ci.stages.skeleton import Found, Step, combined

Sources = Mapping[str, pl.DataFrame]
Fetch = Callable[[int, list[str]], dict[str, pl.DataFrame]]


@dataclass(frozen=True)
class SourceSettings:
    holes: int = 200
    late: int = 40
    randoms: int = 5
    random_insts: int = 20
    per_column: int = 12
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
    """The plugin's ``levels`` as the baseline takes them: columns, factor and window; none when it declares none."""
    levels = getattr(plugin, "levels", None)
    if levels is None:
        return (), 0.0, 0
    if not isinstance(levels, Levels):
        raise TypeError("a data plugin's levels must be an research_ci.api.Levels")
    return levels.columns, levels.factor, levels.window


def column_names(sources: Sources) -> list[str]:
    """Every value column as ``<source>.<column>``, the name grades and cards give it."""
    return [f"{name}.{c}" for name, table in sources.items() for c in values(table)]


@dataclass(frozen=True)
class Seen:
    """What an admission step reads: the data plugin, its full fetch and column names, the settings, the random draws
    the steps share in order, and the fetch as of a time and insts, ``None`` when the plugin cannot replay. The
    baseline and the independent source's answers are worked out once, when a step first needs them."""

    plugin: Any
    sources: Sources
    names: list[str]
    settings: SourceSettings
    rng: np.random.Generator
    fetch: Fetch | None

    @cached_property
    def baseline(self) -> tuple[list[Check], pl.DataFrame]:
        return baseline(self.sources, *declared_levels(self.plugin))

    @cached_property
    def tolerance(self) -> int:
        return span_ns(span(self.settings.tolerance))

    @cached_property
    def reconciled(self) -> tuple[pl.DataFrame, Reference, list[dict[str, Any]]]:
        """The requests to the independent source, its answers, and how each compares with our rows."""
        s = self.settings
        reqs = rc.requests(self.sources, self.baseline[1], s.per_column, s.max_anomalies, self.tolerance, s.seed)
        ref = self.plugin.reference(reqs.select("rid", *REQUEST_COLUMNS))
        return reqs, ref, rc.compare(reqs, self.sources, ref, self.tolerance)

    @property
    def has_reference(self) -> bool:
        return callable(getattr(self.plugin, "reference", None))


def generic(seen: Seen) -> Found:
    """Generic checks of every source's rows, and the anomalies they flag: long gaps and near-duplicate events of a
    source and an inst, and level shifts in the declared levels. The independent source is asked about the
    anomalies."""
    checks, anomalies = seen.baseline
    return Found(list(checks), shown={"anomalies": {k: n for k, n in anomalies.group_by("kind").len().iter_rows()}})


def replayed(seen: Seen) -> Found:
    """The plugin's fetch as of past times, for some insts, must equal the full fetch cut at those times, source by
    source."""
    if seen.fetch is None:
        return Found([Check("replay as of past times", "fail", "the plugin cannot fetch as of a time and insts")])
    s = seen.settings
    cuts = cut_points(seen.sources, seen.rng, s.holes, s.late, s.randoms, s.random_insts)
    check, results = replay(seen.fetch, seen.sources, cuts)
    return Found([check], shown={"replay": results})


def replay_canary(seen: Seen) -> Found:
    """Delete, with hindsight, five events of one inst from its busiest source in the full fetch: the replay must
    notice."""
    if seen.fetch is None:
        return Found()
    name = "canary: replay catches a hindsight deletion"
    e = events(rows(seen.sources)).filter(pl.col("n") >= 20)
    if e.is_empty():
        return Found([Check(name, "skip", "no inst with enough events")])
    insts = e["inst"].unique().sort().to_list()
    inst = insts[int(seen.rng.integers(len(insts)))]
    found = e.filter(pl.col("inst") == inst)
    busiest = found.group_by("source").len().sort(["len", "source"], descending=[True, False])["source"][0]
    times = found.filter(pl.col("source") == busiest)["event_time"].sort()
    i = int(len(times) * seen.rng.uniform(0.3, 0.7))
    lo, hi = times[i], times[min(i + 4, len(times) - 1)]
    gone = (pl.col("inst") == inst) & pl.col("event_time").is_between(lo, hi)
    cut = {**seen.sources, busiest: seen.sources[busiest].filter(~gone)}
    check, _ = replay(seen.fetch, cut, [Cut(int(pd.Timestamp(lo).value), "canary", (inst,))])
    what = f"deleting {inst} from {busiest} from {lo:%Y-%m-%d %H:%M}"
    if check.status == "fail":
        return Found([Check(name, "pass", f"{what} was caught")])
    return Found([Check(name, "fail", f"{what} went unnoticed")])


def agrees(seen: Seen) -> Found:
    """An independent source, asked about random cells of every column and about the anomalies, must agree: it grades
    each column, and confirms the level shifts that are real."""
    if not seen.has_reference:
        check = Check("independent source agrees", "warn", "the plugin names no independent source")
        return Found([check], shown={"grades": dict.fromkeys(seen.names, "unverified")})
    reqs, _, results = seen.reconciled
    check, grades, table = rc.summarize(results, seen.names)
    shifts = reqs.filter(pl.col("kind") == "shift").select("rid", "source", "column", "inst", "end")
    ok = {r["rid"] for r in results if r["kind"] == "shift" and r["status"] == "match"}
    shown = {
        "grades": grades,
        "reference": {"note": getattr(seen.plugin, "reference_note", ""), "counts": table},
        "mismatches": [r for r in results if r["status"] == "mismatch"][:20],
        "confirmed_shifts": [
            [src, c, e, end.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")]
            for rid, src, c, e, end in shifts.iter_rows()
            if rid in ok
        ],
    }
    return Found([check], shown=shown)


def agreement_canary(seen: Seen) -> Found:
    """Alter, on our side, three cells the source agreed with: the comparison must now disagree on each."""
    if not seen.has_reference:
        return Found()
    reqs, ref, results = seen.reconciled
    name = "canary: reconciliation catches altered values"
    agreed = [r["rid"] for r in results if r["kind"] == "sample" and r["status"] == "match"][:3]
    if not agreed:
        return Found([Check(name, "skip", "the source agreed with no sampled cell")])
    picked = reqs.filter(pl.col("rid").is_in(agreed))
    altered = dict(seen.sources)
    for r in picked.iter_rows(named=True):
        hit = (
            (pl.col("inst") == r["inst"])
            & (pl.col("event_time") >= lit_time(int(pd.Timestamp(r["start"]).value)))
            & (pl.col("event_time") <= lit_time(int(pd.Timestamp(r["end"]).value)))
        )
        c = pl.col(r["column"])
        altered[r["source"]] = altered[r["source"]].with_columns(pl.when(hit).then(c * 1.01 + 1e-9).otherwise(c))
    again = rc.compare(picked, altered, ref, seen.tolerance)
    caught = sum(r["status"] == "mismatch" for r in again)
    if caught == len(agreed):
        return Found([Check(name, "pass", f"{caught} of {len(agreed)} altered cells caught")])
    return Found([Check(name, "fail", f"only {caught} of {len(agreed)} altered cells caught")])


STEPS = (
    Step("baseline", "warns", generic, "Gaps, near-duplicates, and level shifts in every source"),
    Step("replay as of past times", "blocks", replayed, "Fetches as of past times match the full fetch"),
    Step("canary: replay", "blocks", replay_canary, "The replay catches a deletion made with hindsight"),
    Step("independent source agrees", "blocks", agrees, "Another source agrees; grades every column", "reference"),
    Step("canary: reconciliation", "blocks", agreement_canary, "The comparison catches altered values", "reference"),
)


def admit(use: Use, plugin: Any, sources: Sources, settings: SourceSettings | None = None) -> dict[str, Any]:
    s = settings or SourceSettings()
    started = time.monotonic()
    names = column_names(sources)
    spec: dict[str, Any] = {
        "plugin": use.name,
        "options": dict(use.options),
        "code": code_hash(plugin),
        "rows": sum(t.height for t in sources.values()),
        "insts": rows(sources)["inst"].n_unique(),
        "sources": {name: values(t) for name, t in sources.items()},
        "settings": asdict(s),
    }

    def fetch(at: int, insts: list[str]) -> dict[str, pl.DataFrame]:
        return source_tables(plugin.fetch(as_of=from_ns(at), insts=insts))

    seen = Seen(plugin, sources, names, s, np.random.default_rng(s.seed), fetch if can_replay(plugin) else None)
    found = combined(step.run(seen) for step in STEPS)
    spec |= found.shown
    spec["checks"] = [asdict(c) for c in found.checks]
    failed = [c.name for c in found.checks if c.status == "fail"]
    status = "not admitted" if failed else "admissible"
    notes = [f"failed: {', '.join(failed)}"] if failed else ["admitted once, by approving the data of an idea"]
    spec["verdict"] = {"status": status, "notes": notes}
    spec["seconds"] = round(time.monotonic() - started, 1)
    return spec
