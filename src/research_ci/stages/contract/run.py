"""Deriving a contract: load the plugins it names, test them, and work out what the data can decide.

Nothing here knows what a field or a strategy means; labels, targets, costs and candidates are plugins. Nothing here
runs a backtest: a consumer the contract names only declares the decision times, and the algo layer runs it. Derive
builds what its steps read, runs its default list (``stages/contract/defaults.py``) and its floor list
(``stages/contract/floor.py``) side by side, and puts what they found into a JSON-ready spec. Planning numbers come
from dispersion only; no level of any target is reported.
"""

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
import polars as pl

from research_ci.api.contract import FLOOR_ALTERNATIVES, Contract, Effect, Label, Use, as_use, span
from research_ci.api.plugins import ConsumerPlugin, LabelPlugin, TargetPlugin
from research_ci.kernel.holdout.tiers import Tier, Tiers, visible
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import from_ns, span_ns, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.worlds import floor_world
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages import skeleton
from research_ci.stages.common import (
    DEFAULT_TARGET,
    DeriveError,
    default_workers,
    iso,
    label_values,
    median,
    norm_candidates,
    quantile_times,
    target_use,
)
from research_ci.stages.contract import defaults, floor
from research_ci.stages.data import defaults as data_defaults
from research_ci.store import vault


@dataclass(frozen=True)
class Settings:
    """How hard derive tests plugins: the cut points of the universe and label checks. When data counts as old comes
    from the data layer's defaults (``STALE_AFTER``)."""

    workers: int = field(default_factory=lambda: int(os.environ.get("RCI_WORKERS", default_workers())))
    universe_cuts: int = 3
    label_cuts: int = 2


@dataclass
class Derived:
    spec: dict[str, Any]
    deps: dict[str, str]


def derive(contract: Contract, registry: Registry | None = None, settings: Settings | None = None) -> Derived:
    return _Derivation(contract, registry, settings or Settings()).run()


class _Derivation:
    def __init__(self, c: Contract, registry: Registry | None, settings: Settings) -> None:
        self.c, self.registry, self.s = c, registry, settings
        self.deps: dict[str, str] = {}
        self.spec: dict[str, Any] = {"claim": c.claim}

    def run(self) -> Derived:
        started = time.monotonic()
        seen = self._seen()
        with ThreadPoolExecutor(max_workers=self.s.workers) as pool:
            listed = skeleton.ordered(floor.STEPS, defaults.CHECKS)
            jobs = [pool.submit(step.run, read) for step, read in floor.runs(listed, seen)]
            found = skeleton.combined(job.result() for job in jobs)
        self.spec["decisive"] = found.shown["decisive"]
        warnings = found.warnings + self._forward()
        self._rest(found.shown["decisive"], [(c.name, c.detail) for c in found.checks if c.status == "fail"])
        self.spec["checks"] = [asdict(ch) for ch in found.checks]
        self.spec["warnings"] = warnings
        self.spec["seconds"] = round(time.monotonic() - started, 1)
        return Derived(self.spec, dict(sorted(self.deps.items())))

    def _seen(self) -> floor.Seen:
        """What the steps read: the explore data, the decision times, the candidates, and the plugins."""
        fields, latest = self._data()
        consumer = self._plugin(as_use(self.c.consumer)) if self.c.consumer is not None else None
        period, times = self._decisions(consumer)
        universe, cands, cuts, times = self._universe(fields, times)
        use = as_use(self.c.costs)
        costs = self._plugin(use)
        round_trip = None if cands is None else median(pl.Series(costs.round_trip(cands)))
        self.spec["costs"] = {"plugin": use.name, "options": dict(use.options), "round_trip_median": round_trip}
        labels = {label.name: self._plugin(as_use(label.use)) for label in self.c.labels}
        self.spec["labels"] = [
            {"name": label.name, "plugin": as_use(label.use).name, "horizons": list(label.horizons)}
            for label in self.c.labels
        ]
        assert cands is not None
        decisive = tuple(self._decisive(e, labels, fields, cands) for e in self.c.decisive)
        return floor.Seen(
            contract=self.c,
            registry=self.registry,
            fields=fields,
            latest=latest,
            now=self.now,
            start=self.start,
            end=self.explore_end,
            period=period,
            times=times,
            candidates=cands,
            universe=universe,
            cuts=cuts,
            label_cuts=self.s.label_cuts,
            labels=labels,
            costs=costs,
            decisive=decisive,
            heavy=data_defaults.CONCENTRATION,
        )

    def _plugin(self, use: Use) -> Any:
        plugin = load(use, self.registry)
        self.deps[f"plugin:{use.name}"] = code_hash(plugin)
        return plugin

    def _data(self) -> tuple[pl.DataFrame, int]:
        source = as_use(self.c.data)
        fields = field_table(self._plugin(source).fetch())
        digest, _ = vault.put(source.name, fields)
        self.deps["data"] = digest
        latest = int(fields.select(pl.col("available_at").max().dt.epoch("ns")).item())
        self.explore_end = latest if self.c.explore_end == "latest" else to_ns(pd.Timestamp(self.c.explore_end))
        seal_end = to_ns(pd.Timestamp(self.c.seal_end)) if self.c.seal_end else None
        explore = visible(fields, Tier.EXPLORE, Tiers(self.explore_end, seal_end))
        self.start = int(explore.select(pl.col("available_at").min().dt.epoch("ns")).item())
        self.now = time.time_ns()
        age = (self.now - latest) / pw.DAY_NS
        self.spec["data"] = {
            "source": source.name,
            "hash": digest,
            "rows": explore.height,
            "insts": explore["inst"].n_unique(),
            "first": iso(self.start),
            "explore_end": iso(self.explore_end),
            "age_days": round(age, 1),
        }
        return explore, latest

    def _decisions(self, consumer: ConsumerPlugin | None) -> tuple[int, list[int]]:
        if consumer is not None:
            timing = consumer.timing()
            every, offset, note = timing.every, timing.offset, timing.note
            origin = f"consumer {as_use(self.c.consumer).name}" if self.c.consumer is not None else ""
        else:
            assert self.c.execution is not None
            e = self.c.execution
            every, offset, note, origin = e.every, e.offset, e.note, "execution"
        period = span_ns(span(every))
        times = pw.decision_grid(period, span_ns(span(offset)), self.start, self.explore_end)
        self.spec["decisions"] = {"every": every, "offset": offset, "note": note, "from": origin, "count": len(times)}
        return period, times

    def _universe(
        self, fields: pl.DataFrame, times: list[int]
    ) -> tuple[Any, pl.DataFrame | None, list[int], list[int]]:
        """The universe as the steps read it: how it computes candidates, the candidates at the decision times, the
        times its checks cut the data at, and the decision times that have candidates."""
        u = self.c.universe
        if u is None:
            return None, None, [], times
        use = as_use(u.use)
        plugin = self._plugin(use)

        def candidates(f: pl.DataFrame, ts: list[int]) -> pl.DataFrame:
            return norm_candidates(plugin.candidates(f, [from_ns(t) for t in ts]))

        full = candidates(fields, times)
        missing = [col for col in (u.clusters, *u.groups) if col not in full.columns]
        if missing:
            raise DeriveError(f"the universe has no column {', '.join(missing)}")
        per_t = full.group_by("t").agg(n=pl.len(), clusters=pl.col(u.clusters).n_unique())
        self.spec["universe"] = {
            "plugin": use.name,
            "clusters": u.clusters,
            "membership": u.membership,
            "shape": u.shape,
            "world": floor_world(u.clusters, 0, u.shape).describe(),
            "times": per_t.height,
            "candidates_median": median(per_t["n"]),
            "clusters_median": median(per_t["clusters"]),
        }
        cuts = quantile_times(per_t["t"], self.s.universe_cuts)
        present = sorted(int(t) for t in per_t.select(pl.col("t").dt.epoch("ns"))["t"])
        return candidates, full, cuts, present

    def _decisive(
        self, effect: Effect, labels: dict[str, LabelPlugin], fields: pl.DataFrame, cands: pl.DataFrame
    ) -> floor.Decisive:
        """A decisive effect's label rows at each horizon, joined with the candidates, and the target it names."""
        label = _label(self.c, effect.on)
        use = target_use(effect, self.c)
        target: TargetPlugin = self._plugin(use)
        found = label_values(labels[label.name], fields, cands, [span(h) for h in label.horizons])
        rows = tuple(
            (h, v.join(cands, on=["t", "inst"], how="inner", suffix="_candidate"))
            for h, v in zip(label.horizons, found, strict=True)
        )
        return floor.Decisive(effect, label, use, target, rows)

    def _rest(self, decisive: list[dict[str, Any]], failed: list[tuple[str, str]]) -> None:
        c = self.c
        affected = {a for u in c.used or () for a in u.affects}
        confirm_only = bool(affected & {"claim", "labels", "costs"}) or (
            c.universe is not None and c.universe.membership == "static"
        )
        floors = floor_world(c.universe.clusters, 0, c.universe.shape).explains if c.universe else FLOOR_ALTERNATIVES
        self.spec["alternatives"] = {"idea": [asdict(a) for a in c.alternatives], "floor": list(floors)}
        self.spec["used"] = [asdict(u) for u in c.used or ()]
        self.spec["history"] = "explore-only" if confirm_only else "can confirm"
        self.spec["seal_end"] = c.seal_end
        self.spec["budget"] = c.budget
        self.spec["verdict"] = _verdict(decisive, failed)

    def _forward(self) -> list[str]:
        """The forward source the contract names, and whether it is installed; a warning when it is not."""
        f = self.c.forward
        self.spec["forward"] = None
        if f is None:
            return []
        source = as_use(f.source)
        try:
            plugin = load(source, self.registry)
        except ValueError as exc:
            self.spec["forward"] = {"source": source.name, "installed": False}
            return [f"forward source {source.name}: {exc}"]
        self.deps[f"plugin:{source.name}"] = code_hash(plugin)
        self.spec["forward"] = {"source": source.name, "installed": True}
        return []


def _verdict(decisive: list[dict[str, Any]], failed: list[tuple[str, str]]) -> dict[str, Any]:
    """Decidable when no check failed and every decisive effect's noise can be measured on some horizon; blocked
    otherwise, each failure with why and what to do."""
    notes = [
        f"{name} failed: {why}; fix the contract, or the plugin it names, and derive again" for name, why in failed
    ]
    measurable = True
    for d in decisive:
        measured = [r["horizon"] for r in d["horizons"] if math.isfinite(r["mde"])]
        if measured:
            notes.append(
                f"{d['on']}: noise measured at {', '.join(measured)}; feat passes it where it beats random and no lab "
                "eval or null world rejects it"
            )
        else:
            notes.append(
                f"{d['on']}: too few blocks on every horizon to measure the noise; give the contract a longer explore "
                "period or a shorter horizon"
            )
            measurable = False
    return {"status": "decidable" if measurable and not failed else "blocked", "notes": notes}


def _label(c: Contract, name: str) -> Label:
    return next(label for label in c.labels if label.name == name)


def current_deps(contract: Contract, recorded: dict[str, str], registry: Registry | None = None) -> dict[str, str]:
    """The dependencies as they are now, keyed like the recorded ones: plugin code and the data snapshot (which must
    still be in the vault; data is not fetched again)."""
    now: dict[str, str] = {}
    for key in recorded:
        if key.startswith("plugin:"):
            name = key.removeprefix("plugin:")
            now[key] = code_hash(load(use_named(contract, name), registry))
    digest = recorded.get("data", "")
    now["data"] = digest if vault.has(as_use(contract.data).name, digest) else "missing"
    return now


def use_named(contract: Contract, name: str) -> Use:
    uses = [as_use(contract.data), as_use(contract.costs)]
    uses += [as_use(contract.consumer)] if contract.consumer is not None else []
    uses += [as_use(contract.universe.use)] if contract.universe is not None else []
    uses += [as_use(label.use) for label in contract.labels]
    uses += [as_use(e.target) for e in contract.decisive if e.target is not None]
    uses += [as_use(contract.forward.source)] if contract.forward is not None else []
    if contract.universe is not None:
        uses.append(Use(DEFAULT_TARGET, {"cluster": contract.universe.clusters}))
    for use in uses:
        if use.name == name:
            return use
    raise DeriveError(f"the contract no longer names the plugin {name!r}; run `rci contract derive` again")
