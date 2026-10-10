"""Deriving a contract: load the plugins it names, test them, and work out what the data can decide.

Nothing here knows what a field or a strategy means; labels, targets, costs and candidates are plugins. Nothing here
runs a backtest: a consumer the contract names only declares the decision times, and the algo layer runs it. The
result is a JSON-ready spec. Planning numbers come from dispersion only; no level of any target is reported.
"""

import math
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import FLOOR_ALTERNATIVES, Contract, ContractError, Effect, Label, Use, as_use, span
from research_ci.api.plugins import ConsumerPlugin, CostModel, LabelPlugin, TargetPlugin, UniversePlugin
from research_ci.eval import protocol as pr
from research_ci.eval.data import concentration as cc
from research_ci.kernel.holdout.tiers import Tier, Tiers, visible
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import TIME, from_ns, span_ns, time_series, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages.defaults import data as defaults
from research_ci.store import vault

DEFAULT_TARGET = "ref.mean"


class DeriveError(ContractError):
    pass


@dataclass(frozen=True)
class Settings:
    """How hard derive tests plugins: the cut points of the universe and label checks. When data counts as old comes
    from the data layer's defaults (``STALE_AFTER``)."""

    workers: int = field(default_factory=lambda: int(os.environ.get("RCI_WORKERS", _default_workers())))
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
        self.checks: list[pr.Check] = []
        self.later: list[tuple[int, Future[pr.Check]]] = []
        self.warnings: list[str] = []
        self.deps: dict[str, str] = {}
        self.spec: dict[str, Any] = {"claim": c.claim}
        self.labels: dict[str, LabelPlugin] = {}

    def run(self) -> Derived:
        started = time.monotonic()
        fields = self._data()
        consumer = self._plugin(as_use(self.c.consumer)) if self.c.consumer is not None else None
        times = self._decisions(consumer)
        with ThreadPoolExecutor(max_workers=self.s.workers) as pool:
            self.pool = pool
            cands = self._universe(times)
            self.costs: CostModel = self._plugin(as_use(self.c.costs))
            self._costs(cands)
            self._labels(fields, cands)
            assert cands is not None
            decisive = [self._label_effect(effect, fields, cands) for effect in self.c.decisive]
        for i, (at, job) in enumerate(self.later):
            self.checks.insert(at + i, job.result())
        self.spec["decisive"] = decisive
        self._rest(decisive)
        self.spec["checks"] = [asdict(ch) for ch in self.checks]
        self.spec["warnings"] = self.warnings
        self.spec["seconds"] = round(time.monotonic() - started, 1)
        return Derived(self.spec, dict(sorted(self.deps.items())))

    def _plugin(self, use: Use) -> Any:
        plugin = load(use, self.registry)
        self.deps[f"plugin:{use.name}"] = code_hash(plugin)
        return plugin

    def _data(self) -> pl.DataFrame:
        source = as_use(self.c.data)
        fields = field_table(self._plugin(source).fetch())
        digest, _ = vault.put(source.name, fields)
        self.deps["data"] = digest
        latest = int(fields.select(pl.col("available_at").max().dt.epoch("ns")).item())
        self.explore_end = latest if self.c.explore_end == "latest" else to_ns(pd.Timestamp(self.c.explore_end))
        seal_end = to_ns(pd.Timestamp(self.c.seal_end)) if self.c.seal_end else None
        explore = visible(fields, Tier.EXPLORE, Tiers(self.explore_end, seal_end))
        self.fields = explore
        self.start = int(explore.select(pl.col("available_at").min().dt.epoch("ns")).item())
        now = time.time_ns()
        age = (now - latest) / pw.DAY_NS
        self.spec["data"] = {
            "source": source.name,
            "hash": digest,
            "rows": explore.height,
            "entities": explore["entity"].n_unique(),
            "first": _iso(self.start),
            "explore_end": _iso(self.explore_end),
            "age_days": round(age, 1),
        }
        fresh = defaults.freshness(latest, now)
        self.checks += fresh.checks
        self.warnings += fresh.warnings
        return explore

    def _decisions(self, consumer: ConsumerPlugin | None) -> list[int]:
        if consumer is not None:
            timing = consumer.timing()
            every, offset, note = timing.every, timing.offset, timing.note
            origin = f"consumer {as_use(self.c.consumer).name}" if self.c.consumer is not None else ""
        else:
            assert self.c.execution is not None
            e = self.c.execution
            every, offset, note, origin = e.every, e.offset, e.note, "execution"
        self.period = span_ns(span(every))
        times = pw.decision_grid(self.period, span_ns(span(offset)), self.start, self.explore_end)
        self.spec["decisions"] = {"every": every, "offset": offset, "note": note, "from": origin, "count": len(times)}
        self.times = times
        return times

    def _universe(self, times: list[int]) -> pl.DataFrame | None:
        u = self.c.universe
        if u is None:
            return None
        use = as_use(u.use)
        plugin: UniversePlugin = self._plugin(use)

        def candidates(f: pl.DataFrame, ts: list[int]) -> pl.DataFrame:
            return _norm_candidates(plugin.candidates(f, [from_ns(t) for t in ts]))

        full = candidates(self.fields, times)
        missing = [col for col in (u.clusters, *u.groups) if col not in full.columns]
        if missing:
            raise DeriveError(f"the universe has no column {', '.join(missing)}")
        per_t = full.group_by("t").agg(n=pl.len(), clusters=pl.col(u.clusters).n_unique())
        cuts = _quantile_times(per_t["t"], self.s.universe_cuts)
        self._later(pr.universe_pit, candidates, self.fields, full, cuts, self.period)
        if cuts:
            self._later(pr.universe_causal, candidates, self.fields, full, cuts[len(cuts) // 2])
        self.spec["universe"] = {
            "plugin": use.name,
            "clusters": u.clusters,
            "membership": u.membership,
            "times": per_t.height,
            "candidates_median": _median(per_t["n"]),
            "clusters_median": _median(per_t["clusters"]),
        }
        if u.membership == "static":
            self.warnings.append("the candidate list was fixed with hindsight; history is explore-grade for it")
        self.times = sorted(int(t) for t in per_t.select(pl.col("t").dt.epoch("ns"))["t"])
        return full

    def _later(self, check: Callable[..., pr.Check], *args: Any) -> None:
        """Run a check in the pool alongside the rest; its result keeps its place among the checks."""
        self.later.append((len(self.checks), self.pool.submit(check, *args)))

    def _costs(self, cands: pl.DataFrame | None) -> None:
        use = as_use(self.c.costs)
        round_trip = None if cands is None else _median(pl.Series(self.costs.round_trip(cands)))
        self.spec["costs"] = {"plugin": use.name, "options": dict(use.options), "round_trip_median": round_trip}

    def _labels(self, fields: pl.DataFrame, cands: pl.DataFrame | None) -> None:
        """Every declared label: its worked examples must hold and bite, and it must not read past ``t + h``."""
        info = []
        for label in self.c.labels:
            use = as_use(label.use)
            plugin: LabelPlugin = self._plugin(use)
            self.labels[label.name] = plugin
            self.checks.extend(pr.label_cases(label.name, plugin.values, plugin.cases()))
            if cands is not None:
                cuts = _quantile_times(time_series(self.times), self.s.label_cuts)
                for h in sorted({label.horizons[0], label.horizons[-1]}, key=pd.Timedelta):
                    horizon = span(h)
                    self.checks.append(pr.label_truncation(label.name, plugin.values, fields, cands, horizon, cuts))
            info.append({"name": label.name, "plugin": use.name, "horizons": list(label.horizons)})
        self.spec["labels"] = info

    def _label_effect(self, effect: Effect, fields: pl.DataFrame, cands: pl.DataFrame) -> dict[str, Any]:
        label = _label(self.c, effect.on)
        plugin = self.labels[label.name]
        assert self.c.universe is not None
        use = as_use(effect.target) if effect.target else Use(DEFAULT_TARGET, {"cluster": self.c.universe.clusters})
        target: TargetPlugin = self._plugin(use)
        width = self.explore_end - self.start
        rows, shortest, heavy, infinite = [], None, {}, []
        for h in label.horizons:
            horizon = span(h)
            values = plugin.values(fields, cands, horizon)
            joined = values.join(cands, on=["t", "entity"], how="inner", suffix="_candidate")
            shortest = shortest or (horizon, joined)
            raw = _per_t(target.reduce(joined, horizon, self.costs))
            per_t = raw.filter(pl.col("value").is_finite())
            if per_t.height < raw.height:
                infinite.append(f"{raw.height - per_t.height} at {h}")
            p = pw.power(pw.block_se(per_t, span_ns(horizon)), width, span_ns(horizon))
            top = cc.shares(joined, self.c.universe.clusters, span_ns(horizon))
            top_share = top.head(1).rows()
            if top_share and top_share[0][1] > defaults.CONCENTRATION:
                heavy.setdefault(top_share[0][0], []).append(f"{top_share[0][1]:.0%} at {h}")
            rows.append({"horizon": h, "decisions": per_t.height, **asdict(p), "concentration": top_share[:1]})
        if infinite:
            self.warnings.append(
                f"{effect.on}: the number is not finite at some decision times ({', '.join(infinite)}), left out of "
                "the power; rci data check looks at whether the data is fit for research"
            )
        for name, where in heavy.items():
            self.warnings.append(
                f"{effect.on}: {self.c.universe.clusters} {name} carries {' and '.join(where)} of the noise the MDE "
                "rests on; rci data check looks at whether its data is fit for research"
            )
        assert shortest is not None
        h0, table = shortest
        self.checks.append(pr.target_reads_score(use.name, lambda t: target.reduce(t, h0, self.costs), table))
        out = {"kind": "label", **_effect(effect, target), "target": use.name, "horizons": rows}
        return out | {"label": {"name": label.name, "plugin": as_use(label.use).name}}

    def _rest(self, decisive: list[dict[str, Any]]) -> None:
        c = self.c
        affected = {a for u in c.used or () for a in u.affects}
        confirm_only = bool(affected & {"claim", "labels", "costs"}) or (
            c.universe is not None and c.universe.membership == "static"
        )
        self.spec["alternatives"] = {"idea": [asdict(a) for a in c.alternatives], "floor": list(FLOOR_ALTERNATIVES)}
        self.spec["used"] = [asdict(u) for u in c.used or ()]
        self.spec["history"] = "explore-only" if confirm_only else "can confirm"
        self.spec["seal_end"] = c.seal_end
        self.spec["forward"] = self._forward()
        self.spec["budget"] = c.budget
        failed = [c.name for c in self.checks if c.status == "fail"]
        self.spec["verdict"] = _verdict(decisive, failed)

    def _forward(self) -> dict[str, Any] | None:
        f = self.c.forward
        if f is None:
            return None
        source = as_use(f.source)
        try:
            plugin = load(source, self.registry)
        except ValueError as exc:
            self.warnings.append(f"forward source {source.name}: {exc}")
            return {"source": source.name, "installed": False}
        self.deps[f"plugin:{source.name}"] = code_hash(plugin)
        return {"source": source.name, "installed": True}


def _verdict(decisive: list[dict[str, Any]], failed: list[str]) -> dict[str, Any]:
    """Decidable when no check failed and every decisive effect's noise can be measured on some horizon; blocked
    otherwise."""
    notes = [f"failed checks: {', '.join(failed)}"] if failed else []
    measurable = True
    for d in decisive:
        measured = [r["horizon"] for r in d["horizons"] if math.isfinite(r["mde"])]
        if measured:
            notes.append(f"{d['on']}: noise measured at {', '.join(measured)}; feat passes it when it beats random")
        else:
            notes.append(f"{d['on']}: too few blocks on every horizon to measure the noise")
            measurable = False
    return {"status": "decidable" if measurable and not failed else "blocked", "notes": notes}


def _label(c: Contract, name: str) -> Label:
    return next(label for label in c.labels if label.name == name)


def _effect(effect: Effect, target: TargetPlugin) -> dict[str, Any]:
    """The effect as cards show it, with the target's units (``per``, ``rolls``) when it declares them."""
    return {"on": effect.on, "per": getattr(target, "per", None), "rolls": bool(getattr(target, "rolls", False))}


def _per_t(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    out = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    if not {"t", "value"} <= set(out.columns):
        raise DeriveError("a target must return columns t and value")
    return out.select(pl.col("t").cast(TIME), pl.col("value").cast(pl.Float64)).drop_nulls().sort("t")


def _norm_candidates(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    frame = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
    missing = {"t", "entity"} - set(frame.columns)
    if missing:
        raise DeriveError(f"the universe must return columns t and entity; missing {', '.join(sorted(missing))}")
    return frame.with_columns(pl.col("t").dt.convert_time_zone("UTC").cast(TIME), pl.col("entity").cast(pl.String))


def _quantile_times(times: pl.Series, k: int) -> list[int]:
    ns = sorted(int(x) for x in times.cast(pl.Int64).to_list())
    if not ns or k < 1:
        return []
    return sorted({ns[int(len(ns) * (i + 1) / (k + 1))] for i in range(k)})


def _iso(ns: int) -> str:
    return from_ns(ns).strftime("%Y-%m-%dT%H:%MZ")


def _median(series: pl.Series) -> float:
    return float(series.cast(pl.Float64).median() or 0.0)  # type: ignore[arg-type]


def _default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def current_deps(contract: Contract, recorded: dict[str, str], registry: Registry | None = None) -> dict[str, str]:
    """The dependencies as they are now, keyed like the recorded ones: plugin code and the data snapshot (which must
    still be in the vault; data is not fetched again)."""
    now: dict[str, str] = {}
    for key in recorded:
        if key.startswith("plugin:"):
            name = key.removeprefix("plugin:")
            now[key] = code_hash(load(_use_named(contract, name), registry))
    digest = recorded.get("data", "")
    now["data"] = digest if vault.has(as_use(contract.data).name, digest) else "missing"
    return now


def _use_named(contract: Contract, name: str) -> Use:
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


def jsonable(obj: Any) -> Any:
    """Plain JSON values as the ledger keeps them: keys sorted, tuples become lists, and numbers that are not finite
    become null. A card rendered from this reads the same as one rendered from the ledger."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in sorted(((str(k), v) for k, v in obj.items()), key=lambda kv: kv[0])}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating | np.integer):
        return jsonable(obj.item())
    return obj
