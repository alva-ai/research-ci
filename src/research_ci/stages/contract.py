"""Deriving a contract: load the plugins it names, test them, and work out what the data can decide.

Nothing here knows what a field or a strategy means; labels, targets, costs, candidates and the consumer are plugins.
The result is a JSON-ready spec. Planning numbers come from dispersion only; no level of any target is reported.
"""

import math
import os
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import (
    FLOOR_ALTERNATIVES,
    Branch,
    Contract,
    ContractError,
    Effect,
    Label,
    Perturb,
    Use,
    as_use,
    span,
)
from research_ci.api.plugins import ConsumerPlugin, ConsumerResult, CostModel, LabelPlugin, TargetPlugin, UniversePlugin
from research_ci.kernel.holdout.tiers import Tier, Tiers, visible
from research_ci.kernel.judge import power as pw
from research_ci.kernel.judge import protocol as pr
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import TIME, from_ns, lit_time, span_ns, time_series, to_ns
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.store import vault
from research_ci.store.workspace import hash_file

DEFAULT_TARGET = "ref.mean"


class DeriveError(ContractError):
    pass


@dataclass(frozen=True)
class Settings:
    """How hard derive tests plugins and how many runs the consumer pilot takes. ``pilot`` lists (parts, runs per
    part): the explore span is split into ``parts`` windows and each gets ``runs`` perturbed runs."""

    workers: int = field(default_factory=lambda: int(os.environ.get("RCI_WORKERS", _default_workers())))
    universe_cuts: int = 3
    label_cuts: int = 2
    consumer_cuts: int = 2
    pilot: tuple[tuple[int, int], ...] = ((1, 8), (2, 4), (4, 2))
    stale_after: str = "2 days"


@dataclass
class Derived:
    spec: dict[str, Any]
    deps: dict[str, str]


def derive(
    contract: Contract, root: Path, registry: Registry | None = None, settings: Settings | None = None
) -> Derived:
    return _Derivation(contract, Path(root), registry, settings or Settings()).run()


class _Derivation:
    def __init__(self, c: Contract, root: Path, registry: Registry | None, settings: Settings) -> None:
        self.c, self.root, self.registry, self.s = c, root, registry, settings
        self.checks: list[pr.Check] = []
        self.later: list[tuple[int, Future[pr.Check]]] = []
        self.warnings: list[str] = []
        self.deps: dict[str, str] = {}
        self.spec: dict[str, Any] = {"claim": c.claim, "template": c.template}
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
            self._baseline_files()
            self._labels(fields, cands)
            decisive = []
            for effect in self.c.decisive:
                assert cands is not None
                if effect.on == "consumer":
                    assert consumer is not None
                    decisive.append(self._consumer_effect(effect, consumer, fields, cands))
                else:
                    decisive.append(self._label_effect(effect, fields, cands))
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
        age = (time.time_ns() - latest) / pw.DAY_NS
        self.spec["data"] = {
            "source": source.name,
            "hash": digest,
            "rows": explore.height,
            "entities": explore["entity"].n_unique(),
            "first": _iso(self.start),
            "explore_end": _iso(self.explore_end),
            "age_days": round(age, 1),
        }
        stale = age > span(self.s.stale_after).total_seconds() / 86_400
        status: pr.Status = "warn" if stale else "pass"
        self.checks.append(pr.Check("data is current", status, f"latest data is {age:.1f} days old"))
        if stale:
            self.warnings.append(f"the data source has nothing newer than {_iso(latest)}; collect live data first")
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

    def _baseline_files(self) -> None:
        b = self.c.baseline
        if b is None:
            return
        files = {"config": b.config, **{f"code:{Path(p).name}": p for p in b.code}}
        info: dict[str, Any] = {"score": b.score, "files": {}}
        for key, raw in files.items():
            if raw is None:
                continue
            path = self._path(raw)
            if not path.is_file():
                raise DeriveError(f"baseline {key} not found: {path}")
            digest = hash_file(path)
            self.deps[f"baseline:{key}"] = digest
            info["files"][key] = {"path": str(path), "hash": digest}
        if b.live_config and b.config:
            info["live_config"] = _compare_configs(self._path(b.config), self._path(b.live_config))
            self.deps["baseline:live_config"] = hash_file(self._path(b.live_config))
        self.spec["baseline"] = info

    def _path(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

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
        rows = []
        for h in label.horizons:
            horizon = span(h)
            values = plugin.values(fields, cands, horizon)
            joined = values.join(cands, on=["t", "entity"], how="inner", suffix="_candidate")
            per_t = _per_t(target.reduce(joined, horizon, self.costs))
            p = pw.power(pw.block_se(per_t, span_ns(horizon)), width, span_ns(horizon), effect.at_least)
            rows.append({"horizon": h, "decisions": per_t.height, **asdict(p)})
        out = {"kind": "label", **_effect(effect), "target": use.name, "horizons": rows}
        return out | {"label": {"name": label.name, "plugin": as_use(label.use).name}}

    def _consumer_effect(
        self, effect: Effect, consumer: ConsumerPlugin, fields: pl.DataFrame, cands: pl.DataFrame
    ) -> dict[str, Any]:
        run: pr.Run = consumer.run
        submit = self.pool.submit
        base = run(fields, None)
        scores = self._baseline_scores(cands)
        cuts = _quantile_times(time_series(self.times), self.s.consumer_cuts)
        middle = _busiest(fields, self.times, self.period)
        jobs: dict[str, Future[Any]] = {
            "again": submit(consumer.run, fields, None, True),
            "pit": submit(pr.consumer_pit, run, fields, base, cuts, 2 * self.period),
            "causal": submit(pr.causal, run, fields, base, middle, self.period),
        }
        if scores is not None:
            jobs["passthrough"] = submit(pr.passthrough, run, fields, base, scores)
        assert effect.pilot is not None
        pilot = None
        if scores is not None:
            if isinstance(effect.design, Branch):
                pilot = self._branch_pilot(run, fields, base, scores, effect.pilot, effect.design)
            else:
                pilot = self._pilot(run, fields, scores, effect.pilot)
        self.checks.append(pr.deterministic(base, jobs["again"].result()))
        self.checks.extend(pr.adds_up(base))
        self.checks.append(pr.fees_match(base, self.costs))
        held = _held(base.positions, cands)
        for name, plugin in self.labels.items():
            values = plugin.values(fields, held, pd.Timedelta(self.period, unit="ns"))
            check = pr.matches_labels(base, values, self.period)
            self.checks.append(pr.Check(f"{check.name} ({name})", check.status, check.detail))
        for key in ("pit", "causal", "passthrough"):
            if key in jobs:
                self.checks.append(jobs[key].result())
        out: dict[str, Any] = {"kind": "consumer", **_effect(effect), "baseline_run": _summary(base)}
        if pilot is None:
            self.warnings.append("the baseline has no scores to perturb, so the consumer's noise is not measured")
            return out
        out["pilot"] = result = pilot()
        if result["design"] == "branch":
            out["power"] = asdict(pw.branch_power(result, effect.at_least))
        else:
            full_days = (self.explore_end - self.start) / pw.DAY_NS
            sd = result["windows"][0]["sd"]
            out["power"] = asdict(pw.consumer_power(sd, full_days, result["beta"], effect.at_least))
        return out

    def _branch_pilot(
        self,
        run: pr.Run,
        fields: pl.DataFrame,
        base: ConsumerResult,
        scores: pl.DataFrame,
        change: Perturb,
        design: Branch,
    ) -> Callable[[], dict[str, Any]]:
        """Branch from the baseline's own path: the same scores up to each branch point, the uninformative change for
        ``horizon``, the same scores again after it; the difference is counted until ``tail`` after the change."""
        every, horizon = span_ns(span(design.every)), span_ns(span(design.horizon))
        tail = span_ns(span(design.tail))
        held = base.positions.filter(pl.col("weight") != 0)
        first = int(held.select(pl.col("t").min().dt.epoch("ns")).item()) if held.height else self.start
        points = list(range(first, self.explore_end - horizon - tail + 1, every))
        plan = []
        for seed, t0 in enumerate(points, start=1):
            inside = (pl.col("t") > lit_time(t0)) & (pl.col("t") <= lit_time(t0 + horizon))
            changed = pr.perturb(scores.filter(inside), change.kind, change.share, horizon, seed)
            branch_scores = pl.concat([scores.filter(~inside), changed]).sort("t", "entity")
            plan.append((t0, self.pool.submit(run, fields, branch_scores)))

        def collect() -> dict[str, Any]:
            diffs, overlaps = [], []
            for t0, job in plan:
                diff, overlap = pr.branch_diff(base, job.result(), t0, t0 + horizon + tail)
                diffs.append(diff)
                overlaps.append(overlap)
            arr = np.array(diffs)
            sd = float(arr.std(ddof=1)) if arr.size > 1 else math.nan
            return {
                "design": "branch",
                **asdict(change),
                **asdict(design),
                "branches": len(diffs),
                "independence": min(1.0, every / (horizon + tail)),
                "horizon_days": horizon / pw.DAY_NS,
                "every_days": every / pw.DAY_NS,
                "sd": sd,
                "mean_shift": float(arr.mean()) if arr.size else math.nan,
                "overlap": float(np.median(overlaps)) if overlaps else math.nan,
            }

        return collect

    def _baseline_scores(self, cands: pl.DataFrame) -> pl.DataFrame | None:
        b = self.c.baseline
        if b is None or b.score not in cands.columns:
            return None
        rows = cands.filter(pl.col("openable")) if "openable" in cands.columns else cands
        return rows.select("t", "entity", pl.col(b.score).alias("score")).drop_nulls("score")

    def _pilot(
        self, run: pr.Run, fields: pl.DataFrame, scores: pl.DataFrame, change: Perturb
    ) -> Callable[[], dict[str, Any]]:
        block = span_ns(span(change.block))
        plan = []
        for parts, reps in self.s.pilot:
            edges = np.linspace(self.start, self.explore_end, parts + 1).astype("int64").tolist()
            for i in range(parts):
                lo, hi = edges[i], edges[i + 1]
                after = pl.col("available_at") >= lit_time(lo) if i == 0 else pl.col("available_at") > lit_time(lo)
                f = fields.filter(after & (pl.col("available_at") <= lit_time(hi)))
                s = scores.filter((pl.col("t") > lit_time(lo)) & (pl.col("t") <= lit_time(hi)))
                base = self.pool.submit(run, f, None)
                noisy = [
                    self.pool.submit(run, f, pr.perturb(s, change.kind, change.share, block, seed))
                    for seed in range(1, reps + 1)
                ]
                plan.append((parts, (hi - lo) / pw.DAY_NS, base, noisy))

        def collect() -> dict[str, Any]:
            windows = []
            for parts in sorted({p for p, *_ in plan}):
                rows = [(days, base, noisy) for p, days, base, noisy in plan if p == parts]
                centered, shifts = [], []
                for _, base, noisy in rows:
                    b = pr.total(base.result())
                    diffs = np.array([pr.total(n.result()) - b for n in noisy])
                    shifts.append(float(diffs.mean()))
                    centered.extend((diffs - diffs.mean()).tolist())
                dof = len(centered) - len(rows)
                sd = math.sqrt(sum(x * x for x in centered) / dof) if dof > 0 else math.nan
                mean_shift = float(np.mean(shifts))
                windows.append(
                    {"parts": parts, "days": rows[0][0], "runs": len(centered), "sd": sd, "mean_shift": mean_shift}
                )
            longest = sorted(windows, key=lambda w: -w["days"])[:2]
            beta = pw.scaling({w["days"]: w["sd"] for w in longest})
            return {"design": "full_path", **asdict(change), "windows": windows, "beta": beta}

        return collect

    def _rest(self, decisive: list[dict[str, Any]]) -> None:
        c = self.c
        affected = {a for u in c.used or () for a in u.affects}
        confirm_only = (
            bool(affected & {"claim", "labels", "costs"})
            or ("baseline" in affected and c.baseline is not None)
            or (c.universe is not None and c.universe.membership == "static")
        )
        floor = ["chance", "timing"] + (["fewer"] if c.consumer is not None or c.baseline is not None else [])
        assert set(floor) <= set(FLOOR_ALTERNATIVES)
        self.spec["alternatives"] = {"idea": [asdict(a) for a in c.alternatives], "floor": floor}
        self.spec["used"] = [asdict(u) for u in c.used or ()]
        self.spec["history"] = "explore-only" if confirm_only else "can confirm"
        self.spec["seal_end"] = c.seal_end
        self.spec["forward"] = self._forward()
        deadline_days = span(c.deadline).total_seconds() / 86_400
        self.spec["deadline_days"] = deadline_days
        self.spec["budget"] = c.budget
        self.spec["skip"] = list(c.skip)
        failed = [c.name for c in self.checks if c.status == "fail"]
        self.spec["verdict"] = _verdict(decisive, confirm_only, deadline_days, failed)

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


def _verdict(
    decisive: list[dict[str, Any]], confirm_only: bool, deadline_days: float, failed: list[str]
) -> dict[str, Any]:
    notes, feasible = [f"blocked by failed checks: {', '.join(failed)}"] if failed else [], []
    for d in decisive:
        if d["kind"] == "label":
            finite = [r for r in d["horizons"] if math.isfinite(r["forward_days"])]
            if not finite:
                notes.append(f"{d['on']}: too few blocks on every horizon to measure the noise")
                feasible.append(False)
                continue
            best = min(finite, key=lambda r: r["forward_days"])
            history_ok = any(r["mde"] <= d["at_least"] for r in finite)
            ok = (history_ok and not confirm_only) or best["forward_days"] <= deadline_days
            where = "history" if history_ok and not confirm_only else "new data"
            notes.append(
                f"{d['on']}: best horizon {best['horizon']}, about {best['forward_days']:.0f} days of new data"
            )
        else:
            p = d.get("power")
            if p is None:
                notes.append("consumer: no pilot, the wait cannot be estimated")
                feasible.append(False)
                continue
            history_ok = p["history_mde"] <= d["at_least"]
            months = p["forward_months"]
            ok = (history_ok and not confirm_only) or (months is not None and months * 30.4 <= deadline_days)
            where = "history" if history_ok and not confirm_only else "new data"
            wait = "never at this noise" if months is None else f"about {months:.1f} months of new data"
            notes.append(f"consumer: history detects {p['history_mde']:.1%}/yr; {wait}")
        feasible.append(ok)
        notes[-1] += f" ({where})"
    if failed:
        return {"status": "blocked", "notes": notes}
    return {"status": "decidable" if all(feasible) else "insufficient", "notes": notes}


def _held(positions: pl.DataFrame, cands: pl.DataFrame) -> pl.DataFrame:
    """Every position held, with the candidate columns a label needs: from the positions table when the consumer
    reports them, otherwise from the candidates (and then positions no longer among them get no label)."""
    if {"long", "short"} <= set(positions.columns):
        return positions
    return positions.join(cands, on=["t", "entity"], how="inner", suffix="_candidate")


def _label(c: Contract, name: str) -> Label:
    return next(label for label in c.labels if label.name == name)


def _effect(effect: Effect) -> dict[str, Any]:
    return {"on": effect.on, "at_least": effect.at_least, "unit": effect.unit}


def _per_t(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    out = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    if not {"t", "value"} <= set(out.columns):
        raise DeriveError("a target must return columns t and value")
    return out.select(pl.col("t").cast(TIME), pl.col("value").cast(pl.Float64)).drop_nulls().sort("t")


def _summary(result: ConsumerResult) -> dict[str, Any]:
    net = result.net
    epoch = net.select(pl.col("t").dt.epoch("ns"))["t"]
    days = (int(epoch.max()) - int(epoch.min())) / pw.DAY_NS if net.height > 1 else math.nan  # type: ignore[arg-type]
    total = pr.total(result)
    parts = {c: float(net[c].sum()) for c in pr.COMPONENTS if c in net.columns}
    return {
        "total": total,
        "per_year": total * pw.YEAR_DAYS / days if days and days > 0 else math.nan,
        "marked": list(result.marked),
        "max_drawdown": pr.max_drawdown(result),
        "turnover": result.turnover,
        "components": parts,
        "trades": result.trades.height,
        "policies": dict(result.policies),
    }


def _compare_configs(a: Path, b: Path) -> dict[str, Any]:
    fa, fb = _leaves(tomllib.loads(a.read_text())), _leaves(tomllib.loads(b.read_text()))
    common = sorted(set(fa) & set(fb))
    differ = [k for k in common if fa[k] != fb[k]]
    return {"common": len(common), "differ": differ, "only_live": len(set(fb) - set(fa))}


def _leaves(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _leaves(v)
        else:
            out[k] = sorted(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else v
    return out


def _norm_candidates(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    frame = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
    missing = {"t", "entity"} - set(frame.columns)
    if missing:
        raise DeriveError(f"the universe must return columns t and entity; missing {', '.join(sorted(missing))}")
    return frame.with_columns(pl.col("t").dt.convert_time_zone("UTC").cast(TIME), pl.col("entity").cast(pl.String))


def _busiest(fields: pl.DataFrame, times: list[int], period: int) -> int:
    """A decision time in the middle third whose next period brings the most kinds of field, so a causality test
    there touches every input, including sparse ones such as settlements."""
    middle = times[len(times) // 3 : 2 * len(times) // 3] or times
    t0 = times[0]
    at = fields.select(pl.col("available_at").dt.epoch("ns").alias("a"), "field")
    slot = t0 + (pl.col("a") - 1 - t0) // period * period
    counts = at.with_columns(slot=slot).group_by("slot").agg(pl.col("field").n_unique())
    kinds = dict(zip(counts["slot"].to_list(), counts["field"].to_list(), strict=True))
    return max(middle, key=lambda t: (kinds.get(t, 0), -t))


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


def current_deps(
    contract: Contract, root: Path, recorded: dict[str, str], registry: Registry | None = None
) -> dict[str, str]:
    """The dependencies as they are now, keyed like the recorded ones: plugin code, baseline files, the data snapshot
    (which must still be in the vault; data is not fetched again)."""
    now: dict[str, str] = {}
    for key in recorded:
        if key.startswith("plugin:"):
            name = key.removeprefix("plugin:")
            now[key] = code_hash(load(_use_named(contract, name), registry))
    digest = recorded.get("data", "")
    now["data"] = digest if vault.has(as_use(contract.data).name, digest) else "missing"
    b = contract.baseline
    if b is not None:
        files = {"config": b.config, **{f"code:{Path(p).name}": p for p in b.code}, "live_config": b.live_config}
        for key, raw in files.items():
            if raw is None:
                continue
            path = Path(raw).expanduser()
            path = path if path.is_absolute() else (root / path).resolve()
            now[f"baseline:{key}"] = hash_file(path) if path.is_file() else "missing"
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
    """Plain JSON values: tuples become lists, and numbers that are not finite become null."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating | np.integer):
        return jsonable(obj.item())
    return obj
