"""Checking an idea's data against its approved contract.

The source must be admitted first. Then the engine checks what the contract and its plugins declare (freshness, legs
of one asset, the grade of every field), runs the idea's own tests if it has any, and turns every problem into a
treatment that is applied by default. Each treatment's influence on the contract's decisive number, in units of the
smallest effect the data can detect (MDE), decides whether a person must look: below the threshold it is applied and
logged. Only differences are reported, never the level of the number.
"""

import importlib.util
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import Contract, ContractError, Use, as_use, span
from research_ci.api.data import DataTest, pits_of, tests_of
from research_ci.api.plugins import CostModel, LabelPlugin, TargetPlugin
from research_ci.kernel.checks import invariants as inv
from research_ci.kernel.checks import treatments as tr
from research_ci.kernel.checks.baseline import baseline
from research_ci.kernel.checks.pits import BUILTIN
from research_ci.kernel.holdout.tiers import Tier, Tiers, visible
from research_ci.kernel.judge import power as pw
from research_ci.kernel.judge.protocol import Check
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import from_ns, span_ns, to_ns
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages.contract import DEFAULT_TARGET, _norm_candidates, _per_t, _use_named
from research_ci.stages.source import SourceSettings, admit
from research_ci.store import vault
from research_ci.store.sources import Sources, source_key

Stat = Callable[[pl.DataFrame, list[tr.Treatment]], dict[str, pl.DataFrame]]


class DataError(ContractError):
    pass


@dataclass(frozen=True)
class DataSettings:
    source: SourceSettings = field(default_factory=SourceSettings)
    materiality: float = 0.5
    stale_after: str = "2 days"
    hedge_ratio: float = 1.2
    hedge_corr: float = 0.3
    hedge_window: int = 72
    hedge_min_obs: int = 48
    mask_around: str = "24h"
    seeds: int = 2


@dataclass
class Checked:
    spec: dict[str, Any]
    deps: dict[str, str]


def check_data(
    contract: Contract,
    root: Path,
    registry: Registry | None = None,
    settings: DataSettings | None = None,
    sources: Sources | None = None,
    keep: tuple[str, ...] = (),
) -> Checked:
    """``keep`` names treatments a person chose not to apply; their influence is still measured and shown."""
    return _DataCheck(contract, Path(root), registry, settings or DataSettings(), sources or Sources(), keep).run()


class _DataCheck:
    def __init__(
        self,
        c: Contract,
        root: Path,
        registry: Registry | None,
        s: DataSettings,
        sources: Sources,
        keep: tuple[str, ...] = (),
    ) -> None:
        self.c, self.root, self.registry, self.s, self.sources, self.keep = c, root, registry, s, sources, keep
        self.checks: list[Check] = []
        self.warnings: list[str] = []
        self.deps: dict[str, str] = {}
        self.spec: dict[str, Any] = {}

    def run(self) -> Checked:
        started = time.monotonic()
        use = as_use(self.c.data)
        plugin = self._plugin(use)
        everything = field_table(plugin.fetch())
        digest, _ = vault.put(use.name, everything)
        self.deps["data"] = digest
        source = self._source(use, plugin, everything)
        fields = self._explore(everything, digest)
        grades = source["grades"]
        cands = self._candidates(fields)
        stat = self._stat(fields, cands)
        treatments = self._treatments(fields, cands, source)
        self.spec["tests"] = self._tests(fields, treatments)
        unknown = sorted(set(self.keep) - {t.name for t in treatments})
        if unknown:
            raise DataError(f"no treatment named {', '.join(map(repr, unknown))}; see the DataCard for the names")
        applied = [t for t in treatments if t.name not in self.keep]
        self.spec["kept"] = sorted(self.keep)
        self._grades(fields, grades, stat)
        self.spec["treatments"] = self._influence(stat, fields, treatments, applied)
        self.spec["power"] = self._power(stat, fields, applied)
        panel = tr.apply_all(applied, fields)
        panel_hash, _ = vault.put(f"{use.name}.panel", panel)
        excluded = self._exclusions(cands, applied)
        exclusions_hash, _ = vault.put(f"{use.name}.exclusions", excluded)
        self.spec["panel"] = {"hash": panel_hash, "rows": panel.height, "exclusions": exclusions_hash}
        self.deps["panel"] = panel_hash
        self.deps["exclusions"] = exclusions_hash
        self.spec["checks"] = [asdict(ch) for ch in self.checks]
        self.spec["warnings"] = self.warnings
        self.spec["materiality"] = self.s.materiality
        self.spec["verdict"] = self._verdict(source)
        self.spec["seconds"] = round(time.monotonic() - started, 1)
        return Checked(self.spec, dict(sorted(self.deps.items())))

    def _plugin(self, use: Use) -> Any:
        plugin = load(use, self.registry)
        self.deps[f"plugin:{use.name}"] = code_hash(plugin)
        return plugin

    def _source(self, use: Use, plugin: Any, everything: pl.DataFrame) -> dict[str, Any]:
        key = source_key(use.name, dict(use.options), code_hash(plugin))
        record = self.sources.latest(key)
        if record is None:
            record = self.sources.record(key, admit(use, plugin, everything, self.s.source))
            record = {**record, "approved": None}
        spec = record["payload"]["spec"]
        approved = record["approved"]
        if spec["verdict"]["status"] != "admissible":
            status = "not admitted"
        else:
            status = "admitted" if approved is not None else "waiting"
        self.deps["source"] = record["hash"]
        self.spec["source"] = {
            "plugin": use.name,
            "key": key,
            "admission": record["hash"],
            "status": status,
            "failed": [c["name"] for c in spec["checks"] if c["status"] == "fail"],
            "approval": None if approved is None else approved["payload"]["approval"],
        }
        return {"spec": spec, "status": status, "grades": spec["grades"]}

    def _explore(self, everything: pl.DataFrame, digest: str) -> pl.DataFrame:
        latest = int(everything.select(pl.col("available_at").max().dt.epoch("ns")).item())
        self.explore_end = latest if self.c.explore_end == "latest" else to_ns(pd.Timestamp(self.c.explore_end))
        seal_end = to_ns(pd.Timestamp(self.c.seal_end)) if self.c.seal_end else None
        fields = visible(everything, Tier.EXPLORE, Tiers(self.explore_end, seal_end))
        self.start = int(fields.select(pl.col("available_at").min().dt.epoch("ns")).item())
        age = (time.time_ns() - latest) / pw.DAY_NS
        self.spec["data"] = {
            "source": as_use(self.c.data).name,
            "hash": digest,
            "rows": fields.height,
            "entities": fields["entity"].n_unique(),
            "first": _iso(self.start),
            "explore_end": _iso(self.explore_end),
            "age_days": round(age, 1),
        }
        stale = age > span(self.s.stale_after).total_seconds() / 86_400
        self.checks.append(Check("data is current", "warn" if stale else "pass", f"latest data is {age:.1f} days old"))
        if stale:
            self.warnings.append(f"the data source has nothing newer than {_iso(latest)}; collect live data first")
        return fields

    def _candidates(self, fields: pl.DataFrame) -> pl.DataFrame | None:
        u = self.c.universe
        if u is None:
            return None
        if self.c.consumer is not None:
            timing = load(as_use(self.c.consumer), self.registry).timing()
            every, offset = timing.every, timing.offset
        else:
            assert self.c.execution is not None
            every, offset = self.c.execution.every, self.c.execution.offset
        self.times = pw.decision_grid(span_ns(span(every)), span_ns(span(offset)), self.start, self.explore_end)
        self.universe = self._plugin(as_use(u.use))
        return self._universe(fields)

    def _universe(self, fields: pl.DataFrame) -> pl.DataFrame:
        return _norm_candidates(self.universe.candidates(fields, [from_ns(t) for t in self.times]))

    def _stat(self, fields: pl.DataFrame, cands: pl.DataFrame | None) -> Stat | None:
        """The decisive numbers per horizon, as a function of treated data: every label effect's target per decision
        time. Consumer effects are not measured here yet."""
        effects = [e for e in self.c.decisive if e.on != "consumer"]
        if cands is None or not effects:
            if any(e.on == "consumer" for e in self.c.decisive):
                self.warnings.append("influence on a consumer effect is not measured yet; treatments apply as listed")
            return None
        costs: CostModel = self._plugin(as_use(self.c.costs))
        plans = []
        for effect in effects:
            label = next(lab for lab in self.c.labels if lab.name == effect.on)
            values: LabelPlugin = self._plugin(as_use(label.use))
            assert self.c.universe is not None
            use = as_use(effect.target) if effect.target else Use(DEFAULT_TARGET, {"cluster": self.c.universe.clusters})
            target: TargetPlugin = self._plugin(use)
            plans.append((effect.on, values, target, label.horizons))

        def stat(f: pl.DataFrame, ts: list[tr.Treatment]) -> dict[str, pl.DataFrame]:
            c = tr.apply_candidates(ts, self._universe(f) if f is not fields else cands)
            out = {}
            for on, values, target, horizons in plans:
                for h in horizons:
                    horizon = span(h)
                    v = values.values(f, c, horizon).join(c, on=["t", "entity"], how="inner", suffix="_candidate")
                    out[f"{on} {h}"] = _per_t(target.reduce(v, horizon, costs))
            return out

        return stat

    def _treatments(self, fields: pl.DataFrame, cands: pl.DataFrame | None, source: dict[str, Any]) -> list[Any]:
        out: list[tr.Treatment] = []
        _, anomalies = baseline(fields)
        shifts = anomalies.filter(pl.col("kind") == "shift")
        unconfirmed = shifts.join(_confirmed(source["spec"]), on=["entity", "field", "end"], how="anti")
        if unconfirmed.height:
            reason = f"{unconfirmed.height} level shifts the independent source did not confirm"
            out.append(tr.mask_windows("mask unconfirmed level shifts", reason, unconfirmed, self._around(), fields))
        if shifts.height > unconfirmed.height:
            self.checks.append(
                Check("level shifts", "pass", f"{shifts.height - unconfirmed.height} confirmed by the source; kept")
            )
        price = getattr(self.universe, "hedge_price", None) if cands is not None else None
        if cands is not None and price and {"long", "short"} <= set(cands.columns):
            assert self.c.universe is not None
            keep = (self.c.universe.clusters,) if self.c.universe.clusters in cands.columns else ()
            legs = inv.hedged_legs(
                cands,
                fields,
                price,
                self.s.hedge_ratio,
                self.s.hedge_corr,
                self.s.hedge_window,
                self.s.hedge_min_obs,
                keep,
            )
            by = keep[0] if keep else "pair"
            self.checks.append(inv.hedge_check(legs, price, self.s.hedge_ratio, self.s.hedge_corr, by))
            bad = legs.filter(~pl.col("ok"))
            if bad.height:
                reason = f"{bad.height:,} candidate rows whose legs do not trade as one asset"
                out.append(tr.drop_candidates("hedge guard at each decision time", reason, bad))
        return out

    def _tests(self, fields: pl.DataFrame, treatments: list[tr.Treatment]) -> list[dict[str, Any]]:
        tests, pits = (
            _layer_file(self.root / "data.py", tests_of),
            _layer_file(self.root / "nulls" / "data.py", pits_of),
        )
        self.pits, self.tests = pits, tests
        out = []
        for t in tests:
            flagged = t.check(fields)
            row: dict[str, Any] = {"name": t.name, "covers": [list(c) for c in t.covers], "on_fail": t.on_fail}
            row["flagged"] = flagged.height
            if flagged.height and t.on_fail == "mask":
                treatments.append(
                    tr.mask_rows(f"test {t.name}: mask", f"{flagged.height} rows flagged", flagged, fields)
                )
            elif flagged.height and t.on_fail == "drop_entities":
                ents = flagged["entity"].unique().to_list()
                treatments.append(tr.drop_entities(f"test {t.name}: drop", f"{len(ents)} entities", ents, fields))
            out.append(row)
        return out

    def _grades(self, fields: pl.DataFrame, grades: dict[str, str], stat: Stat | None) -> None:
        present = sorted(fields["field"].unique().to_list())
        self.spec["grades"] = {f: grades.get(f, "unverified") for f in present}
        weak = [f for f in present if self.spec["grades"][f] != "verified"]
        if not weak:
            self.checks.append(Check("field grades", "pass", "every field agrees with an independent source"))
            return
        base = stat(fields, []) if stat is not None else None
        used = [f for f in weak if stat is None or _moves(stat, base, fields.filter(pl.col("field") != f))]
        self.spec["capped_by"] = used
        if used:
            self.checks.append(
                Check(
                    "field grades", "warn", f"the decisive number reads {', '.join(used)}, not verified; explore-grade"
                )
            )
        else:
            self.checks.append(Check("field grades", "pass", f"unverified fields are not read: {', '.join(weak)}"))

    def _influence(
        self, stat: Stat | None, fields: pl.DataFrame, ts: list[tr.Treatment], applied: list[tr.Treatment]
    ) -> list[dict[str, Any]]:
        """Each treatment's influence, and that of the ones applied together; kept treatments are measured too."""
        rows = []
        kept = set(self.keep)
        if stat is None:
            for t in self.spec["tests"]:
                t["status"] = "not calibrated"
            return [
                {
                    "name": t.name,
                    "on": t.on,
                    "reason": t.reason,
                    "rows": t.rows,
                    "influence": None,
                    "kept": t.name in kept,
                }
                for t in ts
            ]
        self.base = stat(fields, [])
        self.after = stat(tr.apply_all(applied, fields), applied) if applied else self.base
        width = self.explore_end - self.start
        self.mde_before = {k: _mde(v, k, width) for k, v in self.base.items()}
        self.mde_after = {k: _mde(v, k, width) for k, v in self.after.items()}
        self.mde = {k: _smaller(self.mde_before[k], self.mde_after.get(k, math.nan)) for k in self.base}
        for t in ts:
            head = {"name": t.name, "on": t.on, "reason": t.reason, "rows": t.rows, "kept": t.name in kept}
            rows.append(head | self._shift(stat, fields, [t]))
        if len(applied) > 1 or (applied and kept):
            together = self._shift(stat, fields, applied, self.after)
            rows.append({"name": "all applied", "on": "both", "reason": "", "rows": 0, "kept": False} | together)
        for t in self.tests:
            self._calibrate(t, stat, fields)
        return rows

    def _shift(
        self, stat: Stat, fields: pl.DataFrame, ts: list[tr.Treatment], after: dict[str, pl.DataFrame] | None = None
    ) -> dict[str, Any]:
        after = after if after is not None else stat(tr.apply_all(ts, fields), ts)
        out = {}
        for k, base in self.base.items():
            d = _mean_gap(base, after.get(k))
            out[k] = {"bp": abs(d) * 1e4, "mde": abs(d) / self.mde[k] if self.mde[k] else math.nan}
        largest = max((v["mde"] for v in out.values() if math.isfinite(v["mde"])), default=math.nan)
        return {"influence": out, "largest": largest}

    def _calibrate(self, test: DataTest, stat: Stat, fields: pl.DataFrame) -> None:
        """Every pit a test claims to cover, planted in clean copies: those that move the decisive number by the
        threshold or more must be flagged."""
        flagged = set(test.check(fields)["entity"].unique().to_list())
        clean = fields.filter(~pl.col("entity").is_in(sorted(flagged)))
        base = stat(clean, [])
        results = []
        for category, name in test.covers:
            plants = [(f"built-in {category}", BUILTIN[category])] if category in BUILTIN else []
            plants += [
                (f"pit {p.category}: {p.note}" if p.note else f"pit {p.category}", p.plant)
                for p in self.pits
                if p.category == category
            ]
            if not plants:
                results.append({"covers": [category, name], "pit": None, "status": "no pit"})
                continue
            for label, plant in plants:
                for seed in range(self.s.seeds):
                    rng = np.random.default_rng(seed)
                    try:
                        mutated = field_table(plant(clean, name, rng))
                    except (ValueError, pl.exceptions.PolarsError) as exc:
                        results.append(
                            {"covers": [category, name], "pit": label, "status": "no target", "note": str(exc)}
                        )
                        continue
                    harm = max(
                        (abs(_mean_gap(base[k], v)) / self.mde[k] for k, v in stat(mutated, []).items() if self.mde[k]),
                        default=0.0,
                    )
                    caught = test.check(mutated).height > 0
                    status = "caught" if caught else ("missed" if harm >= self.s.materiality else "harmless")
                    results.append(
                        {"covers": [category, name], "pit": label, "seed": seed, "harm": harm, "status": status}
                    )
        for r in self.spec["tests"]:
            if r["name"] == test.name:
                r["calibration"] = results
                bad = [x for x in results if x["status"] in ("missed", "no pit")]
                r["status"] = "returned" if bad else "holds"

    def _power(self, stat: Stat | None, fields: pl.DataFrame, ts: list[tr.Treatment]) -> list[dict[str, Any]]:
        if stat is None:
            return []
        after = self.after
        width = self.explore_end - self.start
        rows = []
        for effect in self.c.decisive:
            if effect.on == "consumer":
                continue
            label = next(lab for lab in self.c.labels if lab.name == effect.on)
            for h in label.horizons:
                k, hn = f"{effect.on} {h}", span_ns(span(h))
                before = pw.power(pw.block_se(self.base[k], hn), width, hn, effect.at_least)
                now = pw.power(pw.block_se(after[k], hn), width, hn, effect.at_least)
                rows.append(
                    {
                        "on": effect.on,
                        "horizon": h,
                        "mde_before": before.mde,
                        "mde": now.mde,
                        "forward_days": now.forward_days,
                        "decisions": after[k].height,
                    }
                )
        return rows

    def _exclusions(self, cands: pl.DataFrame | None, ts: list[tr.Treatment]) -> pl.DataFrame:
        empty = pl.DataFrame(schema={"t": pl.Datetime("ns", "UTC"), "entity": pl.String})
        if cands is None:
            return empty
        kept = tr.apply_candidates(ts, cands.select("t", "entity"))
        return cands.select("t", "entity").join(kept, on=["t", "entity"], how="anti").sort("t", "entity")

    def _verdict(self, source: dict[str, Any]) -> dict[str, Any]:
        notes, decide = [], []
        status = self.spec["source"]["status"]
        if status == "not admitted":
            notes.append(f"source {self.spec['source']['plugin']} not admitted: fix the plugin and check again")
        elif status == "waiting":
            notes.append(f"source {self.spec['source']['plugin']} waits for a person: rci source approve")
        returned = [t["name"] for t in self.spec["tests"] if t.get("status") == "returned"]
        if returned:
            notes.append(f"tests returned: {', '.join(returned)} miss pits they claim to cover")
        failed = [c.name for c in self.checks if c.status == "fail"]
        if failed:
            notes.append(f"failed checks: {', '.join(failed)}")
        for t in self.spec["treatments"]:
            if t.get("influence") and t["name"] != "all applied" and t["largest"] >= self.s.materiality:
                choice = "kept, not applied" if t["kept"] else "applied"
                decide.append(f"{t['name']} ({choice}): moves the decisive number by {t['largest']:.2f} MDE")
        for t in self.spec["tests"]:
            if t["on_fail"] == "ask" and t["flagged"]:
                decide.append(f"test {t['name']} flagged {t['flagged']} rows")
        self.spec["decide"] = decide
        if notes:
            return {"status": "blocked", "notes": notes}
        if decide:
            return {"status": "decide", "notes": ["a person decides; approving applies every treatment listed"]}
        return {"status": "ready", "notes": ["approving freezes the treated data; feat reads only that"]}

    def _around(self) -> int:
        return span_ns(span(self.s.mask_around))


def current_data_deps(
    contract: Contract, recorded: dict[str, str], spec: dict[str, Any], registry: Registry | None = None
) -> dict[str, str]:
    """The data check's dependencies as they are now, keyed like the recorded ones."""
    now: dict[str, str] = {}
    for key in recorded:
        if key.startswith("plugin:"):
            now[key] = code_hash(load(_use_named(contract, key.removeprefix("plugin:")), registry))
    name = as_use(contract.data).name
    now["data"] = recorded["data"] if vault.has(name, recorded["data"]) else "missing"
    now["panel"] = recorded["panel"] if vault.has(f"{name}.panel", recorded["panel"]) else "missing"
    if "exclusions" in recorded:
        found = vault.has(f"{name}.exclusions", recorded["exclusions"])
        now["exclusions"] = recorded["exclusions"] if found else "missing"
    record = Sources().latest(spec["source"]["key"])
    now["source"] = "missing" if record is None else record["hash"]
    return now


def _confirmed(spec: dict[str, Any]) -> pl.DataFrame:
    """Level shifts the independent source confirmed at admission: ``entity``, ``field``, ``end``."""
    rows = spec.get("confirmed_shifts", [])
    frame = pl.DataFrame(rows, schema={"entity": pl.String, "field": pl.String, "end": pl.String}, orient="row")
    return frame.with_columns(pl.col("end").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", "UTC")))


def _layer_file(path: Path, read: Callable[[Any], list[Any]]) -> list[Any]:
    if not path.is_file():
        return []
    spec = importlib.util.spec_from_file_location(f"_rci_{path.parent.name}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise DataError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return read(module)
    except ValueError as exc:
        raise DataError(f"{path}: {exc}") from exc


def _moves(stat: Stat, base: dict[str, pl.DataFrame] | None, without: pl.DataFrame) -> bool:
    try:
        other = stat(without, [])
    except Exception:  # noqa: BLE001
        return True
    if base is None:
        return True
    for k, v in base.items():
        o = other.get(k)
        if o is None or o.height != v.height:
            return True
        j = v.join(o, on="t", how="inner", suffix="_x")
        a, b = j["value"].to_numpy(), j["value_x"].to_numpy()
        if j.height != v.height or not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            return True
    return False


def _mean_gap(base: pl.DataFrame, other: pl.DataFrame | None) -> float:
    if other is None or other.is_empty():
        return math.nan
    j = base.join(other, on="t", how="inner", suffix="_x")
    return float((j["value_x"] - j["value"]).mean()) if j.height else math.nan  # type: ignore[arg-type]


def _mde(per_t: pl.DataFrame, key: str, width: int) -> float:
    """The smallest effect this horizon can detect; it does not depend on the effect asked for."""
    h = span_ns(span(key.rsplit(" ", 1)[1]))
    return pw.power(pw.block_se(per_t, h), width, h, 1.0).mde


def _smaller(a: float, b: float) -> float:
    """The finer of two resolutions: a treatment matters if it moves the number by more than the verdict on the
    treated data, or on the untreated data, could resolve."""
    finite = [x for x in (a, b) if math.isfinite(x) and x > 0]
    return min(finite) if finite else math.nan


def _iso(ns: int) -> str:
    return from_ns(ns).strftime("%Y-%m-%dT%H:%MZ")
