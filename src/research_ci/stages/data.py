"""Checking an idea's data against its approved contract.

The source must pass admission (fidelity: the rows are what the venue reported), which runs once per source and is
approved with the first idea's data, and the decisive number may read only fields an independent source verified, or
the history is explore-grade. Then the engine runs its default checks
(``defaults/data.py``; fitness: whether faithful rows can be used for research) and the idea's own tests, if it has
any, and every problem they find becomes a treatment that is applied by default, or an item a person decides. Each
treatment's influence on the contract's decisive number, in units of the smallest effect the data can detect (MDE),
decides whether a person must look: below the threshold it is applied and logged. A treatment that changes identities
(a split) is weighed against the untreated data, and every other one on top of the splits that are applied. Only
differences are reported, never the level of the number; the power after is computed again on the treated data.
"""

import importlib.util
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import Contract, ContractError, Use, as_use, span
from research_ci.api.data import DataTest, pits_of, tests_of
from research_ci.api.plugins import CostModel, LabelPlugin, TargetPlugin
from research_ci.eval.data import influence as inf
from research_ci.eval.data import treatments as tr
from research_ci.eval.data.pits import BUILTIN
from research_ci.kernel.holdout.tiers import Tier, Tiers, visible
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import from_ns, span_ns, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.check import Check
from research_ci.runner import research
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages.contract import DEFAULT_TARGET, _norm_candidates, _per_t, _use_named
from research_ci.stages.defaults import data as defaults
from research_ci.stages.source import SourceSettings, admit
from research_ci.store import vault
from research_ci.store.sources import Sources, source_key

LEGS = ("entity", "long", "short")
ALL = "all applied"
SPLITS = "splits applied"
RAW = ""
PER_T = {"t": pl.Datetime("ns", "UTC"), "value": pl.Float64}
TESTS_FILE = "data.py"
PITS_FILE = Path("nulls") / "data.py"
TESTS_FORMAT = (
    "data.py defines TESTS = [DataTest(name, check, covers, on_fail), ...], where check(fields) returns the flagged "
    "rows (entity, event_time, optionally field), covers names the (category, field) pairs it catches, and on_fail is "
    "mask, drop_entities, or ask; `rci methods` lists the methods it can call"
)
PITS_FORMAT = (
    "nulls/data.py defines PITS = [Pit(category, plant), ...], where plant(fields, field, rng) returns a copy of the "
    "data with the pit planted in one entity; `rci methods` lists the built-in pits"
)


class DataError(ContractError):
    pass


@dataclass(frozen=True)
class DataSettings:
    source: SourceSettings = field(default_factory=SourceSettings)
    materiality: float = 0.5
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
        self.universe: Any = None
        self.decisive: defaults.Decisive | None = None
        self.inspect: list[str] = []

    def run(self) -> Checked:
        started = time.monotonic()
        found = research.warning([self.root / TESTS_FILE, self.root / PITS_FILE], self.root, self.registry)
        self.warnings += [found] if found else []
        self.tests = _layer_file(self.root / TESTS_FILE, "TESTS", tests_of, TESTS_FORMAT)
        self.pits = _layer_file(self.root / PITS_FILE, "PITS", pits_of, PITS_FORMAT)
        use = as_use(self.c.data)
        plugin = self._plugin(use)
        self.source_plugin = plugin
        everything = field_table(plugin.fetch())
        digest, _ = vault.put(use.name, everything)
        self.deps["data"] = digest
        source = self._source(use, plugin, everything)
        fields = self._explore(everything, digest)
        grades = source["grades"]
        cands = self._candidates(fields)
        numbers = self._stat(fields, cands)
        treatments = self._defaults(fields, cands, source)
        self.spec["tests"] = self._tests(fields, treatments)
        unknown = sorted(set(self.keep) - {t.name for t in treatments})
        if unknown:
            raise DataError(f"no treatment named {', '.join(map(repr, unknown))}; see the DataCard for the names")
        applied = [t for t in treatments if t.name not in self.keep]
        self.spec["kept"] = sorted(self.keep)
        self._grades(fields, grades, numbers)
        panel = tr.apply_all(applied, fields)
        self.spec["treatments"] = self._influence(numbers, fields, treatments, applied, panel)
        self.spec["power"] = self._power(numbers)
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
            "code": spec["code"],
            "fields": spec["fields"],
            "entities": spec["entities"],
            "rows": spec["rows"],
            "checks": spec["checks"],
            "reference": spec.get("reference", {}).get("note", ""),
        }
        return {"spec": spec, "status": status, "grades": spec["grades"]}

    def _explore(self, everything: pl.DataFrame, digest: str) -> pl.DataFrame:
        self.now = time.time_ns()
        self.latest = latest = int(everything.select(pl.col("available_at").max().dt.epoch("ns")).item())
        self.explore_end = latest if self.c.explore_end == "latest" else to_ns(pd.Timestamp(self.c.explore_end))
        seal_end = to_ns(pd.Timestamp(self.c.seal_end)) if self.c.seal_end else None
        fields = visible(everything, Tier.EXPLORE, Tiers(self.explore_end, seal_end))
        self.start = int(fields.select(pl.col("available_at").min().dt.epoch("ns")).item())
        age = (self.now - latest) / pw.DAY_NS
        self.spec["data"] = {
            "source": as_use(self.c.data).name,
            "hash": digest,
            "rows": fields.height,
            "entities": fields["entity"].n_unique(),
            "first": _iso(self.start),
            "explore_end": _iso(self.explore_end),
            "age_days": round(age, 1),
        }
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

    def _stat(self, fields: pl.DataFrame, cands: pl.DataFrame | None) -> "_Decisive | None":
        """The decisive numbers per horizon: every decisive effect's label rows, and its target per decision time."""
        if cands is None:
            return None
        costs: CostModel = self._plugin(as_use(self.c.costs))
        plans = []
        for effect in self.c.decisive:
            label = next(lab for lab in self.c.labels if lab.name == effect.on)
            values: LabelPlugin = self._plugin(as_use(label.use))
            assert self.c.universe is not None
            use = as_use(effect.target) if effect.target else Use(DEFAULT_TARGET, {"cluster": self.c.universe.clusters})
            target: TargetPlugin = self._plugin(use)
            plans += [_Plan(f"{effect.on} {h}", values, target, span(h)) for h in label.horizons]
        numbers = _Decisive(fields, cands, self._universe, plans, costs)
        self.decisive = defaults.Decisive(numbers.rows)
        return numbers

    def _defaults(self, fields: pl.DataFrame, cands: pl.DataFrame | None, source: dict[str, Any]) -> list[Any]:
        seen = defaults.Seen(
            contract=self.c,
            fields=fields,
            candidates=cands,
            source=self.source_plugin,
            universe=self.universe,
            admission=source["spec"],
            latest=self.latest,
            now=self.now,
            decisive=self.decisive,
        )
        out: list[tr.Treatment] = []
        for check in defaults.CHECKS:
            found = check(replace(seen, prior=tuple(t for t in out if t.name not in self.keep)))
            self.checks += found.checks
            self.warnings += found.warnings
            self.inspect += found.decide
            self.spec.update(found.shown)
            out += found.treatments
        return out

    def _tests(self, fields: pl.DataFrame, treatments: list[tr.Treatment]) -> list[dict[str, Any]]:
        out = []
        for t in self.tests:
            try:
                flagged = t.check(fields)
            except Exception as exc:
                raise DataError(f"data test {t.name!r} failed: {type(exc).__name__}: {exc}; {TESTS_FORMAT}") from exc
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

    def _grades(self, fields: pl.DataFrame, grades: dict[str, str], numbers: "_Decisive | None") -> None:
        present = sorted(fields["field"].unique().to_list())
        self.spec["grades"] = {f: grades.get(f, "unverified") for f in present}
        weak = [f for f in present if self.spec["grades"][f] != "verified"]
        if not weak:
            self.checks.append(Check("field grades", "pass", "every field agrees with an independent source"))
            return
        if numbers is None:
            used = weak
        else:
            used = [f for f in weak if _moves(numbers.recompute, numbers.base(), fields.filter(pl.col("field") != f))]
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
        self,
        numbers: "_Decisive | None",
        fields: pl.DataFrame,
        ts: list[tr.Treatment],
        applied: list[tr.Treatment],
        panel: pl.DataFrame,
    ) -> list[dict[str, Any]]:
        """Each treatment's influence, and that of the ones applied together; kept treatments are measured too. A
        split is weighed against the untreated data, and every other treatment on top of the splits that are applied:
        a series the split renames is not the other treatment's doing. These numbers per decision time come from one
        stacked frame, and every gap from one group_by. The ones applied together are computed again on the treated
        data ``panel``, labels and all; that gives the power after."""
        kept = set(self.keep)
        if numbers is None:
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
        splits = [t for t in applied if _renames_rows(t)]
        sets = {t.name: [t] if _renames_rows(t) else [*splits, t] for t in ts}
        if len(splits) > 1:
            sets[SPLITS] = splits
        on_top = SPLITS if len(splits) > 1 else splits[0].name if splits else RAW
        against = {t.name: RAW if _renames_rows(t) else on_top for t in ts}
        frames = [numbers.base_frame().select(pl.lit(RAW).alias("set"), "key", "t", "value"), numbers.per_t(sets)]
        numbers.tables.clear()
        self.base = numbers.base()
        self.after = numbers.recompute(panel, applied) if applied else self.base
        together = len(applied) > 1 or bool(applied and kept)
        if together:
            frames.append(_stacked(ALL, self.after))
            against[ALL] = RAW
        width = self.explore_end - self.start
        self.mde_before = {k: _mde(v, k, width) for k, v in self.base.items()}
        self.mde_after = {k: _mde(v, k, width) for k, v in self.after.items()}
        self.mde = {k: _smaller(self.mde_before[k], self.mde_after.get(k, math.nan)) for k in self.base}
        gaps = _gaps(pl.concat(frames), against)
        rows = []
        for t in ts:
            head = {"name": t.name, "on": t.on, "reason": t.reason, "rows": t.rows, "kept": t.name in kept}
            rows.append(head | self._shift(gaps, t.name))
        if together:
            rows.append({"name": ALL, "on": "both", "reason": "", "rows": 0, "kept": False} | self._shift(gaps, ALL))
        for t in self.tests:
            self._calibrate(t, numbers, fields)
        return rows

    def _shift(self, gaps: pl.DataFrame, name: str) -> dict[str, Any]:
        found = dict(gaps.filter(pl.col("set") == name).select("key", "gap").iter_rows())
        out = {}
        for k in self.base:
            d = found.get(k, math.nan)
            out[k] = {"bp": abs(d) * 1e4, "mde": abs(d) / self.mde[k] if self.mde[k] else math.nan}
            if math.isinf(d):
                out[k]["unbounded"] = True
        largest = max((v["mde"] for v in out.values() if not math.isnan(v["mde"])), default=math.nan)
        return {"influence": out, "largest": largest}

    def _calibrate(self, test: DataTest, numbers: "_Decisive", fields: pl.DataFrame) -> None:
        """Every pit a test claims to cover, planted in clean copies: those that move the decisive number by the
        threshold or more must be flagged. Each pit changes the data in its own way, so the numbers are computed again
        for each."""
        flagged = set(test.check(fields)["entity"].unique().to_list())
        clean = fields.filter(~pl.col("entity").is_in(sorted(flagged)))
        base = numbers.recompute(clean)
        results = []
        for category, name in test.covers:
            plants = [(f"built-in {category}", BUILTIN[category])] if category in defaults.PITS else []
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
                    moved = numbers.recompute(mutated).items()
                    harm = max((abs(_mean_gap(base[k], v)) / self.mde[k] for k, v in moved if self.mde[k]), default=0.0)
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

    def _power(self, numbers: "_Decisive | None") -> list[dict[str, Any]]:
        if numbers is None:
            return []
        after = self.after
        width = self.explore_end - self.start
        rows = []
        for effect in self.c.decisive:
            label = next(lab for lab in self.c.labels if lab.name == effect.on)
            for h in label.horizons:
                k, hn = f"{effect.on} {h}", span_ns(span(h))
                before = pw.power(_se(self.base[k], hn), width, hn)
                now = pw.power(_se(after[k], hn), width, hn)
                rows.append(
                    {
                        "on": effect.on,
                        "horizon": h,
                        "mde_before": before.mde,
                        "mde": now.mde,
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
        decide += self.inspect
        self.spec["decide"] = decide
        if notes:
            return {"status": "blocked", "notes": notes}
        admits = [f"approving also admits {self.spec['source']['plugin']} for every idea"]
        admits = admits if status == "waiting" else []
        if decide:
            return {
                "status": "decide",
                "notes": ["a person decides; approving applies every treatment listed", *admits],
            }
        return {"status": "ready", "notes": ["approving freezes the treated data; feat reads only that", *admits]}


@dataclass(frozen=True)
class _Plan:
    key: str
    values: LabelPlugin
    target: TargetPlugin
    horizon: pd.Timedelta


@dataclass
class _Table:
    """Every key's label rows on one version of the data, the field table ``renames`` leave: the candidates with their
    row (``_c``), the label columns of all keys stacked with the candidate row and the key (``_k``) and their own row
    (``_r``), each key's label columns, and the label rows each treatment touches."""

    renames: tuple[tr.Treatment, ...]
    fields: pl.DataFrame
    cands: pl.DataFrame
    labels: pl.DataFrame
    columns: list[list[str]]
    hits: dict[str, pl.Series] = field(default_factory=dict)


class _Decisive:
    """The decisive numbers. Their label rows are computed once on the explore data and, when treatments rename rows
    (a split), once more on the data those leave. A treatment that drops field rows or candidates leaves out the label
    rows that read what it drops (``eval/data/influence.py``), so weighing it runs only the target again. ``recompute``
    computes everything again, labels and all: for the treated data, and for changes no treatment describes, such as
    a field left out or a planted pit."""

    def __init__(
        self,
        fields: pl.DataFrame,
        cands: pl.DataFrame,
        universe: Callable[[pl.DataFrame], pl.DataFrame],
        plans: list[_Plan],
        costs: CostModel,
    ) -> None:
        self.fields, self.cands, self.universe, self.plans, self.costs = fields, cands, universe, plans, costs
        self.keys = [p.key for p in plans]
        self.key_table = pl.DataFrame(
            {
                "_k": list(range(len(plans))),
                "key": self.keys,
                "h_ns": [int(p.horizon.value) for p in plans],
            },
            schema={"_k": pl.UInt32, "key": pl.Enum(self.keys), "h_ns": pl.Int64},
        )
        self.tables: dict[tuple[str, ...], _Table] = {}
        self._base: pl.DataFrame | None = None

    def rows(self, ts: Sequence[tr.Treatment], columns: Sequence[str]) -> pl.DataFrame:
        """Every key's label rows after ``ts``, stacked: ``key``, ``h_ns`` (the horizon in ns), ``t`` and ``columns``,
        from the label or the candidates; text columns of the candidates come as categoricals."""
        table = self._table(ts)
        dropped = self._dropped(table, ts)
        own = [c for c in columns if c in table.labels.columns]
        theirs = [c for c in columns if c not in own and c != "t"]
        labels = table.labels.select("_r", "_c", "_k", *own)
        labels = labels.filter(~pl.col("_r").is_in(dropped.implode())) if dropped.len() else labels
        text = [c for c in theirs if table.cands.schema[c] == pl.String]
        cand = table.cands.select("t", *theirs).with_columns(pl.col(text).cast(pl.Categorical))
        cand = cand.select(pl.all().gather(labels["_c"]))
        k = self.key_table
        out = (
            labels.select("_k", *own)
            .hstack(cand)
            .with_columns(
                key=pl.col("_k").replace_strict(k["_k"], k["key"], return_dtype=k.schema["key"]),
                h_ns=pl.col("_k").replace_strict(k["_k"], k["h_ns"], return_dtype=pl.Int64),
            )
        )
        return out.select("key", "h_ns", "t", *columns)

    def base(self) -> dict[str, pl.DataFrame]:
        return _by_key(self.base_frame(), self.keys)

    def base_frame(self) -> pl.DataFrame:
        if self._base is None:
            self._base = self.per_t({RAW: []}).drop("set")
        return self._base

    def per_t(self, sets: Mapping[str, Sequence[tr.Treatment]]) -> pl.DataFrame:
        """The number per decision time of every key after each named set of treatments, stacked: ``set``, ``key``,
        ``t``, ``value``. The target runs once per set and key; the label rows come from the tables."""
        out = [pl.DataFrame(schema={"set": pl.String, "key": pl.String, **PER_T})]
        by_table: dict[tuple[str, ...], list[str]] = {}
        for name, ts in sets.items():
            by_table.setdefault(_renames(ts), []).append(name)
        for names in by_table.values():
            table = self._table(sets[names[0]])
            dropped = {name: self._dropped(table, sets[name]) for name in names}
            for k, plan in enumerate(self.plans):
                joined = self._joined(table, k)
                for name in names:
                    d = dropped[name]
                    rows = joined.filter(~pl.col("_r").is_in(d.implode())) if d.len() else joined
                    v = _per_t(plan.target.reduce(rows.drop("_r"), plan.horizon, self.costs))
                    out.append(v.select(pl.lit(name).alias("set"), pl.lit(plan.key).alias("key"), "t", "value"))
        return pl.concat(out)

    def recompute(self, f: pl.DataFrame, ts: Sequence[tr.Treatment] = ()) -> dict[str, pl.DataFrame]:
        """The numbers on the field table ``f``, with the candidate treatments in ``ts`` applied."""
        c = tr.apply_candidates(ts, self.universe(f))
        return {
            p.key: _per_t(
                p.target.reduce(
                    p.values.values(f, c, p.horizon).join(c, on=["t", "entity"], how="inner", suffix="_candidate"),
                    p.horizon,
                    self.costs,
                )
            )
            for p in self.plans
        }

    def _table(self, ts: Sequence[tr.Treatment]) -> _Table:
        renames = tuple(t for t in ts if _renames_rows(t))
        name = tuple(t.name for t in renames)
        if name not in self.tables:
            f = tr.apply_all(renames, self.fields) if renames else self.fields
            self.tables[name] = self._build(f, self.universe(f) if renames else self.cands, renames)
        return self.tables[name]

    def _build(self, f: pl.DataFrame, c: pl.DataFrame, renames: tuple[tr.Treatment, ...]) -> _Table:
        indexed = c.with_row_index("_c")
        index = indexed.select("t", "entity", "_c")
        frames, columns = [], []
        for k, plan in enumerate(self.plans):
            v = plan.values.values(f, c, plan.horizon)
            columns.append(v.columns)
            frames.append(v.join(index, on=["t", "entity"]).drop("t", "entity").with_columns(_k=pl.lit(k, pl.UInt32)))
        labels = pl.concat(frames, how="diagonal_relaxed").with_row_index("_r")
        return _Table(renames, f, indexed, labels, columns)

    def _joined(self, table: _Table, k: int) -> pl.DataFrame:
        """One key's label rows joined with their candidates, as a target reads them, with the label row ``_r``."""
        labels = table.labels.filter(pl.col("_k") == k)
        columns = table.columns[k]
        own = [c for c in columns if c not in ("t", "entity")]
        theirs = [c for c in table.cands.columns if c not in ("t", "entity", "_c")]
        cand = table.cands.select(pl.col("t", "entity", *theirs).gather(labels["_c"]))
        cand = cand.rename({c: f"{c}_candidate" for c in theirs if c in own})
        out = labels.select("_r", *own).hstack(cand)
        return out.select("_r", *columns, *(c for c in cand.columns if c not in ("t", "entity")))

    def _dropped(self, table: _Table, ts: Sequence[tr.Treatment]) -> pl.Series:
        """The label rows of ``table`` that the treatments in ``ts`` that drop rows touch."""
        drops = [t for t in ts if t.removed is not None]
        self._touch(table, [t for t in drops if t.name not in table.hits])
        parts = [table.hits[t.name] for t in drops]
        return pl.concat(parts).unique() if parts else pl.Series("_r", [], dtype=pl.UInt32)

    def _touch(self, table: _Table, ts: Sequence[tr.Treatment]) -> None:
        if not ts:
            return
        found = [pl.DataFrame(schema={"treatment": pl.String, "_r": pl.UInt32})]
        on_fields = [t for t in ts if t.on == "fields"]
        if on_fields:
            gone = pl.concat(
                [
                    _removed(t, self.fields).select(
                        "entity", "field", "event_time", "available_at", treatment=pl.lit(t.name)
                    )
                    for t in on_fields
                ]
            )
            spans = inf.spans(table.fields, reduce(lambda g, r: r.apply(g), table.renames, gone), by=["treatment"])
            legs = [c for c in LEGS if c in table.cands.columns]
            near = pl.any_horizontal(pl.col(c).is_in(spans["entity"].unique().implode()) for c in legs)
            rows = table.cands.filter(near).select("_c", "t", *legs)
            rows = table.labels.select("_r", "_c", "_k").join(rows, on="_c").join(self.key_table, on="_k")
            found.append(inf.touched(rows, spans, legs, by=["treatment"]))
        on_cands = [t for t in ts if t.on == "candidates"]
        if on_cands:
            gone = pl.concat([_removed(t, table.cands).select("_c", treatment=pl.lit(t.name)) for t in on_cands])
            found.append(table.labels.select("_r", "_c").join(gone, on="_c").select("treatment", "_r"))
        hits = pl.concat(found).partition_by("treatment", as_dict=True, include_key=False)
        table.hits.update({t.name: hits[(t.name,)]["_r"] if (t.name,) in hits else found[0]["_r"] for t in ts})


def _removed(t: tr.Treatment, table: pl.DataFrame) -> pl.DataFrame:
    assert t.removed is not None
    return t.removed(table)


def _renames_rows(t: tr.Treatment) -> bool:
    """A treatment that changes identities instead of dropping rows, such as a split."""
    return t.on == "fields" and t.removed is None


def _renames(ts: Sequence[tr.Treatment]) -> tuple[str, ...]:
    return tuple(t.name for t in ts if _renames_rows(t))


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


def _layer_file(path: Path, name: str, read: Callable[[Any], list[Any]], fmt: str) -> list[Any]:
    """An optional file of the idea; one that exists must load and declare ``name``, and a failure says what it must
    look like."""
    if not path.is_file():
        return []
    spec = importlib.util.spec_from_file_location(f"_rci_{path.parent.name}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise DataError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise DataError(f"{path} failed to load: {type(exc).__name__}: {exc}; {fmt}") from exc
    if not hasattr(module, name):
        raise DataError(f"{path} defines no {name}; {fmt}")
    try:
        return read(module)
    except ValueError as exc:
        raise DataError(f"{path}: {exc}; {fmt}") from exc


def _moves(
    numbers: Callable[[pl.DataFrame], dict[str, pl.DataFrame]], base: dict[str, pl.DataFrame], without: pl.DataFrame
) -> bool:
    try:
        other = numbers(without)
    except Exception:  # noqa: BLE001
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


def _gaps(per_t: pl.DataFrame, against: Mapping[str, str]) -> pl.DataFrame:
    """The mean gap per decision time between each set's number and that of the set it is measured ``against``, per
    key, from the stacked numbers of all of them: ``set``, ``key``, ``gap``."""
    pairs = pl.DataFrame(
        {"set": list(against), "ref": list(against.values())}, schema={"set": pl.String, "ref": pl.String}
    )
    ref = per_t.rename({"set": "ref", "value": "value_ref"})
    joined = per_t.join(pairs, on="set").join(ref, on=["ref", "key", "t"], how="inner")
    return joined.group_by("set", "key").agg((pl.col("value") - pl.col("value_ref")).mean().alias("gap"))


def _stacked(name: str, by_key: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(
        [v.select(pl.lit(name).alias("set"), pl.lit(k).alias("key"), "t", "value") for k, v in by_key.items()]
    )


def _by_key(per_t: pl.DataFrame, keys: Sequence[str]) -> dict[str, pl.DataFrame]:
    parts = per_t.partition_by("key", as_dict=True, include_key=False)
    return {k: parts[(k,)].select("t", "value") if (k,) in parts else pl.DataFrame(schema=PER_T) for k in keys}


def _mde(per_t: pl.DataFrame, key: str, width: int) -> float:
    """The smallest effect this horizon can detect; it does not depend on the effect asked for."""
    h = span_ns(span(key.rsplit(" ", 1)[1]))
    return pw.power(_se(per_t, h), width, h).mde


def _se(per_t: pl.DataFrame, h: int) -> float:
    """No noise can be measured on a number that is not finite somewhere."""
    return pw.block_se(per_t, h) if per_t["value"].is_finite().all() else math.nan


def _smaller(a: float, b: float) -> float:
    """The finer of two resolutions: a treatment matters if it moves the number by more than the verdict on the
    treated data, or on the untreated data, could resolve."""
    finite = [x for x in (a, b) if math.isfinite(x) and x > 0]
    return min(finite) if finite else math.nan


def _iso(ns: int) -> str:
    return from_ns(ns).strftime("%Y-%m-%dT%H:%MZ")
