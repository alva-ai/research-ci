"""Checking an idea's data against its approved contract.

The source must pass admission (fidelity: the rows are what the venue reported, ``stages/data/source.py``), which runs
once per source and is approved with the first idea's data, and the decisive number may read only fields an independent
source verified, or the history is explore-grade. The data check builds what its steps read, then runs its floor list
(``stages/data/floor.py``) with its default list (``stages/data/defaults.py``; fitness: whether faithful rows can be
used for research) where the floor marks it: first the steps that find treatments, then, on the data with every
treatment found, the steps that weigh them, and last, once nothing blocks, the calibration of the decisive test on the
treated data.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from research_ci.api.contract import Contract, Use, as_use, span
from research_ci.api.data import pits_of
from research_ci.api.evals import evals_of
from research_ci.api.plugins import CostModel
from research_ci.kernel.holdout.tiers import Tier, Tiers, visible
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import from_ns, span_ns, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.check import Check
from research_ci.runner import research
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages import skeleton
from research_ci.stages.common import iso, norm_candidates, target_use
from research_ci.stages.contract.run import use_named
from research_ci.stages.data import defaults, floor
from research_ci.stages.data import treatments as tr
from research_ci.stages.data.floor import ALL, DataError, DataSettings
from research_ci.stages.data.source import admit
from research_ci.stages.feat import run as fs
from research_ci.store import vault
from research_ci.store.sources import Sources, source_key
from research_ci.validation import load as ev

EVALS_FILE = "data.py"
PITS_FILE = Path("nulls") / "data.py"
EVALS_FORMAT = floor.EVALS_FORMAT
PITS_FORMAT = (
    "nulls/data.py defines PITS = [Pit(category, plant), ...], where plant(fields, field, rng) returns a copy of the "
    "data with the pit planted in one entity; `rci methods` lists the built-in pits"
)
Key = Callable[[fs.Frozen], dict[str, str]]


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
    feat_settings: fs.FeatSettings | None = None,
    key: Key | None = None,
) -> Checked:
    """``keep`` names treatments a person chose not to apply; their influence is still measured and shown. ``key``
    says what a calibration of the decisive test rests on, as feat compares it."""
    check = _DataCheck(contract, Path(root), registry, settings or DataSettings(), sources or Sources(), keep)
    return check.run(feat_settings, key or (lambda frozen: fs.calibration_key(frozen, feat_settings)))


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
        self.deps: dict[str, str] = {}
        self.spec: dict[str, Any] = {}

    def run(self, feat_settings: fs.FeatSettings | None, key: Key) -> Checked:
        started = time.monotonic()
        seen = self._seen()
        found = self._find(seen)
        unknown = sorted(set(self.keep) - {t.name for t in found.treatments})
        if unknown:
            raise DataError(f"no treatment named {', '.join(map(repr, unknown))}; see the DataCard for the names")
        applied = [t for t in found.treatments if t.name not in self.keep]
        panel = tr.apply_all(applied, seen.fields)
        treated = replace(seen, prior=tuple(applied), found=tuple(found.treatments), panel=panel)
        weighed = skeleton.combined(skeleton.run(step, treated) for step in floor.WEIGH)
        found.shown.pop("validated", None)
        self.spec |= {"evals": []} | found.shown | weighed.shown
        self.spec["kept"] = sorted(self.keep)
        use = as_use(self.c.data)
        panel_hash, _ = vault.put(f"{use.name}.panel", panel)
        excluded = _exclusions(seen.candidates, applied)
        exclusions_hash, _ = vault.put(f"{use.name}.exclusions", excluded)
        self.spec["panel"] = {"hash": panel_hash, "rows": panel.height, "exclusions": exclusions_hash}
        self.deps["panel"] = panel_hash
        self.deps["exclusions"] = exclusions_hash
        checks = found.checks + weighed.checks
        self.spec["checks"] = [asdict(ch) for ch in checks]
        self.spec["warnings"] = found.warnings + weighed.warnings
        self.spec["materiality"] = self.s.materiality
        self.spec["verdict"] = self._verdict([c for c in checks if c.status == "fail"], found.decide)
        if self.spec["verdict"]["status"] != "blocked" and self.c.universe is not None:
            feat = feat_settings or fs.FeatSettings()

            def frozen() -> fs.Frozen:
                return fs.frozen(self.c, panel_hash, exclusions_hash, self.registry, feat_settings)

            treated = replace(treated, frozen=frozen, key=key, feat=feat)
        last = skeleton.combined(skeleton.run(step, treated) for step in floor.LAST)
        if "calibration" in last.shown:
            self._calibrated(last.shown["calibration"])
        self.spec["seconds"] = round(time.monotonic() - started, 1)
        return Checked(self.spec, dict(sorted(self.deps.items())))

    def _seen(self) -> floor.Seen:
        """What the steps read: the idea's files, the explore data, the source's admission, the candidates, and the
        decisive numbers."""
        warning = research.warning([self.root / EVALS_FILE, self.root / PITS_FILE], self.root, self.registry)
        evals = _layer_file(self.root, EVALS_FILE, "EVALS", lambda m: evals_of(m, "data"), EVALS_FORMAT)
        pits = _layer_file(self.root, PITS_FILE, "PITS", pits_of, PITS_FORMAT)
        try:
            ev.refuse_untested(evals, self.root, EVALS_FILE, "; nothing was recorded")
        except ev.EvalsError as exc:
            raise DataError(str(exc)) from exc
        use = as_use(self.c.data)
        plugin = self._plugin(use)
        everything = field_table(plugin.fetch())
        digest, _ = vault.put(use.name, everything)
        self.deps["data"] = digest
        admission = self._source(use, plugin, everything)
        fields = self._explore(everything, digest)
        cands, universe = self._candidates(fields)
        numbers = self._numbers(fields, cands, universe)
        return floor.Seen(
            contract=self.c,
            fields=fields,
            candidates=cands,
            source=plugin,
            universe=universe,
            admission=admission,
            latest=self.latest,
            now=self.now,
            decisive=None if numbers is None else floor.Decisive(numbers.rows),
            settings=self.s,
            research=warning,
            root=self.root,
            registry=self.registry,
            evals=tuple(evals),
            pits=tuple(pits),
            builtin=tuple(defaults.PITS),
            numbers=numbers,
        )

    def _find(self, seen: floor.Seen) -> skeleton.Found:
        """The steps that find treatments, in order; each sees the treatments found before it that will be applied,
        and what the steps before it showed."""
        found = skeleton.Found()
        for step in skeleton.ordered(floor.FIND, defaults.CHECKS):
            prior = tuple(t for t in found.treatments if t.name not in self.keep)
            ran = skeleton.run(step, replace(seen, prior=prior, earlier=found.shown), defaults.CHECKS)
            found = skeleton.combined([found, ran])
        return found

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
        return spec

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
            "first": iso(self.start),
            "explore_end": iso(self.explore_end),
            "age_days": round(age, 1),
        }
        return fields

    def _candidates(self, fields: pl.DataFrame) -> tuple[pl.DataFrame | None, Any]:
        u = self.c.universe
        if u is None:
            return None, None
        if self.c.consumer is not None:
            timing = load(as_use(self.c.consumer), self.registry).timing()
            every, offset = timing.every, timing.offset
        else:
            assert self.c.execution is not None
            every, offset = self.c.execution.every, self.c.execution.offset
        self.times = pw.decision_grid(span_ns(span(every)), span_ns(span(offset)), self.start, self.explore_end)
        universe = self._plugin(as_use(u.use))
        return self._universe(universe, fields), universe

    def _universe(self, universe: Any, fields: pl.DataFrame) -> pl.DataFrame:
        return norm_candidates(universe.candidates(fields, [from_ns(t) for t in self.times]))

    def _numbers(self, fields: pl.DataFrame, cands: pl.DataFrame | None, universe: Any) -> tr.Numbers | None:
        """The decisive numbers per horizon: every decisive effect's label rows, and its target per decision time."""
        if cands is None:
            return None
        costs: CostModel = self._plugin(as_use(self.c.costs))
        plans = []
        for effect in self.c.decisive:
            label = next(lab for lab in self.c.labels if lab.name == effect.on)
            values = self._plugin(as_use(label.use))
            target = self._plugin(target_use(effect, self.c))
            plans += [tr.Plan(f"{effect.on} {h}", values, target, span(h)) for h in label.horizons]

        def candidates(f: pl.DataFrame) -> pl.DataFrame:
            return self._universe(universe, f)

        return tr.Numbers(fields, cands, candidates, plans, costs, self.explore_end - self.start)

    def _verdict(self, failed: list[Check], inspect: list[str]) -> dict[str, Any]:
        """Blocked when the source is not admitted or a check failed, each with what failed, why, and what to do;
        otherwise ready, or a person decides."""
        notes, decide = [], []
        status = self.spec["source"]["status"]
        if status == "not admitted":
            names = ", ".join(self.spec["source"]["failed"])
            notes.append(
                f"source {self.spec['source']['plugin']} not admitted: its admission failed {names} (the source lines "
                "above say why); fix the data plugin, or its reference source, and check again"
            )
        fixes = {row["name"]: row for row in self.spec["evals"]}
        for c in failed:
            row = fixes.get(c.name)
            if row is not None and c.by in ("case", "library"):
                v = row["validation"]
                notes.append(f"{c.name} ({c.by} eval) is invalid: {v['summary']}; {v['fix']}")
            else:
                notes.append(f"{c.name} ({c.by or 'floor'}) failed: {c.detail}; fix what it names and check again")
        for t in self.spec["treatments"]:
            if t.get("influence") and t["name"] != ALL and t["largest"] >= self.s.materiality:
                choice = "kept, not applied" if t["kept"] else "applied"
                decide.append(f"{t['name']} ({choice}): moves the decisive number by {t['largest']:.2f} MDE")
        decide += inspect
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

    def _calibrated(self, calibration: dict[str, Any]) -> None:
        """The decisive test's calibration goes into the record; one that fails blocks the check with the reason."""
        self.spec["calibration"] = calibration
        if not calibration["ok"]:
            why = "; ".join(calibration["notes"])
            note = (
                "the decisive test cannot be calibrated on this data (decisive test calibration, floor): "
                f"{why}; change the target or the contract"
            )
            self.spec["verdict"] = {"status": "blocked", "notes": [note]}


def _exclusions(cands: pl.DataFrame | None, ts: list[tr.Treatment]) -> pl.DataFrame:
    empty = pl.DataFrame(schema={"t": pl.Datetime("ns", "UTC"), "entity": pl.String})
    if cands is None:
        return empty
    kept = tr.apply_candidates(ts, cands.select("t", "entity"))
    return cands.select("t", "entity").join(kept, on=["t", "entity"], how="anti").sort("t", "entity")


def current_data_deps(
    contract: Contract, recorded: dict[str, str], spec: dict[str, Any], registry: Registry | None = None
) -> dict[str, str]:
    """The data check's dependencies as they are now, keyed like the recorded ones."""
    now: dict[str, str] = {}
    for key in recorded:
        if key.startswith("plugin:"):
            now[key] = code_hash(load(use_named(contract, key.removeprefix("plugin:")), registry))
    name = as_use(contract.data).name
    now["data"] = recorded["data"] if vault.has(name, recorded["data"]) else "missing"
    now["panel"] = recorded["panel"] if vault.has(f"{name}.panel", recorded["panel"]) else "missing"
    if "exclusions" in recorded:
        found = vault.has(f"{name}.exclusions", recorded["exclusions"])
        now["exclusions"] = recorded["exclusions"] if found else "missing"
    record = Sources().latest(spec["source"]["key"])
    now["source"] = "missing" if record is None else record["hash"]
    return now


def _layer_file(root: Path, rel: str | Path, name: str, read: Callable[[Any], list[Any]], fmt: str) -> list[Any]:
    """An optional file of the idea; one that exists must load and declare ``name``, and a failure says what it must
    look like."""
    path = Path(root) / rel
    if not path.is_file():
        return []
    try:
        module = ev.load_module(path, root)
    except ev.EvalsError as exc:
        raise DataError(f"{exc}; {fmt}") from exc
    if not hasattr(module, name):
        raise DataError(f"{rel} defines no {name}; {fmt}")
    try:
        return read(module)
    except ValueError as exc:
        raise DataError(f"{rel}: {exc}; {fmt}") from exc
