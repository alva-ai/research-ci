"""Checking an idea's data against its approved contract.

The source must pass admission (fidelity: the rows are what the venue reported, ``stages/data/source.py``), which runs
once per source and is approved with the first idea's data, and the decisive number may read only columns an
independent source verified, or the history is explore-grade. Before anything heavy runs, the data check refuses a
``data.py`` without a lab eval, or with one whose tests fail (``stages/lab.py``). It then builds what its steps read,
the explore sources, ``wide`` on the contract's decision times and the candidates, and runs its floor list
(``stages/data/floor.py``) with its default list (``stages/data/defaults.py``; fitness: whether faithful rows can be
used for research) where the floor marks it: first the steps that find treatments, the default list, the pits the
evals cover, the library evals and the lab evals, then, on the data with every treatment found, the steps that weigh
them, and last, once nothing blocks, the calibration of the decisive test on the treated data. The freeze pins the
treated sources, ``wide`` built on them, and the candidate rows the treatments exclude.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from research_ci.api.contract import Contract, Use, as_use, horizons_of, span
from research_ci.api.evals import evals_of
from research_ci.api.plugins import CostModel, Sources
from research_ci.eval.data.baseline import rows
from research_ci.kernel.holdout.tiers import Tier, Tiers, visible
from research_ci.kernel.tables import source_tables
from research_ci.kernel.time.asof import wide
from research_ci.kernel.time.clock import span_ns, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.check import Check
from research_ci.runner import research
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages import lab, skeleton
from research_ci.stages.common import cadence, iso, norm_candidates, target_use
from research_ci.stages.contract.run import use_named
from research_ci.stages.data import defaults, floor
from research_ci.stages.data import treatments as tr
from research_ci.stages.data.floor import ALL, DataError, DataSettings
from research_ci.stages.data.source import admit
from research_ci.stages.feat import run as fs
from research_ci.store import vault
from research_ci.store.sources import Sources as Admissions
from research_ci.store.sources import source_key
from research_ci.validation import load as ev

EVALS_FILE = "data.py"
EVALS_FORMAT = floor.EVALS_FORMAT
NOTHING = "; nothing was recorded"
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
    admissions: Admissions | None = None,
    keep: tuple[str, ...] = (),
    feat_settings: fs.FeatSettings | None = None,
    key: Key | None = None,
) -> Checked:
    """``keep`` names treatments a person chose not to apply; their influence is still measured and shown. ``key``
    says what a calibration of the decisive test rests on, as feat compares it."""
    check = _DataCheck(contract, Path(root), registry, settings or DataSettings(), admissions or Admissions(), keep)
    return check.run(feat_settings, key or (lambda frozen: fs.calibration_key(frozen, feat_settings)))


class _DataCheck:
    def __init__(
        self,
        c: Contract,
        root: Path,
        registry: Registry | None,
        s: DataSettings,
        admissions: Admissions,
        keep: tuple[str, ...] = (),
    ) -> None:
        self.c, self.root, self.registry, self.s, self.keep = c, root, registry, s, keep
        self.admissions = admissions
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
        treated = tr.apply_all(applied, seen.sources)
        after = replace(seen, prior=tuple(applied), found=tuple(found.treatments), treated=treated)
        weighed = skeleton.combined(skeleton.run(step, after) for step in floor.WEIGH)
        found.shown.pop("validated", None)
        self.spec |= {"evals": []} | found.shown | weighed.shown
        self.spec["kept"] = sorted(self.keep)
        name = as_use(self.c.data).name
        treated_hash, _ = vault.put(f"{name}.treated", treated)
        wide_hash, _ = vault.put(f"{name}.wide", {"wide": wide(treated, seen.times)})
        exclusions_hash, _ = vault.put(f"{name}.exclusions", {"exclusions": _exclusions(seen.candidates, applied)})
        self.spec["treated"] = {
            "sources": treated_hash,
            "wide": wide_hash,
            "rows": sum(t.height for t in treated.values()),
            "exclusions": exclusions_hash,
        }
        self.deps |= {"treated": treated_hash, "wide": wide_hash, "exclusions": exclusions_hash}
        checks = found.checks + weighed.checks
        self.spec["checks"] = [asdict(ch) for ch in checks]
        self.spec["warnings"] = found.warnings + weighed.warnings
        self.spec["materiality"] = self.s.materiality
        self.spec["verdict"] = self._verdict([c for c in checks if c.status == "fail"], found.decide)
        if self.spec["verdict"]["status"] != "blocked" and self.c.universe is not None:
            feat = feat_settings or fs.FeatSettings()

            def frozen() -> fs.Frozen:
                return fs.frozen(self.c, treated_hash, wide_hash, exclusions_hash, self.registry, feat_settings)

            after = replace(after, frozen=frozen, key=key, feat=feat)
        last = skeleton.combined(skeleton.run(step, after) for step in floor.LAST)
        if "calibration" in last.shown:
            self._calibrated(last.shown["calibration"])
        self.spec["seconds"] = round(time.monotonic() - started, 1)
        return Checked(self.spec, dict(sorted(self.deps.items())))

    def _seen(self) -> floor.Seen:
        """What the steps read: the idea's files, the explore data, the source's admission, the candidates, and the
        decisive numbers. A ``data.py`` without a valid lab eval is refused first, before the data is fetched."""
        warning = research.warning([self.root / EVALS_FILE], self.root, self.registry)
        evals = _layer_file(self.root, EVALS_FILE, "EVALS", lambda m: evals_of(m, "data"), EVALS_FORMAT)
        try:
            ev.refuse_untested(evals, self.root, EVALS_FILE, NOTHING)
            validated = ev.validate(evals, self.root, self.registry)
        except ev.EvalsError as exc:
            raise DataError(f"{str(exc).removesuffix(NOTHING)}{NOTHING}") from exc
        ready = lab.Lab("data", EVALS_FILE, (self.root / EVALS_FILE).is_file(), tuple(evals), validated)
        refused = lab.refusal(ready)
        if refused:
            raise DataError(f"{refused}{NOTHING}")
        use = as_use(self.c.data)
        plugin = self._plugin(use)
        everything = source_tables(plugin.fetch())
        digest, _ = vault.put(use.name, everything)
        self.deps["data"] = digest
        admission = self._source(use, plugin, everything)
        sources = self._explore(everything, digest)
        timing, _ = cadence(self.c, self._plugin)
        self.times = pw.decision_grid(
            span_ns(span(timing.every)), span_ns(span(timing.offset)), self.start, self.explore_end
        )
        self.wide = wide(sources, self.times)
        cands, universe = self._candidates(sources)
        horizons = horizons_of(self.c, timing.every)
        numbers = self._numbers(sources, cands, universe, horizons)
        return floor.Seen(
            contract=self.c,
            horizons=horizons,
            sources=sources,
            times=self.times,
            wide=self.wide,
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
            validated=validated,
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

    def _source(self, use: Use, plugin: Any, everything: Sources) -> dict[str, Any]:
        key = source_key(use.name, dict(use.options), code_hash(plugin))
        record = self.admissions.latest(key)
        if record is None:
            record = self.admissions.record(key, admit(use, plugin, everything, self.s.source))
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
            "sources": spec["sources"],
            "insts": spec["insts"],
            "rows": spec["rows"],
            "checks": spec["checks"],
            "reference": spec.get("reference", {}).get("note", ""),
        }
        return spec

    def _explore(self, everything: Sources, digest: str) -> dict[str, pl.DataFrame]:
        self.now = time.time_ns()
        known = rows(everything).select(pl.col("available_at").dt.epoch("ns"))["available_at"]
        self.latest = latest = int(known.max())  # type: ignore[arg-type]
        self.explore_end = latest if self.c.explore_end == "latest" else to_ns(pd.Timestamp(self.c.explore_end))
        seal_end = to_ns(pd.Timestamp(self.c.seal_end)) if self.c.seal_end else None
        sources = visible(everything, Tier.EXPLORE, Tiers(self.explore_end, seal_end))
        explored = rows(sources)
        self.start = int(explored.select(pl.col("available_at").min().dt.epoch("ns")).item())
        age = (self.now - latest) / pw.DAY_NS
        self.spec["data"] = {
            "source": as_use(self.c.data).name,
            "hash": digest,
            "sources": list(sources),
            "rows": explored.height,
            "insts": explored["inst"].n_unique(),
            "first": iso(self.start),
            "explore_end": iso(self.explore_end),
            "age_days": round(age, 1),
        }
        return sources

    def _candidates(self, sources: Sources) -> tuple[pl.DataFrame | None, Any]:
        u = self.c.universe
        if u is None:
            return None, None
        universe = self._plugin(as_use(u.use))
        return norm_candidates(universe.candidates(self.wide, sources)), universe

    def _universe(self, universe: Any, sources: Sources) -> pl.DataFrame:
        return norm_candidates(universe.candidates(wide(sources, self.times), sources))

    def _numbers(
        self, sources: Sources, cands: pl.DataFrame | None, universe: Any, horizons: tuple[str, ...]
    ) -> tr.Numbers | None:
        """The decisive numbers per effect and horizon: the outcome rows, and each effect's target per decision
        time."""
        if cands is None:
            return None
        assert self.c.outcome is not None
        costs: CostModel = self._plugin(as_use(self.c.costs))
        outcome = self._plugin(as_use(self.c.outcome))
        plans = [
            tr.Plan(f"{effect.name} {h}", self._plugin(target_use(effect, self.c)), span(h))
            for effect in self.c.decisive
            for h in horizons
        ]

        def candidates(f: Sources) -> pl.DataFrame:
            return self._universe(universe, f)

        return tr.Numbers(sources, cands, candidates, outcome, plans, costs, self.explore_end - self.start)

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
            if row is not None and c.by in ("lab", "library"):
                notes.append(_eval_note(row))
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


def _eval_note(row: dict[str, Any]) -> str:
    """Why an eval of ``data.py`` blocks the check: a lab eval's veto, or an eval that missed a pit it covers."""
    if row.get("examples"):
        return (
            f"lab eval {row['name']!r} (slot 3) vetoes the data: H0: {row['h0']}; it flagged {row['flagged']:,} rows "
            f"that the treatments before it left, such as {', '.join(row['examples'])}; with on_fail='block' any "
            "flagged row blocks the check: fix the data or the data plugin and check again; if the eval is wrong, fix "
            "it and its CASES"
        )
    v = row["validation"]
    slot = "slot 4, tests of lab evals" if row["by"] == "lab" else "slot 2, its pits"
    return f"{row['by']} eval {row['name']!r} ({slot}) is invalid: {v['summary']}; {v['fix']}"


def _exclusions(cands: pl.DataFrame | None, ts: list[tr.Treatment]) -> pl.DataFrame:
    """The candidate rows the candidate treatments drop."""
    empty = pl.DataFrame(schema={"t": pl.Datetime("ns", "UTC"), "inst": pl.String})
    if cands is None:
        return empty
    kept = tr.apply_candidates(ts, cands).select("t", "inst")
    return cands.select("t", "inst").join(kept, on=["t", "inst"], how="anti").sort("t", "inst")


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
    for kind in ("treated", "wide", "exclusions"):
        now[kind] = recorded[kind] if vault.has(f"{name}.{kind}", recorded[kind]) else "missing"
    record = Admissions().latest(spec["source"]["key"])
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
