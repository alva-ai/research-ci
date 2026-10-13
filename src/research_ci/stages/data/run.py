"""Checking an idea's data against its approved contract.

The source must pass admission (fidelity: the rows are what the venue reported, ``stages/data/source.py``), which runs
once per source and is approved with the first idea's data, and the decisive number may read only columns an
independent source verified, or the history is explore-grade. Before anything heavy runs, the data check refuses a
``data.py`` without a lab eval, or with one whose tests fail (``stages/lab.py``). It then builds what its steps read,
the explore sources, the decision grid (the contract's clock, or the decision times per inst the idea's sampler,
``grid`` in ``data.py``, gives), ``wide`` on it and the candidates, and runs its floor list (``stages/data/floor.py``)
with its default list (``stages/data/defaults.py``; fitness: whether faithful rows can be used for research) where the
floor marks it: first the steps that find treatments, the default list, the pits the evals cover, the library evals
and the lab evals, then, on the data with every treatment found, the steps that weigh them, and last, once nothing
blocks, the calibration of the decisive test on the treated data. What the card approves is stored in the vault as
parquet (``PRODUCTS``), and the freeze pins it: the treated sources, the decision grid on them, ``wide`` built on it,
the candidates the treatments leave, their outcomes, and the candidate rows the treatments exclude; feat reads these
and computes none of them again. Outcome horizons stay in clock time.
"""

import inspect
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
from research_ci.kernel.time.asof import Grid, wide
from research_ci.kernel.time.clock import TIME, span_ns, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.check import Check
from research_ci.runner import research
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages import lab, skeleton
from research_ci.stages.common import cadence, iso, median, norm_candidates, outcome_values, target_use
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
HOUR_NS = 3_600_000_000_000
PRODUCTS = ("treated", "grid", "wide", "candidates", "outcomes", "exclusions")
GRID_FORMAT = (
    "data.py may define a sampler, grid(sources), or grid(sources, wide) with wide on the contract's clock, which "
    "returns the decision times per inst: a DataFrame (polars or pandas) or a polars LazyFrame with columns t and "
    "inst, each t from rows known by then (available_at <= t); without one the decisions follow the contract's clock"
)
UNEVEN = (
    "the decision grid is uneven: the shuffles, blocks and the MDE assume an even one, so results on it are "
    "approximate (docs/todo.md, item 21)"
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
        self.spec["products"] = self._products(seen, treated, applied)
        products = {k: p["hash"] for k, p in self.spec["products"].items()}
        self.deps |= products
        checks = found.checks + weighed.checks
        self.spec["checks"] = [asdict(ch) for ch in checks]
        self.spec["warnings"] = found.warnings + weighed.warnings + ([] if self.spec["grid"]["even"] else [UNEVEN])
        self.spec["materiality"] = self.s.materiality
        self.spec["verdict"] = self._verdict([c for c in checks if c.status == "fail"], found.decide)
        if self.spec["verdict"]["status"] != "blocked" and self.c.universe is not None:
            feat = feat_settings or fs.FeatSettings()

            def frozen() -> fs.Frozen:
                return fs.frozen(self.c, products, self.registry, feat_settings)

            after = replace(after, frozen=frozen, key=key, feat=feat)
        last = skeleton.combined(skeleton.run(step, after) for step in floor.LAST)
        if "calibration" in last.shown:
            self._calibrated(last.shown["calibration"])
        self.spec["seconds"] = round(time.monotonic() - started, 1)
        return Checked(self.spec, dict(sorted(self.deps.items())))

    def _products(self, seen: floor.Seen, treated: Sources, applied: list[tr.Treatment]) -> dict[str, Any]:
        """What the card approves, stored in the vault by content: the treated sources, the decision grid on them
        (``t | inst``, whose shape the card shows) and ``wide`` built on it, the candidates on it that the treatments
        leave, their outcome at every horizon (``h``), and the candidate rows the treatments exclude; each with its
        hash, folder and rows."""
        built = wide(treated, self._grid(treated))
        grid = built.select("t", "inst")
        self.spec["grid"] = _shape(grid) | {"by": self.by}
        excluded = _exclusions(seen.candidates, applied)
        cands = pl.DataFrame(schema={"t": TIME, "inst": pl.String})
        outcomes = pl.DataFrame(schema={"t": TIME, "inst": pl.String, "h": pl.String, "y": pl.Float64, "y_end": TIME})
        if seen.universe is not None:
            assert self.c.outcome is not None
            cands = norm_candidates(seen.universe.candidates(built, treated))
            cands = cands.join(excluded, on=["t", "inst"], how="anti").sort("t", "inst")
            outcomes = _outcomes(self._plugin(as_use(self.c.outcome)), treated, cands, seen.horizons)
        tables = {
            "treated": treated,
            "grid": {"grid": grid},
            "wide": {"wide": built},
            "candidates": {"candidates": cands},
            "outcomes": {"outcomes": outcomes},
            "exclusions": {"exclusions": excluded},
        }
        name = as_use(self.c.data).name
        out = {}
        for kind, product in tables.items():
            digest, folder = vault.put(f"{name}.{kind}", product)
            out[kind] = {"hash": digest, "path": str(folder), "rows": sum(t.height for t in product.values())}
        return out

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
        self.clock = pw.decision_grid(
            span_ns(span(timing.every)), span_ns(span(timing.offset)), self.start, self.explore_end
        )
        self.sampler = _sampler(self.root)
        self.by = "the contract's clock, every " + timing.every if self.sampler is None else "grid in data.py"
        self.times = self._grid(sources)
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
            sampler=None if self.sampler is None else self._sampled,
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
        return norm_candidates(universe.candidates(wide(sources, self._grid(sources)), sources))

    def _grid(self, sources: Sources) -> Grid:
        """The decision times ``wide`` is built on: the contract's clock, or the grid the idea's sampler gives on
        ``sources``."""
        return self.clock if self.sampler is None else self._sampled(sources)

    def _sampled(self, sources: Sources) -> pl.DataFrame:
        """The grid ``t | inst`` the idea's sampler gives on ``sources``, within the explore span; a sampler that fails
        or returns something else is refused, saying what it must return."""
        assert self.sampler is not None
        given: dict[str, Any] = {"sources": dict(sources)}
        if len(inspect.signature(self.sampler).parameters) == 2:
            given["wide"] = wide(sources, self.clock)
        try:
            out = self.sampler(*given.values())
            out = out.collect() if isinstance(out, pl.LazyFrame) else out
            out = pl.from_pandas(out) if isinstance(out, pd.DataFrame) else out
            if not isinstance(out, pl.DataFrame):
                raise TypeError(f"it returned {type(out).__name__}")
            grid = out.select(pl.col("t").dt.convert_time_zone("UTC").cast(TIME), pl.col("inst").cast(pl.String))
        except Exception as exc:
            raise DataError(f"{EVALS_FILE}: grid failed: {type(exc).__name__}: {exc}; {GRID_FORMAT}{NOTHING}") from exc
        within = pl.col("t").dt.epoch("ns").is_between(self.start, self.explore_end, closed="right")
        return grid.unique().filter(within).sort("t", "inst")

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
        return {"status": "ready", "notes": ["approving freezes the products; feat reads only those", *admits]}

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


def _outcomes(outcome: Any, sources: Sources, cands: pl.DataFrame, horizons: tuple[str, ...]) -> pl.DataFrame:
    """The outcome of every candidate at every horizon, stacked, with the horizon in ``h``."""
    found = outcome_values(outcome, sources, cands, [span(h) for h in horizons])
    return pl.concat(
        [out.with_columns(h=pl.lit(h)) for h, out in zip(horizons, found, strict=True)], how="diagonal_relaxed"
    )


def _sampler(root: Path) -> Callable[..., Any] | None:
    """The idea's sampler, ``grid`` in ``data.py``, if it defines one."""
    path = root / EVALS_FILE
    if not path.is_file():
        return None
    try:
        found = getattr(ev.load_module(path, root), "grid", None)
    except ev.EvalsError as exc:
        raise DataError(f"{exc}; {GRID_FORMAT}{NOTHING}") from exc
    if found is None:
        return None
    asks = tuple(inspect.signature(found).parameters) if callable(found) else ()
    if asks not in (("sources",), ("sources", "wide")):
        raise DataError(f"{EVALS_FILE}: grid takes {', '.join(asks) or 'nothing'}; {GRID_FORMAT}{NOTHING}")
    return found


def _shape(grid: pl.DataFrame) -> dict[str, Any]:
    """How the decision times fall: how many, how many per inst (min, median, max), the longest gap between an inst's
    decision times, and whether every gap is the same."""
    per = grid.group_by("inst").len()["len"]
    gaps = grid.sort("inst", "t").select(pl.col("t").diff().over("inst").dt.total_nanoseconds()).drop_nulls()["t"]
    longest = int(gaps.max()) if gaps.len() else 0  # type: ignore[arg-type]
    return {
        "times": grid["t"].n_unique(),
        "rows": grid.height,
        "insts": per.len(),
        "per_inst": [int(per.min() or 0), median(per), int(per.max() or 0)],  # type: ignore[arg-type]
        "longest_gap": f"{longest / HOUR_NS:g}h",
        "even": gaps.n_unique() <= 1,
    }


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
    for kind in PRODUCTS:
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
