"""The feat layer: features judged on the frozen data by a rule fixed before any is seen.

The rule is the minimal bar, fixed by the contract: the contract's decisive effect, computed with the submitted
scores, must beat the engine's floor world (chance and timing: scores shuffled across clusters at each decision time,
the same way within a horizon) with ``z ≥ z_need``, which rises with the submission count. Before the first submission
is judged, the decisive effect is calibrated in the floor world. Diagnostics, the engine's defaults and the idea's
``CRITERIA`` (``feat.py``, optional), are judged against the floor and the idea's ``NULLS`` (``nulls/feat.py``) on
every submission and shown, never deciding. Each submission names its own file of features: the engine checks that
its scores rank the candidates and read nothing ahead of their time, then judges.
"""

import importlib.util
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from research_ci.api.contract import FLOOR_ALTERNATIVES, Contract, ContractError, Effect, Use, as_use, span
from research_ci.api.nulls import nulls_of
from research_ci.api.plugins import CostModel, LabelPlugin, TargetPlugin
from research_ci.api.study import TABLE_KEYS, Criterion, criteria_of
from research_ci.eval import protocol as pr
from research_ci.eval.calibration.stand_in import calibrate
from research_ci.kernel.hashing import hash_frame, hash_obj
from research_ci.kernel.time.clock import TIME, from_ns, lit_time, span_ns, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.judge import CriterionError, Evaluate, judge, per_t
from research_ci.kernel.verdict.permute import Permutation
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages.contract import DEFAULT_TARGET, _effect, _norm_candidates, _quantile_times, _use_named
from research_ci.stages.defaults.feat import defaults
from research_ci.store import vault
from research_ci.store.workspace import hash_file

FEAT_FILE = "feat.py"
NULLS_FILE = Path("nulls") / "feat.py"


class FeatError(ContractError):
    pass


@dataclass(frozen=True)
class FeatSettings:
    """The evaluation tables keep at most ``max_times`` decision times (every k-th when there are more). Calibrating
    the decisive effect runs on ``calibration_times`` of them: ``calibration_reps`` stand-ins and planted outputs, each
    against ``calibration_nulls`` shuffles. ``cuts`` decision times check that features read nothing ahead."""

    workers: int = field(default_factory=lambda: int(os.environ.get("RCI_WORKERS", _default_workers())))
    max_times: int = 1000
    calibration_times: int = 300
    calibration_reps: int = 100
    calibration_nulls: int = 30
    floor_reps: int = 200
    min_blocks: int = 5
    cuts: int = 3
    z_base: float = pw.Z_BASE


@dataclass(frozen=True)
class World:
    """A null world as the engine runs it: scores shuffled within ``bucket`` (``None``: each decision time), whole
    ``clusters`` at a time, the same way within ``block``; it claims to keep what lives in ``keeps``. It judges the
    diagnostics named in ``criteria``, or every diagnostic when empty."""

    name: str
    explains: tuple[str, ...]
    bucket: str | None
    clusters: str | None
    block: str | None
    keeps: str | None
    criteria: tuple[str, ...]
    reps: int
    by: str
    note: str = ""

    def block_ns(self, h_ns: int) -> int:
        if self.block is None:
            return 1
        return h_ns if self.block == "horizon" else span_ns(span(self.block))

    def describe(self) -> str:
        return f"PermuteWithin({self.bucket or 'time'}, clusters={self.clusters}, block={self.block})"


@dataclass
class Frozen:
    """What feat reads: the treated data and the candidates the data layer froze, and one evaluation table per label
    and horizon (``t``, ``entity``, ``y`` and its components, the candidate columns) on every ``stride``-th decision
    time whose label is complete before the explore data ends."""

    contract: Contract
    fields: pl.DataFrame
    candidates: pl.DataFrame
    tables: dict[str, dict[str, tuple[pl.DataFrame, int]]]
    components: dict[str, list[str]]
    start: int
    end: int
    stride: int
    deps: dict[str, str]
    registry: Registry | None


def frozen(
    contract: Contract,
    panel: str,
    exclusions: str,
    registry: Registry | None = None,
    settings: "FeatSettings | None" = None,
) -> Frozen:
    s = settings or FeatSettings()
    if contract.universe is None:
        raise FeatError("feat judges scores per candidate; the contract names no universe")
    name = as_use(contract.data).name
    fields = vault.get(f"{name}.panel", panel)
    excluded = vault.get(f"{name}.exclusions", exclusions)
    deps: dict[str, str] = {}

    def plugin(use: Use) -> Any:
        found = load(use, registry)
        deps[f"plugin:{use.name}"] = code_hash(found)
        return found

    start = int(fields.select(pl.col("available_at").min().dt.epoch("ns")).item())
    latest = int(fields.select(pl.col("available_at").max().dt.epoch("ns")).item())
    end = latest if contract.explore_end == "latest" else to_ns(pd.Timestamp(contract.explore_end))
    if contract.consumer is not None:
        timing = plugin(as_use(contract.consumer)).timing()
        every, offset = timing.every, timing.offset
    else:
        assert contract.execution is not None
        every, offset = contract.execution.every, contract.execution.offset
    times = pw.decision_grid(span_ns(span(every)), span_ns(span(offset)), start, end)
    universe = plugin(as_use(contract.universe.use))
    for effect in contract.decisive:
        plugin(_target_use(effect, contract))
    plugin(as_use(contract.costs))
    cands = _norm_candidates(universe.candidates(fields, [from_ns(t) for t in times]))
    cands = cands.join(excluded.select("t", "entity"), on=["t", "entity"], how="anti").sort("t", "entity")
    present = sorted(int(t) for t in cands.select(pl.col("t").dt.epoch("ns").unique())["t"])
    stride = max(1, math.ceil(len(present) / s.max_times))
    kept = pl.Series("t", present[::stride], dtype=pl.Int64)
    sampled = cands.filter(pl.col("t").dt.epoch("ns").is_in(kept.implode()))
    extra = [c for c in cands.columns if c not in ("t", "entity")]
    renamed = {c: f"{c}_candidate" for c in extra if c in TABLE_KEYS or c.startswith("y_")}
    columns = sampled.rename(renamed)
    tables: dict[str, dict[str, tuple[pl.DataFrame, int]]] = {}
    components: dict[str, list[str]] = {}
    for label in contract.labels:
        values: LabelPlugin = plugin(as_use(label.use))
        tables[label.name] = {}
        for h in label.horizons:
            h_ns = span_ns(span(h))
            out = _frame(values.values(fields, sampled, span(h)))
            parts = [c for c in out.columns if c not in ("t", "entity", "value")]
            components[label.name] = parts
            out = out.filter(pl.col("t") <= lit_time(end - h_ns))
            y = out.select(
                pl.col("t").cast(TIME),
                pl.col("entity").cast(pl.String),
                pl.col("value").cast(pl.Float64).alias("y"),
                *[pl.col(c).cast(pl.Float64).alias(f"y_{c}") for c in parts],
            )
            table = y.join(columns, on=["t", "entity"], how="inner").sort("t", "entity")
            tables[label.name][h] = (table, h_ns)
    return Frozen(contract, fields, cands, tables, components, start, end, stride, deps, registry)


def calibration(frozen: Frozen, settings: FeatSettings | None = None) -> dict[str, Any]:
    """The decisive effect in the floor world before anything is judged: its target must read the scores, a stand-in
    that carries only each decision time's level must die as often as chance allows, and a planted signal must live.
    ``ok`` is false, with ``notes`` saying why, when any of them does not hold."""
    s = settings or FeatSettings()
    started = time.monotonic()
    checks = [_reads_score(e, frozen) for e in frozen.contract.decisive]
    units = {e.on: _target(e, frozen)[2] for e in frozen.contract.decisive}
    rows = [r | units[r["effect"]] for r in _calibrate(frozen, s)]
    notes = [f"{c.name}: {c.detail}" for c in checks if c.status == "fail"]
    notes += [_returned(r) for r in rows if not r.get("ok")]
    return {
        "rows": rows,
        "checks": [asdict(c) for c in checks],
        "ok": not notes,
        "notes": notes,
        "seconds": round(time.monotonic() - started, 1),
    }


def calibration_key(frozen: Frozen, settings: FeatSettings | None = None) -> dict[str, str]:
    """What a calibration rests on beyond the stage's dependencies: the code of every plugin feat runs and the
    calibration settings. A calibration is reused while these are unchanged."""
    s = settings or FeatSettings()
    return frozen.deps | {"settings": hash_obj(asdict(s) | {"workers": None})}


def files(root: Path) -> dict[str, str]:
    """The idea's criteria and nulls files, ``none`` when absent."""
    feat, nulls = Path(root) / FEAT_FILE, Path(root) / NULLS_FILE
    return {
        "feat_file": hash_file(feat) if feat.is_file() else "none",
        "nulls_file": hash_file(nulls) if nulls.is_file() else "none",
    }


def submit(
    root: Path,
    frozen: Frozen,
    path: Path,
    n: int,
    earlier: dict[str, int],
    settings: FeatSettings | None = None,
) -> dict[str, Any]:
    """Judge one file of features as the ``n``-th submission of the layer. ``earlier`` maps the scores of earlier
    submissions (by hash) to their number; the same scores again are refused, since they are not a new trial, and so
    is a null world that cannot run."""
    s = settings or FeatSettings()
    started = time.monotonic()
    criteria = _criteria(Path(root))
    diagnostics = _diagnostics(criteria, frozen.contract)
    worlds = [_floor(frozen, s)] + _worlds(Path(root), frozen, set(diagnostics))
    features = _features_of(path)
    scores = _scores(features(frozen.fields, frozen.candidates), frozen.candidates)
    digest = hash_frame(scores)
    if digest in earlier:
        raise FeatError(f"the same scores as submission {earlier[digest]}; they would not be a new trial")
    floors = _floors(features, frozen, scores, s)
    out: dict[str, Any] = {
        "n": n,
        "features": {"path": str(Path(path).resolve()), "hash": hash_file(path)},
        "scores": {"hash": digest, **floors["coverage"]},
        "floors": [asdict(ch) for ch in floors["checks"]],
        "need": pw.z_need(n, s.z_base),
        "stride": frozen.stride,
        "worlds": [asdict(w) for w in worlds],
        "coverage": asdict(_coverage(frozen.contract, worlds)),
    }
    if floors["leak"]:
        out["verdict"] = _verdict("TAINTED", ["the scores read data from after their decision time"])
        return out | {"seconds": round(time.monotonic() - started, 1)}
    if floors["degenerate"]:
        out["verdict"] = _verdict("FAILED", ["the scores do not rank the candidates"])
        return out | {"seconds": round(time.monotonic() - started, 1)}
    with ThreadPoolExecutor(max_workers=s.workers) as pool:
        decisive_jobs = [_decisive_submit(e, frozen, scores, out["need"], s, pool) for e in frozen.contract.decisive]
        results = _judge_all(frozen, scores, diagnostics, worlds, s, out["need"])
        out["decisive"] = [job() for job in decisive_jobs]
    out["criteria"] = results
    out["verdict"] = _combine(out["decisive"], out["need"])
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


def explore(frozen: Frozen, path: Path, settings: FeatSettings | None = None) -> dict[str, Any]:
    """The checks a submission must pass before it is judged, without judging it: nothing about how it would do."""
    s = settings or FeatSettings()
    features = _features_of(path)
    scores = _scores(features(frozen.fields, frozen.candidates), frozen.candidates)
    floors = _floors(features, frozen, scores, s)
    return {
        "features": {"path": str(Path(path).resolve()), "hash": hash_file(path)},
        "scores": {"hash": hash_frame(scores), **floors["coverage"]},
        "floors": [asdict(ch) for ch in floors["checks"]],
    }


def _criteria(root: Path) -> list[Criterion]:
    return _layer_file(root / FEAT_FILE, criteria_of, required=False)


def _diagnostics(criteria: list[Criterion], c: Contract) -> dict[str, tuple[str, Criterion]]:
    """Every diagnostic by name, and whose it is: the engine's defaults, then the idea's criteria, which replace a
    default of the same name."""
    return {cr.name: ("engine", cr) for cr in defaults(c)} | {cr.name: ("idea", cr) for cr in criteria}


def _label_of(cr: Criterion, frozen: Frozen) -> str:
    names = [label.name for label in frozen.contract.labels]
    if not names:
        raise FeatError("the contract declares no labels; criteria need one")
    label = cr.label or names[0]
    if label not in names:
        raise FeatError(f"criterion {cr.name}: no label {label!r} in the contract ({', '.join(names)})")
    return label


def _horizons(cr: Criterion, frozen: Frozen) -> tuple[str, ...]:
    label = _label_of(cr, frozen)
    declared = next(lab.horizons for lab in frozen.contract.labels if lab.name == label)
    if cr.horizons is None:
        return declared
    unknown = [h for h in cr.horizons if h not in declared]
    if unknown:
        raise FeatError(f"criterion {cr.name}: horizons {', '.join(unknown)} are not among {', '.join(declared)}")
    return cr.horizons


def _floor(frozen: Frozen, s: FeatSettings) -> World:
    assert frozen.contract.universe is not None
    return World(
        name="floor",
        explains=FLOOR_ALTERNATIVES,
        bucket=None,
        clusters=frozen.contract.universe.clusters,
        block="horizon",
        keeps=None,
        criteria=(),
        reps=s.floor_reps,
        by="engine",
        note="scores shuffled across clusters at each decision time, the same way within a horizon",
    )


def _worlds(root: Path, frozen: Frozen, names: set[str]) -> list[World]:
    nulls = _layer_file(root / NULLS_FILE, nulls_of, required=False)
    ids = {a.id for a in frozen.contract.alternatives} | set(FLOOR_ALTERNATIVES)
    columns = set(next(iter(next(iter(frozen.tables.values())).values()))[0].columns)
    out, problems = [], []
    for i, n in enumerate(nulls, start=1):
        if n.explains not in ids:
            problems.append(
                f"null {i}: {n.explains!r} is not an alternative of the contract ({', '.join(sorted(ids))})"
            )
        w = n.world
        bucket = None if w.bucket_by == "time" else w.bucket_by
        keeps = None if n.keeps in (None, "time") else n.keeps
        missing = [col for col in (bucket, w.clusters, keeps) if col is not None and col not in columns]
        if missing:
            problems.append(f"null {i}: the evaluation table has no column {', '.join(missing)}")
        unknown = [name for name in n.criteria if name not in names]
        if unknown:
            problems.append(f"null {i}: no diagnostic named {', '.join(unknown)}")
        out.append(
            World(
                name=f"null {i}",
                explains=(n.explains,),
                bucket=bucket,
                clusters=w.clusters,
                block=w.block,
                keeps=keeps,
                criteria=tuple(n.criteria),
                reps=n.reps,
                by="idea",
                note=n.note,
            )
        )
    if problems:
        raise FeatError(f"{NULLS_FILE}: {'; '.join(problems)}")
    return out


def _coverage(c: Contract, worlds: list[World]) -> pr.Check:
    """Which world answers which alternative. One that no world answers is a warning, not a failure: only the
    decisive effect decides, and the worlds of the idea judge only diagnostics."""
    lines, missing = [], []
    for alt in [*FLOOR_ALTERNATIVES, *(a.id for a in c.alternatives)]:
        kept_by = [w.name for w in worlds if alt in w.explains]
        lines.append(f"{alt}: {', '.join(kept_by) or 'none'}")
        missing += [] if kept_by else [alt]
    if missing:
        return pr.Check("alternatives covered", "warn", f"{' · '.join(lines)}; no world answers {', '.join(missing)}")
    return pr.Check("alternatives covered", "pass", " · ".join(lines))


def _calibrate(frozen: Frozen, s: FeatSettings) -> list[dict[str, Any]]:
    """Each decisive effect in the floor world, on its shortest horizon: a stand-in that carries only each decision
    time's level must die as often as chance allows, and a planted signal must live."""
    floor = _floor(frozen, s)
    out = []
    for effect in frozen.contract.decisive:
        evaluate, _, _ = _target(effect, frozen)
        h = min(evaluate, key=span)
        table, h_ns = frozen.tables[effect.on][h]
        table = _thin(table, s.calibration_times)
        shuffle = Permutation(table, floor.bucket, floor.clusters, floor.block_ns(h_ns)).index
        args = (evaluate[h], table, h_ns, floor.keeps, shuffle, s.calibration_reps, s.calibration_nulls, s.z_base)
        out.append({"effect": effect.on, "world": floor.name, "horizon": h, **_safe_calibrate(*args)})
    return out


def _safe_calibrate(*args: Any) -> dict[str, Any]:
    try:
        return calibrate(*args)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _thin(table: pl.DataFrame, at_most: int) -> pl.DataFrame:
    times = table["t"].unique().sort()
    step = max(1, math.ceil(times.len() / at_most))
    return table.filter(pl.col("t").is_in(times.gather_every(step).implode()))


def _returned(r: dict[str, Any]) -> str:
    if "error" in r:
        return f"{r['effect']} in the {r['world']} world: {r['error']}"
    missed = []
    if r["stand_in_dies"] < r["dies_at_least"]:
        missed.append(f"a stand-in died {r['stand_in_dies']:.0%} of the time, needs ≥ {r['dies_at_least']:.0%}")
    if r["planted_lives"] < 0.8:
        missed.append(f"a planted signal lived {r['planted_lives']:.0%} of the time, needs ≥ 80%")
    return f"{r['effect']} in the {r['world']} world at {r['horizon']}: {'; '.join(missed)}"


def _reads_score(effect: Effect, frozen: Frozen) -> pr.Check:
    """Derive already blocks a target that ignores the scores; feat checks again on its own evaluation table."""
    evaluate, use, _ = _target(effect, frozen)
    h = next(iter(frozen.tables[effect.on]))
    table, _ = frozen.tables[effect.on][h]
    return pr.target_reads_score(use.name, lambda t: per_t(evaluate[h], t), table)


def _target_use(effect: Effect, c: Contract) -> Use:
    assert c.universe is not None
    return as_use(effect.target) if effect.target else Use(DEFAULT_TARGET, {"cluster": c.universe.clusters})


def _target(effect: Effect, frozen: Frozen) -> tuple[dict[str, Evaluate], Use, dict[str, Any]]:
    """The effect's target per horizon of its label, the plugin it names, and the effect as cards show it."""
    c = frozen.contract
    use = _target_use(effect, c)
    target: TargetPlugin = load(use, frozen.registry)
    costs: CostModel = load(as_use(c.costs), frozen.registry)
    parts = frozen.components.get(effect.on, [])
    back = {"y": "value"} | {f"y_{p}": p for p in parts}

    def make(h: str) -> Evaluate:
        horizon = span(h)
        return lambda table: target.reduce(table.rename(back), horizon, costs)

    return {h: make(h) for h in frozen.tables[effect.on]}, use, _effect(effect, target)


def _features_of(path: Path) -> Callable[[pl.DataFrame, pl.DataFrame], Any]:
    found = _layer_file(Path(path), lambda m: getattr(m, "features", None), required=True)
    if not callable(found):
        raise FeatError(f"{path} must define features(fields, candidates)")
    return found


def _scores(raw: Any, cands: pl.DataFrame) -> pl.DataFrame:
    frame = pl.from_pandas(raw) if isinstance(raw, pd.DataFrame) else raw
    if not isinstance(frame, pl.DataFrame) or not {"t", "entity", "score"} <= set(frame.columns):
        raise FeatError("features must return a DataFrame with columns t, entity, score")
    out = frame.select(
        pl.col("t").dt.convert_time_zone("UTC").cast(TIME),
        pl.col("entity").cast(pl.String),
        pl.col("score").cast(pl.Float64),
    )
    if out.select(pl.struct("t", "entity").is_duplicated().any()).item():
        raise FeatError("features return more than one score for some candidate at some time")
    return out.join(cands.select("t", "entity"), on=["t", "entity"], how="semi").sort("t", "entity")


def _floors(
    features: Callable[[pl.DataFrame, pl.DataFrame], Any], frozen: Frozen, scores: pl.DataFrame, s: FeatSettings
) -> dict[str, Any]:
    checks: list[pr.Check] = []
    cands = frozen.candidates
    finite = scores.filter(pl.col("score").is_finite())
    distinct = finite.group_by("t").agg(pl.col("score").n_unique().alias("k"))
    times = cands["t"].n_unique()
    ranks = distinct.filter(pl.col("k") >= 2).height / times if times else 0.0
    coverage = {"rows": finite.height, "share": finite.height / cands.height if cands.height else 0.0, "ranks": ranks}
    degenerate = ranks < 0.5
    status: pr.Status = "fail" if degenerate else "pass"
    detail = f"scores {coverage['share']:.0%} of candidate rows; ranks candidates at {ranks:.0%} of decision times"
    checks.append(pr.Check("scores rank candidates", status, detail))
    epoch = cands.select(pl.col("t").dt.epoch("ns").unique())["t"]
    cuts = _quantile_times(epoch, s.cuts)
    with ThreadPoolExecutor(max_workers=s.workers) as pool:
        jobs = [(cut, pool.submit(_cut_scores, features, frozen.fields, cands, cut)) for cut in cuts]
        problems = []
        for cut, job in jobs:
            try:
                cut_scores = job.result()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{_iso(cut)}: features failed on the cut data: {exc}")
                continue
            before = scores.filter(pl.col("t") <= lit_time(cut))
            mismatch = pr._frame_mismatch(before, cut_scores, ["t", "entity"], ["score"])
            if mismatch:
                problems.append(f"{_iso(cut)}: {mismatch}")
    leak = bool(problems)
    detail = "; ".join(problems) if leak else f"the same scores with the data cut at {len(cuts)} decision times"
    checks.append(pr.Check("scores read nothing ahead", "fail" if leak else "pass", detail))
    return {"checks": checks, "coverage": coverage, "leak": leak, "degenerate": degenerate}


def _cut_scores(
    features: Callable[[pl.DataFrame, pl.DataFrame], Any], fields: pl.DataFrame, cands: pl.DataFrame, cut: int
) -> pl.DataFrame:
    known = fields.filter(pl.col("available_at") <= lit_time(cut))
    before = cands.filter(pl.col("t") <= lit_time(cut))
    out = _scores(features(known, before), before)
    return out.filter(pl.col("t") <= lit_time(cut))


def _judge_all(
    frozen: Frozen,
    scores: pl.DataFrame,
    diagnostics: dict[str, tuple[str, Criterion]],
    worlds: list[World],
    s: FeatSettings,
    need: float,
) -> list[dict[str, Any]]:
    scored = {
        label: {
            h: (table.join(scores, on=["t", "entity"], how="left").sort("t", "entity"), h_ns)
            for h, (table, h_ns) in by_h.items()
        }
        for label, by_h in frozen.tables.items()
    }
    jobs: list[tuple[str, str, World, Future[dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=s.workers) as pool:
        for name, (by, cr) in diagnostics.items():
            label = _label_of(cr, frozen)
            tables = {h: scored[label][h] for h in _horizons(cr, frozen)}
            for w in worlds:
                if w.criteria and name not in w.criteria:
                    continue

                def index(h: str, w: World = w, tables: dict[str, tuple[pl.DataFrame, int]] = tables) -> Any:
                    table, h_ns = tables[h]
                    return Permutation(table, w.bucket, w.clusters, w.block_ns(h_ns)).index

                args = (cr.evaluate, tables, index, w.reps, s.min_blocks, cr.combine)
                jobs.append((name, by, w, pool.submit(_safe_judge, *args)))
        out = []
        for name, by, w, job in jobs:
            result = job.result()
            result["passed"] = math.isfinite(result.get("z", math.nan)) and result["z"] >= need
            out.append({"criterion": name, "by": by, "world": w.name, "explains": list(w.explains), **result})
        return out


def _safe_judge(*args: Any) -> dict[str, Any]:
    try:
        return judge(*args)
    except CriterionError as exc:
        raise FeatError(str(exc)) from exc


def _decisive_submit(
    effect: Effect, frozen: Frozen, scores: pl.DataFrame, need: float, s: FeatSettings, pool: ThreadPoolExecutor
) -> Callable[[], dict[str, Any]]:
    evaluate, use, head = _target(effect, frozen)
    tables = {
        h: (table.join(scores, on=["t", "entity"], how="left").sort("t", "entity"), h_ns)
        for h, (table, h_ns) in frozen.tables[effect.on].items()
    }
    floor = _floor(frozen, s)

    def index(h: str) -> Any:
        table, h_ns = tables[h]
        return Permutation(table, floor.bucket, floor.clusters, floor.block_ns(h_ns)).index

    job = pool.submit(judge, evaluate, tables, index, s.floor_reps, s.min_blocks, "max")

    def collect() -> dict[str, Any]:
        """The family's z against the floor decides; the effect shown is the gain over random, the target with these
        scores less its mean in the floor world, at the best counted horizon."""
        result = job.result()
        rows = [r | {"gain": r["real"] - r["null_mean"]} for r in result["horizons"]]
        best = next((r for r in rows if r["h"] == result["best"]), None)
        z = result["z"]
        return head | {
            "kind": "label",
            "target": use.name,
            "effect": best["gain"] if best else math.nan,
            "se": best["se"] if best else math.nan,
            "z": z,
            "best": result["best"],
            "horizons": rows,
            "outcome": "passed" if math.isfinite(z) and z >= need else "failed",
        }

    return collect


def _combine(decisive: list[dict[str, Any]], need: float) -> dict[str, Any]:
    lines = [
        f"{d['on']}: beats random, z {_num(d['z'])} ≥ {need:.2f}"
        if d["outcome"] == "passed"
        else f"{d['on']}: does not beat random, z {_num(d['z'])} < {need:.2f}"
        for d in decisive
    ]
    return _verdict("PASSED" if all(d["outcome"] == "passed" for d in decisive) else "FAILED", lines)


def _verdict(status: str, notes: list[str]) -> dict[str, Any]:
    return {"status": status, "notes": notes}


def _num(x: Any) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.4g}"


def _frame(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame


def _layer_file(path: Path, read: Callable[[Any], Any], required: bool) -> Any:
    if not path.is_file():
        if required:
            raise FeatError(f"{path} does not exist")
        return []
    spec = importlib.util.spec_from_file_location(f"_rci_feat_{path.parent.name}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise FeatError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return read(module)
    except ValueError as exc:
        raise FeatError(f"{path}: {exc}") from exc


def _iso(ns: int) -> str:
    return from_ns(ns).strftime("%Y-%m-%dT%H:%MZ")


def _default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def current_deps(contract: Contract, root: Path, recorded: dict[str, str], registry: Registry | None) -> dict:
    """A submission's plugin code and the idea's criteria and nulls files as they are now, keyed like the recorded
    ones."""
    now: dict[str, str] = {}
    for key in recorded:
        if key.startswith("plugin:"):
            now[key] = code_hash(load(_use_named(contract, key.removeprefix("plugin:")), registry))
    return now | files(root)
