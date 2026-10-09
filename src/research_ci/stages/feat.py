"""The feat layer: features judged on the frozen data by criteria and null worlds fixed before any is seen.

Registering fixes the method: ``CRITERIA`` from ``feat.py``, ``NULLS`` from ``nulls/feat.py``, and the engine's floor
(chance and timing: scores shuffled across clusters at each decision time, the same way within a horizon). Every
alternative in the contract must be kept by a world that judges a required criterion, and every criterion is
calibrated in every world that judges it. Each submission names its own file of features: the engine checks that its
scores read nothing ahead of their time, judges every criterion against every world that applies, and measures the
contract's decisive effect.
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

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import Branch, Contract, ContractError, Effect, Use, as_use, span
from research_ci.api.nulls import PermuteWithin, nulls_of
from research_ci.api.plugins import ConsumerPlugin, ConsumerResult, CostModel, LabelPlugin, TargetPlugin
from research_ci.api.study import TABLE_KEYS, Criterion, criteria_of
from research_ci.kernel.hashing import hash_frame
from research_ci.kernel.judge import power as pw
from research_ci.kernel.judge import protocol as pr
from research_ci.kernel.judge.calibrate import calibrate
from research_ci.kernel.judge.nulls import Permutation
from research_ci.kernel.judge.verdict import CriterionError, Evaluate, decide, judge, per_t, with_scores
from research_ci.kernel.time.clock import TIME, from_ns, lit_time, span_ns, to_ns
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages.contract import DEFAULT_TARGET, _norm_candidates, _quantile_times, _use_named
from research_ci.store import vault
from research_ci.store.workspace import hash_file

FLOOR = ("chance", "timing")
FEAT_FILE = "feat.py"
NULLS_FILE = Path("nulls") / "feat.py"


class FeatError(ContractError):
    pass


@dataclass(frozen=True)
class FeatSettings:
    """The evaluation tables keep at most ``max_times`` decision times (every k-th when there are more). Calibration
    runs on ``calibration_times`` of them: ``calibration_reps`` stand-ins and planted outputs, each against
    ``calibration_nulls`` shuffles. ``consumer_reps`` runs with the contract's uninformative change measure how much
    the consumer moves by chance. ``cuts`` decision times check that features read nothing ahead."""

    workers: int = field(default_factory=lambda: int(os.environ.get("RCI_WORKERS", _default_workers())))
    max_times: int = 1000
    calibration_times: int = 300
    calibration_reps: int = 100
    calibration_nulls: int = 30
    floor_reps: int = 200
    min_blocks: int = 5
    consumer_reps: int = 20
    cuts: int = 3
    z_base: float = pw.Z_BASE


@dataclass(frozen=True)
class World:
    """A null world as the engine runs it: scores shuffled within ``bucket`` (``None``: each decision time), whole
    ``clusters`` at a time, the same way within ``block``; it claims to keep what lives in ``keeps``."""

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


def method(root: Path, frozen: Frozen, settings: FeatSettings | None = None) -> dict[str, Any]:
    """Register how the layer is judged: what each world keeps, which alternative it answers, and whether every
    criterion can tell real information from what a world keeps."""
    s = settings or FeatSettings()
    c = frozen.contract
    started = time.monotonic()
    checks: list[pr.Check] = []
    criteria = _criteria(Path(root))
    worlds = [_floor(frozen, criteria, s)] + _worlds(Path(root), frozen, criteria, checks)
    rows = [_criterion_row(cr, frozen) for cr in criteria]
    if not any(cr.required for cr in criteria):
        checks.append(pr.Check("required criteria", "fail", "no criterion is required; nothing would decide"))
    checks.append(_coverage(c, worlds, criteria))
    calibration = _calibrate_all(frozen, criteria, worlds, s)
    bad = [f"{r['criterion']} in {r['world']}" for r in calibration if not r.get("ok")]
    detail = f"{len(calibration)} pairs hold" if not bad else f"returned: {', '.join(bad)}"
    checks.append(pr.Check("calibration", "fail" if bad else "pass", detail))
    decisive = [_decisive_method(e, frozen, checks) for e in c.decisive]
    tables = {
        label: {h: {"times": t["t"].n_unique(), "rows": t.height} for h, (t, _) in by_h.items()}
        for label, by_h in frozen.tables.items()
    }
    failed = [ch.name for ch in checks if ch.status == "fail"]
    spec = {
        "criteria": rows,
        "worlds": [asdict(w) for w in worlds],
        "alternatives": [{"id": a.id, "text": a.text} for a in c.alternatives],
        "calibration": calibration,
        "decisive": decisive,
        "tables": tables,
        "stride": frozen.stride,
        "settings": asdict(s) | {"workers": None},
        "checks": [asdict(ch) for ch in checks],
        "verdict": {
            "status": "blocked" if failed else "ready",
            "notes": [f"failed: {', '.join(failed)}"] if failed else ["approving freezes the criteria and the worlds"],
        },
        "seconds": round(time.monotonic() - started, 1),
    }
    return spec


def method_deps(root: Path, frozen: Frozen) -> dict[str, str]:
    """What a method rests on: the criteria and nulls files, and the code of every plugin feat runs."""
    nulls = Path(root) / NULLS_FILE
    files = {
        "feat_file": hash_file(Path(root) / FEAT_FILE),
        "nulls_file": hash_file(nulls) if nulls.is_file() else "none",
    }
    return dict(sorted((frozen.deps | files).items()))


def submit(
    root: Path,
    frozen: Frozen,
    method_spec: dict[str, Any],
    path: Path,
    n: int,
    earlier: dict[str, int],
    settings: FeatSettings | None = None,
) -> dict[str, Any]:
    """Judge one file of features as the ``n``-th submission of the layer. ``earlier`` maps the scores of earlier
    submissions (by hash) to their number; the same scores again are refused, since they are not a new trial."""
    s = settings or FeatSettings()
    started = time.monotonic()
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
    }
    if floors["leak"]:
        out["verdict"] = _verdict("TAINTED", ["the scores read data from after their decision time"])
        return out | {"seconds": round(time.monotonic() - started, 1)}
    if floors["degenerate"]:
        out["verdict"] = _verdict("FAILED", ["the scores do not rank the candidates"])
        return out | {"seconds": round(time.monotonic() - started, 1)}
    criteria = {cr.name: cr for cr in _criteria(Path(root))}
    worlds = [World(**w) for w in method_spec["worlds"]]
    with ThreadPoolExecutor(max_workers=s.workers) as pool:
        decisive_jobs = [_decisive_submit(e, frozen, scores, out["need"], s, pool) for e in frozen.contract.decisive]
        results = _judge_all(frozen, scores, criteria, worlds, s, out["need"])
        out["decisive"] = [job() for job in decisive_jobs]
    out["criteria"] = results
    out["verdict"] = _combine(results, out["decisive"], criteria)
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
    return _layer_file(root / FEAT_FILE, criteria_of, required=True)


def _criterion_row(cr: Criterion, frozen: Frozen) -> dict[str, Any]:
    label = _label_of(cr, frozen)
    return {
        "name": cr.name,
        "label": label,
        "horizons": list(_horizons(cr, frozen)),
        "combine": cr.combine,
        "required": cr.required,
        "note": cr.note,
    }


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


def _floor(frozen: Frozen, criteria: list[Criterion], s: FeatSettings) -> World:
    assert frozen.contract.universe is not None
    return World(
        name="floor",
        explains=FLOOR,
        bucket=None,
        clusters=frozen.contract.universe.clusters,
        block="horizon",
        keeps=None,
        criteria=tuple(cr.name for cr in criteria),
        reps=s.floor_reps,
        by="engine",
        note="scores shuffled across clusters at each decision time, the same way within a horizon",
    )


def _worlds(root: Path, frozen: Frozen, criteria: list[Criterion], checks: list[pr.Check]) -> list[World]:
    nulls = _layer_file(root / NULLS_FILE, nulls_of, required=False)
    ids = {a.id for a in frozen.contract.alternatives} | set(FLOOR)
    names = {cr.name for cr in criteria}
    columns = set(next(iter(next(iter(frozen.tables.values())).values()))[0].columns)
    out, problems = [], []
    for i, n in enumerate(nulls, start=1):
        if not isinstance(n.world, PermuteWithin):
            problems.append(f"null {i}: {type(n.world).__name__} worlds are not run at feat yet")
            continue
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
            problems.append(f"null {i}: no criterion named {', '.join(unknown)}")
        out.append(
            World(
                name=f"null {i}",
                explains=(n.explains,),
                bucket=bucket,
                clusters=w.clusters,
                block=w.block,
                keeps=keeps,
                criteria=tuple(n.criteria) or tuple(sorted(names)),
                reps=n.reps,
                by="idea",
                note=n.note,
            )
        )
    status: pr.Status = "fail" if problems else "pass"
    checks.append(pr.Check("null worlds", status, "; ".join(problems) or f"{len(out)} declared, all runnable"))
    return [w for w in out if not any(p.startswith(f"{w.name}:") for p in problems)]


def _coverage(c: Contract, worlds: list[World], criteria: list[Criterion]) -> pr.Check:
    required = {cr.name for cr in criteria if cr.required}
    lines, missing = [], []
    for alt in [*FLOOR, *(a.id for a in c.alternatives)]:
        kept_by = [w.name for w in worlds if alt in w.explains and required & set(w.criteria)]
        lines.append(f"{alt}: {', '.join(kept_by) or 'none'}")
        if not kept_by:
            missing.append(alt)
    if missing:
        detail = f"no world that judges a required criterion keeps {', '.join(missing)}"
        return pr.Check("alternatives covered", "fail", detail)
    return pr.Check("alternatives covered", "pass", " · ".join(lines))


def _calibrate_all(
    frozen: Frozen, criteria: list[Criterion], worlds: list[World], s: FeatSettings
) -> list[dict[str, Any]]:
    jobs: list[tuple[Criterion, World, str, Future[dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=s.workers) as pool:
        for cr in criteria:
            label, horizons = _label_of(cr, frozen), _horizons(cr, frozen)
            h = min(horizons, key=lambda x: span(x))
            table, h_ns = frozen.tables[label][h]
            table = _thin(table, s.calibration_times)
            for w in worlds:
                if cr.name not in w.criteria:
                    continue
                shuffle = Permutation(table, w.bucket, w.clusters, w.block_ns(h_ns)).index
                args = (cr.evaluate, table, h_ns, w.keeps, shuffle, s.calibration_reps, s.calibration_nulls, s.z_base)
                jobs.append((cr, w, h, pool.submit(_safe_calibrate, *args)))
        return [{"criterion": cr.name, "world": w.name, "horizon": h, **job.result()} for cr, w, h, job in jobs]


def _safe_calibrate(*args: Any) -> dict[str, Any]:
    try:
        return calibrate(*args)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _thin(table: pl.DataFrame, at_most: int) -> pl.DataFrame:
    times = table["t"].unique().sort()
    step = max(1, math.ceil(times.len() / at_most))
    return table.filter(pl.col("t").is_in(times.gather_every(step).implode()))


def _decisive_method(effect: Effect, frozen: Frozen, checks: list[pr.Check]) -> dict[str, Any]:
    out: dict[str, Any] = {"on": effect.on, "at_least": effect.at_least, "unit": effect.unit}
    if effect.on != "consumer":
        evaluate, use = _target(effect, frozen)
        out |= {"kind": "label", "target": use.name, "horizons": list(frozen.tables[effect.on])}
        h = next(iter(frozen.tables[effect.on]))
        table, _ = frozen.tables[effect.on][h]
        rng = np.random.default_rng(0)
        a = per_t(evaluate[h], with_scores(table, rng.standard_normal(table.height)))
        b = per_t(evaluate[h], with_scores(table, rng.standard_normal(table.height)))
        if a.height == b.height and np.allclose(a["value"].to_numpy(), b["value"].to_numpy()):
            detail = f"{use.name} gives the same number for any scores; the decisive effect cannot be measured at feat"
            checks.append(pr.Check("target reads the score", "fail", detail))
        else:
            checks.append(pr.Check("target reads the score", "pass", f"{use.name} moves with the scores"))
        return out
    assert effect.pilot is not None
    reference = "the baseline" if _baseline_scores(frozen) is not None else "random changes to the submitted scores"
    out |= {"kind": "consumer", "pilot": asdict(effect.pilot), "reference": reference}
    if isinstance(effect.design, Branch):
        out["design"] = {"kind": "branch", **asdict(effect.design)}
        if _baseline_scores(frozen) is None:
            checks.append(pr.Check("decisive design", "fail", "a branch design needs the baseline's scores"))
    else:
        out["design"] = {"kind": "full_path"}
    return out


def _target(effect: Effect, frozen: Frozen) -> tuple[dict[str, Evaluate], Use]:
    c = frozen.contract
    assert c.universe is not None
    use = as_use(effect.target) if effect.target else Use(DEFAULT_TARGET, {"cluster": c.universe.clusters})
    target: TargetPlugin = load(use, frozen.registry)
    costs: CostModel = load(as_use(c.costs), frozen.registry)
    parts = frozen.components.get(effect.on, [])
    back = {"y": "value"} | {f"y_{p}": p for p in parts}

    def make(h: str) -> Evaluate:
        horizon = span(h)
        return lambda table: target.reduce(table.rename(back), horizon, costs)

    return {h: make(h) for h in frozen.tables[effect.on]}, use


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
    criteria: dict[str, Criterion],
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
    jobs: list[tuple[str, World, Future[dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=s.workers) as pool:
        for name, cr in criteria.items():
            label = _label_of(cr, frozen)
            tables = {h: scored[label][h] for h in _horizons(cr, frozen)}
            for w in worlds:
                if name not in w.criteria:
                    continue

                def index(h: str, w: World = w, tables: dict[str, tuple[pl.DataFrame, int]] = tables) -> Any:
                    table, h_ns = tables[h]
                    return Permutation(table, w.bucket, w.clusters, w.block_ns(h_ns)).index

                args = (cr.evaluate, tables, index, w.reps, s.min_blocks, cr.combine)
                jobs.append((name, w, pool.submit(_safe_judge, *args)))
        out = []
        for name, w, job in jobs:
            result = job.result()
            result["passed"] = math.isfinite(result.get("z", math.nan)) and result["z"] >= need
            out.append({"criterion": name, "world": w.name, "explains": list(w.explains), **result})
        return out


def _safe_judge(*args: Any) -> dict[str, Any]:
    try:
        return judge(*args)
    except CriterionError as exc:
        raise FeatError(str(exc)) from exc


def _decisive_submit(
    effect: Effect, frozen: Frozen, scores: pl.DataFrame, need: float, s: FeatSettings, pool: ThreadPoolExecutor
) -> Callable[[], dict[str, Any]]:
    head = {"on": effect.on, "at_least": effect.at_least, "unit": effect.unit}
    if effect.on != "consumer":
        evaluate, use = _target(effect, frozen)
        tables = {
            h: (table.join(scores, on=["t", "entity"], how="left").sort("t", "entity"), h_ns)
            for h, (table, h_ns) in frozen.tables[effect.on].items()
        }
        floor = _floor(frozen, [], s)

        def index(h: str) -> Any:
            table, h_ns = tables[h]
            return Permutation(table, floor.bucket, floor.clusters, floor.block_ns(h_ns)).index

        job = pool.submit(judge, evaluate, tables, index, s.floor_reps, s.min_blocks, "max")

        def collect_label() -> dict[str, Any]:
            """Each counted horizon is decided with the family's z; the effect is the target with these scores."""
            result = job.result()
            rows = [r for r in result["horizons"] if r["counted"]]
            for r in rows:
                r["outcome"] = decide(r["real"], r["se"], result["z"], effect.at_least, need, s.z_base)
            outcomes = {r["outcome"] for r in rows}
            outcome = "passed" if "passed" in outcomes else "failed" if outcomes == {"failed"} else "insufficient"
            best = max(rows, key=lambda r: r["real"] - s.z_base * r["se"], default=None)
            value, se = (best["real"], best["se"]) if best else (math.nan, math.nan)
            found = {"kind": "label", "target": use.name, "effect": value, "se": se, "z": result["z"]}
            return (
                head | found | {"best": best["h"] if best else None, "horizons": result["horizons"], "outcome": outcome}
            )

        return collect_label
    consumer: ConsumerPlugin = load(as_use(frozen.contract.consumer), frozen.registry)  # type: ignore[arg-type]
    assert effect.pilot is not None
    mine = scores.filter(pl.col("score").is_finite())
    base_scores = _baseline_scores(frozen)
    if isinstance(effect.design, Branch):
        assert base_scores is not None
        return _branch(effect, frozen, consumer, mine, base_scores, need, s, pool, head)
    fields, run = frozen.fields, consumer.run
    reference = base_scores if base_scores is not None else mine
    block = span_ns(span(effect.pilot.block))
    real = pool.submit(run, fields, mine)
    base = pool.submit(run, fields, None) if base_scores is not None else None
    changed = [
        pool.submit(run, fields, pr.perturb(reference, effect.pilot.kind, effect.pilot.share, block, seed))
        for seed in range(1, s.consumer_reps + 1)
    ]

    def collect_consumer() -> dict[str, Any]:
        mine_total = pr.total(real.result())
        null = np.array([pr.total(r.result()) for r in changed])
        mu, sd = float(null.mean()), float(null.std(ddof=1))
        years = (frozen.end - frozen.start) / pw.DAY_NS / pw.YEAR_DAYS
        if base is not None:
            gain, se = mine_total - pr.total(base.result()), sd
            over = "the baseline"
        else:
            gain, se = mine_total - mu, sd * math.sqrt(1 + 1 / len(null))
            over = "the mean of runs with the change applied to these scores"
        z = (mine_total - mu) / sd if sd > 0 else math.nan
        effect_y, se_y = gain / years, se / years
        outcome = decide(effect_y, se_y, z, effect.at_least, need, s.z_base)
        return head | {
            "kind": "consumer",
            "design": "full_path",
            "over": over,
            "runs": len(null),
            "effect": effect_y,
            "se": se_y,
            "z": z,
            "null_sd": sd / years,
            "years": years,
            "run": _summary(real.result(), years),
            "outcome": outcome,
        }

    return collect_consumer


def _branch(
    effect: Effect,
    frozen: Frozen,
    consumer: ConsumerPlugin,
    mine: pl.DataFrame,
    base_scores: pl.DataFrame,
    need: float,
    s: FeatSettings,
    pool: ThreadPoolExecutor,
    head: dict[str, Any],
) -> Callable[[], dict[str, Any]]:
    """Branch from the baseline's own path: its scores up to each branch point, the submitted scores (or the
    uninformative change) for ``horizon``, its scores again after; the difference is counted until ``tail`` after."""
    assert isinstance(effect.design, Branch) and effect.pilot is not None
    design, pilot = effect.design, effect.pilot
    every, horizon, tail = (span_ns(span(x)) for x in (design.every, design.horizon, design.tail))
    fields, run = frozen.fields, consumer.run
    base = run(fields, None)
    held = base.positions.filter(pl.col("weight") != 0)
    first = int(held.select(pl.col("t").min().dt.epoch("ns")).item()) if held.height else frozen.start
    plan = []
    for seed, t0 in enumerate(range(first, frozen.end - horizon - tail + 1, every), start=1):
        inside = (pl.col("t") > lit_time(t0)) & (pl.col("t") <= lit_time(t0 + horizon))
        outside = base_scores.filter(~inside)
        real = pl.concat([outside, mine.filter(inside)]).sort("t", "entity")
        changed = pr.perturb(base_scores.filter(inside), pilot.kind, pilot.share, horizon, seed)
        null = pl.concat([outside, changed]).sort("t", "entity")
        plan.append((t0, pool.submit(run, fields, real), pool.submit(run, fields, null)))

    def collect() -> dict[str, Any]:
        mine_d, null_d = [], []
        for t0, a, b in plan:
            mine_d.append(pr.branch_diff(base, a.result(), t0, t0 + horizon + tail)[0])
            null_d.append(pr.branch_diff(base, b.result(), t0, t0 + horizon + tail)[0])
        a_arr, b_arr = np.array(mine_d), np.array(null_d)
        years = horizon / pw.DAY_NS / pw.YEAR_DAYS
        n_eff = len(a_arr) * min(1.0, every / (horizon + tail))
        sd = float(b_arr.std(ddof=1)) if b_arr.size > 1 else math.nan
        se = sd / math.sqrt(n_eff) if n_eff > 0 else math.nan
        z = (float(a_arr.mean()) - float(b_arr.mean())) / se if se > 0 else math.nan
        effect_y, se_y = float(a_arr.mean()) / years, se / years
        return head | {
            "kind": "consumer",
            "design": "branch",
            "over": "the baseline",
            "branches": len(a_arr),
            "effect": effect_y,
            "se": se_y,
            "z": z,
            "null_sd": sd / years,
            "outcome": decide(effect_y, se_y, z, effect.at_least, need, s.z_base),
        }

    return collect


def _baseline_scores(frozen: Frozen) -> pl.DataFrame | None:
    b = frozen.contract.baseline
    cands = frozen.candidates
    if b is None or b.score not in cands.columns:
        return None
    rows = cands.filter(pl.col("openable")) if "openable" in cands.columns else cands
    return rows.select("t", "entity", pl.col(b.score).cast(pl.Float64).alias("score")).drop_nulls("score")


def _summary(result: ConsumerResult, years: float) -> dict[str, Any]:
    total = pr.total(result)
    return {
        "per_year": total / years if years > 0 else math.nan,
        "max_drawdown": pr.max_drawdown(result),
        "turnover": result.turnover,
        "trades": result.trades.height,
    }


def _combine(results: list[dict[str, Any]], decisive: list[dict[str, Any]], criteria: dict[str, Criterion]) -> dict:
    notes = []
    for r in results:
        if criteria[r["criterion"]].required and not r["passed"]:
            explains = ", ".join(r["explains"])
            notes.append(f"{r['criterion']} does not beat {r['world']} ({explains}): z {_num(r.get('z'))}")
    if notes:
        return _verdict("FAILED", notes)
    outcomes = [d["outcome"] for d in decisive]
    lines = [f"{d['on']}: {d['outcome']} (effect {_num(d['effect'])} ± {_num(d['se'])})" for d in decisive]
    if any(o == "failed" for o in outcomes):
        return _verdict("FAILED", lines)
    if all(o == "passed" for o in outcomes):
        return _verdict("PASSED", lines)
    return _verdict("INSUFFICIENT", [*lines, "the data cannot decide yet: more data, or a person decides"])


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


def current_method_deps(contract: Contract, root: Path, recorded: dict[str, str], registry: Registry | None) -> dict:
    """The method's dependencies as they are now, keyed like the recorded ones."""
    now: dict[str, str] = {}
    for key in recorded:
        if key.startswith("plugin:"):
            now[key] = code_hash(load(_use_named(contract, key.removeprefix("plugin:")), registry))
    feat, nulls = Path(root) / FEAT_FILE, Path(root) / NULLS_FILE
    now["feat_file"] = hash_file(feat) if feat.is_file() else "missing"
    now["nulls_file"] = hash_file(nulls) if nulls.is_file() else "none"
    return now
