"""The feat layer: features judged on the frozen data by a rule fixed before any is seen.

Each submission names its own file of features: the engine reads it, and the idea's evals and null worlds, for
row-by-row code, which the card shows as a warning, validates the idea's case evals by their tests, hands ``features``
the frozen data and the candidates as polars DataFrames, and runs its floor list (``stages/floor/feat.py``) with the
default diagnostics (``stages/defaults/feat.py``) where the floor marks them: the screen, the floors on the scores,
then the decisive test, which alone decides, next to the diagnostics and the idea's evals, which are shown. Only the
candidates the features scored, with a finite score, are judged, in the real world and in every null world alike.
What the calls cost, their wall time and the peak memory of the full one, is shown, never judged.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from research_ci.api.contract import FLOOR_ALTERNATIVES, Contract, Use, as_use, span
from research_ci.api.evals import TABLE_KEYS, Eval, evals_of
from research_ci.api.nulls import nulls_of
from research_ci.api.plugins import LabelPlugin
from research_ci.kernel.hashing import hash_frame, hash_obj
from research_ci.kernel.time.clock import TIME, from_ns, lit_time, span_ns, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.judge import judge_all
from research_ci.runner import evals as ev
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages import steps
from research_ci.stages.common import label_values, norm_candidates, target_use
from research_ci.stages.contract import use_named
from research_ci.stages.defaults import feat as defaults
from research_ci.stages.floor import feat as floor
from research_ci.stages.floor.feat import (
    EVALS_FORMAT,
    EXPLORED,
    FEAT_FILE,
    NO_TRIAL,
    NULLS_FILE,
    SHAPE,
    FeatError,
    FeatSettings,
    Frozen,
    World,
)
from research_ci.stages.floor.feat import screen as screen
from research_ci.store import vault
from research_ci.store.workspace import hash_file

JUDGED = ("decisive", "evals")
NULLS_FORMAT = (
    "nulls/feat.py defines NULLS = [Null(alternative, PermuteWithin(column, clusters=..., block='horizon')), ...], "
    "one world for an alternative explanation in the contract; the world shuffles the scores within the buckets of a "
    "column of the evaluation table"
)
EVAL_FILES = "eval:"


def frozen(
    contract: Contract,
    panel: str,
    exclusions: str,
    registry: Registry | None = None,
    settings: FeatSettings | None = None,
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
        plugin(target_use(effect, contract))
    plugin(as_use(contract.costs))
    cands = norm_candidates(universe.candidates(fields, [from_ns(t) for t in times]))
    cands = cands.join(excluded.select("t", "entity"), on=["t", "entity"], how="anti").sort("t", "entity")
    present = sorted(int(t) for t in cands.select(pl.col("t").dt.epoch("ns").unique())["t"])
    stride = max(1, math.ceil(len(present) / s.max_times))
    kept = pl.Series("t", present[::stride], dtype=pl.Int64)
    sampled = cands.filter(pl.col("t").dt.epoch("ns").is_in(kept.implode()))
    extra = [c for c in cands.columns if c not in ("t", "entity")]
    renamed = {c: f"{c}_candidate" for c in extra if c in TABLE_KEYS or c.startswith("y_")}
    columns = sampled.rename(renamed)
    tables: floor.Tables = {}
    components: dict[str, list[str]] = {}
    for label in contract.labels:
        values: LabelPlugin = plugin(as_use(label.use))
        tables[label.name] = {}
        found = label_values(values, fields, sampled, [span(h) for h in label.horizons])
        for h, out in zip(label.horizons, found, strict=True):
            h_ns = span_ns(span(h))
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
    return Frozen(contract, fields, cands, tables, components, start, end, stride, deps, registry, panel, exclusions)


def calibration(frozen: Frozen, settings: FeatSettings | None = None) -> dict[str, Any]:
    """The decisive test in the floor world before anything is judged (``calibration`` in ``stages/floor/feat.py``):
    the data check runs it on the treated data, and feat again when anything it rests on changed."""
    return floor.calibration(frozen, settings or FeatSettings())


def calibration_key(frozen: Frozen, settings: FeatSettings | None = None) -> dict[str, str]:
    """What a calibration rests on beyond the contract and the engine: the treated data, the code of every plugin feat
    runs, and the calibration settings. A calibration is reused while these are unchanged."""
    s = settings or FeatSettings()
    data = {"panel": frozen.panel, "exclusions": frozen.exclusions}
    return frozen.deps | data | {"settings": hash_obj(asdict(s) | {"workers": None})}


def files(root: Path) -> dict[str, str]:
    """The idea's evals and nulls files, ``none`` when absent."""
    feat, nulls = Path(root) / FEAT_FILE, Path(root) / NULLS_FILE
    return {
        "feat_file": hash_file(feat) if feat.is_file() else "none",
        "nulls_file": hash_file(nulls) if nulls.is_file() else "none",
    }


def eval_files(root: Path, result: dict[str, Any]) -> dict[str, str]:
    """The module of every case eval a submission ran, as it is now, by its file in the idea: ``missing`` when gone."""
    names = {r["validation"]["file"] for r in result.get("evals", []) if r.get("by") == "case" and "validation" in r}
    out = {}
    for name in sorted(n for n in names if n and n != FEAT_FILE):
        path = Path(name) if Path(name).is_absolute() else Path(root) / name
        out[f"{EVAL_FILES}{name}"] = hash_file(path) if path.is_file() else "missing"
    return out


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
    seen = floor.Seen(frozen, s, Path(root), Path(path))
    warnings = _run(floor.SCREEN, seen).warnings
    evals = _evals(Path(root), frozen.registry)
    named = [(step.name.format(clusters=seen.clusters), step) for step in defaults.CHECKS]
    names = {name for name, _ in named} | {e.name for e in evals}
    worlds = [floor.floor_world(frozen, s)] + _worlds(Path(root), frozen, names)
    features = _features_of(Path(root), path, NO_TRIAL)
    validated = _validated(Path(root), evals, frozen.registry)
    scores, cost = floor.computed(features, frozen, Path(path), after=NO_TRIAL)
    digest = hash_frame(scores)
    if digest in earlier:
        raise FeatError(f"the same scores as submission {earlier[digest]}; they would not be a new trial{NO_TRIAL}")
    need = pw.z_need(n, s.z_base)
    seen = replace(seen, features=features, scores=scores, cost=cost, need=need, worlds=tuple(worlds))
    seen = replace(seen, validated=validated)
    floors, shown = _run(floor.FLOORS, seen), _run(floor.SHOWN, seen)
    out: dict[str, Any] = {
        "n": n,
        "features": {"path": str(Path(path).resolve()), "hash": hash_file(path)},
        "scores": {"hash": digest, **floors.shown["scores"]},
        "floors": [asdict(ch) for ch in floors.checks],
        "cost": shown.shown["cost"] | {"cut_seconds": floors.shown["cuts"]},
        "need": need,
        "stride": frozen.stride,
        "worlds": [asdict(w) for w in worlds],
        "coverage": shown.shown["coverage"],
        "warnings": warnings,
    }
    failed = {c.name: c for c in floors.checks if c.status == "fail"}
    if "scores read nothing ahead" in failed:
        why = failed["scores read nothing ahead"].detail
        note = f"the scores read data from after their decision time (scores read nothing ahead, floor): {why}"
        out["verdict"] = _verdict("TAINTED", [f"{note}; compute each score from rows with available_at <= t"])
        return out | {"seconds": round(time.monotonic() - started, 1)}
    if "scores rank candidates" in failed:
        why = failed["scores rank candidates"].detail
        note = f"the scores do not rank the candidates (scores rank candidates, floor): {why}, needs ≥ 50%"
        out["verdict"] = _verdict("FAILED", [f"{note}; give two or more candidates distinct scores at each time"])
        return out | {"seconds": round(time.monotonic() - started, 1)}
    judged = _judge(seen, named, evals)
    out["decisive"] = judged.shown["decisive"]
    out["evals"] = judged.shown["evals"]
    out["verdict"] = _combine(out["decisive"], need)
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


def explore(root: Path, frozen: Frozen, path: Path, settings: FeatSettings | None = None) -> dict[str, Any]:
    """The checks a submission must pass before it is judged, without judging it: nothing about how it would do."""
    s = settings or FeatSettings()
    seen = floor.Seen(frozen, s, Path(root), Path(path), after=EXPLORED)
    warnings = _run(floor.SCREEN, seen).warnings
    features = _features_of(Path(root), path, EXPLORED)
    scores, cost = floor.computed(features, frozen, Path(path), after=EXPLORED)
    seen = replace(seen, features=features, scores=scores, cost=cost)
    floors = _run(floor.FLOORS, seen)
    return {
        "features": {"path": str(Path(path).resolve()), "hash": hash_file(path)},
        "scores": {"hash": hash_frame(scores), **floors.shown["scores"]},
        "floors": [asdict(ch) for ch in floors.checks],
        "cost": asdict(cost) | {"cut_seconds": floors.shown["cuts"]},
        "warnings": warnings,
    }


def _run(listed: Sequence[steps.Step[floor.Seen]], seen: floor.Seen) -> steps.Found:
    return steps.combined(steps.run(step, seen) for step in listed)


def _judge(seen: floor.Seen, named: list[tuple[str, steps.Step[floor.Seen]]], evals: list[Eval]) -> steps.Found:
    """The decisive test and the evals side by side: each step says what it needs judged, and all of it is judged
    together, every shuffle once for each eval and world that reads it. An idea eval named like a default diagnostic
    is judged in its place. Only the candidates the features scored are judged: the rows of the evaluation tables
    without a finite score are left out, so they are left out in every world alike."""
    scores = seen.scores
    assert scores is not None
    scored = scores.filter(pl.col("score").is_finite())
    tables = {
        label: {
            h: (table.join(scored, on=["t", "entity"], how="inner").sort("t", "entity"), h_ns)
            for h, (table, h_ns) in by_h.items()
        }
        for label, by_h in seen.frozen.tables.items()
    }
    idea = {e.name: e for e in evals}
    rest = tuple(e for e in evals if e.name not in {name for name, _ in named})
    judging = replace(seen, scored=tables, evals=rest)
    found: list[steps.Found] = []
    for item in floor.JUDGE:
        if isinstance(item, steps.Step):
            found.append(item.run(judging))
        else:
            found += [_default(judging, name, step, idea) for name, step in named]
    pending = [p for f in found for key in JUDGED for p in f.shown.get(key, [])]
    asked = [p for p in pending if p.request is not None]
    results = dict(zip(map(id, asked), judge_all([p.request for p in asked], seen.settings.workers), strict=True))
    rows = {id(p): p.shown(results.get(id(p))) for p in pending}
    for f in found:
        f.shown |= {key: [row for p in f.shown[key] for row in rows[id(p)]] for key in JUDGED if key in f.shown}
    out = steps.combined(found)
    out.shown["evals"] = [r for f in found for r in f.shown.get("evals", [])]
    return out


def _default(seen: floor.Seen, name: str, step: steps.Step[floor.Seen], idea: dict[str, Eval]) -> steps.Found:
    if name in idea:
        return floor.idea_eval(seen, idea[name])
    return step.run(replace(seen, name=name))


def _evals(root: Path, registry: Registry | None) -> list[Eval]:
    """The idea's ``EVALS`` in ``feat.py``, none without the file; a case eval with no tests is refused."""
    found = _layer_file(root, root / FEAT_FILE, _declared("EVALS", lambda m: evals_of(m, "feat")), False, EVALS_FORMAT)
    try:
        ev.refuse_untested(found, root, FEAT_FILE, NO_TRIAL)
    except ev.EvalsError as exc:
        raise FeatError(str(exc)) from exc
    return found


def _validated(root: Path, evals: list[Eval], registry: Registry | None) -> dict[str, ev.Validation]:
    try:
        return ev.validate(evals, root, registry)
    except ev.EvalsError as exc:
        raise FeatError(f"{exc}{NO_TRIAL}") from exc


def _declared(name: str, read: Callable[[Any], list[Any]]) -> Callable[[Any], list[Any]]:
    """A layer file that exists must declare ``name``; an empty list is fine."""

    def checked(module: Any) -> list[Any]:
        if not hasattr(module, name):
            raise ValueError(f"it defines no {name}")
        return read(module)

    return checked


def _worlds(root: Path, frozen: Frozen, names: set[str]) -> list[World]:
    nulls = _layer_file(root, root / NULLS_FILE, _declared("NULLS", nulls_of), required=False, fmt=NULLS_FORMAT)
    ids = {a.id for a in frozen.contract.alternatives} | set(FLOOR_ALTERNATIVES)
    columns = set(next(iter(next(iter(frozen.tables.values())).values()))[0].columns) | {"score"}
    out, problems = [], []
    for i, n in enumerate(nulls, start=1):
        if n.explains not in ids:
            problems.append(
                f"null {i}: {n.explains!r} is not an alternative of the contract ({', '.join(sorted(ids))})"
            )
        w = n.world
        bucket = None if w.bucket_by == "time" else w.bucket_by
        keeps = None if n.keeps in (None, "time") else n.keeps
        missing = [col for col in dict.fromkeys((bucket, w.clusters, keeps)) if col is not None and col not in columns]
        if missing:
            problems.append(
                f"null {i}: the evaluation table has no column {', '.join(missing)}; its columns are "
                f"{', '.join(sorted(columns))}, and 'time' buckets by decision time"
            )
        unknown = [name for name in n.evals if name not in names]
        if unknown:
            problems.append(f"null {i}: no eval named {', '.join(unknown)}; the evals are {', '.join(sorted(names))}")
        out.append(
            World(
                name=f"null {i}",
                explains=(n.explains,),
                bucket=bucket,
                clusters=w.clusters,
                block=w.block,
                keeps=keeps,
                evals=tuple(n.evals),
                reps=n.reps,
                by="idea",
                note=n.note,
            )
        )
    if problems:
        raise FeatError(f"{NULLS_FILE}: {'; '.join(problems)}{NO_TRIAL}")
    return out


def _features_of(root: Path, path: Path, after: str) -> floor.Features:
    found = _layer_file(root, Path(path), lambda m: getattr(m, "features", None), required=True, fmt=SHAPE, after=after)
    if not callable(found):
        raise FeatError(f"{path} defines no function features; {SHAPE}{after}")
    return found


def _combine(decisive: list[dict[str, Any]], need: float) -> dict[str, Any]:
    lines = [
        f"{d['on']}: beats random, z {_num(d['z'])} ≥ {need:.2f}"
        if d["outcome"] == "passed"
        else f"{d['on']}: does not beat random, z {_num(d['z'])} < {need:.2f} (decisive test, floor): {_effect(d)}"
        for d in decisive
    ]
    return _verdict("PASSED" if all(d["outcome"] == "passed" for d in decisive) else "FAILED", lines)


def _effect(d: dict[str, Any]) -> str:
    """What the decisive effect came to, for a FAILED verdict: the gain over random and its error at the best horizon,
    or why no z could be computed."""
    if d.get("best") is None:
        return "no horizon has enough blocks for a z; the evaluation tables are too short for this contract"
    per = f" per {d['per']}" if d.get("per") else ""
    scale, unit = (1e4, " bp") if d.get("per") else (1.0, "")
    gain, se = d["effect"] * scale, d["se"] * scale
    return f"gain over random {gain:.3g}{unit}{per} ± {se:.3g} at {d['best']}"


def _verdict(status: str, notes: list[str]) -> dict[str, Any]:
    return {"status": status, "notes": notes}


def _num(x: Any) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.4g}"


def _layer_file(
    root: Path, path: Path, read: Callable[[Any], Any], required: bool, fmt: str, after: str = NO_TRIAL
) -> Any:
    """A file of the idea, loaded with the idea's directory on the import path, and read; a failure names the file,
    says what it must look like, and spends nothing."""
    if not path.is_file():
        if required:
            raise FeatError(f"{path} does not exist; {fmt}")
        return []
    try:
        module = ev.load_module(path, root)
    except ev.EvalsError as exc:
        raise FeatError(f"{exc}; {fmt}{after}") from exc
    try:
        return read(module)
    except ValueError as exc:
        raise FeatError(f"{_named(path, root)}: {exc}; {fmt}{after}") from exc


def _named(path: Path, root: Path) -> str:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def current_deps(contract: Contract, root: Path, recorded: dict[str, str], registry: Registry | None) -> dict:
    """A submission's plugin code, the idea's evals and nulls files, and its case evals' modules as they are now,
    keyed like the recorded ones."""
    now: dict[str, str] = {}
    for key in recorded:
        if key.startswith("plugin:"):
            now[key] = code_hash(load(use_named(contract, key.removeprefix("plugin:")), registry))
        if key.startswith(EVAL_FILES):
            name = key.removeprefix(EVAL_FILES)
            path = Path(name) if Path(name).is_absolute() else Path(root) / name
            now[key] = hash_file(path) if path.is_file() else "missing"
    return now | files(root)
