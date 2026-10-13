"""The feat layer: features judged on the frozen data by a rule fixed before any is seen.

Each submission names its own file of features: the engine reads it, and the idea's evals and null worlds, for
row-by-row code, which the card shows as a warning; refuses, before the submission is counted, a ``feat.py`` without a
lab eval or with one that fails its tests, and a contract alternative without a null world; hands ``features`` the
frozen ``wide`` and the candidates as polars DataFrames, and runs its floor list (``stages/feat/floor.py``) with the
default diagnostics (``stages/feat/defaults.py``) where the floor marks them: the floors on the scores, then the
decisive test (slot 1), the diagnostics and library evals, which are shown (slot 2), and the lab evals and null worlds,
which can veto (slot 3). The verdict is horizon by horizon (``kernel/verdict/rule.py``). Only the candidates the
features scored, with a finite score, are judged, in the real world and in every null world alike; for events the
tables keep every candidate row, the scores missing between the events, and the scores move in time with their gaps,
so statistics and targets judge the rows with a finite score in each world. What the calls cost, their wall time and
the peak memory of the full one, is shown, never judged.
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from research_ci.api.contract import FLOOR_ALTERNATIVES, Contract, Use, as_use, horizons_of, span
from research_ci.api.evals import TABLE_KEYS, Eval, evals_of
from research_ci.api.nulls import Null, PermuteWithin, nulls_of
from research_ci.eval.data.baseline import rows
from research_ci.kernel.hashing import hash_frame, hash_obj
from research_ci.kernel.time.clock import lit_time, span_ns, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict import rule
from research_ci.kernel.verdict.judge import judge_all
from research_ci.kernel.verdict.worlds import World, floor_world
from research_ci.runner.plugins import Registry, code_hash, load
from research_ci.stages import lab, skeleton
from research_ci.stages.common import cadence, parts, target_use
from research_ci.stages.contract.run import use_named
from research_ci.stages.feat import defaults, floor
from research_ci.stages.feat.floor import (
    EVALS_FORMAT,
    EXPLORED,
    FEAT_FILE,
    NO_TRIAL,
    NULLS_FILE,
    SHAPE,
    FeatError,
    FeatSettings,
    Frozen,
)
from research_ci.stages.feat.floor import screen as screen
from research_ci.store import vault
from research_ci.store.workspace import hash_file
from research_ci.validation import load as ev

JUDGED = ("decisive", "shown", "vetoes")
NULLS_FORMAT = (
    "nulls/feat.py defines NULLS = [Null(alternative, PermuteWithin(column, clusters=..., block='horizon'), "
    "evals=(...)), ...], one world for every alternative explanation in the contract; the world shuffles the scores "
    "within the buckets of a column of the evaluation table, or, as ShiftInTime(min_shift=...), shifts each "
    "inst's scores in time, and judges the evals it names, or the decisive effect"
)
EVAL_FILES = "eval:"


def frozen(
    contract: Contract,
    products: Mapping[str, str],
    registry: Registry | None = None,
    settings: FeatSettings | None = None,
) -> Frozen:
    """What feat reads, the products the data freeze pinned, by hash: the treated sources, the decision grid, ``wide``
    built on it, the candidates the treatments leave, and their outcomes; with the evaluation tables built from them,
    one per horizon. Nothing of these is computed again."""
    s = settings or FeatSettings()
    if contract.universe is None or contract.outcome is None:
        raise FeatError("feat judges scores per candidate; the contract names no universe or no outcome")
    name = as_use(contract.data).name
    sources = vault.get(f"{name}.treated", products["treated"])
    grid = vault.get(f"{name}.grid", products["grid"])["grid"]
    table = vault.get(f"{name}.wide", products["wide"])["wide"]
    cands = vault.get(f"{name}.candidates", products["candidates"])["candidates"]
    outcomes = vault.get(f"{name}.outcomes", products["outcomes"])["outcomes"]
    deps: dict[str, str] = {}

    def plugin(use: Use) -> Any:
        found = load(use, registry)
        deps[f"plugin:{use.name}"] = code_hash(found)
        return found

    known = rows(sources).select(pl.col("available_at").dt.epoch("ns"))["available_at"]
    start, latest = int(known.min()), int(known.max())  # type: ignore[arg-type]
    end = latest if contract.explore_end == "latest" else to_ns(pd.Timestamp(contract.explore_end))
    timing, _ = cadence(contract, plugin)
    for effect in contract.decisive:
        plugin(target_use(effect, contract))
    plugin(as_use(contract.costs))
    present = sorted(int(t) for t in cands.select(pl.col("t").dt.epoch("ns").unique())["t"])
    stride = max(1, math.ceil(len(present) / s.max_times))
    kept = pl.Series("t", present[::stride], dtype=pl.Int64)
    sampled = cands.filter(pl.col("t").dt.epoch("ns").is_in(kept.implode()))
    extra = [c for c in cands.columns if c not in ("t", "inst")]
    renamed = {c: f"{c}_candidate" for c in extra if c in TABLE_KEYS or c.startswith("y_")}
    columns = sampled.rename(renamed)
    tables: floor.Tables = {}
    for h in horizons_of(contract, timing.every):
        h_ns = span_ns(span(h))
        out = outcomes.filter((pl.col("h") == h) & (pl.col("t") <= lit_time(end - h_ns))).drop("h")
        tables[h] = (out.join(columns, on=["t", "inst"], how="inner").sort("t", "inst"), h_ns)
    found = parts(outcomes.columns)
    return Frozen(contract, sources, table, cands, grid, tables, found, start, end, stride, deps, registry, products)


def calibration(frozen: Frozen, settings: FeatSettings | None = None) -> dict[str, Any]:
    """The decisive test in the floor world before anything is judged (``calibration`` in ``stages/feat/floor.py``):
    the data check runs it on the treated data, and feat again when anything it rests on changed."""
    return floor.calibration(frozen, settings or FeatSettings())


def calibration_key(frozen: Frozen, settings: FeatSettings | None = None) -> dict[str, str]:
    """What a calibration rests on beyond the contract and the engine: the data freeze's products, the code of every
    plugin feat runs, and the calibration settings. A calibration is reused while these are unchanged."""
    s = settings or FeatSettings()
    return frozen.deps | dict(frozen.products) | {"settings": hash_obj(asdict(s) | {"workers": None})}


def files(root: Path) -> dict[str, str]:
    """The idea's evals and nulls files, ``none`` when absent."""
    feat, nulls = Path(root) / FEAT_FILE, Path(root) / NULLS_FILE
    return {
        "feat_file": hash_file(feat) if feat.is_file() else "none",
        "nulls_file": hash_file(nulls) if nulls.is_file() else "none",
    }


def eval_files(root: Path, result: dict[str, Any]) -> dict[str, str]:
    """The module of every lab eval a submission ran, as it is now, by its file in the idea: ``missing`` when gone."""
    names = {r["validation"]["file"] for r in result.get("tests", [])}
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
    again: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Judge one file of features as the ``n``-th submission of the layer. ``earlier`` maps the scores of earlier
    submissions of this version (by hash) to their number; the same scores again are refused, since they are not a new
    trial. ``again`` maps those of earlier versions: the same scores are judged again under this version as the
    submission they were, without a new trial, since a reopened contract judges them by its rules. Refused too are,
    before anything runs, a ``feat.py`` without a valid lab eval, an alternative of the contract without a null
    world, and a null world that cannot run."""
    s = settings or FeatSettings()
    started = time.monotonic()
    seen = floor.Seen(frozen, s, Path(root), Path(path))
    warnings = _run(floor.SCREEN, seen).warnings
    evals, validated = _lab_ready(Path(root), frozen.registry)
    named = [(step.name.format(clusters=seen.clusters), step) for step in defaults.CHECKS if step.needs == seen.shape]
    names = {name for name, _ in named} | {e.name for e in evals}
    worlds = [floor.floor_world(frozen, s)] + _worlds(Path(root), frozen, names)
    seen = replace(seen, worlds=tuple(worlds), validated=validated)
    covered = _covered(_run(floor.COVERED, seen))
    features = _features_of(Path(root), path, NO_TRIAL)
    scores, cost = floor.computed(features, frozen, Path(path), after=NO_TRIAL)
    digest = hash_frame(scores)
    if digest in earlier:
        raise FeatError(f"the same scores as submission {earlier[digest]}; they would not be a new trial{NO_TRIAL}")
    rejudged = (again or {}).get(digest)
    n = rejudged or n
    need = pw.z_need(n, s.z_base)
    seen = replace(seen, features=features, scores=scores, cost=cost, need=need)
    floors, shown = _run(floor.FLOORS, seen), _run(floor.SHOWN, seen)
    out: dict[str, Any] = {
        "n": n,
        "again": rejudged is not None,
        "features": {"path": str(Path(path).resolve()), "hash": hash_file(path)},
        "scores": {"hash": digest, **floors.shown["scores"]},
        "floors": [asdict(ch) for ch in floors.checks],
        "cost": shown.shown["cost"] | {"cut_seconds": floors.shown["cuts"]},
        "need": need,
        "stride": frozen.stride,
        "worlds": [asdict(w) for w in worlds],
        "coverage": covered.shown["coverage"],
        "tests": [
            {"name": e.name, "file": validated[e.name].file, "h0": e.h0, "validation": validated[e.name].record()}
            for e in evals
            if e.name in validated and validated[e.name].status != "trusted"
        ],
        "warnings": warnings,
    }
    failed = {c.name: c for c in floors.checks if c.status == "fail"}
    if "scores read nothing ahead" in failed:
        why = failed["scores read nothing ahead"].detail
        note = f"the scores read data from after their decision time (scores read nothing ahead, floor): {why}"
        out["verdict"] = rule.verdict("TAINTED", [f"{note}; compute each score from rows with available_at <= t"])
        return out | {"seconds": round(time.monotonic() - started, 1)}
    if "scores rank candidates" in failed:
        why = failed["scores rank candidates"].detail
        said, fix = floor.RANKS[seen.shape]
        need = "" if seen.shape == "events" else ", needs ≥ 50%"
        note = f"{said} (scores rank candidates, floor): {why}{need}"
        out["verdict"] = rule.verdict("FAILED", [f"{note}; {fix}"])
        return out | {"seconds": round(time.monotonic() - started, 1)}
    judged = _judge(seen, named, evals)
    out |= {key: judged.shown.get(key, []) for key in JUDGED}
    out["verdict"] = rule.combine(out["decisive"], out["vetoes"], need)
    if out["verdict"]["status"] == "PASSED":
        out["scores"] |= stored(frozen, scores)
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


def stored(frozen: Frozen, scores: pl.DataFrame) -> dict[str, str]:
    """A PASSED submission's scores, ``t | inst | score``, kept in the vault as parquet by content, so accepting it
    pins them for the next layer: their vault hash and folder."""
    digest, folder = vault.put(scores_kind(frozen.contract), {"scores": scores})
    return {"stored": digest, "path": str(folder)}


def scores_kind(contract: Contract) -> str:
    return f"{as_use(contract.data).name}.scores"


def ready(root: Path, contract: Contract, registry: Registry | None = None) -> None:
    """Refuses, before anything heavy runs and before a submission is counted, a ``feat.py`` without a lab eval or
    with one that fails its tests, and an alternative of the contract without a null world."""
    _lab_ready(Path(root), registry)
    nulls = _layer_file(root, root / NULLS_FILE, _declared("NULLS", nulls_of), required=False, fmt=NULLS_FORMAT)
    unknown = _unknown(nulls, contract)
    if unknown:
        raise FeatError(f"{NULLS_FILE}: {'; '.join(unknown)}{NO_TRIAL}")
    u = contract.universe
    clusters, shape = (u.clusters, u.shape) if u is not None else ("inst", "cross")
    worlds = [floor_world(clusters, 0, shape)] + [_world(i, n) for i, n in enumerate(nulls, start=1)]
    _covered(skeleton.Found([floor.coverage(contract, worlds)]))


def _lab_ready(root: Path, registry: Registry | None) -> tuple[list[Eval], dict[str, ev.Validation]]:
    evals = _evals(root, registry)
    validated = _validated(root, evals, registry)
    refused = lab.refusal(lab.Lab("feat", FEAT_FILE, (root / FEAT_FILE).is_file(), tuple(evals), validated))
    if refused:
        raise FeatError(f"{refused}{NO_TRIAL}")
    return evals, validated


def _covered(found: skeleton.Found) -> skeleton.Found:
    failed = [c.detail for c in found.checks if c.status == "fail"]
    if failed:
        raise FeatError(f"{'; '.join(failed)}{NO_TRIAL}")
    return found


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


def _run(listed: Sequence[skeleton.Step[floor.Seen]], seen: floor.Seen) -> skeleton.Found:
    return skeleton.combined(skeleton.run(step, seen) for step in listed)


def _judge(seen: floor.Seen, named: list[tuple[str, skeleton.Step[floor.Seen]]], evals: list[Eval]) -> skeleton.Found:
    """The decisive test, the evals and the null worlds side by side: each step says what it needs judged, and all of
    it is judged together, every shuffle once for each eval and world that reads it. An idea eval named like a default
    diagnostic is judged in its place. Only the candidates the features scored are judged: the rows of the evaluation
    tables without a finite score are left out, so they are left out in every world alike. For events every row stays,
    its score NaN where the features gave none, since those rows are where a shift in time moves the events to."""
    scores = seen.scores
    assert scores is not None
    scored = scores.filter(pl.col("score").is_finite())
    events = seen.shape == "events"
    tables = {h: (_scored(table, scored, events), h_ns) for h, (table, h_ns) in seen.frozen.tables.items()}
    idea = {e.name: e for e in evals}
    rest = tuple(e for e in evals if e.name not in {name for name, _ in named})
    judging = replace(seen, scored=tables, evals=rest)
    found: list[skeleton.Found] = []
    for item in floor.JUDGE:
        if isinstance(item, skeleton.Step):
            found.append(item.run(judging))
        else:
            found += [_default(judging, name, step, idea) for name, step in named]
    pending = [p for f in found for key in JUDGED for p in f.shown.get(key, [])]
    asked = [p for p in pending if p.request is not None]
    results = dict(zip(map(id, asked), judge_all([p.request for p in asked], seen.settings.workers), strict=True))
    rows = {id(p): p.shown(results.get(id(p))) for p in pending}
    for f in found:
        f.shown |= {key: [row for p in f.shown[key] for row in rows[id(p)]] for key in JUDGED if key in f.shown}
    out = skeleton.combined(found)
    out.shown |= {key: [r for f in found for r in f.shown.get(key, [])] for key in JUDGED}
    return out


def _scored(table: pl.DataFrame, scored: pl.DataFrame, events: bool) -> pl.DataFrame:
    """An evaluation table with the finite scores: only their rows, or for events every row, NaN where none."""
    if not events:
        return table.join(scored, on=["t", "inst"], how="inner").sort("t", "inst")
    joined = table.join(scored, on=["t", "inst"], how="left").sort("t", "inst")
    return joined.with_columns(pl.col("score").fill_null(float("nan")))


def _default(seen: floor.Seen, name: str, step: skeleton.Step[floor.Seen], idea: dict[str, Eval]) -> skeleton.Found:
    if name in idea:
        return floor.idea_eval(seen, idea[name])
    return step.run(replace(seen, name=name))


def _evals(root: Path, registry: Registry | None) -> list[Eval]:
    """The idea's ``EVALS`` in ``feat.py``, none without the file; a lab eval with no h0, no tests, or still the example
    is refused."""
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
    columns = set(next(iter(frozen.tables.values()))[0].columns) | {"score"}
    out, problems = [], _unknown(nulls, frozen.contract)
    for i, n in enumerate(nulls, start=1):
        world = _world(i, n)
        asked = (world.bucket, world.clusters, world.keeps)
        missing = [col for col in dict.fromkeys(asked) if col is not None and col not in columns]
        if missing:
            problems.append(
                f"null {i}: the evaluation table has no column {', '.join(missing)}; its columns are "
                f"{', '.join(sorted(columns))}, and 'time' buckets by decision time"
            )
        unknown = [name for name in n.evals if name not in names]
        if unknown:
            problems.append(f"null {i}: no eval named {', '.join(unknown)}; the evals are {', '.join(sorted(names))}")
        out.append(world)
    if problems:
        raise FeatError(f"{NULLS_FILE}: {'; '.join(problems)}{NO_TRIAL}")
    return out


def _unknown(nulls: list[Null], contract: Contract) -> list[str]:
    """The worlds that name an alternative the contract does not have."""
    ids = {a.id for a in contract.alternatives} | set(FLOOR_ALTERNATIVES)
    return [
        f"null {i}: {n.explains!r} is not an alternative of the contract ({', '.join(sorted(ids))})"
        for i, n in enumerate(nulls, start=1)
        if n.explains not in ids
    ]


def _world(i: int, n: Null) -> World:
    w = n.world
    shift = None if isinstance(w, PermuteWithin) else w.min_shift
    return World(
        name=f"null {i}",
        explains=(n.explains,),
        bucket=None if not isinstance(w, PermuteWithin) or w.bucket_by == "time" else w.bucket_by,
        clusters=w.clusters if isinstance(w, PermuteWithin) else None,
        block=w.block if isinstance(w, PermuteWithin) else None,
        keeps=None if n.keeps in (None, "time") else n.keeps,
        evals=tuple(n.evals),
        reps=n.reps,
        by="idea",
        note=n.note,
        shift=shift,
    )


def _features_of(root: Path, path: Path, after: str) -> floor.Features:
    found = _layer_file(root, Path(path), lambda m: getattr(m, "features", None), required=True, fmt=SHAPE, after=after)
    if not callable(found):
        raise FeatError(f"{path} defines no function features; {SHAPE}{after}")
    return found


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
    """A submission's plugin code, the idea's evals and nulls files, and its lab evals' modules as they are now,
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
