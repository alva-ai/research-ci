"""The steps a feat submission takes, around its default list (``stages/feat/defaults.py``), slot by slot.

Before anything runs, and before the submission is counted: ``SCREEN`` refuses a features file that cannot be a trial;
the checks of the lab's evals (``stages/lab.py``) refuse a ``feat.py`` without a lab eval (slot 3) or with one that
fails its tests (slot 4); ``COVERED`` refuses a contract alternative without a null world (slot 3). ``FLOORS`` checks
the scores, the premises of slot 1: they must rank what the universe's shape compares (the candidates at each time, each
inst's times, or events) and read nothing ahead of their time; a submission that fails one is TAINTED or FAILED and
nothing is judged. ``JUDGE`` holds the decisive test, the one bar, fixed by the contract (slot 1): the contract's
decisive effect, computed with the submitted scores, must beat the engine's floor world with ``z ≥ z_need``, which rises
with the submission count; it passes at the horizons whose z against the family's null reaches the bar. The floor world
follows the shape: for ``cross``, chance and timing, scores shuffled across clusters at each decision time, the same way
within a horizon; for ``series`` and ``events``, chance, each inst's scores shifted in time by at least the horizon.
Next to it slot 2, the default diagnostics and the library evals ``feat.py`` lists, judged in the floor world and shown,
never deciding; and slot 3, the lab evals of ``feat.py``, judged in the floor world, and the null worlds of
``nulls/feat.py``, each judging the evals it names or, naming none, the decisive effect: each must reach ``z ≥ 2``
(``VETO_Z``) at every horizon it covers, or the submission fails there (``kernel/verdict/rule.py``). Only the candidates
the features gave a finite score are judged, in every world alike; for events every candidate row stays in the tables,
and the rows without a score are where the events can move to. ``SHOWN`` says what the features cost. Before the first
submission is judged, the data check calibrates the decisive test in the floor world (``calibration``).
"""

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

import pandas as pd
import polars as pl

from research_ci.api.contract import Contract, ContractError, Effect, Use, as_use, span
from research_ci.api.evals import FORMAT, Eval, source_of
from research_ci.api.plugins import CostModel, Sources, TargetPlugin
from research_ci.eval import protocol as pr
from research_ci.kernel.time.asof import known, wide
from research_ci.kernel.time.clock import TIME, lit_time
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict import rule, worlds
from research_ci.kernel.verdict.judge import (
    Batched,
    EvalError,
    Evaluate,
    Request,
    Sequential,
    Statistic,
    per_t,
)
from research_ci.kernel.verdict.worlds import World, lanes
from research_ci.runner import research
from research_ci.runner.meter import Cost, measured
from research_ci.runner.plugins import Registry, load
from research_ci.stages import lab
from research_ci.stages.common import UNSCORED, default_workers, iso, owner, quantile_times, target_use, units
from research_ci.stages.skeleton import DEFAULTS, Found, Step
from research_ci.validation import load as ev
from research_ci.validation.calibration import EVENTS, calibrate

FEAT_FILE = "feat.py"
NULLS_FILE = Path("nulls") / "feat.py"
EXAMPLE = Path("research") / "example.py"
SCORE_COLUMNS = ("t", "inst", "score")
NO_TRIAL = "; no submission was spent"
EXPLORED = "; nothing was recorded"
SHAPE = (
    "a features file defines features(wide, candidates), which gets polars DataFrames and returns a DataFrame "
    "(polars or pandas) or a polars LazyFrame with columns t, inst, score; research/example.py shows one"
)
EVALS_FORMAT = FORMAT["feat"]
Features = Callable[[pl.DataFrame, pl.DataFrame], Any]
Tables = dict[str, tuple[pl.DataFrame, int]]
RANKS: dict[str, tuple[str, str]] = {
    "cross": ("the scores do not rank the candidates", "give two or more candidates distinct scores at each time"),
    "series": (
        "the scores do not time the insts",
        "give each inst's scores two or more distinct values over time",
    ),
    "events": (
        "the scores mark no events",
        "score the candidate rows at the events and leave the others without a score",
    ),
}


class FeatError(ContractError):
    pass


@dataclass(frozen=True)
class FeatSettings:
    """The evaluation tables keep at most ``max_times`` decision times (every k-th when there are more). Calibrating
    the decisive effect runs on ``calibration_times`` of them: ``calibration_reps`` stand-ins and planted outputs, each
    against ``calibration_nulls`` shuffles. ``cuts`` decision times check that features read nothing ahead. The
    decisive test, and every lab eval in the floor world, draws its shuffles ``floor_batch`` at a time, up to
    ``floor_reps``, until its decision is settled, and a null world of the idea up to as many as it says; a diagnostic
    of slot 2 takes ``shown_reps`` in the floor world."""

    workers: int = field(default_factory=lambda: int(os.environ.get("RCI_WORKERS", default_workers())))
    max_times: int = 1000
    calibration_times: int = 300
    calibration_reps: int = 100
    calibration_nulls: int = 30
    floor_reps: int = 200
    floor_batch: int = 50
    shown_reps: int = 50
    min_blocks: int = 5
    cuts: int = 3
    z_base: float = pw.Z_BASE


@dataclass
class Frozen:
    """What feat reads: the treated sources, ``wide`` built on them and the candidates the data layer froze, the
    decision grid of ``wide`` (``t | inst``), and one evaluation table per horizon (``t``, ``inst``, ``y``, its parts
    and ``y_end``, the candidate columns) on every ``stride``-th decision time whose outcome is complete before the
    explore data ends; ``parts`` names the outcome's parts, and ``products`` the hashes of the data freeze's products it
    was read from."""

    contract: Contract
    sources: Sources
    wide: pl.DataFrame
    candidates: pl.DataFrame
    grid: pl.DataFrame
    tables: Tables
    parts: list[str]
    start: int
    end: int
    stride: int
    deps: dict[str, str]
    registry: Registry | None
    products: Mapping[str, str] = field(default_factory=dict)

    def inputs(self, cut: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """What a features file gets: ``wide`` and the candidates, as polars DataFrames, copies that share their
        memory; with ``cut``, ``wide`` built again on the sources cut there, on the frozen grid up to it, and the
        candidates up to it."""
        if cut is None:
            return self.wide.clone(), self.candidates.clone()
        until = pl.col("t") <= lit_time(cut)
        return wide(known(self.sources, cut), self.grid.filter(until)), self.candidates.filter(until)


@dataclass(frozen=True)
class Seen:
    """What a feat step reads: the frozen data, the settings, the idea's directory, the features file, and what a
    refusal says was not spent. Once the features ran: their function, their scores, and what the full call cost.
    For judging: the bar, the worlds, the evaluation tables with the scores, the idea's evals not judged in place of a
    default diagnostic and their validations, and the name a default diagnostic is judged under."""

    frozen: Frozen
    settings: FeatSettings
    root: Path
    path: Path
    after: str = NO_TRIAL
    features: Features | None = None
    scores: pl.DataFrame | None = None
    cost: Cost | None = None
    need: float = math.nan
    worlds: tuple[World, ...] = ()
    scored: Tables = field(default_factory=dict)
    evals: tuple[Eval, ...] = ()
    validated: Mapping[str, ev.Validation] = field(default_factory=dict)
    name: str = ""

    @property
    def clusters(self) -> str:
        assert self.frozen.contract.universe is not None
        return self.frozen.contract.universe.clusters

    @property
    def shape(self) -> str:
        assert self.frozen.contract.universe is not None
        return self.frozen.contract.universe.shape

    def judge(self, evaluate: Evaluate) -> Found:
        """A default diagnostic, ``evaluate`` on the outcome, judged under ``name``."""
        e = Eval(evaluate, self.name, combine="max")
        return judged(self, e, "default")


@dataclass(frozen=True)
class Pending:
    """What a judging step asks for: the stage judges every step's ``request`` together (``judge_all``), each shuffle
    once for all of them, then ``shown`` turns the result, or the exception, into the rows of the card. Without a
    request nothing is judged, and ``shown`` gets ``None``."""

    request: Request | None
    shown: Callable[[dict[str, Any] | Exception | None], list[dict[str, Any]]]


def screen_file(seen: Seen) -> Found:
    """What is refused before anything runs, a features file that does not exist, is still the example, or does not
    parse; and the card's warnings: row-by-row code in it, in the idea's evals and null worlds, or in the research
    modules they import."""
    return Found(warnings=screen(seen.root, seen.path, seen.frozen.registry, seen.after))


def ranks(seen: Seen) -> Found:
    """The scores must rank what the shape compares: the candidates, with two or more distinct values, at half the
    decision times or more (cross); each inst's decision times, with two or more distinct values over time, for half
    the insts or more (series); or mark events, some of the candidate rows and not all (events)."""
    assert seen.scores is not None
    cands = seen.frozen.candidates
    finite = seen.scores.filter(pl.col("score").is_finite())
    by = "inst" if seen.shape == "series" else "t"
    distinct = finite.group_by(by).agg(pl.col("score").n_unique().alias("k"))
    total = cands[by].n_unique()
    share = distinct.filter(pl.col("k") >= 2).height / total if total else 0.0
    coverage = {"rows": finite.height, "share": finite.height / cands.height if cands.height else 0.0, "ranks": share}
    scored = f"scores {coverage['share']:.0%} of candidate rows"
    if seen.shape == "events":
        failed = not 0 < finite.height < cands.height
        detail = f"{scored}: {finite.height:,} of {cands.height:,} are events"
    elif seen.shape == "series":
        failed = share < 0.5
        detail = f"{scored}; varies over time for {share:.0%} of insts"
    else:
        failed = share < 0.5
        detail = f"{scored}; ranks candidates at {share:.0%} of decision times"
    return Found([pr.Check("scores rank candidates", "fail" if failed else "pass", detail)], shown={"scores": coverage})


def nothing_ahead(seen: Seen) -> Found:
    """The features, run again on the data cut at a few decision times, must give the same scores up to each cut."""
    assert seen.features is not None and seen.scores is not None
    epoch = seen.frozen.candidates.select(pl.col("t").dt.epoch("ns").unique())["t"]
    cuts = quantile_times(epoch, seen.settings.cuts)
    seconds: list[float] = []
    with ThreadPoolExecutor(max_workers=seen.settings.workers) as pool:
        jobs = [(cut, pool.submit(computed, seen.features, seen.frozen, seen.path, cut)) for cut in cuts]
        problems = []
        for cut, job in jobs:
            try:
                cut_scores, cost = job.result()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{iso(cut)}: features failed on the cut data: {exc}")
                continue
            seconds.append(cost.seconds)
            before = seen.scores.filter(pl.col("t") <= lit_time(cut))
            mismatch = pr._frame_mismatch(before, cut_scores, ["t", "inst"], ["score"])
            if mismatch:
                problems.append(f"{iso(cut)}: {mismatch}")
    leak = bool(problems)
    detail = "; ".join(problems) if leak else f"the same scores with the data cut at {len(cuts)} decision times"
    return Found([pr.Check("scores read nothing ahead", "fail" if leak else "pass", detail)], shown={"cuts": seconds})


def decisive_test(seen: Seen) -> Found:
    """The decisive effect, with the submitted scores, against the floor world: the family's z over the horizons must
    reach the bar, and it passes at each horizon whose z against the same null (the best of each shuffle's
    standardized horizons) does. Shuffles are drawn a batch at a time until every horizon's decision is settled
    (``kernel/verdict/sequential.py``) or all are drawn. The effect shown is the gain over random, the target with
    these scores less its mean in the floor world, at the best counted horizon."""
    s = seen.settings
    floor = floor_world(seen.frozen, s)
    found = Found(shown={"decisive": []})
    for effect in seen.frozen.contract.decisive:
        evaluate, use, head = target(effect, seen.frozen)
        settle = Sequential(s.floor_batch, seen.need, horizons=True)
        request = Request(evaluate, lanes(seen.scored, floor), s.floor_reps, s.min_blocks, "max", settle)
        found.shown["decisive"].append(Pending(request, partial(_decided, use, head, seen.need)))
    return found


def library_evals(seen: Seen) -> Found:
    """The library evals ``feat.py`` lists, judged in the floor world like the default diagnostics and shown, never
    deciding; a null world that names one judges it too, and can veto."""
    found = Found(shown={"shown": [], "vetoes": []})
    for e in seen.evals:
        if source_of(e.of) == "library":
            found = _joined(found, idea_eval(seen, e))
    return found


def lab_evals(seen: Seen) -> Found:
    """Each lab eval ``feat.py`` lists, a test of the claim: judged in the floor world, and in every null world that
    names it, it must reach ``z ≥ 2`` at every horizon it covers, or the submission fails there. One named like a
    default diagnostic is judged in its place instead."""
    found = Found(shown={"shown": [], "vetoes": []})
    for e in seen.evals:
        if source_of(e.of) == "lab":
            found = _joined(found, idea_eval(seen, e))
    return found


def null_worlds(seen: Seen) -> Found:
    """Each null world of ``nulls/feat.py`` that names no eval judges the decisive effect itself, with the submitted
    scores shuffled its way: it must reach ``z ≥ 2`` at every horizon, or the submission fails there. A world that
    names evals judges them where they are judged."""
    s = seen.settings
    found = Found(shown={"vetoes": []})
    for w in seen.worlds:
        if w.by == "engine" or w.evals:
            continue
        for effect in seen.frozen.contract.decisive:
            evaluate, _, head = target(effect, seen.frozen)
            veto = Sequential(s.floor_batch, rule.VETO_Z, horizons=True)
            request = Request(evaluate, lanes(seen.scored, w), w.reps, s.min_blocks, "each", veto)
            found.shown["vetoes"].append(Pending(request, partial(_world_row, effect, head, w, seen)))
    return found


def idea_eval(seen: Seen, e: Eval) -> Found:
    """An eval of the idea judged: a lab eval in slot 3, with its validation; a library eval in slot 2."""
    by = source_of(e.of)
    return judged(seen, e, by, seen.validated.get(e.name) if by == "lab" else None)


def alternatives(seen: Seen) -> Found:
    """Which world answers which alternative: every alternative of the contract needs a null world in
    ``nulls/feat.py``, or the submission is refused before it is counted; the floor world answers chance, and for a
    cross shape timing."""
    check = coverage(seen.frozen.contract, seen.worlds)
    return Found([check], shown={"coverage": asdict(check)})


def coverage(c: Contract, judging: Sequence[World]) -> pr.Check:
    """Which of ``judging`` answers each alternative of the contract, and the engine's; failed, with what to write,
    when one has none."""
    lines, missing = [], []
    floors = dict.fromkeys(alt for w in judging if w.by == "engine" for alt in w.explains)
    for alt in [*floors, *(a.id for a in c.alternatives)]:
        kept_by = [w.name for w in judging if alt in w.explains]
        lines.append(f"{alt}: {', '.join(kept_by) or 'none'}")
        missing += [] if kept_by else [alt]
    if not missing:
        return pr.Check("alternatives covered", "pass", " · ".join(lines))
    texts = {a.id: a.text for a in c.alternatives}
    named = "; ".join(f"{alt} ({texts.get(alt, '')})" for alt in missing)
    clusters = c.universe.clusters if c.universe is not None else "<cluster column>"
    world = f'PermuteWithin("time", clusters="{clusters}", block="horizon")'
    if c.universe is not None and c.universe.shape != "cross":
        world = 'ShiftInTime(min_shift="<a duration the alternative needs, such as 72h>")'
    detail = (
        f"slot 3, null worlds: no world answers {named}: every alternative explanation of the contract needs one "
        f'in {NULLS_FILE}, such as NULLS = [Null("{missing[0]}", {world}, evals=("<an eval it judges>",))], which '
        "keeps what the alternative says and shuffles the rest; it judges the evals it names, or the decisive effect "
        "when it names none"
    )
    return pr.Check("alternatives covered", "fail", detail)


def cost(seen: Seen) -> Found:
    """What the full features call cost, its wall time and peak memory; with the seconds of the calls on cut data,
    shown, never judged."""
    assert seen.cost is not None
    return Found(shown={"cost": asdict(seen.cost)})


SCREEN = (Step("screen", "blocks", screen_file, "Refuses a missing, example, or unparsable file"),)
COVERED = (
    Step(
        "alternatives covered",
        "blocks",
        alternatives,
        "Every alternative of the contract has a null world",
        "nulls/feat.py",
        slot=3,
    ),
)
FLOORS = (
    Step("scores rank candidates", "blocks", ranks, "Scores rank candidates, or times, or mark events"),
    Step("scores read nothing ahead", "blocks", nothing_ahead, "Same scores with the data cut early"),
)
JUDGE = (
    Step("decisive test", "decides", decisive_test, "The decisive effect beats the floor world: z ≥ z_need"),
    DEFAULTS,
    Step("library evals", "shown", library_evals, "Each library eval feat.py lists, vs the floor", "feat.py", slot=2),
    Step("lab evals", "vetoes", lab_evals, "Each lab eval reaches z ≥ 2 where it covers", "feat.py", slot=3),
    Step("null worlds", "vetoes", null_worlds, "Each world's judgement reaches z ≥ 2", "nulls/feat.py", slot=3),
)
SHOWN = (Step("cost", "shown", cost, "Wall time and peak memory of the features call"),)
STEPS = (*SCREEN, *lab.READY, *COVERED, *FLOORS, *JUDGE, *SHOWN)


def screen(root: Path, path: Path, registry: Registry | None = None, after: str = NO_TRIAL) -> list[str]:
    """What is refused before anything runs, a features file that does not exist, is still the example, or does not
    parse; and the card's warnings: row-by-row code in it, in the idea's evals and null worlds, or in the research
    modules they import."""
    path = Path(path)
    if not path.is_file():
        raise FeatError(f"{path} does not exist; {SHAPE}{after}")
    if research.still_example(path):
        raise FeatError(
            f"{path} is still the example; write your own: copy {EXAMPLE} to research/<name>.py, delete its "
            f"`{research.EXAMPLE_MARKER}` line, and replace every TODO{after}"
        )
    root = Path(root)
    found = research.warning([path, root / FEAT_FILE, root / NULLS_FILE], root, registry)
    return [found] if found else []


def judged(seen: Seen, e: Eval, by: str, validation: "ev.Validation | None" = None) -> Found:
    """An eval judged in the floor world and in every null world that names it. A lab eval, and any eval in a world of
    the idea, can veto (slot 3): it draws its shuffles until its votes against ``VETO_Z`` are settled. A default
    diagnostic or a library eval in the floor world is shown (slot 2), on ``shown_reps`` shuffles. Worlds that shuffle
    alike are judged once and shown under each name."""
    s = seen.settings
    label = column_of(e, seen.frozen)
    tables = {h: _as_y(seen.scored[h], label) for h in horizons_of(e, seen.frozen)}
    combine = e.combine or "max"
    alike: dict[tuple[Any, ...], Request] = {}
    found = Found(shown={"shown": [], "vetoes": []})
    for w in seen.worlds:
        floor = w.by == "engine"
        if not floor and e.name not in w.evals:
            continue
        veto = by == "lab" or not floor
        reps = (s.floor_reps if floor else w.reps) if veto else s.shown_reps
        settle = Sequential(s.floor_batch, rule.VETO_Z, horizons=combine == "each") if veto else None
        key = (w.bucket, w.clusters, w.block, w.shift, reps, veto)
        if key not in alike:
            alike[key] = Request(e.of, lanes(tables, w), reps, s.min_blocks, combine, settle)
        row = partial(_eval_row, e, by, validation, w, seen, label, veto)
        found.shown["vetoes" if veto else "shown"].append(Pending(alike[key], row))
    return found


def column_of(e: Eval, frozen: Frozen) -> str:
    """The column an eval reads as ``y``: the outcome, or one of its parts."""
    column = e.label or "y"
    if column != "y" and column not in frozen.parts:
        known = ", ".join(["y", *frozen.parts])
        raise FeatError(f"eval {e.name}: label {column!r} is not the outcome or one of its parts ({known}){NO_TRIAL}")
    return column


def _as_y(lane: tuple[pl.DataFrame, int], column: str) -> tuple[pl.DataFrame, int]:
    """A horizon's table with ``column`` read as ``y``."""
    table, h_ns = lane
    return (table if column == "y" else table.with_columns(pl.col(column).alias("y")), h_ns)


def horizons_of(e: Eval, frozen: Frozen) -> tuple[str, ...]:
    declared = tuple(frozen.tables)
    if e.horizons is None:
        return declared
    unknown = [h for h in e.horizons if h not in declared]
    if unknown:
        raise FeatError(f"eval {e.name}: horizons {', '.join(unknown)} are not among {', '.join(declared)}{NO_TRIAL}")
    return tuple(e.horizons)


def floor_world(frozen: Frozen, s: FeatSettings) -> World:
    u = frozen.contract.universe
    assert u is not None
    return worlds.floor_world(u.clusters, s.shown_reps, u.shape)


def target(effect: Effect, frozen: Frozen) -> tuple[dict[str, Evaluate], Use, dict[str, Any]]:
    """The effect's target per horizon, the plugin it names, and the effect as cards show it."""
    c = frozen.contract
    use = target_use(effect, c)
    plugin: TargetPlugin = load(use, frozen.registry)
    costs: CostModel = load(as_use(c.costs), frozen.registry)
    batched = getattr(plugin, "batched", None) if owner(plugin, "batched") is owner(plugin, "reduce") else None

    def make(h: str) -> Evaluate:
        horizon = span(h)
        form = cast(Batched | None, batched(horizon, costs)) if callable(batched) else None
        return Statistic(lambda table: plugin.reduce(table, horizon, costs), form)

    return {h: make(h) for h in frozen.tables}, use, units(effect, plugin)


def calibration(frozen: Frozen, s: FeatSettings) -> dict[str, Any]:
    """The decisive effect in the floor world before anything is judged: its target must read the scores, a stand-in
    that carries only each decision time's level must die as often as chance allows, and a planted signal must live.
    ``ok`` is false, with ``notes`` saying why, when any of them does not hold."""
    started = time.monotonic()
    checks = [_reads_score(e, frozen) for e in frozen.contract.decisive]
    effects = {e.name: target(e, frozen)[2] for e in frozen.contract.decisive}
    rows = [r | effects[r["effect"]] for r in _calibrate(frozen, s)]
    notes = [f"{c.name}: {c.detail}" for c in checks if c.status == "fail"]
    notes += [_returned(r) for r in rows if not r.get("ok")]
    return {
        "rows": rows,
        "checks": [asdict(c) for c in checks],
        "ok": not notes,
        "notes": notes,
        "seconds": round(time.monotonic() - started, 1),
    }


def computed(
    features: Features, frozen: Frozen, path: Path, cut: int | None = None, after: str = ""
) -> tuple[pl.DataFrame, Cost]:
    """The scores of ``features`` on what the engine hands it, collected and checked, and what the call cost, a
    returned LazyFrame's collection included; the peak memory only for the full call, which runs alone. A failure says
    what to fix, and ``after`` what it did not cost."""
    table, cands = frozen.inputs(cut)

    def call() -> Any:
        raw = features(table, cands)
        return raw.collect() if isinstance(raw, pl.LazyFrame) else raw

    try:
        raw, spent = measured(call, memory=cut is None)
    except Exception as exc:
        raise FeatError(f"{path.name}: features(wide, candidates) failed: {type(exc).__name__}: {exc}{after}") from exc
    asked = frozen.candidates if cut is None else frozen.candidates.filter(pl.col("t") <= lit_time(cut))
    try:
        return _scores(raw, asked), spent
    except FeatError as exc:
        raise FeatError(f"{exc}{after}") from exc


def _decided(
    use: Use, head: dict[str, Any], need: float, result: dict[str, Any] | Exception | None
) -> list[dict[str, Any]]:
    if isinstance(result, Exception):
        raise result
    assert result is not None
    return [_outcome(use, head, result, need)]


def _outcome(use: Use, head: dict[str, Any], result: dict[str, Any], need: float) -> dict[str, Any]:
    rows = _horizons(result, need)
    for r in rows:
        r["passed"] = bool(r["counted"]) and math.isfinite(r["adjusted"]) and r["adjusted"] >= need
    best = next((r for r in rows if r["h"] == result["best"]), None)
    z = result["z"]
    return head | {
        "kind": "outcome",
        "target": use.name,
        "effect": best["gain"] if best else math.nan,
        "se": best["se"] if best else math.nan,
        "z": z,
        "best": result["best"],
        "horizons": rows,
        "passing": [r["h"] for r in rows if r["passed"]],
        "shuffles": result["shuffles"],
        **({"distinct": result["distinct"]} if "distinct" in result else {}),
        "outcome": "passed" if math.isfinite(z) and z >= need else "failed",
    }


def _horizons(result: dict[str, Any], bar: float) -> list[dict[str, Any]]:
    """Each horizon of a judgement with its gain over the null and its MDE at ``bar``, from the shuffles drawn."""
    return [r | {"gain": r["real"] - r["null_mean"], "mde": pw.mde(r["null_sd"], bar)} for r in result["horizons"]]


def _failed(name: str, by: str, result: Exception) -> FeatError:
    what = str(result) if isinstance(result, EvalError) else f"{type(result).__name__}: {result}"
    tested = "; it passed its tests, so add a case like this table to them" if by == "lab" else ""
    return FeatError(f"eval {name!r} failed: {what}{tested}; {EVALS_FORMAT}{NO_TRIAL}")


def _eval_row(
    e: Eval,
    by: str,
    validation: "ev.Validation | None",
    w: World,
    seen: Seen,
    label: str,
    veto: bool,
    result: dict[str, Any] | Exception | None,
) -> list[dict[str, Any]]:
    assert result is not None
    if isinstance(result, Exception):
        raise _failed(e.name, by, result) from result
    bar = rule.VETO_Z if veto else seen.need
    row: dict[str, Any] = {"name": e.name, "by": by, "h0": e.h0, "label": label, "combine": e.combine or "max"}
    row |= _world(w, seen) | result | {"horizons": _horizons(result, bar), "slot": 3 if veto else 2}
    if veto:
        row["votes"] = rule.votes(row)
        row["passed"] = all(row["votes"].values())
    else:
        row["passed"] = math.isfinite(result.get("z", math.nan)) and result["z"] >= seen.need
    return [row | ({"validation": validation.record()} if validation is not None else {})]


def _world_row(
    effect: Effect, head: dict[str, Any], w: World, seen: Seen, result: dict[str, Any] | Exception | None
) -> list[dict[str, Any]]:
    assert result is not None
    if isinstance(result, Exception):
        raise _failed(f"the decisive effect {effect.name}", "decisive", result) from result
    row: dict[str, Any] = {"name": effect.name, "by": "decisive", "h0": "", "label": "y", "combine": "each"}
    row |= head | _world(w, seen) | result | {"horizons": _horizons(result, rule.VETO_Z), "slot": 3}
    row["votes"] = rule.votes(row)
    row["passed"] = all(row["votes"].values())
    return [row]


def _world(w: World, seen: Seen) -> dict[str, Any]:
    """Which world judged, the alternatives it answers, and what the alternative of an idea's world says."""
    texts = {a.id: a.text for a in seen.frozen.contract.alternatives}
    alternative = "; ".join(texts.get(x, "") for x in w.explains) if w.by != "engine" else ""
    return {"world": w.name, "explains": list(w.explains), "alternative": alternative}


def _joined(a: Found, b: Found) -> Found:
    for key, rows in b.shown.items():
        a.shown[key] = a.shown.get(key, []) + rows
    return a


def _calibrate(frozen: Frozen, s: FeatSettings) -> list[dict[str, Any]]:
    """Each decisive effect in the floor world, on its shortest horizon: a stand-in that carries only what the world
    keeps (each decision time's level, or for a shift in time each inst's) must die as often as chance allows, and a
    planted signal must live; for events both are events, at a share of the rows. A shift with fewer offsets than
    calibration shuffles draws each offset once."""
    floor = floor_world(frozen, s)
    assert frozen.contract.universe is not None
    events = EVENTS if frozen.contract.universe.shape == "events" else 0.0
    out = []
    for effect in frozen.contract.decisive:
        evaluate, _, _ = target(effect, frozen)
        h = min(evaluate, key=span)
        table, h_ns = frozen.tables[h]
        table = _thin(table, s.calibration_times)
        shuffle, distinct = worlds.shuffler(table, floor, h_ns)
        nulls = min(s.calibration_nulls, distinct or s.calibration_nulls)
        args = (evaluate[h], table, h_ns, floor.keeps, shuffle, s.calibration_reps, nulls, s.z_base)
        found = _safe_calibrate(*args, shifts=floor.shift is not None, events=events)
        out.append({"effect": effect.name, "world": floor.name, "horizon": h, **found})
    return out


def _safe_calibrate(*args: Any, **options: Any) -> dict[str, Any]:
    try:
        return calibrate(*args, **options)
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
    """Derive already blocks a target that ignores the scores; feat checks again on its own evaluation table, for
    events with rows left without a score."""
    evaluate, use, _ = target(effect, frozen)
    h, (table, _) = next(iter(frozen.tables.items()))
    assert frozen.contract.universe is not None
    missing = UNSCORED if frozen.contract.universe.shape == "events" else 0.0
    return pr.target_reads_score(use.name, lambda t: per_t(evaluate[h], t), table, missing)


def _scores(raw: Any, cands: pl.DataFrame) -> pl.DataFrame:
    frame = pl.from_pandas(raw) if isinstance(raw, pd.DataFrame) else raw
    if not isinstance(frame, pl.DataFrame):
        raise FeatError(f"features returned {type(raw).__name__}; {SHAPE}")
    missing = [c for c in SCORE_COLUMNS if c not in frame.columns]
    if missing:
        got = ", ".join(frame.columns) or "no columns"
        raise FeatError(f"features returned {got}, without {', '.join(missing)}; {SHAPE}")
    try:
        out = frame.select(
            pl.col("t").dt.convert_time_zone("UTC").cast(TIME),
            pl.col("inst").cast(pl.String),
            pl.col("score").cast(pl.Float64),
        )
    except pl.exceptions.PolarsError as exc:
        raise FeatError(
            f"features must return t as a timezone-aware datetime, inst as text, and score as a number: {exc}; {SHAPE}"
        ) from exc
    if out.select(pl.struct("t", "inst").is_duplicated().any()).item():
        raise FeatError("features return more than one score for some candidate at some time")
    return out.join(cands.select("t", "inst"), on=["t", "inst"], how="semi").sort("t", "inst")
