"""The steps a feat submission takes, around its default list (``stages/defaults/feat.py``).

``SCREEN`` refuses a features file that cannot be a trial before anything runs. ``FLOORS`` checks the scores: they
must rank the candidates and read nothing ahead of their time; a submission that fails one is TAINTED or FAILED and
nothing is judged. ``JUDGE`` holds the decisive test, the one bar, fixed by the contract: the contract's decisive
effect, computed with the submitted scores, must beat the engine's floor world (chance and timing: scores shuffled
across clusters at each decision time, the same way within a horizon) with ``z ≥ z_need``, which rises with the
submission count. Next to it the diagnostics, the default list and the idea's ``EVALS`` (``feat.py``; a case eval
only when its tests pass, an invalid one is shown as invalid and not judged), are judged against the floor world and
the idea's ``NULLS`` (``nulls/feat.py``) and shown, never deciding. Only the candidates the features gave a finite score
are judged, in every world alike. ``SHOWN`` says which world answers which alternative, and what the features cost.
Before the first submission is judged, the data check calibrates the decisive test in the floor world
(``calibration``).
"""

import math
import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

import pandas as pd
import polars as pl

from research_ci.api.contract import FLOOR_ALTERNATIVES, Contract, ContractError, Effect, Use, as_use, span
from research_ci.api.evals import FORMAT, Eval
from research_ci.api.plugins import CostModel, TargetPlugin
from research_ci.eval import protocol as pr
from research_ci.eval.calibration.stand_in import calibrate
from research_ci.kernel.time.clock import TIME, lit_time, span_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.judge import (
    Batched,
    EvalError,
    Evaluate,
    Index,
    Lane,
    Request,
    Sequential,
    Statistic,
    per_t,
)
from research_ci.kernel.verdict.permute import Layout, Permutation
from research_ci.runner import evals as ev
from research_ci.runner import research
from research_ci.runner.meter import Cost, measured
from research_ci.runner.plugins import Registry, load
from research_ci.stages.common import default_workers, iso, owner, quantile_times, target_use, units
from research_ci.stages.steps import DEFAULTS, Found, Step

FEAT_FILE = "feat.py"
NULLS_FILE = Path("nulls") / "feat.py"
EXAMPLE = Path("research") / "example.py"
SCORE_COLUMNS = ("t", "entity", "score")
NO_TRIAL = "; no submission was spent"
EXPLORED = "; nothing was recorded"
SHAPE = (
    "a features file defines features(fields, candidates), which gets polars DataFrames and returns a DataFrame "
    "(polars or pandas) or a polars LazyFrame with columns t, entity, score; research/example.py shows one"
)
EVALS_FORMAT = FORMAT["feat"]
Features = Callable[[pl.DataFrame, pl.DataFrame], Any]
Tables = dict[str, dict[str, tuple[pl.DataFrame, int]]]


class FeatError(ContractError):
    pass


@dataclass(frozen=True)
class FeatSettings:
    """The evaluation tables keep at most ``max_times`` decision times (every k-th when there are more). Calibrating
    the decisive effect runs on ``calibration_times`` of them: ``calibration_reps`` stand-ins and planted outputs, each
    against ``calibration_nulls`` shuffles. ``cuts`` decision times check that features read nothing ahead. The
    decisive test draws its shuffles ``floor_batch`` at a time, up to ``floor_reps``, until its decision is settled;
    a diagnostic takes ``shown_reps`` in the floor world, and as many as each null world of the idea says."""

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


@dataclass(frozen=True)
class World:
    """A null world as the engine runs it: scores shuffled within ``bucket`` (``None``: each decision time), whole
    ``clusters`` at a time, the same way within ``block``; it claims to keep what lives in ``keeps``. It judges the
    evals named in ``evals``, or every eval when empty."""

    name: str
    explains: tuple[str, ...]
    bucket: str | None
    clusters: str | None
    block: str | None
    keeps: str | None
    evals: tuple[str, ...]
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
    tables: Tables
    components: dict[str, list[str]]
    start: int
    end: int
    stride: int
    deps: dict[str, str]
    registry: Registry | None
    panel: str = ""
    exclusions: str = ""

    def inputs(self, cut: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """What a features file gets: the frozen data and the candidates, as polars DataFrames, copies that share
        their memory; with ``cut``, only the rows known by then and the candidates up to it."""
        if cut is None:
            return self.fields.clone(), self.candidates.clone()
        return (
            self.fields.filter(pl.col("available_at") <= lit_time(cut)),
            self.candidates.filter(pl.col("t") <= lit_time(cut)),
        )


@dataclass(frozen=True)
class Seen:
    """What a feat step reads: the frozen data, the settings, the idea's directory, the features file, and what a
    refusal says was not spent. Once the features ran: their function, their scores, and what the full call cost.
    For judging: the bar, the worlds, the evaluation tables with the scores, the idea's evals and their validations,
    and the name a default diagnostic is judged under."""

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

    def judge(self, evaluate: Evaluate) -> Found:
        """A default diagnostic, ``evaluate`` on the label of the first decisive effect, judged under ``name``."""
        e = Eval(evaluate, self.name, label=self.frozen.contract.decisive[0].on, combine="max")
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
    """The scores must rank the candidates, with two or more distinct values, at half the decision times or more."""
    assert seen.scores is not None
    cands = seen.frozen.candidates
    finite = seen.scores.filter(pl.col("score").is_finite())
    distinct = finite.group_by("t").agg(pl.col("score").n_unique().alias("k"))
    times = cands["t"].n_unique()
    share = distinct.filter(pl.col("k") >= 2).height / times if times else 0.0
    coverage = {"rows": finite.height, "share": finite.height / cands.height if cands.height else 0.0, "ranks": share}
    detail = f"scores {coverage['share']:.0%} of candidate rows; ranks candidates at {share:.0%} of decision times"
    return Found(
        [pr.Check("scores rank candidates", "fail" if share < 0.5 else "pass", detail)], shown={"scores": coverage}
    )


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
            mismatch = pr._frame_mismatch(before, cut_scores, ["t", "entity"], ["score"])
            if mismatch:
                problems.append(f"{iso(cut)}: {mismatch}")
    leak = bool(problems)
    detail = "; ".join(problems) if leak else f"the same scores with the data cut at {len(cuts)} decision times"
    return Found([pr.Check("scores read nothing ahead", "fail" if leak else "pass", detail)], shown={"cuts": seconds})


def decisive_test(seen: Seen) -> Found:
    """The decisive effect, with the submitted scores, against the floor world: the family's z over the horizons must
    reach the bar. Shuffles are drawn a batch at a time until the decision is settled (``kernel/verdict/sequential.py``)
    or all are drawn. The effect shown is the gain over random, the target with these scores less its mean in the
    floor world, at the best counted horizon."""
    s = seen.settings
    floor = floor_world(seen.frozen, s)
    found = Found(shown={"decisive": []})
    for effect in seen.frozen.contract.decisive:
        evaluate, use, head = target(effect, seen.frozen)
        rule = Sequential(s.floor_batch, seen.need)
        request = Request(evaluate, lanes(seen.scored[effect.on], floor), s.floor_reps, s.min_blocks, "max", rule)
        found.shown["decisive"].append(Pending(request, partial(_decided, use, head, seen.need)))
    return found


def idea_evals(seen: Seen) -> Found:
    """The idea's ``EVALS``, judged like the diagnostics of the default list; one named like a default diagnostic is
    judged in its place instead. A case eval runs only when it passed its tests: an invalid one is shown with the test
    it failed, what it expected and got, and how to fix it, and with no numbers."""
    found = Found(shown={"evals": []})
    for e in seen.evals:
        found.shown["evals"] += idea_eval(seen, e).shown["evals"]
    return found


def idea_eval(seen: Seen, e: Eval) -> Found:
    by = ev.source_of(e.of)
    v = seen.validated.get(e.name, ev.Validation("trusted"))
    if v.status != "invalid":
        return judged(seen, e, by, v)
    row = {"eval": e.name, "by": by, "validation": v.record(), "invalid": True, "fix": v.fix(), "passed": False}
    return Found(shown={"evals": [Pending(None, lambda _: [row])]})


def alternatives(seen: Seen) -> Found:
    """Which world answers which alternative. One that no world answers is a warning, not a failure: only the
    decisive test decides, and the worlds of the idea judge only diagnostics."""
    c = seen.frozen.contract
    lines, missing = [], []
    for alt in [*FLOOR_ALTERNATIVES, *(a.id for a in c.alternatives)]:
        kept_by = [w.name for w in seen.worlds if alt in w.explains]
        lines.append(f"{alt}: {', '.join(kept_by) or 'none'}")
        missing += [] if kept_by else [alt]
    if missing:
        check = pr.Check("alternatives covered", "warn", f"{' · '.join(lines)}; no world answers {', '.join(missing)}")
    else:
        check = pr.Check("alternatives covered", "pass", " · ".join(lines))
    return Found(shown={"coverage": asdict(check)})


def cost(seen: Seen) -> Found:
    """What the full features call cost, its wall time and peak memory; with the seconds of the calls on cut data,
    shown, never judged."""
    assert seen.cost is not None
    return Found(shown={"cost": asdict(seen.cost)})


SCREEN = (Step("screen", "blocks", screen_file, "Refuses a missing, example, or unparsable file"),)
FLOORS = (
    Step("scores rank candidates", "blocks", ranks, "Scores rank candidates at half the times"),
    Step("scores read nothing ahead", "blocks", nothing_ahead, "Same scores with the data cut early"),
)
JUDGE = (
    Step("decisive test", "decides", decisive_test, "The decisive effect beats the floor world: z ≥ z_need"),
    DEFAULTS,
    Step("the idea's evals", "shown", idea_evals, "Each valid idea eval, judged like a diagnostic", "feat.py"),
)
SHOWN = (
    Step("alternatives covered", "warns", alternatives, "Every alternative of the contract has a world"),
    Step("cost", "shown", cost, "Wall time and peak memory of the features call"),
)
STEPS = (*SCREEN, *FLOORS, *JUDGE, *SHOWN)


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
    """An eval judged in the floor world and in every null world that judges it; shown, never deciding. Worlds that
    shuffle alike are judged once and shown under each name."""
    label = label_of(e, seen.frozen)
    tables = {h: seen.scored[label][h] for h in horizons_of(e, seen.frozen)}
    alike: dict[tuple[Any, ...], Request] = {}
    found = Found(shown={"evals": []})
    for w in seen.worlds:
        if w.evals and e.name not in w.evals:
            continue
        key = (w.bucket, w.clusters, w.block, w.reps)
        if key not in alike:
            alike[key] = Request(e.of, lanes(tables, w), w.reps, seen.settings.min_blocks, e.combine or "max")
        found.shown["evals"].append(Pending(alike[key], partial(_shown_row, e, by, validation, w, seen.need)))
    return found


def lanes(tables: dict[str, tuple[pl.DataFrame, int]], w: World) -> dict[str, Lane]:
    """The world's shuffle of each horizon's table; two evals or worlds on the same table and layout share it."""
    out = {}
    for h, (table, h_ns) in tables.items():
        layout = Layout(w.bucket, w.clusters, w.block_ns(h_ns))
        out[h] = Lane(table, h_ns, (id(table), layout), partial(_shuffle, table, layout))
    return out


def label_of(e: Eval, frozen: Frozen) -> str:
    names = [label.name for label in frozen.contract.labels]
    if not names:
        raise FeatError("the contract declares no labels; evals need one")
    label = e.label or names[0]
    if label not in names:
        raise FeatError(f"eval {e.name}: no label {label!r} in the contract ({', '.join(names)}){NO_TRIAL}")
    return label


def horizons_of(e: Eval, frozen: Frozen) -> tuple[str, ...]:
    label = label_of(e, frozen)
    declared = next(lab.horizons for lab in frozen.contract.labels if lab.name == label)
    if e.horizons is None:
        return declared
    unknown = [h for h in e.horizons if h not in declared]
    if unknown:
        raise FeatError(f"eval {e.name}: horizons {', '.join(unknown)} are not among {', '.join(declared)}{NO_TRIAL}")
    return tuple(e.horizons)


def floor_world(frozen: Frozen, s: FeatSettings) -> World:
    assert frozen.contract.universe is not None
    return World(
        name="floor",
        explains=FLOOR_ALTERNATIVES,
        bucket=None,
        clusters=frozen.contract.universe.clusters,
        block="horizon",
        keeps=None,
        evals=(),
        reps=s.shown_reps,
        by="engine",
        note="scores shuffled across clusters at each decision time, the same way within a horizon",
    )


def target(effect: Effect, frozen: Frozen) -> tuple[dict[str, Evaluate], Use, dict[str, Any]]:
    """The effect's target per horizon of its label, the plugin it names, and the effect as cards show it."""
    c = frozen.contract
    use = target_use(effect, c)
    plugin: TargetPlugin = load(use, frozen.registry)
    costs: CostModel = load(as_use(c.costs), frozen.registry)
    parts = frozen.components.get(effect.on, [])
    back = {"y": "value"} | {f"y_{p}": p for p in parts}

    batched = getattr(plugin, "batched", None) if owner(plugin, "batched") is owner(plugin, "reduce") else None

    def make(h: str) -> Evaluate:
        horizon = span(h)
        form = cast(Batched | None, batched(horizon, costs)) if callable(batched) else None
        return Statistic(
            lambda table: plugin.reduce(table.rename(back), horizon, costs),
            None if form is None else _renamed(form, back),
        )

    return {h: make(h) for h in frozen.tables[effect.on]}, use, units(effect, plugin)


def calibration(frozen: Frozen, s: FeatSettings) -> dict[str, Any]:
    """The decisive effect in the floor world before anything is judged: its target must read the scores, a stand-in
    that carries only each decision time's level must die as often as chance allows, and a planted signal must live.
    ``ok`` is false, with ``notes`` saying why, when any of them does not hold."""
    started = time.monotonic()
    checks = [_reads_score(e, frozen) for e in frozen.contract.decisive]
    effects = {e.on: target(e, frozen)[2] for e in frozen.contract.decisive}
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
    fields, cands = frozen.inputs(cut)

    def call() -> Any:
        raw = features(fields, cands)
        return raw.collect() if isinstance(raw, pl.LazyFrame) else raw

    try:
        raw, spent = measured(call, memory=cut is None)
    except Exception as exc:
        raise FeatError(
            f"{path.name}: features(fields, candidates) failed: {type(exc).__name__}: {exc}{after}"
        ) from exc
    asked = frozen.candidates if cut is None else frozen.candidates.filter(pl.col("t") <= lit_time(cut))
    try:
        return _scores(raw, asked), spent
    except FeatError as exc:
        raise FeatError(f"{exc}{after}") from exc


def _renamed(form: Batched, back: dict[str, str]) -> Batched:
    return lambda table: form(table.rename(back))


def _shuffle(table: pl.DataFrame, layout: Layout) -> Index:
    return Permutation(table, layout.bucket, layout.clusters, layout.block_ns).index


def _decided(
    use: Use, head: dict[str, Any], need: float, result: dict[str, Any] | Exception | None
) -> list[dict[str, Any]]:
    if isinstance(result, Exception):
        raise result
    assert result is not None
    return [_outcome(use, head, result, need)]


def _outcome(use: Use, head: dict[str, Any], result: dict[str, Any], need: float) -> dict[str, Any]:
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
        "shuffles": result["shuffles"],
        "outcome": "passed" if math.isfinite(z) and z >= need else "failed",
    }


def _shown_row(
    e: Eval,
    by: str,
    validation: "ev.Validation | None",
    w: World,
    need: float,
    result: dict[str, Any] | Exception | None,
) -> list[dict[str, Any]]:
    assert result is not None
    if isinstance(result, Exception):
        what = str(result) if isinstance(result, EvalError) else f"{type(result).__name__}: {result}"
        tested = "; it passed its tests, so add a case like this table to them" if by == "case" else ""
        raise FeatError(f"eval {e.name!r} failed: {what}{tested}; {EVALS_FORMAT}{NO_TRIAL}") from result
    passed = math.isfinite(result.get("z", math.nan)) and result["z"] >= need
    row = {"eval": e.name, "by": by, "world": w.name, "explains": list(w.explains), **result, "passed": passed}
    return [row | ({"validation": validation.record()} if validation is not None and by != "default" else {})]


def _calibrate(frozen: Frozen, s: FeatSettings) -> list[dict[str, Any]]:
    """Each decisive effect in the floor world, on its shortest horizon: a stand-in that carries only each decision
    time's level must die as often as chance allows, and a planted signal must live."""
    floor = floor_world(frozen, s)
    out = []
    for effect in frozen.contract.decisive:
        evaluate, _, _ = target(effect, frozen)
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
    evaluate, use, _ = target(effect, frozen)
    h = next(iter(frozen.tables[effect.on]))
    table, _ = frozen.tables[effect.on][h]
    return pr.target_reads_score(use.name, lambda t: per_t(evaluate[h], t), table)


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
            pl.col("entity").cast(pl.String),
            pl.col("score").cast(pl.Float64),
        )
    except pl.exceptions.PolarsError as exc:
        raise FeatError(
            f"features must return t as a timezone-aware datetime, entity as text, and score as a number: {exc}; "
            f"{SHAPE}"
        ) from exc
    if out.select(pl.struct("t", "entity").is_duplicated().any()).item():
        raise FeatError("features return more than one score for some candidate at some time")
    return out.join(cands.select("t", "entity"), on=["t", "entity"], how="semi").sort("t", "entity")
