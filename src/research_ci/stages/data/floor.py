"""The steps the data check takes on every idea's data, around its default list (``stages/data/defaults.py``).

``FIND`` finds what to treat: the warning the idea's research files gave, the default list, then the idea's own evals
(``EVALS`` in ``data.py``), each validated before it runs: a case eval by its tests, and every eval that claims to
cover a kind of problem by the pits of that kind it must catch; an invalid one does not run and blocks the check.
Every problem they find becomes a treatment that is applied by default, or an item a person decides. ``WEIGH`` works on
the data with every treatment found: the field grades, each treatment's influence on the contract's decisive number in
units of the smallest effect the data can detect (MDE), and the power before and after. Below the threshold a
treatment is applied and logged. A treatment that changes identities (a split) is weighed against the untreated data,
and every other one on top of the splits that are applied. Only differences are reported, never the level of the
number. ``LAST`` runs once nothing blocks: the decisive test, calibrated on the treated data.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from research_ci.api.contract import Contract, ContractError, span
from research_ci.api.data import Pit
from research_ci.api.evals import FORMAT, Eval
from research_ci.eval.data.pits import BUILTIN
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import span_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.check import Check
from research_ci.runner.plugins import Registry
from research_ci.stages.data import treatments as tr
from research_ci.stages.data.source import SourceSettings
from research_ci.stages.feat.floor import FeatSettings, Frozen, calibration
from research_ci.stages.skeleton import DEFAULTS, Found, Step
from research_ci.validation import load as ev

ALL = "all applied"
SPLITS = "splits applied"
RAW = ""
EVALS_FORMAT = FORMAT["data"]
FLAGS = ("entity", "event_time")


class DataError(ContractError):
    pass


@dataclass(frozen=True)
class DataSettings:
    source: SourceSettings = field(default_factory=SourceSettings)
    materiality: float = 0.5
    seeds: int = 2


@dataclass(frozen=True)
class Decisive:
    """The label rows of the contract's decisive numbers, keyed by label and horizon, such as ``carry 24h``:
    ``rows(treatments, columns)`` stacks every key's rows as the treatments leave them, with ``key``, ``h_ns`` (the
    horizon in ns), ``t`` and the named columns of the label or the candidates."""

    rows: Callable[[Sequence[tr.Treatment], Sequence[str]], pl.DataFrame]


@dataclass(frozen=True)
class Seen:
    """What a check reads: the contract, the explore data, the candidates (``None`` without a universe), the data and
    universe plugins, the source's admission, when the newest row became known, and now (both in ns); the decisive
    numbers (``None`` without a universe), and the treatments the checks before it found that will be applied.

    The floor's steps read more: the settings; the warning the idea's research files gave when they were read; the
    idea's directory, its evals and pits, and the categories of the built-in pits; what the steps before showed
    (``earlier``); the decisive numbers as the floor weighs them; every treatment found, kept ones too, and the treated
    data, once every treatment is found; and how to read the treated data as feat does, and what a calibration of the
    decisive test on it rests on."""

    contract: Contract
    fields: pl.DataFrame
    candidates: pl.DataFrame | None
    source: Any
    universe: Any
    admission: dict[str, Any]
    latest: int
    now: int
    decisive: Decisive | None = None
    prior: tuple[tr.Treatment, ...] = ()
    settings: DataSettings = field(default_factory=DataSettings)
    research: str = ""
    root: Path | None = None
    registry: Registry | None = None
    evals: tuple[Eval, ...] = ()
    pits: tuple[Pit, ...] = ()
    builtin: tuple[str, ...] = ()
    earlier: Mapping[str, Any] = field(default_factory=dict)
    numbers: tr.Numbers | None = None
    found: tuple[tr.Treatment, ...] = ()
    panel: pl.DataFrame | None = None
    frozen: Callable[[], Frozen] | None = None
    key: Callable[[Frozen], dict[str, str]] | None = None
    feat: FeatSettings | None = None


def row_loops(seen: Seen) -> Found:
    """The warning the idea's ``data.py`` and ``nulls/data.py``, and the research modules they import, gave when the
    data check read them, before anything else ran: every place they go row by row, with its file and line."""
    return Found(warnings=[seen.research] if seen.research else [])


def validated(seen: Seen) -> Found:
    """Each of the idea's ``EVALS`` before it runs: an eval of the engine's library is trusted; a case eval runs its
    tests, the cases and properties of its module that run it (kept by the hash of its files, so an unchanged one is
    not run again). An eval that claims to cover a kind of problem in a field (``covers``) must flag every pit of that
    kind, built in (the categories ``PITS`` in ``stages/data/defaults.py`` lists) or from the idea's own ``PITS``,
    planted in a clean copy of the data, that moves the decisive number by the materiality threshold; the pits are
    tests of the eval too. An eval that fails a test is invalid: it does not run, and the check is blocked."""
    if not seen.evals:
        return Found(shown={"validated": {}})
    assert seen.root is not None
    try:
        found = ev.validate(seen.evals, seen.root, seen.registry)
    except ev.EvalsError as exc:
        raise DataError(f"{exc}; nothing was recorded") from exc
    covering = [e for e in seen.evals if e.covers]
    numbers = seen.numbers
    if covering and numbers is not None:
        mde = _pit_resolution(numbers, seen.fields, seen.prior)
        found |= {e.name: found[e.name].with_tests(_pits(seen, numbers, e, mde)) for e in covering}
    elif covering:
        skipped = Check("pits", "skip", "not planted: the contract names no universe, so there is no decisive number")
        found |= {e.name: found[e.name].with_tests([skipped]) for e in covering}
    return Found(shown={"validated": found})


def idea_evals(seen: Seen) -> Found:
    """Each of the idea's ``EVALS`` that its validation let through, on the explore data: the rows it flags are
    masked, their entities dropped, or a person asked, as its ``on_fail`` says. An invalid one is a failed check that
    says which test failed, what it expected and got, and how to fix it; it flags nothing."""
    validations: Mapping[str, ev.Validation] = seen.earlier.get("validated", {})
    found = Found(shown={"evals": []})
    for e in seen.evals:
        v, by = validations[e.name], ev.source_of(e.of)
        row: dict[str, Any] = {"name": e.name, "by": by, "covers": [list(c) for c in e.covers], "on_fail": e.on_fail}
        row |= {"validation": v.record(), "flagged": None, "note": e.note}
        found.shown["evals"].append(row)
        if v.status == "invalid":
            found.checks.append(Check(e.name, "fail", f"{v.summary()}; it did not run; {v.fix()}", by))
            continue
        flagged = _flagged(e, seen.fields)
        row["flagged"] = flagged.height
        detail = f"{v.summary()} · flagged {flagged.height:,} rows"
        if not flagged.height:
            found.checks.append(Check(e.name, "pass", detail, by))
        elif e.on_fail == "mask":
            reason = f"{flagged.height:,} rows flagged"
            found.treatments.append(tr.mask_rows(f"eval {e.name}: mask", reason, flagged, seen.fields))
            found.checks.append(Check(e.name, "warn", f"{detail}; masked", by))
        elif e.on_fail == "drop_entities":
            ents = flagged["entity"].unique().to_list()
            found.treatments.append(
                tr.drop_entities(f"eval {e.name}: drop", f"{len(ents)} entities", ents, seen.fields)
            )
            found.checks.append(Check(e.name, "warn", f"{detail}; their {len(ents)} entities dropped", by))
        else:
            found.decide.append(f"eval {e.name} flagged {flagged.height:,} rows")
            found.checks.append(Check(e.name, "warn", f"{detail}; a person decides", by))
    return found


def _flagged(e: Eval, fields: pl.DataFrame) -> pl.DataFrame:
    """The rows an eval flags, or a refusal that says what it must return."""
    try:
        out = e.of(fields)
        out = out.collect() if isinstance(out, pl.LazyFrame) else out
    except Exception as exc:
        raise DataError(
            f"eval {e.name!r} failed on the explore data: {type(exc).__name__}: {exc}; it passed its tests, so add a "
            f"case like this data to them and fix it; {EVALS_FORMAT}"
        ) from exc
    if not isinstance(out, pl.DataFrame) or any(c not in out.columns for c in FLAGS):
        got = ", ".join(out.columns) if isinstance(out, pl.DataFrame) else type(out).__name__
        raise DataError(f"eval {e.name!r} returned {got}, not the rows it flags; {EVALS_FORMAT}")
    return out


def field_grades(seen: Seen) -> Found:
    """The decisive number may read only fields the independent source verified; one that reads another field makes
    the history explore-grade. A field it reads is one whose absence moves it."""
    grades = seen.admission["grades"]
    present = sorted(seen.fields["field"].unique().to_list())
    shown: dict[str, Any] = {"grades": {f: grades.get(f, "unverified") for f in present}}
    weak = [f for f in present if shown["grades"][f] != "verified"]
    if not weak:
        return Found([Check("field grades", "pass", "every field agrees with an independent source")], shown=shown)
    numbers = seen.numbers
    if numbers is None:
        used = weak
    else:
        used = [f for f in weak if _moves(numbers.recompute, numbers.base(), seen.fields.filter(pl.col("field") != f))]
    shown["capped_by"] = used
    if used:
        detail = f"the decisive number reads {', '.join(used)}, not verified; explore-grade"
        return Found([Check("field grades", "warn", detail)], shown=shown)
    return Found([Check("field grades", "pass", f"unverified fields are not read: {', '.join(weak)}")], shown=shown)


def influence(seen: Seen) -> Found:
    """Each treatment's influence on the decisive number, and that of the ones applied together, in MDE; kept
    treatments are measured too. A split is weighed against the untreated data, and every other treatment on top of
    the splits that are applied: a series the split renames is not the other treatment's doing. These numbers per
    decision time come from one stacked frame, and every gap from one group_by. The ones applied together are computed
    again on the treated data, labels and all; that gives the power after."""
    applied, ts = list(seen.prior), list(seen.found)
    kept = {t.name for t in ts} - {t.name for t in applied}
    numbers = seen.numbers
    if numbers is None:
        rows = [
            {"name": t.name, "on": t.on, "reason": t.reason, "rows": t.rows, "influence": None, "kept": t.name in kept}
            for t in ts
        ]
        return Found(shown={"treatments": rows})
    assert seen.panel is not None
    splits = [t for t in applied if tr.renames_rows(t)]
    sets = {t.name: [t] if tr.renames_rows(t) else [*splits, t] for t in ts}
    if len(splits) > 1:
        sets[SPLITS] = splits
    on_top = SPLITS if len(splits) > 1 else splits[0].name if splits else RAW
    against = {t.name: RAW if tr.renames_rows(t) else on_top for t in ts}
    frames = [numbers.base_frame().select(pl.lit(RAW).alias("set"), "key", "t", "value"), numbers.per_set(sets)]
    numbers.tables.clear()
    after = numbers.treated(seen.panel, applied)
    together = len(applied) > 1 or bool(applied and kept)
    if together:
        frames.append(_stacked(ALL, after))
        against[ALL] = RAW
    mde = numbers.resolution(seen.panel, applied)
    gaps = _gaps(pl.concat(frames), against)
    rows = []
    for t in ts:
        head = {"name": t.name, "on": t.on, "reason": t.reason, "rows": t.rows, "kept": t.name in kept}
        rows.append(head | _shift(gaps, t.name, mde))
    if together:
        rows.append({"name": ALL, "on": "both", "reason": "", "rows": 0, "kept": False} | _shift(gaps, ALL, mde))
    return Found(shown={"treatments": rows})


def power(seen: Seen) -> Found:
    """The MDE of each decisive effect at each horizon, before and after the treatments that are applied."""
    numbers = seen.numbers
    if numbers is None:
        return Found(shown={"power": []})
    assert seen.panel is not None
    after = numbers.treated(seen.panel, list(seen.prior))
    base = numbers.base()
    rows = []
    for effect in seen.contract.decisive:
        label = next(lab for lab in seen.contract.labels if lab.name == effect.on)
        for h in label.horizons:
            k, hn = f"{effect.on} {h}", span_ns(span(h))
            before = pw.power(tr.se(base[k], hn), numbers.width, hn)
            now = pw.power(tr.se(after[k], hn), numbers.width, hn)
            rows.append(
                {"on": effect.on, "horizon": h, "mde_before": before.mde, "mde": now.mde, "decisions": after[k].height}
            )
    return Found(shown={"power": rows})


def decisive_calibration(seen: Seen) -> Found:
    """The decisive test in the floor world on the treated data, before any features exist: the contract's decisive
    target must read the scores, a stand-in that carries only each decision time's level must die as often as chance
    allows, and a planted signal must live. A calibration that fails blocks the data check; feat reuses one that holds
    while nothing it rests on changes."""
    if seen.frozen is None or seen.key is None or seen.feat is None:
        return Found()
    try:
        frozen = seen.frozen()
    except ContractError as exc:
        return Found(shown={"calibration": {"ok": False, "notes": [str(exc)], "rows": [], "checks": [], "key": {}}})
    return Found(shown={"calibration": calibration(frozen, seen.feat) | {"key": seen.key(frozen)}})


FIND = (
    Step("row-by-row code", "warns", row_loops, "Row-by-row code in the idea's data files", "data.py, nulls/data.py"),
    DEFAULTS,
    Step("eval validation", "blocks", validated, "Case evals pass their tests; covered pits are caught", "data.py"),
    Step("the idea's evals", "treats", idea_evals, "Masks, drops, or asks about what a valid eval flags", "data.py"),
)
WEIGH = (
    Step("field grades", "warns", field_grades, "The decisive number reads only verified fields"),
    Step("treatment influence", "warns", influence, "Each treatment's move of the decisive number, in MDE"),
    Step("power", "shown", power, "MDE per horizon, before and after the treatments"),
)
LAST = (Step("decisive test calibration", "blocks", decisive_calibration, "Stand-ins die and planted signals live"),)
STEPS = (*FIND, *WEIGH, *LAST)


def _pit_resolution(numbers: tr.Numbers, fields: pl.DataFrame, prior: Sequence[tr.Treatment]) -> dict[str, float]:
    """The MDE pits are weighed in: the finer of the MDE on the explore data and on the data with the treatments of
    the default list applied."""
    base = {k: tr.mde(v, k, numbers.width) for k, v in numbers.base().items()}
    if not prior:
        return base
    after = numbers.recompute(tr.apply_all(prior, fields), prior)
    return {k: tr.smaller(b, tr.mde(after[k], k, numbers.width) if k in after else math.nan) for k, b in base.items()}


def _pits(seen: Seen, numbers: tr.Numbers, e: Eval, mde: dict[str, float]) -> list[Check]:
    """Every pit of the kinds ``e`` covers, planted in a copy of the data without the entities it flags, with two seeds
    (``seeds``): each must be flagged, unless it moves the decisive number by less than the threshold."""
    flagged = set(_flagged(e, seen.fields)["entity"].unique().to_list())
    clean = seen.fields.filter(~pl.col("entity").is_in(sorted(flagged)))
    base = numbers.recompute(clean)
    out: list[Check] = []
    threshold = seen.settings.materiality
    for category, name in e.covers:
        plants = [(f"built-in {category}", BUILTIN[category])] if category in seen.builtin else []
        plants += [
            (f"pit {p.category}: {p.note}" if p.note else f"pit {p.category}", p.plant)
            for p in seen.pits
            if p.category == category
        ]
        if not plants:
            kinds = ", ".join(seen.builtin) or "none"
            detail = (
                f"no pit plants {category}, so the claim cannot be checked; the built-in kinds are {kinds}; add a "
                f"Pit({category!r}, plant) to PITS in nulls/data.py, or drop the claim from covers"
            )
            out.append(Check(f"pits: {category} in {name}", "fail", detail))
            continue
        for label, plant in plants:
            for seed in range(seen.settings.seeds):
                test = f"{label} in {name}, seed {seed}"
                try:
                    mutated = field_table(plant(clean, name, np.random.default_rng(seed)))
                except (ValueError, pl.exceptions.PolarsError) as exc:
                    out.append(Check(test, "skip", f"could not be planted: {exc}"))
                    continue
                moved = numbers.recompute(mutated).items()
                gaps = [abs(_mean_gap(base[k], v)) / mde[k] for k, v in moved if mde.get(k, math.nan) > 0]
                harm = max((g for g in gaps if math.isfinite(g)), default=0.0)
                if _flagged(e, mutated).height > 0:
                    out.append(Check(test, "pass", "caught"))
                elif harm >= threshold:
                    moves = f"it moves the decisive number {harm:.2f} MDE ≥ {threshold:g}"
                    out.append(Check(test, "fail", f"expected it flagged, got nothing: {moves}"))
                else:
                    out.append(Check(test, "pass", f"not flagged, and harmless: it moves the number {harm:.2f} MDE"))
    return out


def _shift(gaps: pl.DataFrame, name: str, mde: dict[str, float]) -> dict[str, Any]:
    found = dict(gaps.filter(pl.col("set") == name).select("key", "gap").iter_rows())
    out = {}
    for k in mde:
        d = found.get(k, math.nan)
        out[k] = {"bp": abs(d) * 1e4, "mde": abs(d) / mde[k] if mde[k] else math.nan}
        if math.isinf(d):
            out[k]["unbounded"] = True
    largest = max((v["mde"] for v in out.values() if not math.isnan(v["mde"])), default=math.nan)
    return {"influence": out, "largest": largest}


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


def _gaps(frame: pl.DataFrame, against: Mapping[str, str]) -> pl.DataFrame:
    """The mean gap per decision time between each set's number and that of the set it is measured ``against``, per
    key, from the stacked numbers of all of them: ``set``, ``key``, ``gap``."""
    pairs = pl.DataFrame(
        {"set": list(against), "ref": list(against.values())}, schema={"set": pl.String, "ref": pl.String}
    )
    ref = frame.rename({"set": "ref", "value": "value_ref"})
    joined = frame.join(pairs, on="set").join(ref, on=["ref", "key", "t"], how="inner")
    return joined.group_by("set", "key").agg((pl.col("value") - pl.col("value_ref")).mean().alias("gap"))


def _stacked(name: str, by_key: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(
        [v.select(pl.lit(name).alias("set"), pl.lit(k).alias("key"), "t", "value") for k, v in by_key.items()]
    )
