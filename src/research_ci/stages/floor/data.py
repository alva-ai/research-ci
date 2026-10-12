"""The steps the data check takes on every idea's data, around its default list (``stages/defaults/data.py``).

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
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import Contract, ContractError, span
from research_ci.api.data import Pit
from research_ci.api.evals import FORMAT, Eval
from research_ci.api.plugins import CostModel, LabelPlugin, TargetPlugin
from research_ci.eval.data import influence as inf
from research_ci.eval.data import treatments as tr
from research_ci.eval.data.pits import BUILTIN
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import span_ns
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.check import Check
from research_ci.runner import evals as ev
from research_ci.runner.plugins import Registry
from research_ci.stages.common import label_values, per_t
from research_ci.stages.floor.feat import FeatSettings, Frozen, calibration
from research_ci.stages.floor.source import SourceSettings
from research_ci.stages.steps import DEFAULTS, Found, Step

LEGS = ("entity", "long", "short")
ALL = "all applied"
SPLITS = "splits applied"
RAW = ""
PER_T = {"t": pl.Datetime("ns", "UTC"), "value": pl.Float64}
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
    numbers: "Numbers | None" = None
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
    kind, built in (the categories ``PITS`` in ``stages/defaults/data.py`` lists) or from the idea's own ``PITS``,
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
    splits = [t for t in applied if _renames_rows(t)]
    sets = {t.name: [t] if _renames_rows(t) else [*splits, t] for t in ts}
    if len(splits) > 1:
        sets[SPLITS] = splits
    on_top = SPLITS if len(splits) > 1 else splits[0].name if splits else RAW
    against = {t.name: RAW if _renames_rows(t) else on_top for t in ts}
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
            before = pw.power(_se(base[k], hn), numbers.width, hn)
            now = pw.power(_se(after[k], hn), numbers.width, hn)
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


@dataclass(frozen=True)
class Plan:
    key: str
    values: LabelPlugin
    target: TargetPlugin
    horizon: pd.Timedelta


@dataclass
class _Table:
    """Every key's label rows on one version of the data, the field table ``renames`` leave: the candidates with their
    row (``_c``), the label columns of all keys stacked with the candidate row and the key (``_k``) and their own row
    (``_r``), each key's label columns, and the label rows each treatment touches."""

    renames: tuple[tr.Treatment, ...]
    fields: pl.DataFrame
    cands: pl.DataFrame
    labels: pl.DataFrame
    columns: list[list[str]]
    hits: dict[str, pl.Series] = field(default_factory=dict)


class Numbers:
    """The decisive numbers. Their label rows are computed once on the explore data and, when treatments rename rows
    (a split), once more on the data those leave. A treatment that drops field rows or candidates leaves out the label
    rows that read what it drops (``eval/data/influence.py``), so weighing it runs only the target again. ``recompute``
    computes everything again, labels and all: for the treated data, and for changes no treatment describes, such as
    a field left out or a planted pit. ``width`` is the explore data's span in ns."""

    def __init__(
        self,
        fields: pl.DataFrame,
        cands: pl.DataFrame,
        universe: Callable[[pl.DataFrame], pl.DataFrame],
        plans: list[Plan],
        costs: CostModel,
        width: int,
    ) -> None:
        self.fields, self.cands, self.universe, self.plans, self.costs = fields, cands, universe, plans, costs
        self.width = width
        self.keys = [p.key for p in plans]
        self.key_table = pl.DataFrame(
            {
                "_k": list(range(len(plans))),
                "key": self.keys,
                "h_ns": [int(p.horizon.value) for p in plans],
            },
            schema={"_k": pl.UInt32, "key": pl.Enum(self.keys), "h_ns": pl.Int64},
        )
        self.tables: dict[tuple[str, ...], _Table] = {}
        self._base: pl.DataFrame | None = None
        self._treated: dict[str, pl.DataFrame] | None = None
        self._resolution: dict[str, float] | None = None

    def rows(self, ts: Sequence[tr.Treatment], columns: Sequence[str]) -> pl.DataFrame:
        """Every key's label rows after ``ts``, stacked: ``key``, ``h_ns`` (the horizon in ns), ``t`` and ``columns``,
        from the label or the candidates; text columns of the candidates come as categoricals."""
        table = self._table(ts)
        dropped = self._dropped(table, ts)
        own = [c for c in columns if c in table.labels.columns]
        theirs = [c for c in columns if c not in own and c != "t"]
        labels = table.labels.select("_r", "_c", "_k", *own)
        labels = labels.filter(~pl.col("_r").is_in(dropped.implode())) if dropped.len() else labels
        text = [c for c in theirs if table.cands.schema[c] == pl.String]
        cand = table.cands.select("t", *theirs).with_columns(pl.col(text).cast(pl.Categorical))
        cand = cand.select(pl.all().gather(labels["_c"]))
        k = self.key_table
        out = (
            labels.select("_k", *own)
            .hstack(cand)
            .with_columns(
                key=pl.col("_k").replace_strict(k["_k"], k["key"], return_dtype=k.schema["key"]),
                h_ns=pl.col("_k").replace_strict(k["_k"], k["h_ns"], return_dtype=pl.Int64),
            )
        )
        return out.select("key", "h_ns", "t", *columns)

    def base(self) -> dict[str, pl.DataFrame]:
        return _by_key(self.base_frame(), self.keys)

    def base_frame(self) -> pl.DataFrame:
        """Every key's number per decision time on the untreated data, stacked: ``key``, ``t``, ``value``."""
        if self._base is None:
            table = self._raw_table()
            out = [pl.DataFrame(schema={"key": pl.String, **PER_T})]
            out += [
                self._reduced(k, self._joined(table, k)).select("key", "t", "value") for k in range(len(self.plans))
            ]
            self._base = pl.concat(out)
        return self._base

    def treated(self, panel: pl.DataFrame, applied: Sequence[tr.Treatment]) -> dict[str, pl.DataFrame]:
        """The numbers on the treated data ``panel``, labels and all, with the candidate treatments of ``applied``;
        computed once."""
        if self._treated is None:
            self._treated = self.recompute(panel, applied) if applied else self.base()
        return self._treated

    def resolution(self, panel: pl.DataFrame, applied: Sequence[tr.Treatment]) -> dict[str, float]:
        """The MDE of each key a treatment's influence is measured in: the finer of the MDE before and after the
        treatments, since a treatment matters if it moves the number by more than the verdict on the treated data, or
        on the untreated data, could resolve."""
        if self._resolution is None:
            after = self.treated(panel, applied)
            before = {k: _mde(v, k, self.width) for k, v in self.base().items()}
            now = {k: _mde(v, k, self.width) for k, v in after.items()}
            self._resolution = {k: _smaller(before[k], now.get(k, math.nan)) for k in before}
        return self._resolution

    def per_set(self, sets: Mapping[str, Sequence[tr.Treatment]]) -> pl.DataFrame:
        """The number per decision time of every key after each named set of treatments, stacked: ``set``, ``key``,
        ``t``, ``value``. The target runs once per set and key; the label rows come from the tables."""
        out = [pl.DataFrame(schema={"set": pl.String, "key": pl.String, **PER_T})]
        by_table: dict[tuple[str, ...], list[str]] = {}
        for name, ts in sets.items():
            by_table.setdefault(_renames(ts), []).append(name)
        for names in by_table.values():
            table = self._table(sets[names[0]])
            dropped = {name: self._dropped(table, sets[name]) for name in names}
            for k in range(len(self.plans)):
                joined = self._joined(table, k)
                for name in names:
                    d = dropped[name]
                    rows = joined.filter(~pl.col("_r").is_in(d.implode())) if d.len() else joined
                    out.append(self._reduced(k, rows).select(pl.lit(name).alias("set"), "key", "t", "value"))
        return pl.concat(out)

    def _reduced(self, k: int, rows: pl.DataFrame) -> pl.DataFrame:
        """Key ``k``'s target on its label rows: ``key``, ``t``, ``value``."""
        plan = self.plans[k]
        v = per_t(plan.target.reduce(rows.drop("_r"), plan.horizon, self.costs))
        return v.select(pl.lit(plan.key).alias("key"), "t", "value")

    def recompute(self, f: pl.DataFrame, ts: Sequence[tr.Treatment] = ()) -> dict[str, pl.DataFrame]:
        """The numbers on the field table ``f``, with the candidate treatments in ``ts`` applied."""
        c = tr.apply_candidates(ts, self.universe(f))
        return {
            p.key: per_t(
                p.target.reduce(
                    v.join(c, on=["t", "entity"], how="inner", suffix="_candidate"),
                    p.horizon,
                    self.costs,
                )
            )
            for p, v in zip(self.plans, self._values(f, c), strict=True)
        }

    def _table(self, ts: Sequence[tr.Treatment]) -> _Table:
        renames = tuple(t for t in ts if _renames_rows(t))
        if not renames:
            return self._raw_table()
        name = tuple(t.name for t in renames)
        if name not in self.tables:
            f = tr.apply_all(renames, self.fields)
            self.tables[name] = self._build(f, self.universe(f), renames)
        return self.tables[name]

    def _raw_table(self) -> _Table:
        if () not in self.tables:
            self.tables[()] = self._build(self.fields, self.cands, ())
        return self.tables[()]

    def _build(self, f: pl.DataFrame, c: pl.DataFrame, renames: tuple[tr.Treatment, ...]) -> _Table:
        indexed = c.with_row_index("_c")
        index = indexed.select("t", "entity", "_c")
        frames, columns = [], []
        for k, v in enumerate(self._values(f, c)):
            columns.append(v.columns)
            frames.append(v.join(index, on=["t", "entity"]).drop("t", "entity").with_columns(_k=pl.lit(k, pl.UInt32)))
        labels = pl.concat(frames, how="diagonal_relaxed").with_row_index("_r")
        return _Table(renames, f, indexed, labels, columns)

    def _values(self, f: pl.DataFrame, c: pl.DataFrame) -> list[pl.DataFrame]:
        """Every plan's label rows, each label plugin asked once for all of its horizons."""
        out: dict[int, pl.DataFrame] = {}
        by_plugin: dict[int, list[int]] = {}
        for k, plan in enumerate(self.plans):
            by_plugin.setdefault(id(plan.values), []).append(k)
        for ks in by_plugin.values():
            found = label_values(self.plans[ks[0]].values, f, c, [self.plans[k].horizon for k in ks])
            out |= dict(zip(ks, found, strict=True))
        return [out[k] for k in range(len(self.plans))]

    def _joined(self, table: _Table, k: int) -> pl.DataFrame:
        """One key's label rows joined with their candidates, as a target reads them, with the label row ``_r``."""
        labels = table.labels.filter(pl.col("_k") == k)
        columns = table.columns[k]
        own = [c for c in columns if c not in ("t", "entity")]
        theirs = [c for c in table.cands.columns if c not in ("t", "entity", "_c")]
        cand = table.cands.select(pl.col("t", "entity", *theirs).gather(labels["_c"]))
        cand = cand.rename({c: f"{c}_candidate" for c in theirs if c in own})
        out = labels.select("_r", *own).hstack(cand)
        return out.select("_r", *columns, *(c for c in cand.columns if c not in ("t", "entity")))

    def _dropped(self, table: _Table, ts: Sequence[tr.Treatment]) -> pl.Series:
        """The label rows of ``table`` that the treatments in ``ts`` that drop rows touch."""
        drops = [t for t in ts if t.removed is not None]
        self._touch(table, [t for t in drops if t.name not in table.hits])
        parts = [table.hits[t.name] for t in drops]
        return pl.concat(parts).unique() if parts else pl.Series("_r", [], dtype=pl.UInt32)

    def _touch(self, table: _Table, ts: Sequence[tr.Treatment]) -> None:
        if not ts:
            return
        found = [pl.DataFrame(schema={"treatment": pl.String, "_r": pl.UInt32})]
        on_fields = [t for t in ts if t.on == "fields"]
        if on_fields:
            gone = pl.concat(
                [
                    _removed(t, self.fields).select(
                        "entity", "field", "event_time", "available_at", treatment=pl.lit(t.name)
                    )
                    for t in on_fields
                ]
            )
            spans = inf.spans(table.fields, reduce(lambda g, r: r.apply(g), table.renames, gone), by=["treatment"])
            legs = [c for c in LEGS if c in table.cands.columns]
            near = pl.any_horizontal(pl.col(c).is_in(spans["entity"].unique().implode()) for c in legs)
            rows = table.cands.filter(near).select("_c", "t", *legs)
            rows = table.labels.select("_r", "_c", "_k").join(rows, on="_c").join(self.key_table, on="_k")
            found.append(inf.touched(rows, spans, legs, by=["treatment"]))
        on_cands = [t for t in ts if t.on == "candidates"]
        if on_cands:
            gone = pl.concat([_removed(t, table.cands).select("_c", treatment=pl.lit(t.name)) for t in on_cands])
            found.append(table.labels.select("_r", "_c").join(gone, on="_c").select("treatment", "_r"))
        hits = pl.concat(found).partition_by("treatment", as_dict=True, include_key=False)
        table.hits.update({t.name: hits[(t.name,)]["_r"] if (t.name,) in hits else found[0]["_r"] for t in ts})


def _pit_resolution(numbers: Numbers, fields: pl.DataFrame, prior: Sequence[tr.Treatment]) -> dict[str, float]:
    """The MDE pits are weighed in: the finer of the MDE on the explore data and on the data with the treatments of
    the default list applied."""
    base = {k: _mde(v, k, numbers.width) for k, v in numbers.base().items()}
    if not prior:
        return base
    after = numbers.recompute(tr.apply_all(prior, fields), prior)
    return {k: _smaller(b, _mde(after[k], k, numbers.width) if k in after else math.nan) for k, b in base.items()}


def _pits(seen: Seen, numbers: Numbers, e: Eval, mde: dict[str, float]) -> list[Check]:
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


def _removed(t: tr.Treatment, table: pl.DataFrame) -> pl.DataFrame:
    assert t.removed is not None
    return t.removed(table)


def _renames_rows(t: tr.Treatment) -> bool:
    """A treatment that changes identities instead of dropping rows, such as a split."""
    return t.on == "fields" and t.removed is None


def _renames(ts: Sequence[tr.Treatment]) -> tuple[str, ...]:
    return tuple(t.name for t in ts if _renames_rows(t))


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


def _by_key(frame: pl.DataFrame, keys: Sequence[str]) -> dict[str, pl.DataFrame]:
    parts = frame.partition_by("key", as_dict=True, include_key=False)
    return {k: parts[(k,)].select("t", "value") if (k,) in parts else pl.DataFrame(schema=PER_T) for k in keys}


def _mde(values: pl.DataFrame, key: str, width: int) -> float:
    """The smallest effect this horizon can detect; it does not depend on the effect asked for."""
    h = span_ns(span(key.rsplit(" ", 1)[1]))
    return pw.power(_se(values, h), width, h).mde


def _se(values: pl.DataFrame, h: int) -> float:
    """No noise can be measured on a number that is not finite somewhere."""
    return pw.block_se(values, h) if values["value"].is_finite().all() else math.nan


def _smaller(a: float, b: float) -> float:
    """The finer of two resolutions."""
    finite = [x for x in (a, b) if math.isfinite(x) and x > 0]
    return min(finite) if finite else math.nan
