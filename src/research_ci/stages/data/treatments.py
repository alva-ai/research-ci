"""Treatments: what the engine does about a flagged problem, as a function of the sources or of the candidates. They
are applied by default and kept in the freeze; their influence on the decisive number decides whether a person has to
look. On the sources a treatment sets cells to null (a mask), drops rows (of insts, or of spans of them), or splits an
inst in every source; on the candidates it drops rows.

Which outcome rows a treatment touches, so its influence on an effect is weighed on outcomes computed once. An outcome
row at ``t`` with horizon ``h`` may read what its legs report up to ``t + h``, and what is current at ``t``; a source
row is current from when it becomes known until the next row of its source and inst does. A treatment that drops or
masks source rows touches every outcome row of their inst whose window ``[t, t + h]`` meets a span in which such a row
was current; leaving those outcome rows out stands for computing the outcomes again on the treated sources. It errs
towards more influence: a row inside the window that the outcome does not read still counts, and so does a masked
column it does not read.

``Numbers`` are the decisive numbers the treatments are weighed on.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import reduce

import pandas as pd
import polars as pl

from research_ci.api.contract import span
from research_ci.api.evals import Case, Source
from research_ci.api.plugins import CostModel, OutcomePlugin, TargetPlugin
from research_ci.eval import synthetic
from research_ci.eval.data.baseline import rows
from research_ci.eval.synthetic import HOUR_NS, START_NS
from research_ci.kernel.tables import values
from research_ci.kernel.time.clock import TIME, span_ns
from research_ci.kernel.verdict import power as pw
from research_ci.stages.common import Treatment, outcome_values, per_t

Sources = Mapping[str, pl.DataFrame]
SPAN_SCHEMA = {"inst": pl.String, "event_time": TIME, "start": pl.Int64, "end": pl.Int64}
ROW = ["source", "inst", "event_time"]
OPEN = 2**62
LEGS = ("inst", "long", "short")
PER_T = {"t": pl.Datetime("ns", "UTC"), "value": pl.Float64}


def mask_cells(name: str, reason: str, flagged: pl.DataFrame, sources: Sources) -> Treatment:
    """Set the flagged cells to null. Flags have ``source``, ``inst`` and ``event_time``, and ``column`` when they name
    one; a flag that names none masks every value column of its rows."""
    named = flagged.select(
        *ROW, pl.col("column") if "column" in flagged.columns else pl.lit(None, dtype=pl.String).alias("column")
    )
    owned = pl.DataFrame(
        [(s, c) for s, t in sources.items() for c in values(t)],
        schema={"source": pl.String, "_c": pl.String},
        orient="row",
    )
    cells = named.join(owned, on="source").filter(pl.col("column").is_null() | (pl.col("column") == pl.col("_c")))
    cells = cells.select(*ROW, pl.col("_c").alias("column")).with_columns(pl.col("event_time").cast(TIME)).unique()
    by_source = cells.partition_by("source", as_dict=True, include_key=False)

    def masked(source: str, table: pl.DataFrame) -> pl.DataFrame:
        hit = by_source.get((source,))
        if hit is None:
            return table
        marks = hit.with_columns(_m=pl.lit(True)).pivot(on="column", index=["inst", "event_time"], values="_m")
        marks = marks.rename({c: f"_m_{c}" for c in marks.columns if c not in ("inst", "event_time")})
        joined = table.join(marks, on=["inst", "event_time"], how="left", maintain_order="left")
        null = [
            pl.when(pl.col(f"_m_{c}").fill_null(False)).then(None).otherwise(pl.col(c)).alias(c)
            for c in values(table)
            if f"_m_{c}" in joined.columns
        ]
        return joined.with_columns(null).select(table.columns)

    def apply(tables: Sources) -> dict[str, pl.DataFrame]:
        return {s: masked(s, t) for s, t in tables.items()}

    def removed(tables: Sources) -> pl.DataFrame:
        return rows(tables).join(cells.select(ROW).unique(), on=ROW, how="semi")

    return Treatment(name, "sources", reason, removed(sources).height, apply, removed)


def mask_windows(name: str, reason: str, windows: pl.DataFrame, around_ns: int, sources: Sources) -> Treatment:
    """Mask the window's column within ``around_ns`` of each window. A window has ``source``, ``column``, ``inst``,
    ``start`` and ``end``."""
    pad = pl.duration(nanoseconds=around_ns, time_unit="ns")
    w = windows.select(
        "source", "column", "inst", (pl.col("start") - pad).alias("lo"), (pl.col("end") + pad).alias("hi")
    )
    found = rows(sources).join(w, on=["source", "inst"]).filter(pl.col("event_time").is_between("lo", "hi"))
    return mask_cells(name, reason, found.select(*ROW, "column"), sources)


def drop_insts(name: str, reason: str, insts: Sequence[str], sources: Sources) -> Treatment:
    """Drop every row of the named insts, in every source."""
    names = pl.Series(sorted(set(insts)), dtype=pl.String).implode()
    gone = pl.col("inst").is_in(names)
    return _dropping(name, reason, sources, gone)


def drop_spans(name: str, reason: str, spans: pl.DataFrame, sources: Sources) -> Treatment:
    """Drop the rows each span covers, in every source: every row of the span's ``inst`` whose event falls in
    ``[start, end]``."""
    keys = spans.select("inst", pl.col("start").cast(TIME), pl.col("end").cast(TIME))
    on = ["inst", "event_time"]

    def inside(tables: Sources) -> pl.DataFrame:
        found = rows(tables).join(keys, on="inst").filter(pl.col("event_time").is_between("start", "end"))
        return found.select(on).unique()

    def apply(tables: Sources) -> dict[str, pl.DataFrame]:
        gone = inside(tables)
        return {s: t.join(gone, on=on, how="anti", maintain_order="left") for s, t in tables.items()}

    def removed(tables: Sources) -> pl.DataFrame:
        return rows({s: t.join(inside(tables), on=on, how="semi") for s, t in tables.items()})

    return Treatment(name, "sources", reason, removed(sources).height, apply, removed)


def _dropping(name: str, reason: str, sources: Sources, gone: pl.Expr) -> Treatment:
    def apply(tables: Sources) -> dict[str, pl.DataFrame]:
        return {s: t.filter(~gone) for s, t in tables.items()}

    def removed(tables: Sources) -> pl.DataFrame:
        return rows({s: t.filter(gone) for s, t in tables.items()})

    return Treatment(name, "sources", reason, removed(sources).height, apply, removed)


def split_insts(name: str, reason: str, splits: pl.DataFrame, sources: Sources) -> Treatment:
    """Rename each split inst's rows from the split on, in every source. The rows of each split's ``inst`` whose event
    is at or after its ``at`` are renamed to its ``to``, up to the inst's next split. The rows stay; ``rows`` counts
    those renamed. It applies to any table with ``inst`` and ``event_time``, each one of a dict."""
    s = splits.select("inst", pl.col("at").cast(TIME), "to").sort("at")
    names = s["inst"].unique()

    def renamed(f: pl.DataFrame) -> pl.DataFrame:
        g = f.with_row_index("_i")
        touched = g.filter(pl.col("inst").is_in(names.implode())).select("_i", "inst", "event_time")
        hit = touched.sort("event_time").join_asof(
            s, left_on="event_time", right_on="at", by="inst", strategy="backward", check_sortedness=False
        )
        return hit.filter(pl.col("to").is_not_null()).select("_i", "to")

    def one(f: pl.DataFrame) -> pl.DataFrame:
        r = renamed(f)
        return f.with_columns(f["inst"].scatter(r["_i"], r["to"])) if r.height else f

    def apply(tables: Sources) -> dict[str, pl.DataFrame]:
        return {n: one(t) for n, t in tables.items()}

    return Treatment(name, "sources", reason, renamed(rows(sources)).height, apply)


def drop_candidates(name: str, reason: str, bad: pl.DataFrame) -> Treatment:
    """Drop the candidate rows at ``t`` whose ``inst``, or one of whose legs (``long``, ``short``), is a flagged
    ``inst`` at ``t``: a flagged candidate, or a flagged row of ``wide`` a candidate reads."""
    flags = bad.select("t", "inst").unique()

    def hit(c: pl.DataFrame) -> pl.Series:
        indexed = c.with_row_index("_i")
        legs = [indexed.select("_i", "t", pl.col(x).alias("inst")) for x in LEGS if x in c.columns]
        found = pl.concat(legs).join(flags, on=["t", "inst"], how="semi")["_i"]
        return indexed["_i"].is_in(found.implode())

    def apply(c: pl.DataFrame) -> pl.DataFrame:
        return c.filter(~hit(c))

    def removed(c: pl.DataFrame) -> pl.DataFrame:
        return c.filter(hit(c))

    return Treatment(name, "candidates", reason, flags.height, apply, removed)


def apply_all(treatments: Sequence[Treatment], sources: Sources) -> dict[str, pl.DataFrame]:
    """The sources after every treatment of the sources: the masks and drops first, each found on the sources as
    they came, then the splits in order."""
    on = [t for t in treatments if t.on == "sources"]
    ordered = [t for t in on if t.removed is not None] + [t for t in on if t.removed is None]
    return reduce(lambda f, t: t.apply(f), ordered, dict(sources))


def apply_candidates(treatments: Sequence[Treatment], cands: pl.DataFrame) -> pl.DataFrame:
    """The candidates after every candidate treatment."""
    return reduce(lambda c, t: t.apply(c), [t for t in treatments if t.on == "candidates"], cands)


def spans(sources: Sources, removed: pl.DataFrame, by: Sequence[str] = ()) -> pl.DataFrame:
    """When each removed row was current in ``sources``. Columns ``inst``, ``event_time``, ``start`` (when it became
    known) and ``end`` (when the next row of its source and inst did; far in the future for the last), both in ns.
    ``by`` names columns of ``removed`` that set its rows apart, such as the treatment; they come first."""
    keys, group = ["source", "inst"], list(by)
    schema = {**{k: removed.schema[k] for k in group}, **SPAN_SCHEMA}
    known = rows(sources).lazy().join(removed.lazy().select(keys).unique(), on=keys, how="semi")
    known = known.select(*keys, pl.col("available_at").dt.epoch("ns").alias("start")).unique().sort(*keys, "start")
    known = known.with_columns(end=pl.col("start").shift(-1).over(keys).fill_null(OPEN))
    gone = removed.lazy().select(*group, *keys, "event_time", pl.col("available_at").dt.epoch("ns").alias("start"))
    out = gone.join(known, on=[*keys, "start"], how="left").select(list(schema))
    return out.collect().cast(schema)  # type: ignore[arg-type]


def touched(
    rows: pl.DataFrame, found: pl.DataFrame, legs: Sequence[str], h: str = "h_ns", by: Sequence[str] = ()
) -> pl.DataFrame:
    """Outcome rows whose window meets a span of a leg. One row per group and outcome row touched: an outcome row
    (``_r``) whose window ``[t, t + h]`` meets a span (``inst``, ``start``, ``end``) of one of its ``legs``, the columns
    that name insts. ``h`` is the column with each row's horizon in ns, and ``by`` the columns of ``found`` that set
    spans apart, such as the treatment."""
    keys = list(by)
    legs = [c for c in legs if c in rows.columns]
    schema = {**{k: found.schema[k] for k in keys}, "_r": rows.schema["_r"]}
    if found.is_empty() or not legs:
        return pl.DataFrame(schema=schema)
    horizons = rows.lazy().select(pl.col(h).cast(pl.Int64).alias("_h")).unique()
    reach = found.lazy().select(*keys, "inst", "start", "end").join(horizons, how="cross")
    reach = reach.select(*keys, "inst", "_h", (pl.col("start") - pl.col("_h")).alias("lo"), pl.col("end").alias("hi"))
    group = [*keys, "inst", "_h"]
    reach = reach.sort(*group, "lo").with_columns(_prior=pl.col("hi").cum_max().shift(1).over(group))
    fresh = pl.col("_prior").is_null() | (pl.col("lo") > pl.col("_prior"))
    merged = (
        reach.with_columns(_run=fresh.cum_sum().over(group))
        .group_by(*group, "_run")
        .agg(pl.col("lo").min(), pl.col("hi").max())
    )
    insts = found.lazy().select(*keys, "inst").unique()
    asked = rows.lazy().select(
        "_r", pl.col("t").dt.epoch("ns").alias("_t"), pl.col(h).cast(pl.Int64).alias("_h"), *legs
    )
    asked = asked.unpivot(index=["_r", "_t", "_h"], on=legs, value_name="_leg").select(
        "_r", "_t", "_h", pl.col("_leg").alias("inst")
    )
    asked = asked.join(insts, on="inst").sort("_t")
    hit = asked.join_asof(
        merged.sort("lo"), left_on="_t", right_on="lo", by=group, strategy="backward", check_sortedness=False
    )
    out = hit.filter(pl.col("_t") < pl.col("hi")).select(*keys, "_r").unique()
    return out.sort(*keys, "_r").collect().cast(schema)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Plan:
    key: str
    target: TargetPlugin
    horizon: pd.Timedelta


@dataclass
class _Table:
    """Every key's outcome rows on one version of the sources, the one ``renames`` leave: the candidates with their
    row (``_c``), the outcome columns of all keys stacked with the candidate row and the key (``_k``) and their own row
    (``_r``), each key's outcome columns, and the outcome rows each treatment touches."""

    renames: tuple[Treatment, ...]
    sources: dict[str, pl.DataFrame]
    cands: pl.DataFrame
    outcomes: pl.DataFrame
    columns: list[list[str]]
    hits: dict[str, pl.Series] = field(default_factory=dict)


class Numbers:
    """The decisive numbers. Their outcome rows are computed once on the explore sources and, when treatments rename
    rows (a split), once more on the sources those leave. A treatment that drops or masks source rows, or drops
    candidates, leaves out the outcome rows that read what it touches (``spans``, ``touched``), so weighing it runs
    only the target again. ``recompute`` computes everything again, ``wide``, candidates, outcomes and all: for the
    treated sources, and for changes no treatment describes, such as a column left out or a planted pit.
    ``universe`` gives the candidates on sources, building ``wide`` on them; ``width`` is the explore data's span in
    ns."""

    def __init__(
        self,
        sources: Sources,
        cands: pl.DataFrame,
        universe: Callable[[Sources], pl.DataFrame],
        outcome: OutcomePlugin,
        plans: list[Plan],
        costs: CostModel,
        width: int,
    ) -> None:
        self.sources, self.cands, self.universe, self.plans, self.costs = dict(sources), cands, universe, plans, costs
        self.outcome, self.width = outcome, width
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

    def rows(self, ts: Sequence[Treatment], columns: Sequence[str]) -> pl.DataFrame:
        """Every key's outcome rows after ``ts``, stacked: ``key``, ``h_ns`` (the horizon in ns), ``t`` and
        ``columns``, from the outcome or the candidates; text columns of the candidates come as categoricals."""
        table = self._table(ts)
        dropped = self._dropped(table, ts)
        own = [c for c in columns if c in table.outcomes.columns]
        theirs = [c for c in columns if c not in own and c != "t"]
        found = table.outcomes.select("_r", "_c", "_k", *own)
        found = found.filter(~pl.col("_r").is_in(dropped.implode())) if dropped.len() else found
        text = [c for c in theirs if table.cands.schema[c] == pl.String]
        cand = table.cands.select("t", *theirs).with_columns(pl.col(text).cast(pl.Categorical))
        cand = cand.select(pl.all().gather(found["_c"]))
        k = self.key_table
        out = (
            found.select("_k", *own)
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
        """Every key's number per decision time on the untreated sources, stacked: ``key``, ``t``, ``value``."""
        if self._base is None:
            table = self._raw_table()
            out = [pl.DataFrame(schema={"key": pl.String, **PER_T})]
            out += [
                self._reduced(k, self._joined(table, k)).select("key", "t", "value") for k in range(len(self.plans))
            ]
            self._base = pl.concat(out)
        return self._base

    def treated(self, treated: Sources, applied: Sequence[Treatment]) -> dict[str, pl.DataFrame]:
        """The numbers on the treated sources, outcomes and all, with the candidate treatments of ``applied``; computed
        once."""
        if self._treated is None:
            self._treated = self.recompute(treated, applied) if applied else self.base()
        return self._treated

    def resolution(self, treated: Sources, applied: Sequence[Treatment]) -> dict[str, float]:
        """The MDE of each key a treatment's influence is measured in: the finer of the MDE before and after the
        treatments, since a treatment matters if it moves the number by more than the verdict on the treated data, or
        on the untreated data, could resolve."""
        if self._resolution is None:
            after = self.treated(treated, applied)
            before = {k: mde(v, k, self.width) for k, v in self.base().items()}
            now = {k: mde(v, k, self.width) for k, v in after.items()}
            self._resolution = {k: smaller(before[k], now.get(k, math.nan)) for k in before}
        return self._resolution

    def per_set(self, sets: Mapping[str, Sequence[Treatment]]) -> pl.DataFrame:
        """The number per decision time of every key after each named set of treatments, stacked: ``set``, ``key``,
        ``t``, ``value``. The target runs once per set and key; the outcome rows come from the tables."""
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
                    found = joined.filter(~pl.col("_r").is_in(d.implode())) if d.len() else joined
                    out.append(self._reduced(k, found).select(pl.lit(name).alias("set"), "key", "t", "value"))
        return pl.concat(out)

    def _reduced(self, k: int, found: pl.DataFrame) -> pl.DataFrame:
        """Key ``k``'s target on its outcome rows: ``key``, ``t``, ``value``."""
        plan = self.plans[k]
        v = per_t(plan.target.reduce(found.drop("_r"), plan.horizon, self.costs))
        return v.select(pl.lit(plan.key).alias("key"), "t", "value")

    def recompute(self, sources: Sources, ts: Sequence[Treatment] = ()) -> dict[str, pl.DataFrame]:
        """The numbers on ``sources``, with the candidate treatments in ``ts`` applied."""
        c = apply_candidates(ts, self.universe(sources))
        return {
            p.key: per_t(
                p.target.reduce(v.join(c, on=["t", "inst"], how="inner", suffix="_candidate"), p.horizon, self.costs)
            )
            for p, v in zip(self.plans, self._values(sources, c), strict=True)
        }

    def _table(self, ts: Sequence[Treatment]) -> _Table:
        renames = tuple(t for t in ts if renames_rows(t))
        if not renames:
            return self._raw_table()
        name = tuple(t.name for t in renames)
        if name not in self.tables:
            f = apply_all(renames, self.sources)
            self.tables[name] = self._build(f, self.universe(f), renames)
        return self.tables[name]

    def _raw_table(self) -> _Table:
        if () not in self.tables:
            self.tables[()] = self._build(self.sources, self.cands, ())
        return self.tables[()]

    def _build(self, f: dict[str, pl.DataFrame], c: pl.DataFrame, renames: tuple[Treatment, ...]) -> _Table:
        indexed = c.with_row_index("_c")
        index = indexed.select("t", "inst", "_c")
        frames, columns = [], []
        for k, v in enumerate(self._values(f, c)):
            columns.append(v.columns)
            frames.append(v.join(index, on=["t", "inst"]).drop("t", "inst").with_columns(_k=pl.lit(k, pl.UInt32)))
        outcomes = pl.concat(frames, how="diagonal_relaxed").with_row_index("_r")
        return _Table(renames, f, indexed, outcomes, columns)

    def _values(self, f: Sources, c: pl.DataFrame) -> list[pl.DataFrame]:
        """Every plan's outcome rows, the outcome asked once for all of the horizons."""
        horizons = sorted({p.horizon for p in self.plans})
        found = dict(zip(horizons, outcome_values(self.outcome, f, c, horizons), strict=True))
        return [found[p.horizon] for p in self.plans]

    def _joined(self, table: _Table, k: int) -> pl.DataFrame:
        """One key's outcome rows joined with their candidates, as a target reads them, with the outcome row ``_r``."""
        found = table.outcomes.filter(pl.col("_k") == k)
        columns = table.columns[k]
        own = [c for c in columns if c not in ("t", "inst")]
        theirs = [c for c in table.cands.columns if c not in ("t", "inst", "_c")]
        cand = table.cands.select(pl.col("t", "inst", *theirs).gather(found["_c"]))
        cand = cand.rename({c: f"{c}_candidate" for c in theirs if c in own})
        out = found.select("_r", *own).hstack(cand)
        return out.select("_r", *columns, *(c for c in cand.columns if c not in ("t", "inst")))

    def _dropped(self, table: _Table, ts: Sequence[Treatment]) -> pl.Series:
        """The outcome rows of ``table`` that the treatments in ``ts`` that drop or mask rows touch."""
        drops = [t for t in ts if t.removed is not None]
        self._touch(table, [t for t in drops if t.name not in table.hits])
        found = [table.hits[t.name] for t in drops]
        return pl.concat(found).unique() if found else pl.Series("_r", [], dtype=pl.UInt32)

    def _touch(self, table: _Table, ts: Sequence[Treatment]) -> None:
        if not ts:
            return
        found = [pl.DataFrame(schema={"treatment": pl.String, "_r": pl.UInt32})]
        on_sources = [t for t in ts if t.on == "sources"]
        if on_sources:
            gone = pl.concat([_removed(t, self.sources).with_columns(treatment=pl.lit(t.name)) for t in on_sources])
            renamed = reduce(lambda g, r: r.apply({"gone": g})["gone"], table.renames, gone)
            current = spans(table.sources, renamed, by=["treatment"])
            legs = [c for c in LEGS if c in table.cands.columns]
            near = pl.any_horizontal(pl.col(c).is_in(current["inst"].unique().implode()) for c in legs)
            asked = table.cands.filter(near).select("_c", "t", *legs)
            asked = table.outcomes.select("_r", "_c", "_k").join(asked, on="_c").join(self.key_table, on="_k")
            found.append(touched(asked, current, legs, by=["treatment"]))
        on_cands = [t for t in ts if t.on == "candidates"]
        if on_cands:
            gone = pl.concat([_removed(t, table.cands).select("_c", treatment=pl.lit(t.name)) for t in on_cands])
            found.append(table.outcomes.select("_r", "_c").join(gone, on="_c").select("treatment", "_r"))
        hits = pl.concat(found).partition_by("treatment", as_dict=True, include_key=False)
        table.hits.update({t.name: hits[(t.name,)]["_r"] if (t.name,) in hits else found[0]["_r"] for t in ts})


def _removed(t: Treatment, table: Sources | pl.DataFrame) -> pl.DataFrame:
    assert t.removed is not None
    return t.removed(table)


def renames_rows(t: Treatment) -> bool:
    """A treatment that changes identities instead of dropping or masking rows, such as a split."""
    return t.on == "sources" and t.removed is None


def _renames(ts: Sequence[Treatment]) -> tuple[str, ...]:
    return tuple(t.name for t in ts if renames_rows(t))


def _by_key(frame: pl.DataFrame, keys: Sequence[str]) -> dict[str, pl.DataFrame]:
    found = frame.partition_by("key", as_dict=True, include_key=False)
    return {k: found[(k,)].select("t", "value") if (k,) in found else pl.DataFrame(schema=PER_T) for k in keys}


def mde(numbers: pl.DataFrame, key: str, width: int) -> float:
    """The smallest effect this horizon can detect; it does not depend on the effect asked for."""
    h = span_ns(span(key.rsplit(" ", 1)[1]))
    return pw.power(se(numbers, h), width, h).mde


def se(numbers: pl.DataFrame, h: int) -> float:
    """No noise can be measured on a number that is not finite somewhere."""
    return pw.block_se(numbers, h) if numbers["value"].is_finite().all() else math.nan


def smaller(a: float, b: float) -> float:
    """The finer of two resolutions."""
    finite = [x for x in (a, b) if math.isfinite(x) and x > 0]
    return min(finite) if finite else math.nan


def _sources() -> dict[str, pl.DataFrame]:
    return synthetic.sources(Source(insts=3, hours=48), 0)


def _at(hour: int) -> pl.Expr:
    return pl.lit(START_NS + hour * HOUR_NS).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")


def _counts(t: Treatment) -> tuple[int, int]:
    """The rows a treatment says it touches, and the rows it changes or takes away."""
    before = _sources()
    after = t.apply(before)
    changed = [b.join(after[n], on=b.columns, how="anti", nulls_equal=True).height for n, b in before.items()]
    return t.rows, sum(changed)


def _windows() -> pl.DataFrame:
    return pl.DataFrame({"source": ["prices"], "column": ["price"], "inst": ["e0"]}).with_columns(
        start=_at(10), end=_at(11)
    )


def _flags(column: bool = True) -> pl.DataFrame:
    flags = _sources()["prices"].filter(pl.col("inst") == "e1").head(3)
    flags = flags.select(pl.lit("prices").alias("source"), "inst", "event_time")
    return flags.with_columns(column=pl.lit("price")) if column else flags


def _masked(t: Treatment) -> tuple[int, int, int]:
    """What a mask says it touches, the rows it changes, and the price cells it leaves null."""
    return (*_counts(t), t.apply(_sources())["prices"]["price"].null_count())


def _spans() -> pl.DataFrame:
    found = [("e0", 5, 7), ("e1", 40, 41)]
    frame = pl.DataFrame(found, schema=["inst", "start", "end"], orient="row")
    return frame.with_columns((pl.col(c) * HOUR_NS + START_NS).cast(TIME) for c in ("start", "end"))


def _split() -> tuple[int, dict[str, int], bool, bool]:
    """e0 split at hours 20 and 40: rows renamed, rows per inst in both sources, the rest untouched, and applying it
    twice changes nothing more."""
    f = _sources()
    splits = pl.DataFrame([("e0", 20, "e0#2"), ("e0", 40, "e0#3")], schema=["inst", "at", "to"], orient="row")
    t = split_insts("s", "", splits.with_columns((pl.col("at") * HOUR_NS + START_NS).cast(TIME)), f)
    out = t.apply(f)
    counts = {str(e): int(n) for e, n in rows(out).group_by("inst").len().iter_rows()}
    same = all(out[n].drop("inst").equals(f[n].drop("inst")) for n in f)
    again = t.apply(out)
    return t.rows, counts, same, all(again[n].equals(out[n]) for n in out)


def _prices() -> dict[str, pl.DataFrame]:
    """Hourly prices of a and b at hours 0 to 10, each known an hour after its event."""
    found = pl.DataFrame({"inst": ["a"] * 11 + ["b"] * 11, "hour": list(range(11)) * 2})
    hours = [pl.from_epoch((pl.col("hour") + k) * HOUR_NS + START_NS, time_unit="ns") for k in (0, 1)]
    prices = found.select(
        "inst",
        event_time=hours[0].dt.replace_time_zone("UTC"),
        available_at=hours[1].dt.replace_time_zone("UTC"),
        vintage=pl.lit(0),
        price=pl.lit(1.0),
    )
    return {"prices": prices}


def _rows(h: int = 2, legs: bool = False) -> pl.DataFrame:
    """An outcome row of a and of b at every hour from 0 to 10, horizon ``h`` hours; with ``legs``, of a pair whose
    long leg is a."""
    frame = pl.DataFrame({"hour": list(range(11)) * 2, "inst": ["a"] * 11 + ["b"] * 11})
    frame = frame.with_columns(
        t=pl.from_epoch(pl.col("hour") * HOUR_NS + START_NS, time_unit="ns").dt.replace_time_zone("UTC"),
        h_ns=pl.lit(h * HOUR_NS),
    )
    if legs:
        frame = frame.with_columns(long=pl.col("inst"), short=pl.lit("c"), inst=pl.col("inst") + ">c")
    return frame.with_row_index("_r")


def _hit(hours: Sequence[int], h: int = 2, legs: bool = False) -> list[tuple[str, int]]:
    """The outcome rows touched when a's prices of ``hours`` are dropped, as (inst, hour)."""
    f = _prices()
    gone = rows(f).filter((pl.col("inst") == "a") & pl.col("event_time").dt.hour().is_in(list(hours)))
    found = touched(_rows(h, legs), spans(f, gone), ["inst", "long", "short"])
    return _rows(h, legs).join(found, on="_r").select("inst", "hour").sort("hour").rows()


def _apart() -> list[tuple[str, int]]:
    """Spans of two treatments found at once, each touching only its own rows."""
    f = _prices()
    every = rows(f)
    a = every.filter((pl.col("inst") == "a") & (pl.col("event_time").dt.hour() == 3)).with_columns(
        treatment=pl.lit("x")
    )
    b = every.filter((pl.col("inst") == "b") & (pl.col("event_time").dt.hour() == 7)).with_columns(
        treatment=pl.lit("y")
    )
    found = spans(f, pl.concat([a, b]), by=["treatment"])
    hit = touched(_rows(0), found, ["inst"], by=["treatment"])
    return hit.join(_rows(0), on="_r").select("treatment", "hour").sort("treatment", "hour").rows()


CASES = [
    Case(
        "masking a window nulls its column's cells and an hour each side",
        lambda: _masked(mask_windows("m", "", _windows(), HOUR_NS, _sources())),
        (4, 4, 4),
    ),
    Case("masking cells nulls exactly them", lambda: _masked(mask_cells("m", "", _flags(), _sources())), (3, 3, 3)),
    Case(
        "a flag that names no column masks every column of its row",
        lambda: _masked(mask_cells("m", "", _flags(column=False), _sources())),
        (3, 3, 3),
    ),
    Case("dropping an inst drops all its rows", lambda: _counts(drop_insts("d", "", ["e2"], _sources())), (54, 54)),
    Case(
        "dropping spans drops their rows in every source",
        lambda: _counts(drop_spans("d", "", _spans(), _sources())),
        (6, 6),
    ),
    Case(
        "splitting renames the rows from each split on, and only them",
        _split,
        (33, {"e0": 21, "e0#2": 22, "e0#3": 11, "e1": 54, "e2": 54}, True, True),
    ),
    Case(
        "a dropped row touches the rows whose window meets the hour it was current, of its inst only",
        lambda: _hit([5]),
        [("a", 4), ("a", 5), ("a", 6)],
    ),
    Case("the last row stays current", lambda: _hit([10]), [("a", 9), ("a", 10)]),
    Case("a longer horizon reaches further back", lambda: len(_hit([5], h=4)), 5),
    Case("dropped rows next to each other make one span", lambda: _hit([5, 6], h=0), [("a", 6), ("a", 7)]),
    Case("a leg counts like the inst", lambda: _hit([5], legs=True), [("a>c", 4), ("a>c", 5), ("a>c", 6)]),
    Case("each treatment touches only its own rows", _apart, [("x", 4), ("y", 8)]),
]
