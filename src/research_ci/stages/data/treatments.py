"""Treatments: what the engine does about a flagged problem, as a function of the field table or of the candidates.
They are applied by default and kept in the freeze; their influence on the decisive number decides whether a person
has to look.

Which label rows a treatment touches, so its influence on a label effect is weighed on labels computed once. A label
row at ``t`` with horizon ``h`` may read what its legs report up to ``t + h``, and what is current at ``t``; a field
row is current from when it becomes known until the next row of its entity and field does. A treatment that drops
field rows touches every label row of their entity whose window ``[t, t + h]`` meets a span in which a dropped row was
current; leaving those label rows out stands for computing the labels again on the treated data. It errs towards more
influence: a row inside the window that the label does not read still counts.

``Numbers`` are the decisive numbers the treatments are weighed on.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import reduce
from typing import Literal

import pandas as pd
import polars as pl

from research_ci.api.contract import span
from research_ci.api.evals import Case, Source
from research_ci.api.plugins import CostModel, LabelPlugin, TargetPlugin
from research_ci.eval.synthetic import HOUR_NS, START_NS, source
from research_ci.kernel.time.clock import TIME, span_ns
from research_ci.kernel.verdict import power as pw
from research_ci.stages.common import Treatment, label_values, per_t

SPAN_SCHEMA = {"entity": pl.String, "event_time": TIME, "start": pl.Int64, "end": pl.Int64}
OPEN = 2**62
LEGS = ("entity", "long", "short")
PER_T = {"t": pl.Datetime("ns", "UTC"), "value": pl.Float64}


def _dropping(
    name: str,
    on: Literal["fields", "candidates"],
    reason: str,
    table: pl.DataFrame,
    apply: Callable[[pl.DataFrame], pl.DataFrame],
    removed: Callable[[pl.DataFrame], pl.DataFrame],
) -> Treatment:
    return Treatment(name, on, reason, removed(table).height, apply, removed)


def mask_windows(name: str, reason: str, windows: pl.DataFrame, around_ns: int, fields: pl.DataFrame) -> Treatment:
    """Drop the flagged field's rows within ``around_ns`` of each window. A window has ``entity``, ``field``, ``start``
    and ``end``."""
    pad = pl.duration(nanoseconds=around_ns, time_unit="ns")
    keys = ["entity", "field"]
    w = windows.select(*keys, (pl.col("start") - pad).alias("lo"), (pl.col("end") + pad).alias("hi"))

    def hit(f: pl.DataFrame) -> pl.Series:
        g = f.select(*keys, "event_time").with_row_index("_i")
        inside = g.join(w, on=keys).filter(pl.col("event_time").is_between("lo", "hi"))
        return g["_i"].is_in(inside["_i"].unique().implode())

    return _dropping(name, "fields", reason, fields, lambda f: f.filter(~hit(f)), lambda f: f.filter(hit(f)))


def mask_rows(name: str, reason: str, flagged: pl.DataFrame, fields: pl.DataFrame) -> Treatment:
    """Drop the flagged rows. Flags match by ``entity`` and ``event_time``, and by ``field`` too when the flags name
    one."""
    keys = ["entity", "event_time"] + (["field"] if "field" in flagged.columns else [])
    flags = flagged.select(*keys).with_columns(pl.col("event_time").cast(TIME)).unique()

    def apply(f: pl.DataFrame) -> pl.DataFrame:
        return f.join(flags, on=keys, how="anti", maintain_order="left")

    def removed(f: pl.DataFrame) -> pl.DataFrame:
        return f.join(flags, on=keys, how="semi", maintain_order="left")

    return _dropping(name, "fields", reason, fields, apply, removed)


def drop_entities(name: str, reason: str, entities: Sequence[str], fields: pl.DataFrame) -> Treatment:
    """Drop every row of the named entities."""
    names = pl.Series(sorted(set(entities)), dtype=pl.String).implode()
    gone = pl.col("entity").is_in(names)
    return _dropping(name, "fields", reason, fields, lambda f: f.filter(~gone), lambda f: f.filter(gone))


def drop_spans(name: str, reason: str, spans: pl.DataFrame, fields: pl.DataFrame) -> Treatment:
    """Drop the rows each span covers. Every row of the span's ``entity`` whose event falls in ``[start, end]`` goes."""
    keys = spans.select("entity", pl.col("start").cast(TIME), pl.col("end").cast(TIME))

    def hit(f: pl.DataFrame) -> pl.Series:
        g = f.select("entity", "event_time").with_row_index("_i")
        inside = g.join(keys, on="entity").filter(pl.col("event_time").is_between("start", "end"))
        return g["_i"].is_in(inside["_i"].unique().implode())

    return _dropping(name, "fields", reason, fields, lambda f: f.filter(~hit(f)), lambda f: f.filter(hit(f)))


def split_entities(name: str, reason: str, splits: pl.DataFrame, fields: pl.DataFrame) -> Treatment:
    """Rename each split entity's rows from the split on. The rows of each split's ``entity`` whose event is at or after
    its ``at`` are renamed to its ``to``, up to the entity's next split. The rows stay; ``rows`` counts those renamed.
    It applies to any table with ``entity`` and ``event_time``."""
    s = splits.select("entity", pl.col("at").cast(TIME), "to").sort("at")
    names = s["entity"].unique()

    def renamed(f: pl.DataFrame) -> pl.DataFrame:
        g = f.with_row_index("_i")
        touched = g.filter(pl.col("entity").is_in(names.implode())).select("_i", "entity", "event_time")
        hit = touched.sort("event_time").join_asof(
            s, left_on="event_time", right_on="at", by="entity", strategy="backward", check_sortedness=False
        )
        return hit.filter(pl.col("to").is_not_null()).select("_i", "to")

    def apply(f: pl.DataFrame) -> pl.DataFrame:
        r = renamed(f)
        return f.with_columns(f["entity"].scatter(r["_i"], r["to"])) if r.height else f

    return Treatment(name, "fields", reason, renamed(fields).height, apply)


def drop_candidates(name: str, reason: str, bad: pl.DataFrame) -> Treatment:
    """Drop candidate rows (``t``, ``entity``)."""
    flags = bad.select("t", "entity").unique()

    def apply(c: pl.DataFrame) -> pl.DataFrame:
        return c.join(flags, on=["t", "entity"], how="anti", maintain_order="left")

    def removed(c: pl.DataFrame) -> pl.DataFrame:
        return c.join(flags, on=["t", "entity"], how="semi", maintain_order="left")

    return Treatment(name, "candidates", reason, flags.height, apply, removed)


def apply_all(treatments: Sequence[Treatment], fields: pl.DataFrame) -> pl.DataFrame:
    """The field table after every field treatment. The drops come at once, then the rest in order."""
    drops = [t.removed for t in treatments if t.on == "fields" and t.removed is not None]
    if drops:
        indexed = fields.with_row_index("_row")
        gone = pl.concat([removed(indexed).select("_row") for removed in drops])["_row"]
        fields = fields.filter(~pl.repeat(False, fields.height, eager=True).scatter(gone, True))
    return reduce(lambda f, t: t.apply(f), [t for t in treatments if t.on == "fields" and t.removed is None], fields)


def apply_candidates(treatments: Sequence[Treatment], cands: pl.DataFrame) -> pl.DataFrame:
    """The candidates after every candidate treatment."""
    return reduce(lambda c, t: t.apply(c), [t for t in treatments if t.on == "candidates"], cands)


def spans(fields: pl.DataFrame, removed: pl.DataFrame, by: Sequence[str] = ()) -> pl.DataFrame:
    """When each removed row was current in ``fields``. Columns ``entity``, ``event_time``, ``start`` (when it became
    known) and ``end`` (when the next row of its entity and field did; far in the future for the last), both in ns.
    ``by`` names columns of ``removed`` that set its rows apart, such as the treatment; they come first."""
    keys, group = ["entity", "field"], list(by)
    schema = {**{k: removed.schema[k] for k in group}, **SPAN_SCHEMA}
    known = fields.lazy().join(removed.lazy().select(keys).unique(), on=keys, how="semi")
    known = known.select(*keys, pl.col("available_at").dt.epoch("ns").alias("start")).unique().sort(*keys, "start")
    known = known.with_columns(end=pl.col("start").shift(-1).over(keys).fill_null(OPEN))
    gone = removed.lazy().select(*group, *keys, "event_time", pl.col("available_at").dt.epoch("ns").alias("start"))
    out = gone.join(known, on=[*keys, "start"], how="left").select(list(schema))
    return out.collect().cast(schema)  # type: ignore[arg-type]


def touched(
    rows: pl.DataFrame, found: pl.DataFrame, legs: Sequence[str], h: str = "h_ns", by: Sequence[str] = ()
) -> pl.DataFrame:
    """Label rows whose window meets a span of a leg. One row per group and label row touched: a label row (``_r``)
    whose window ``[t, t + h]`` meets a span (``entity``, ``start``, ``end``) of one of its ``legs``, the columns that
    name entities. ``h`` is the column with each row's horizon in ns, and ``by`` the columns of ``found`` that set spans
    apart, such as the treatment."""
    keys = list(by)
    legs = [c for c in legs if c in rows.columns]
    schema = {**{k: found.schema[k] for k in keys}, "_r": rows.schema["_r"]}
    if found.is_empty() or not legs:
        return pl.DataFrame(schema=schema)
    horizons = rows.lazy().select(pl.col(h).cast(pl.Int64).alias("_h")).unique()
    reach = found.lazy().select(*keys, "entity", "start", "end").join(horizons, how="cross")
    reach = reach.select(*keys, "entity", "_h", (pl.col("start") - pl.col("_h")).alias("lo"), pl.col("end").alias("hi"))
    group = [*keys, "entity", "_h"]
    reach = reach.sort(*group, "lo").with_columns(_prior=pl.col("hi").cum_max().shift(1).over(group))
    fresh = pl.col("_prior").is_null() | (pl.col("lo") > pl.col("_prior"))
    merged = (
        reach.with_columns(_run=fresh.cum_sum().over(group))
        .group_by(*group, "_run")
        .agg(pl.col("lo").min(), pl.col("hi").max())
    )
    entities = found.lazy().select(*keys, "entity").unique()
    asked = rows.lazy().select(
        "_r", pl.col("t").dt.epoch("ns").alias("_t"), pl.col(h).cast(pl.Int64).alias("_h"), *legs
    )
    asked = asked.unpivot(index=["_r", "_t", "_h"], on=legs, value_name="_leg").select(
        "_r", "_t", "_h", pl.col("_leg").alias("entity")
    )
    asked = asked.join(entities, on="entity").sort("_t")
    hit = asked.join_asof(
        merged.sort("lo"), left_on="_t", right_on="lo", by=group, strategy="backward", check_sortedness=False
    )
    out = hit.filter(pl.col("_t") < pl.col("hi")).select(*keys, "_r").unique()
    return out.sort(*keys, "_r").collect().cast(schema)  # type: ignore[arg-type]


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

    renames: tuple[Treatment, ...]
    fields: pl.DataFrame
    cands: pl.DataFrame
    labels: pl.DataFrame
    columns: list[list[str]]
    hits: dict[str, pl.Series] = field(default_factory=dict)


class Numbers:
    """The decisive numbers. Their label rows are computed once on the explore data and, when treatments rename rows
    (a split), once more on the data those leave. A treatment that drops field rows or candidates leaves out the label
    rows that read what it drops (``spans``, ``touched``), so weighing it runs only the target again. ``recompute``
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

    def rows(self, ts: Sequence[Treatment], columns: Sequence[str]) -> pl.DataFrame:
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

    def treated(self, panel: pl.DataFrame, applied: Sequence[Treatment]) -> dict[str, pl.DataFrame]:
        """The numbers on the treated data ``panel``, labels and all, with the candidate treatments of ``applied``;
        computed once."""
        if self._treated is None:
            self._treated = self.recompute(panel, applied) if applied else self.base()
        return self._treated

    def resolution(self, panel: pl.DataFrame, applied: Sequence[Treatment]) -> dict[str, float]:
        """The MDE of each key a treatment's influence is measured in: the finer of the MDE before and after the
        treatments, since a treatment matters if it moves the number by more than the verdict on the treated data, or
        on the untreated data, could resolve."""
        if self._resolution is None:
            after = self.treated(panel, applied)
            before = {k: mde(v, k, self.width) for k, v in self.base().items()}
            now = {k: mde(v, k, self.width) for k, v in after.items()}
            self._resolution = {k: smaller(before[k], now.get(k, math.nan)) for k in before}
        return self._resolution

    def per_set(self, sets: Mapping[str, Sequence[Treatment]]) -> pl.DataFrame:
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

    def recompute(self, f: pl.DataFrame, ts: Sequence[Treatment] = ()) -> dict[str, pl.DataFrame]:
        """The numbers on the field table ``f``, with the candidate treatments in ``ts`` applied."""
        c = apply_candidates(ts, self.universe(f))
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

    def _table(self, ts: Sequence[Treatment]) -> _Table:
        renames = tuple(t for t in ts if renames_rows(t))
        if not renames:
            return self._raw_table()
        name = tuple(t.name for t in renames)
        if name not in self.tables:
            f = apply_all(renames, self.fields)
            self.tables[name] = self._build(f, self.universe(f), renames)
        return self.tables[name]

    def _raw_table(self) -> _Table:
        if () not in self.tables:
            self.tables[()] = self._build(self.fields, self.cands, ())
        return self.tables[()]

    def _build(self, f: pl.DataFrame, c: pl.DataFrame, renames: tuple[Treatment, ...]) -> _Table:
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

    def _dropped(self, table: _Table, ts: Sequence[Treatment]) -> pl.Series:
        """The label rows of ``table`` that the treatments in ``ts`` that drop rows touch."""
        drops = [t for t in ts if t.removed is not None]
        self._touch(table, [t for t in drops if t.name not in table.hits])
        parts = [table.hits[t.name] for t in drops]
        return pl.concat(parts).unique() if parts else pl.Series("_r", [], dtype=pl.UInt32)

    def _touch(self, table: _Table, ts: Sequence[Treatment]) -> None:
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
            current = spans(table.fields, reduce(lambda g, r: r.apply(g), table.renames, gone), by=["treatment"])
            legs = [c for c in LEGS if c in table.cands.columns]
            near = pl.any_horizontal(pl.col(c).is_in(current["entity"].unique().implode()) for c in legs)
            rows = table.cands.filter(near).select("_c", "t", *legs)
            rows = table.labels.select("_r", "_c", "_k").join(rows, on="_c").join(self.key_table, on="_k")
            found.append(touched(rows, current, legs, by=["treatment"]))
        on_cands = [t for t in ts if t.on == "candidates"]
        if on_cands:
            gone = pl.concat([_removed(t, table.cands).select("_c", treatment=pl.lit(t.name)) for t in on_cands])
            found.append(table.labels.select("_r", "_c").join(gone, on="_c").select("treatment", "_r"))
        hits = pl.concat(found).partition_by("treatment", as_dict=True, include_key=False)
        table.hits.update({t.name: hits[(t.name,)]["_r"] if (t.name,) in hits else found[0]["_r"] for t in ts})


def _removed(t: Treatment, table: pl.DataFrame) -> pl.DataFrame:
    assert t.removed is not None
    return t.removed(table)


def renames_rows(t: Treatment) -> bool:
    """A treatment that changes identities instead of dropping rows, such as a split."""
    return t.on == "fields" and t.removed is None


def _renames(ts: Sequence[Treatment]) -> tuple[str, ...]:
    return tuple(t.name for t in ts if renames_rows(t))


def _by_key(frame: pl.DataFrame, keys: Sequence[str]) -> dict[str, pl.DataFrame]:
    parts = frame.partition_by("key", as_dict=True, include_key=False)
    return {k: parts[(k,)].select("t", "value") if (k,) in parts else pl.DataFrame(schema=PER_T) for k in keys}


def mde(values: pl.DataFrame, key: str, width: int) -> float:
    """The smallest effect this horizon can detect; it does not depend on the effect asked for."""
    h = span_ns(span(key.rsplit(" ", 1)[1]))
    return pw.power(se(values, h), width, h).mde


def se(values: pl.DataFrame, h: int) -> float:
    """No noise can be measured on a number that is not finite somewhere."""
    return pw.block_se(values, h) if values["value"].is_finite().all() else math.nan


def smaller(a: float, b: float) -> float:
    """The finer of two resolutions."""
    finite = [x for x in (a, b) if math.isfinite(x) and x > 0]
    return min(finite) if finite else math.nan


def _fields() -> pl.DataFrame:
    return source(Source(entities=3, hours=48), 0)


def _at(hour: int) -> pl.Expr:
    return pl.lit(START_NS + hour * HOUR_NS).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")


def _counts(t: Treatment) -> tuple[int, int]:
    f = _fields()
    return t.rows, f.height - t.apply(f).height


def _windows() -> pl.DataFrame:
    return pl.DataFrame({"entity": ["e0"], "field": ["price"]}).with_columns(start=_at(10), end=_at(11))


def _flags() -> pl.DataFrame:
    return (
        _fields()
        .filter((pl.col("entity") == "e1") & (pl.col("field") == "price"))
        .head(3)
        .select("entity", "field", "event_time")
    )


def _spans() -> pl.DataFrame:
    rows = [("e0", 5, 7), ("e1", 40, 41)]
    frame = pl.DataFrame(rows, schema=["entity", "start", "end"], orient="row")
    return frame.with_columns((pl.col(c) * HOUR_NS + START_NS).cast(TIME) for c in ("start", "end"))


def _split() -> tuple[int, dict[str, int], bool, bool]:
    """e0 split at hours 20 and 40: rows renamed, rows per entity, the rest untouched, and applying it twice changes
    nothing more."""
    f = _fields()
    splits = pl.DataFrame([("e0", 20, "e0#2"), ("e0", 40, "e0#3")], schema=["entity", "at", "to"], orient="row")
    t = split_entities("s", "", splits.with_columns((pl.col("at") * HOUR_NS + START_NS).cast(TIME)), f)
    out = t.apply(f)
    counts = {str(e): int(n) for e, n in out.group_by("entity").len().iter_rows()}
    return t.rows, counts, out.drop("entity").equals(f.drop("entity")), t.apply(out).equals(out)


def _prices() -> pl.DataFrame:
    """Hourly prices of a and b at hours 0 to 10, each known an hour after its event."""
    rows = pl.DataFrame({"entity": ["a"] * 11 + ["b"] * 11, "hour": list(range(11)) * 2})
    return rows.select(
        "entity",
        field=pl.lit("price"),
        event_time=pl.from_epoch(pl.col("hour") * HOUR_NS + START_NS, time_unit="ns").dt.replace_time_zone("UTC"),
        available_at=pl.from_epoch((pl.col("hour") + 1) * HOUR_NS + START_NS, time_unit="ns").dt.replace_time_zone(
            "UTC"
        ),
        value=pl.lit(1.0),
    )


def _rows(h: int = 2, legs: bool = False) -> pl.DataFrame:
    """A label row of a and of b at every hour from 0 to 10, horizon ``h`` hours; with ``legs``, of a pair whose
    long leg is a."""
    frame = pl.DataFrame({"hour": list(range(11)) * 2, "entity": ["a"] * 11 + ["b"] * 11})
    frame = frame.with_columns(
        t=pl.from_epoch(pl.col("hour") * HOUR_NS + START_NS, time_unit="ns").dt.replace_time_zone("UTC"),
        h_ns=pl.lit(h * HOUR_NS),
    )
    if legs:
        frame = frame.with_columns(long=pl.col("entity"), short=pl.lit("c"), entity=pl.col("entity") + ">c")
    return frame.with_row_index("_r")


def _hit(hours: Sequence[int], h: int = 2, legs: bool = False) -> list[tuple[str, int]]:
    """The label rows touched when a's prices of ``hours`` are dropped, as (entity, hour)."""
    f = _prices()
    gone = f.filter((pl.col("entity") == "a") & pl.col("event_time").dt.hour().is_in(list(hours)))
    rows = _rows(h, legs)
    found = touched(rows, spans(f, gone), ["entity", "long", "short"])
    return rows.join(found, on="_r").select("entity", "hour").sort("hour").rows()


def _apart() -> list[tuple[str, int]]:
    """Spans of two treatments found at once, each touching only its own rows."""
    f = _prices()
    a = f.filter((pl.col("entity") == "a") & (pl.col("event_time").dt.hour() == 3)).with_columns(treatment=pl.lit("x"))
    b = f.filter((pl.col("entity") == "b") & (pl.col("event_time").dt.hour() == 7)).with_columns(treatment=pl.lit("y"))
    found = spans(f, pl.concat([a, b]), by=["treatment"])
    hit = touched(_rows(0), found, ["entity"], by=["treatment"])
    return hit.join(_rows(0), on="_r").select("treatment", "hour").sort("treatment", "hour").rows()


CASES = [
    Case(
        "masking a window drops its rows and an hour each side",
        lambda: _counts(mask_windows("m", "", _windows(), HOUR_NS, _fields())),
        (4, 4),
    ),
    Case("masking rows drops exactly them", lambda: _counts(mask_rows("m", "", _flags(), _fields())), (3, 3)),
    Case("dropping an entity drops all its rows", lambda: _counts(drop_entities("d", "", ["e2"], _fields())), (54, 54)),
    Case(
        "dropping spans drops their rows of every field",
        lambda: _counts(drop_spans("d", "", _spans(), _fields())),
        (6, 6),
    ),
    Case(
        "splitting renames the rows from each split on, and only them",
        _split,
        (33, {"e0": 21, "e0#2": 22, "e0#3": 11, "e1": 54, "e2": 54}, True, True),
    ),
    Case(
        "a dropped row touches the rows whose window meets the hour it was current, of its entity only",
        lambda: _hit([5]),
        [("a", 4), ("a", 5), ("a", 6)],
    ),
    Case("the last row stays current", lambda: _hit([10]), [("a", 9), ("a", 10)]),
    Case("a longer horizon reaches further back", lambda: len(_hit([5], h=4)), 5),
    Case("dropped rows next to each other make one span", lambda: _hit([5, 6], h=0), [("a", 6), ("a", 7)]),
    Case("a leg counts like the entity", lambda: _hit([5], legs=True), [("a>c", 4), ("a>c", 5), ("a>c", 6)]),
    Case("each treatment touches only its own rows", _apart, [("x", 4), ("y", 8)]),
]
