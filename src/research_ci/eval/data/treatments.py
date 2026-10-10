"""Treatments: what the engine does about a flagged problem, as a function of the field table or of the candidates.
They are applied by default and kept in the freeze; their influence on the decisive number decides whether a person
has to look."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Literal

import polars as pl

from research_ci.api.methods import Case, Source
from research_ci.eval.synthetic import HOUR_NS, START_NS, source
from research_ci.kernel.time.clock import TIME


@dataclass(frozen=True)
class Treatment:
    """``removed`` gives the rows the treatment drops from a field table or a candidate table, so its influence can be
    weighed by leaving out the label rows that read them; a treatment without it changes rows instead, such as a
    split, and only ``apply`` says what it does."""

    name: str
    on: Literal["fields", "candidates"]
    reason: str
    rows: int
    apply: Callable[[pl.DataFrame], pl.DataFrame]
    removed: Callable[[pl.DataFrame], pl.DataFrame] | None = None


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
    """Drop the flagged field's rows within ``around_ns`` of each window (``entity``, ``field``, ``start``, ``end``)."""
    pad = pl.duration(nanoseconds=around_ns, time_unit="ns")
    keys = ["entity", "field"]
    w = windows.select(*keys, (pl.col("start") - pad).alias("lo"), (pl.col("end") + pad).alias("hi"))

    def hit(f: pl.DataFrame) -> pl.Series:
        g = f.select(*keys, "event_time").with_row_index("_i")
        inside = g.join(w, on=keys).filter(pl.col("event_time").is_between("lo", "hi"))
        return g["_i"].is_in(inside["_i"].unique().implode())

    return _dropping(name, "fields", reason, fields, lambda f: f.filter(~hit(f)), lambda f: f.filter(hit(f)))


def mask_rows(name: str, reason: str, flagged: pl.DataFrame, fields: pl.DataFrame) -> Treatment:
    """Drop the flagged rows: by ``entity`` and ``event_time``, and by ``field`` too when the flags name one."""
    keys = ["entity", "event_time"] + (["field"] if "field" in flagged.columns else [])
    flags = flagged.select(*keys).with_columns(pl.col("event_time").cast(TIME)).unique()

    def apply(f: pl.DataFrame) -> pl.DataFrame:
        return f.join(flags, on=keys, how="anti", maintain_order="left")

    def removed(f: pl.DataFrame) -> pl.DataFrame:
        return f.join(flags, on=keys, how="semi", maintain_order="left")

    return _dropping(name, "fields", reason, fields, apply, removed)


def drop_entities(name: str, reason: str, entities: Sequence[str], fields: pl.DataFrame) -> Treatment:
    names = pl.Series(sorted(set(entities)), dtype=pl.String).implode()
    gone = pl.col("entity").is_in(names)
    return _dropping(name, "fields", reason, fields, lambda f: f.filter(~gone), lambda f: f.filter(gone))


def drop_spans(name: str, reason: str, spans: pl.DataFrame, fields: pl.DataFrame) -> Treatment:
    """Drop every row of each span's ``entity`` whose event falls in ``[start, end]``."""
    keys = spans.select("entity", pl.col("start").cast(TIME), pl.col("end").cast(TIME))

    def hit(f: pl.DataFrame) -> pl.Series:
        g = f.select("entity", "event_time").with_row_index("_i")
        inside = g.join(keys, on="entity").filter(pl.col("event_time").is_between("start", "end"))
        return g["_i"].is_in(inside["_i"].unique().implode())

    return _dropping(name, "fields", reason, fields, lambda f: f.filter(~hit(f)), lambda f: f.filter(hit(f)))


def split_entities(name: str, reason: str, splits: pl.DataFrame, fields: pl.DataFrame) -> Treatment:
    """Rename the rows of each split's ``entity`` whose event is at or after its ``at`` to its ``to``, up to the
    entity's next split. The rows stay; ``rows`` counts those renamed. It applies to any table with ``entity`` and
    ``event_time``."""
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
    """The field table after every field treatment: the drops at once, then the rest in order."""
    drops = [t.removed for t in treatments if t.on == "fields" and t.removed is not None]
    if drops:
        indexed = fields.with_row_index("_row")
        gone = pl.concat([removed(indexed).select("_row") for removed in drops])["_row"]
        fields = fields.filter(~pl.repeat(False, fields.height, eager=True).scatter(gone, True))
    return reduce(lambda f, t: t.apply(f), [t for t in treatments if t.on == "fields" and t.removed is None], fields)


def apply_candidates(treatments: Sequence[Treatment], cands: pl.DataFrame) -> pl.DataFrame:
    return reduce(lambda c, t: t.apply(c), [t for t in treatments if t.on == "candidates"], cands)


def _fields() -> pl.DataFrame:
    return source(Source(entities=3, hours=48), 0)


def _at(hour: int) -> pl.Expr:
    return pl.lit(START_NS + hour * HOUR_NS).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")


def _removed(t: Treatment) -> tuple[int, int]:
    f = _fields()
    return t.rows, f.height - t.apply(f).height


def _windows() -> pl.DataFrame:
    return pl.DataFrame({"entity": ["e0"], "field": ["price"]}).with_columns(start=_at(10), end=_at(11))


def _flagged() -> pl.DataFrame:
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


CASES = [
    Case(
        "masking a window drops its rows and an hour each side",
        lambda: _removed(mask_windows("m", "", _windows(), HOUR_NS, _fields())),
        (4, 4),
    ),
    Case("masking rows drops exactly them", lambda: _removed(mask_rows("m", "", _flagged(), _fields())), (3, 3)),
    Case(
        "dropping an entity drops all its rows", lambda: _removed(drop_entities("d", "", ["e2"], _fields())), (54, 54)
    ),
    Case(
        "dropping spans drops their rows of every field",
        lambda: _removed(drop_spans("d", "", _spans(), _fields())),
        (6, 6),
    ),
    Case(
        "splitting renames the rows from each split on, and only them",
        _split,
        (33, {"e0": 21, "e0#2": 22, "e0#3": 11, "e1": 54, "e2": 54}, True, True),
    ),
]
