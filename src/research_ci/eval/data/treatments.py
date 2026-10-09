"""Treatments: what the engine does about a flagged problem, as a function of the field table or of the candidates.
They are applied by default and kept in the freeze; their influence on the decisive number decides whether a person
has to look."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

from research_ci.api.methods import Case, Source
from research_ci.eval.data.baseline import first_vintages
from research_ci.eval.synthetic import HOUR_NS, START_NS, source
from research_ci.kernel.time.clock import TIME


@dataclass(frozen=True)
class Treatment:
    name: str
    on: Literal["fields", "candidates"]
    reason: str
    rows: int
    apply: Callable[[pl.DataFrame], pl.DataFrame]


def mask_windows(name: str, reason: str, windows: pl.DataFrame, around_ns: int, fields: pl.DataFrame) -> Treatment:
    """Drop the flagged field's rows within ``around_ns`` of each window (``entity``, ``field``, ``start``, ``end``)."""
    pad = pl.duration(nanoseconds=around_ns, time_unit="ns")
    keys = ["entity", "field"]
    w = windows.select(*keys, (pl.col("start") - pad).alias("lo"), (pl.col("end") + pad).alias("hi"))

    def apply(f: pl.DataFrame) -> pl.DataFrame:
        g = f.with_row_index("_i")
        touched = g.join(w.select(keys).unique(), on=keys, how="semi")
        hit = touched.join(w, on=keys).filter(pl.col("event_time").is_between("lo", "hi"))["_i"].unique()
        return g.filter(~pl.col("_i").is_in(hit.implode())).drop("_i")

    return Treatment(name, "fields", reason, fields.height - apply(fields).height, apply)


def mask_rows(name: str, reason: str, flagged: pl.DataFrame, fields: pl.DataFrame) -> Treatment:
    """Drop the flagged rows: by ``entity`` and ``event_time``, and by ``field`` too when the flags name one."""
    keys = ["entity", "event_time"] + (["field"] if "field" in flagged.columns else [])
    flags = flagged.select(*keys).with_columns(pl.col("event_time").cast(TIME)).unique()

    def apply(f: pl.DataFrame) -> pl.DataFrame:
        return f.join(flags, on=keys, how="anti")

    return Treatment(name, "fields", reason, fields.height - apply(fields).height, apply)


def drop_entities(name: str, reason: str, entities: Sequence[str], fields: pl.DataFrame) -> Treatment:
    names = sorted(set(entities))

    def apply(f: pl.DataFrame) -> pl.DataFrame:
        return f.filter(~pl.col("entity").is_in(names))

    return Treatment(name, "fields", reason, fields.height - apply(fields).height, apply)


def lag_field(field: str, reason: str, fields: pl.DataFrame) -> Treatment:
    """Each event of ``field`` takes the value of the entity's previous event: if values were stamped one event too
    early, this is what was known; the first event of each entity is dropped."""

    def apply(f: pl.DataFrame) -> pl.DataFrame:
        hit = pl.col("field") == field
        rows = first_vintages(f.filter(hit)).with_columns(pl.col("value").shift(1).over("entity")).drop_nulls("value")
        return pl.concat([f.filter(~hit), rows.select(f.columns)]).sort("available_at", "entity", "field")

    rows = fields.filter(pl.col("field") == field).height
    return Treatment(f"lag {field} by one event", "fields", reason, rows, apply)


def drop_candidates(name: str, reason: str, bad: pl.DataFrame) -> Treatment:
    """Drop candidate rows (``t``, ``entity``)."""
    flags = bad.select("t", "entity").unique()

    def apply(c: pl.DataFrame) -> pl.DataFrame:
        return c.join(flags, on=["t", "entity"], how="anti")

    return Treatment(name, "candidates", reason, flags.height, apply)


def apply_all(treatments: Sequence[Treatment], fields: pl.DataFrame) -> pl.DataFrame:
    for t in treatments:
        if t.on == "fields":
            fields = t.apply(fields)
    return fields


def apply_candidates(treatments: Sequence[Treatment], cands: pl.DataFrame) -> pl.DataFrame:
    for t in treatments:
        if t.on == "candidates":
            cands = t.apply(cands)
    return cands


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


def _lagged() -> list[float]:
    f = _fields()
    after = lag_field("flow", "", f).apply(f).filter((pl.col("entity") == "e0") & (pl.col("field") == "flow"))
    before = f.filter((pl.col("entity") == "e0") & (pl.col("field") == "flow")).sort("event_time")
    shifted = before["value"].head(before.height - 1).to_list()
    return [round(a - b, 12) for a, b in zip(after.sort("event_time")["value"].to_list(), shifted, strict=True)]


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
    Case("lagging a field gives each event the previous value", lambda: set(_lagged()), {0.0}),
]
