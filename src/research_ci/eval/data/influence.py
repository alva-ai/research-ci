"""Which label rows a treatment touches, so its influence on a label effect is weighed on labels computed once. A label
row at ``t`` with horizon ``h`` may read what its legs report up to ``t + h``, and what is current at ``t``; a field
row is current from when it becomes known until the next row of its entity and field does. A treatment that drops
field rows touches every label row of their entity whose window ``[t, t + h]`` meets a span in which a dropped row was
current; leaving those label rows out stands for computing the labels again on the treated data. It errs towards more
influence: a row inside the window that the label does not read still counts."""

from collections.abc import Sequence

import polars as pl

from research_ci.api.methods import Case
from research_ci.eval.synthetic import HOUR_NS, START_NS
from research_ci.kernel.time.clock import TIME

SPAN_SCHEMA = {"entity": pl.String, "event_time": TIME, "start": pl.Int64, "end": pl.Int64}
OPEN = 2**62


def spans(fields: pl.DataFrame, removed: pl.DataFrame, by: Sequence[str] = ()) -> pl.DataFrame:
    """When each removed row was current in ``fields``: ``entity``, ``event_time``, ``start`` (when it became known)
    and ``end`` (when the next row of its entity and field did; far in the future for the last), both in ns. ``by``
    names columns of ``removed`` that set its rows apart, such as the treatment; they come first."""
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
    """The label rows (``_r``) whose window ``[t, t + h]`` meets a span (``entity``, ``start``, ``end``) of one of
    their ``legs``, the columns that name entities; ``h`` is the column with each row's horizon in ns, and ``by`` the
    columns of ``found`` that set spans apart, such as the treatment: one row per group and label row touched."""
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


def _fields() -> pl.DataFrame:
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
    f = _fields()
    gone = f.filter((pl.col("entity") == "a") & pl.col("event_time").dt.hour().is_in(list(hours)))
    rows = _rows(h, legs)
    found = touched(rows, spans(f, gone), ["entity", "long", "short"])
    return rows.join(found, on="_r").select("entity", "hour").sort("hour").rows()


def _apart() -> list[tuple[str, int]]:
    """Spans of two treatments found at once, each touching only its own rows."""
    f = _fields()
    a = f.filter((pl.col("entity") == "a") & (pl.col("event_time").dt.hour() == 3)).with_columns(treatment=pl.lit("x"))
    b = f.filter((pl.col("entity") == "b") & (pl.col("event_time").dt.hour() == 7)).with_columns(treatment=pl.lit("y"))
    found = spans(f, pl.concat([a, b]), by=["treatment"])
    hit = touched(_rows(0), found, ["entity"], by=["treatment"])
    return hit.join(_rows(0), on="_r").select("treatment", "hour").sort("treatment", "hour").rows()


CASES = [
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
