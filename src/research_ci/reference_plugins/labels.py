"""Reference label plugins. They get no privileges: the engine checks them exactly like a lab's labels."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl

from research_ci.api.plugins import LabelCase

TIME = pl.Datetime("ns", "UTC")


class HoldingReturn:
    """``p(t + h) / p(t) - 1`` of the candidate's own entity, each price as it was first known at that moment.

    A price is current until the next one is due: no row when the latest price at ``t`` or ``t + h`` is as old as the
    entity's usual spacing between prices, because the series has stopped or the data does not reach that time."""

    def __init__(self, price: str = "close") -> None:
        self.price = price

    def values(self, fields: pl.DataFrame, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame:
        prices = _known(fields, self.price)
        rows = candidates.select(pl.col("t").cast(TIME), "entity")
        rows = rows.with_columns(end=pl.col("t") + pl.duration(nanoseconds=int(horizon.value)))
        rows = _level(rows, prices, "entity", "t", "p0")
        rows = _level(rows, prices, "entity", "end", "p1")
        out = rows.drop_nulls(["p0", "p1"]).select("t", "entity", (pl.col("p1") / pl.col("p0") - 1).alias("value"))
        return out.sort("t", "entity")

    def cases(self) -> Sequence[LabelCase]:
        prices = [(0, 100.0), (1, 104.0), (2, 110.0), (3, 99.0), (4, 121.0)]
        fields = _fields([("x", self.price, h, v) for h, v in prices])
        cands = _cands([(0, "x"), (2, "x")])
        expected = _rows([(0, "x", 0.10), (2, "x", 0.10)], ["value"])
        return [LabelCase(fields, cands, "2h", expected)]


class Markout:
    """``side * (p(t + h) / p(t) - 1)`` for event rows: candidates carry a ``side`` column (+1 bought, -1 sold)."""

    def __init__(self, price: str = "close") -> None:
        self.price = price

    def values(self, fields: pl.DataFrame, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame:
        base = HoldingReturn(self.price).values(fields, candidates.select("t", "entity"), horizon)
        sides = candidates.select(pl.col("t").cast(TIME), "entity", pl.col("side").cast(pl.Float64))
        return base.join(sides, on=["t", "entity"]).select("t", "entity", pl.col("value") * pl.col("side"))

    def cases(self) -> Sequence[LabelCase]:
        prices = [("x", self.price, h, v) for h, v in [(0, 100.0), (1, 98.0), (2, 95.0), (3, 97.0)]]
        fields = _fields(prices)
        cands = _cands([(0, "x")]).with_columns(side=pl.lit(-1.0))
        return [LabelCase(fields, cands, "2h", _rows([(0, "x", 0.05)], ["value"]))]


class PairCarry:
    """A long/short pair held from ``t`` to ``t + h``, before costs, per unit of one leg's notional.

    ``price``: the long leg's return minus the short leg's. ``flow`` (optional): the flow on the short leg minus the
    flow on the long leg, over events that became known in ``(t, t + h]``, as when shorts receive a positive funding
    rate. Candidates name their legs in columns ``long`` and ``short``."""

    def __init__(self, price: str = "close", flow: str | None = None) -> None:
        self.price, self.flow = price, flow

    def values(self, fields: pl.DataFrame, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame:
        prices = _known(fields, self.price)
        rows = candidates.select(pl.col("t").cast(TIME), "entity", "long", "short")
        rows = rows.with_columns(end=pl.col("t") + pl.duration(nanoseconds=int(horizon.value)))
        for leg in ("long", "short"):
            rows = _level(rows, prices, leg, "t", f"p_{leg}0")
            rows = _level(rows, prices, leg, "end", f"p_{leg}1")
        rows = rows.drop_nulls(["p_long0", "p_long1", "p_short0", "p_short1"])
        price = (pl.col("p_long1") / pl.col("p_long0")) - (pl.col("p_short1") / pl.col("p_short0"))
        rows = rows.with_columns(price.alias("price"))
        parts = ["price"]
        if self.flow is not None:
            flows = _cumulative(fields, self.flow)
            for leg in ("long", "short"):
                rows = _running(rows, flows, leg, "t", f"c_{leg}0")
                rows = _running(rows, flows, leg, "end", f"c_{leg}1")
            flow = (pl.col("c_short1") - pl.col("c_short0")) - (pl.col("c_long1") - pl.col("c_long0"))
            rows = rows.with_columns(flow.alias("flow"))
            parts.append("flow")
        return rows.select("t", "entity", pl.sum_horizontal(parts).alias("value"), *parts).sort("t", "entity")

    def cases(self) -> Sequence[LabelCase]:
        a = [100, 101, 99, 102, 103, 101, 104, 105, 103, 106, 107, 105, 108]
        b = [50, 50.5, 50.2, 51, 51.3, 50.8, 52, 52.4, 51.9, 53, 53.2, 52.8, 54]
        rows = [("a", self.price, h, float(v)) for h, v in enumerate(a)]
        rows += [("b", self.price, h, v) for h, v in enumerate(b)]
        four, eight = [104 / 99 - 52 / 50.2], [107 / 99 - 53.2 / 50.2]
        columns = ["value", "price"]
        if self.flow is not None:
            rows += [("a", self.flow, 5, 0.0002), ("b", self.flow, 5, 0.0006), ("b", self.flow, 10, 0.0004)]
            rows += [("a", self.flow, 12, 0.0001)]
            four, eight, columns = [*four, 0.0004], [*eight, 0.0008], [*columns, "flow"]
        fields = _fields(rows)
        cands = _cands([(2, "a>b", "a", "b")])
        return [
            LabelCase(fields, cands, "4h", _rows([(2, "a>b", sum(four), *four)], columns)),
            LabelCase(fields, cands, "8h", _rows([(2, "a>b", sum(eight), *eight)], columns)),
        ]


def _known(fields: pl.DataFrame, name: str) -> pl.DataFrame:
    """Each entity's latest level by event time, as a step function of when it became known (first vintages), with
    the entity's usual spacing between levels."""
    rows = fields.filter(pl.col("field") == name).sort("vintage")
    rows = rows.unique(subset=["entity", "event_time"], keep="first", maintain_order=True).sort(
        "available_at", "event_time"
    )
    rows = rows.filter(pl.col("event_time") == pl.col("event_time").cum_max().over("entity"))
    out = rows.select("entity", pl.col("available_at").alias("known_at"), "value")
    out = out.unique(subset=["entity", "known_at"], keep="last", maintain_order=True)
    spacing = pl.col("known_at").diff().median().over("entity")
    return out.with_columns(seen=pl.col("known_at"), spacing=spacing).sort("known_at")


def _cumulative(fields: pl.DataFrame, name: str) -> pl.DataFrame:
    rows = fields.filter(pl.col("field") == name).sort("vintage")
    rows = rows.unique(subset=["entity", "event_time"], keep="first", maintain_order=True).sort("available_at")
    out = rows.select("entity", pl.col("available_at").alias("known_at"), pl.col("value").cum_sum().over("entity"))
    return out.unique(subset=["entity", "known_at"], keep="last", maintain_order=True).sort("known_at")


def _level(rows: pl.DataFrame, known: pl.DataFrame, by: str, at: str, name: str) -> pl.DataFrame:
    indexed = rows.with_row_index("_row")
    query = indexed.select("_row", pl.col(by).alias("entity"), pl.col(at).alias("_at")).sort("_at")
    hit = query.join_asof(known, left_on="_at", right_on="known_at", by="entity", check_sortedness=False)
    age = pl.col("_at") - pl.col("seen")
    current = (age < pl.col("spacing")) | (pl.col("spacing").is_null() & (age == pl.duration(nanoseconds=0)))
    value = pl.when(current).then(pl.col("value")).otherwise(None)
    hit = hit.select("_row", value.alias(name))
    return indexed.join(hit, on="_row").drop("_row")


def _running(rows: pl.DataFrame, known: pl.DataFrame, by: str, at: str, name: str) -> pl.DataFrame:
    indexed = rows.with_row_index("_row")
    query = indexed.select("_row", pl.col(by).alias("entity"), pl.col(at).alias("_at")).sort("_at")
    hit = query.join_asof(known, left_on="_at", right_on="known_at", by="entity", check_sortedness=False)
    hit = hit.select("_row", pl.col("value").fill_null(0.0).alias(name))
    return indexed.join(hit, on="_row").drop("_row")


EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _at(hour: int) -> datetime:
    return EPOCH + timedelta(hours=hour)


def _fields(rows: list[tuple[str, str, int, float]]) -> pl.DataFrame:
    frame = pl.DataFrame(
        [(e, f, _at(h), _at(h), v, 0, "") for e, f, h, v in rows],
        schema=["entity", "field", "event_time", "available_at", "value", "vintage", "source"],
        orient="row",
    )
    return frame.with_columns(pl.col("event_time").cast(TIME), pl.col("available_at").cast(TIME))


def _cands(rows: list[tuple]) -> pl.DataFrame:
    names = ["t", "entity", "long", "short"][: len(rows[0])]
    frame = pl.DataFrame([(_at(r[0]), *r[1:]) for r in rows], schema=names, orient="row")
    return frame.with_columns(pl.col("t").cast(TIME))


def _rows(rows: list[tuple], values: list[str]) -> pl.DataFrame:
    frame = pl.DataFrame([(_at(r[0]), *r[1:]) for r in rows], schema=["t", "entity", *values], orient="row")
    return frame.with_columns(pl.col("t").cast(TIME))
