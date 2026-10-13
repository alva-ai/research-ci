"""Reference outcome plugins. They get no privileges: the engine checks them exactly like a lab's outcomes. Each reads
the value columns it names from whichever source holds them, at that source's cadence, and gives ``y``, its parts
``y_<part>``, and ``y_end``, here ``t + h``."""

from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl

from research_ci.api.plugins import OutcomeCase

TIME = pl.Datetime("ns", "UTC")
Sources = Mapping[str, pl.DataFrame]


class HoldingReturn:
    """``p(t + h) / p(t) - 1`` of the candidate's own inst, each price as it was first known at that moment.

    A price is current until the next one is due: no row when the latest price at ``t`` or ``t + h`` is as old as the
    inst's usual spacing between prices, because the series has stopped or the data does not reach that time."""

    def __init__(self, price: str = "close") -> None:
        self.price = price

    def values(self, sources: Sources, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame:
        prices = _known(sources, self.price)
        rows = candidates.select(pl.col("t").cast(TIME), "inst")
        rows = rows.with_columns(y_end=pl.col("t") + pl.duration(nanoseconds=int(horizon.value)))
        rows = _level(rows, prices, "inst", "t", "p0")
        rows = _level(rows, prices, "inst", "y_end", "p1")
        out = rows.drop_nulls(["p0", "p1"]).select("t", "inst", (pl.col("p1") / pl.col("p0") - 1).alias("y"), "y_end")
        return out.sort("t", "inst")

    def cases(self) -> Sequence[OutcomeCase]:
        prices = [(0, 100.0), (1, 104.0), (2, 110.0), (3, 99.0), (4, 121.0)]
        sources = _sources(self.price, [("x", h, v) for h, v in prices])
        cands = _cands([(0, "x"), (2, "x")])
        expected = _rows([(0, "x", 0.10), (2, "x", 0.10)], ["y"])
        return [OutcomeCase(sources, cands, "2h", expected)]


class Markout:
    """``side * (p(t + h) / p(t) - 1)`` for event rows: candidates carry a ``side`` column (+1 bought, -1 sold)."""

    def __init__(self, price: str = "close") -> None:
        self.price = price

    def values(self, sources: Sources, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame:
        base = HoldingReturn(self.price).values(sources, candidates.select("t", "inst"), horizon)
        sides = candidates.select(pl.col("t").cast(TIME), "inst", pl.col("side").cast(pl.Float64))
        return base.join(sides, on=["t", "inst"]).select("t", "inst", pl.col("y") * pl.col("side"), "y_end")

    def cases(self) -> Sequence[OutcomeCase]:
        sources = _sources(self.price, [("x", h, v) for h, v in [(0, 100.0), (1, 98.0), (2, 95.0), (3, 97.0)]])
        cands = _cands([(0, "x")]).with_columns(side=pl.lit(-1.0))
        return [OutcomeCase(sources, cands, "2h", _rows([(0, "x", 0.05)], ["y"]))]


class PairCarry:
    """A long/short pair held from ``t`` to ``t + h``, before costs, per unit of one leg's notional.

    ``y_price``: the long leg's return minus the short leg's, from the column ``price``. ``y_flow`` (with ``flow``):
    the flow on the short leg minus the flow on the long leg, from the column ``flow``, over events that became known
    in ``(t, t + h]``, as when shorts receive a positive funding rate. ``y`` is their sum. Candidates name their legs
    in columns ``long`` and ``short``. ``values_each`` gives every horizon from one pass over what does not depend on
    the horizon: the levels and running flows at ``t``."""

    def __init__(self, price: str = "close", flow: str | None = None) -> None:
        self.price, self.flow = price, flow

    def values(self, sources: Sources, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame:
        return next(self.values_each(sources, candidates, [horizon]))

    def values_each(
        self, sources: Sources, candidates: pl.DataFrame, horizons: Sequence[pd.Timedelta]
    ) -> Iterator[pl.DataFrame]:
        prices = _known(sources, self.price)
        start = candidates.select(pl.col("t").cast(TIME), "inst", "long", "short")
        for leg in ("long", "short"):
            start = _level(start, prices, leg, "t", f"p_{leg}0")
        flows = None if self.flow is None else _cumulative(sources, self.flow)
        if flows is not None:
            for leg in ("long", "short"):
                start = _running(start, flows, leg, "t", f"c_{leg}0")
        return (self._held(start, prices, flows, h) for h in horizons)

    @staticmethod
    def _held(
        start: pl.DataFrame, prices: pl.DataFrame, flows: pl.DataFrame | None, horizon: pd.Timedelta
    ) -> pl.DataFrame:
        rows = start.with_columns(y_end=pl.col("t") + pl.duration(nanoseconds=int(horizon.value)))
        for leg in ("long", "short"):
            rows = _level(rows, prices, leg, "y_end", f"p_{leg}1")
        rows = rows.drop_nulls(["p_long0", "p_long1", "p_short0", "p_short1"])
        price = (pl.col("p_long1") / pl.col("p_long0")) - (pl.col("p_short1") / pl.col("p_short0"))
        rows = rows.with_columns(price.alias("y_price"))
        parts = ["y_price"]
        if flows is not None:
            for leg in ("long", "short"):
                rows = _running(rows, flows, leg, "y_end", f"c_{leg}1")
            flow = (pl.col("c_short1") - pl.col("c_short0")) - (pl.col("c_long1") - pl.col("c_long0"))
            rows = rows.with_columns(flow.alias("y_flow"))
            parts.append("y_flow")
        return rows.select("t", "inst", pl.sum_horizontal(parts).alias("y"), *parts, "y_end").sort("t", "inst")

    def cases(self) -> Sequence[OutcomeCase]:
        a = [100, 101, 99, 102, 103, 101, 104, 105, 103, 106, 107, 105, 108]
        b = [50, 50.5, 50.2, 51, 51.3, 50.8, 52, 52.4, 51.9, 53, 53.2, 52.8, 54]
        prices = [("a", h, float(v)) for h, v in enumerate(a)] + [("b", h, v) for h, v in enumerate(b)]
        four, eight = [104 / 99 - 52 / 50.2], [107 / 99 - 53.2 / 50.2]
        columns = ["y", "y_price"]
        flows = None
        if self.flow is not None:
            flows = [("a", 5, 0.0002), ("b", 5, 0.0006), ("b", 10, 0.0004), ("a", 12, 0.0001)]
            four, eight, columns = [*four, 0.0004], [*eight, 0.0008], [*columns, "y_flow"]
        sources = _sources(self.price, prices, self.flow, flows)
        cands = _cands([(2, "a>b", "a", "b")])
        return [
            OutcomeCase(sources, cands, "4h", _rows([(2, "a>b", sum(four), *four)], columns)),
            OutcomeCase(sources, cands, "8h", _rows([(2, "a>b", sum(eight), *eight)], columns)),
        ]


def _column(sources: Sources, name: str) -> pl.DataFrame:
    """The cells of the value column ``name`` that hold a value, from the source that holds it: ``inst``,
    ``event_time``, ``available_at``, ``vintage``, ``value``; none when no source holds it."""
    found = [t for t in sources.values() if name in t.columns]
    schema = {"inst": pl.String, "event_time": TIME, "available_at": TIME, "vintage": pl.Int64, "value": pl.Float64}
    if not found:
        return pl.DataFrame(schema=schema)
    rows = found[0].select("inst", "event_time", "available_at", "vintage", pl.col(name).alias("value"))
    return rows.drop_nulls("value").cast(schema)  # type: ignore[arg-type]


def _known(sources: Sources, name: str) -> pl.DataFrame:
    """Each inst's latest level by event time, as a step function of when it became known (first vintages), with
    the inst's usual spacing between levels."""
    rows = _column(sources, name).sort("vintage")
    rows = rows.unique(subset=["inst", "event_time"], keep="first", maintain_order=True).sort(
        "available_at", "event_time"
    )
    rows = rows.filter(pl.col("event_time") == pl.col("event_time").cum_max().over("inst"))
    out = rows.select("inst", pl.col("available_at").alias("known_at"), "value")
    out = out.unique(subset=["inst", "known_at"], keep="last", maintain_order=True)
    spacing = pl.col("known_at").diff().median().over("inst")
    return out.with_columns(seen=pl.col("known_at"), spacing=spacing).sort("known_at")


def _cumulative(sources: Sources, name: str) -> pl.DataFrame:
    rows = _column(sources, name).sort("vintage")
    rows = rows.unique(subset=["inst", "event_time"], keep="first", maintain_order=True).sort("available_at")
    out = rows.select("inst", pl.col("available_at").alias("known_at"), pl.col("value").cum_sum().over("inst"))
    return out.unique(subset=["inst", "known_at"], keep="last", maintain_order=True).sort("known_at")


def _level(rows: pl.DataFrame, known: pl.DataFrame, by: str, at: str, name: str) -> pl.DataFrame:
    indexed = rows.with_row_index("_row")
    query = indexed.select("_row", pl.col(by).alias("inst"), pl.col(at).alias("_at")).sort("_at")
    hit = query.join_asof(known, left_on="_at", right_on="known_at", by="inst", check_sortedness=False)
    age = pl.col("_at") - pl.col("seen")
    current = (age < pl.col("spacing")) | (pl.col("spacing").is_null() & (age == pl.duration(nanoseconds=0)))
    value = pl.when(current).then(pl.col("value")).otherwise(None)
    hit = hit.select("_row", value.alias(name))
    return indexed.join(hit, on="_row").drop("_row")


def _running(rows: pl.DataFrame, known: pl.DataFrame, by: str, at: str, name: str) -> pl.DataFrame:
    indexed = rows.with_row_index("_row")
    query = indexed.select("_row", pl.col(by).alias("inst"), pl.col(at).alias("_at")).sort("_at")
    hit = query.join_asof(known, left_on="_at", right_on="known_at", by="inst", check_sortedness=False)
    hit = hit.select("_row", pl.col("value").fill_null(0.0).alias(name))
    return indexed.join(hit, on="_row").drop("_row")


EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _at(hour: int) -> datetime:
    return EPOCH + timedelta(hours=hour)


def _sources(
    price: str,
    prices: list[tuple[str, int, float]],
    flow: str | None = None,
    flows: list[tuple[str, int, float]] | None = None,
) -> dict[str, pl.DataFrame]:
    """Hourly ``prices`` in a source ``prices`` and, with ``flow``, its ``flows`` in a source ``flows``, each known at
    its event."""
    out = {"prices": _table(price, prices)}
    if flow is not None and flows is not None:
        out["flows"] = _table(flow, flows)
    return out


def _table(column: str, rows: list[tuple[str, int, float]]) -> pl.DataFrame:
    frame = pl.DataFrame(
        [(e, _at(h), _at(h), 0, v) for e, h, v in rows],
        schema=["inst", "event_time", "available_at", "vintage", column],
        orient="row",
    )
    return frame.with_columns(pl.col("event_time").cast(TIME), pl.col("available_at").cast(TIME))


def _cands(rows: list[tuple]) -> pl.DataFrame:
    names = ["t", "inst", "long", "short"][: len(rows[0])]
    frame = pl.DataFrame([(_at(r[0]), *r[1:]) for r in rows], schema=names, orient="row")
    return frame.with_columns(pl.col("t").cast(TIME))


def _rows(rows: list[tuple], values: list[str]) -> pl.DataFrame:
    frame = pl.DataFrame([(_at(r[0]), *r[1:]) for r in rows], schema=["t", "inst", *values], orient="row")
    return frame.with_columns(pl.col("t").cast(TIME))
