"""Small plugins on synthetic data for the engine's own tests, with broken variants the protocol tests must catch."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api import ConsumerResult, Hedge, Levels, OutcomeCase, Reference, Timing
from research_ci.kernel.tables import source_tables, values
from research_ci.kernel.time.asof import wide
from research_ci.kernel.time.clock import TIME
from research_ci.reference_plugins.outcomes import PairCarry
from research_ci.reference_plugins.targets import ClusterMean

START = pd.Timestamp("2026-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)
SYMBOLS = ("X", "Y", "Z")
VENUES = ("a", "b", "c")
SETTLE = 8
Sources = Mapping[str, pl.DataFrame]
NOT_KEPT = "funding_interval_hours"


def toy_sources(hours: int = 24 * 25, seed: int = 0) -> dict[str, pl.DataFrame]:
    """Hourly closes (source ``candles``) and, every ``SETTLE`` hours, funding rates and their interval (source
    ``funding``) of three symbols on three venues; insts are ``<venue>:<symbol>``, each row known at its event."""
    rng = np.random.default_rng(seed)
    candles, funding = [], []
    settles = np.arange(SETTLE, hours + 1, SETTLE)
    for symbol in SYMBOLS:
        common = np.cumsum(rng.normal(0, 0.004, hours))
        for i, venue in enumerate(VENUES):
            inst = f"{venue}:{symbol}"
            close = 100 * np.exp(common + np.cumsum(rng.normal(0, 0.0005, hours)))
            level = rng.normal(0.0004 * (i + 1), 0.0001)
            rates = level + rng.normal(0, 0.00005, settles.size)
            candles.append(pl.DataFrame({"inst": inst, "hour": np.arange(1, hours + 1), "close": close}))
            funding.append(
                pl.DataFrame({"inst": inst, "hour": settles, "funding_rate": rates, NOT_KEPT: float(SETTLE)})
            )
    return source_tables({"candles": _timed(pl.concat(candles)), "funding": _timed(pl.concat(funding))})


def _timed(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows of hours since ``START``, each known at its event."""
    at = (pl.lit(START) + pl.duration(hours=pl.col("hour"), time_unit="ns")).cast(TIME)
    return frame.select(
        "inst", at.alias("event_time"), at.alias("available_at"), pl.lit(0).alias("vintage"), pl.exclude("inst", "hour")
    )


class ToyData:
    """Replays as of any time, and answers reconciliation from the venue, here the same generator run apart; the
    venue keeps no history of funding intervals."""

    sources = {"candles": ("close",), "funding": ("funding_rate", NOT_KEPT)}
    levels = Levels(columns=("close",), factor=5.0, window=24, mask="24h")

    reference_note = "the toy venue"

    def __init__(self, hours: int = 24 * 25, seed: int = 0) -> None:
        self.hours, self.seed = hours, seed

    def fetch(self, as_of: Any = None, insts: Sequence[str] | None = None) -> dict[str, pl.DataFrame]:
        tables = self.build(as_of)
        keep = pl.lit(True)
        if as_of is not None:
            keep = keep & (pl.col("available_at") <= pd.Timestamp(as_of))
        if insts is not None:
            keep = keep & pl.col("inst").is_in(list(insts))
        return {name: t.filter(keep) for name, t in tables.items()}

    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        return toy_sources(self.hours, self.seed)

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(toy_sources(self.hours, self.seed), requests)


def venue_reference(venue: Sources, requests: pl.DataFrame) -> Reference:
    """What the toy venue answers from ``venue``; it keeps no history of funding intervals."""
    answered = requests.filter(pl.col("column") != NOT_KEPT)
    asked = answered.select("source", "inst", "start", "end")
    found = {
        name: table.join(asked.filter(pl.col("source") == name).drop("source"), on="inst")
        .filter(pl.col("event_time").is_between(pl.col("start"), pl.col("end")))
        .select("inst", "event_time", *(c for c in values(table) if c != NOT_KEPT))
        .unique()
        for name, table in venue.items()
    }
    return Reference(found, answered.select("rid"))


def _edited(tables: Sources, name: str, how: Any) -> dict[str, pl.DataFrame]:
    return {**tables, name: how(tables[name])}


def timed_sources(hours: int = 24 * 25, seed: int = 0, drift: float = 0.002) -> dict[str, pl.DataFrame]:
    """The toy market with a timing signal planted: each symbol has a regime, +1 or -1 for stretches of one to four
    days, which every inst publishes each hour next to its close as the column ``regime``, and whose closes drift with
    it, by ``drift`` per hour, over the next hour."""
    tables = toy_sources(hours, seed)
    rng = np.random.default_rng(seed + 1)
    plans = []
    for symbol in SYMBOLS:
        sign, regime = float(rng.choice([-1.0, 1.0])), []
        while len(regime) < hours:
            regime += [sign] * int(rng.integers(24, 97))
            sign = -sign
        side = np.array(regime[:hours])
        lift = np.exp(np.concatenate([[0.0], np.cumsum(drift * side)[:-1]]))
        plans.append(pl.DataFrame({"symbol": symbol, "h": np.arange(hours), "lift": lift, "regime": side}))
    plan = pl.concat(plans)
    hour = ((pl.col("event_time") - pl.lit(START)).dt.total_hours() - 1).alias("h")
    candles = tables["candles"].with_columns(hour, symbol=pl.col("inst").str.split(":").list.get(1))
    candles = candles.join(plan, on=["symbol", "h"]).with_columns(close=pl.col("close") * pl.col("lift"))
    candles = candles.select("inst", "event_time", "available_at", "vintage", "close", "regime")
    return source_tables({**tables, "candles": candles})


class ToyTimed(ToyData):
    """The toy market with a planted timing signal (``timed_sources``); the venue reports it too."""

    sources = {"candles": ("close", "regime"), "funding": ("funding_rate", NOT_KEPT)}

    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        return timed_sources(self.hours, self.seed)

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToySeries:
    """The ``insts`` (one asset, by default) at every decision time once a close of theirs is known."""

    def __init__(self, insts: Sequence[str] = ("a:X",)) -> None:
        self.insts = list(insts)

    def candidates(self, wide: pl.DataFrame, sources: Sources) -> pl.DataFrame:
        listed = wide.filter(pl.col("inst").is_in(self.insts) & pl.col("age_candles").is_not_null())
        return listed.select("t", "inst").sort("t", "inst")


class ToyEvents(ToySeries):
    """One asset per symbol, every one a candidate at every decision time; the features score only the events."""

    def __init__(self) -> None:
        super().__init__(("a:X", "a:Y", "a:Z"))


class ToyHindsight(ToyData):
    """Drops a symbol's closes in a window once it knows a jump follows, as an ex-post repair would."""

    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        tables = super().build()
        lo, hi = START + 200 * HOUR, START + 206 * HOUR
        if as_of is not None and pd.Timestamp(as_of) < hi + HOUR:
            return tables
        window = (pl.col("inst") == "b:Y") & pl.col("event_time").is_between(lo, hi)
        return _edited(tables, "candles", lambda t: t.filter(~window))


class ToyNoAsOf(ToyData):
    def fetch(self) -> dict[str, pl.DataFrame]:  # type: ignore[override]
        return super().build()


class ToyCorrupt(ToyData):
    """Its closes for one venue are off by a tenth from what the venue reports."""

    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        hit = pl.col("inst").str.starts_with("c:")
        off = pl.when(hit).then(pl.col("close") * 1.1).otherwise("close")
        return _edited(super().build(), "candles", lambda t: t.with_columns(close=off))


class ToySwapped(ToyData):
    """Venue c lists another asset under the ticker Z: its closes follow their own path at the same level."""

    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        tables = super().build()
        hit = pl.col("inst") == "c:Z"
        candles = tables["candles"]
        other = candles.filter(hit).sort("event_time")
        rng = np.random.default_rng(99)
        path = float(other["close"][0]) * np.exp(np.cumsum(rng.normal(0, 0.01, other.height)))
        swapped = pl.concat([candles.filter(~hit), other.with_columns(close=pl.Series(path))]).sort("available_at")
        return {**tables, "candles": swapped}

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyRelaunched(ToyData):
    """Venue c delists Z after hour 150 and, twelve hours later, lists another asset under the same ticker at a
    thousand times the level. The venue reports both, so the source is faithful."""

    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        z, t = pl.col("inst") == "c:Z", pl.col("event_time")
        gone = z & t.is_between(START + 151 * HOUR, START + 161 * HOUR)
        later = z & (t >= START + 162 * HOUR)
        tables = {name: table.filter(~gone) for name, table in super().build().items()}
        return _edited(
            tables,
            "candles",
            lambda c: c.with_columns(close=pl.when(later).then(pl.col("close") * 1000).otherwise("close")),
        )

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyRenamed(ToyRelaunched):
    """The relaunch of ``ToyRelaunched`` under a name of its own, ``c:Z#2``, as splitting it leaves it."""

    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        later = (pl.col("inst") == "c:Z") & (pl.col("event_time") >= START + 162 * HOUR)
        renamed = pl.when(later).then(pl.lit("c:Z#2")).otherwise("inst")
        return {name: t.with_columns(inst=renamed) for name, t in super().build().items()}


class ToyZeros(ToyData):
    """Venue b reports a close of zero for X three hours running, and so does the venue's API."""

    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        hours = pl.col("event_time").is_between(START + 100 * HOUR, START + 102 * HOUR)
        hit = (pl.col("inst") == "b:X") & hours
        return _edited(
            super().build(), "candles", lambda t: t.with_columns(close=pl.when(hit).then(0.0).otherwise("close"))
        )

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyShort(ToyData):
    """Venue c lists Y for twenty hours only, shorter than the longest horizon."""

    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        brief = pl.col("event_time").is_between(START + 300 * HOUR, START + 319 * HOUR)
        return {name: t.filter((pl.col("inst") != "c:Y") | brief) for name, t in super().build().items()}

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyPairs:
    """Every venue pair of a symbol; long the venue with the lower latest funding rate per hour."""

    peek = False
    hedge = Hedge(price="close", max_ratio=1.2, min_corr=0.3, window=72, min_obs=48, min_move=1e-3)

    def candidates(self, wide: pl.DataFrame, sources: Sources) -> pl.DataFrame:
        found = _next_rate(wide, sources) if self.peek else wide.select("t", "inst", "funding_rate")
        known = found.select("t", "inst", (pl.col("funding_rate") / SETTLE).alias("rate")).drop_nulls("rate")
        parts = pl.col("inst").str.split(":")
        known = known.with_columns(venue=parts.list.get(0), symbol=parts.list.get(1))
        pairs = known.join(known, on=["t", "symbol"], suffix="_b").filter(pl.col("venue") < pl.col("venue_b"))
        lower = pl.col("rate") <= pl.col("rate_b")
        long, short = [pl.when(lower).then(a).otherwise(b) for a, b in (("inst", "inst_b"), ("inst_b", "inst"))]
        lv, sv = [pl.when(lower).then(a).otherwise(b) for a, b in (("venue", "venue_b"), ("venue_b", "venue"))]
        spread = (pl.col("rate") - pl.col("rate_b")).abs()
        out = pairs.select(
            "t",
            pl.format("{}:{}>{}", "symbol", lv, sv).alias("inst"),
            long.alias("long"),
            short.alias("short"),
            "symbol",
            pl.format("{}>{}", lv, sv).alias("venue_pair"),
            spread.alias("spread_h"),
            spread.alias("score"),
            pl.lit(True).alias("openable"),
        )
        return out.sort("t", "inst")


class ToyPeekPairs(ToyPairs):
    peek = True


def _next_rate(wide: pl.DataFrame, sources: Sources) -> pl.DataFrame:
    """At each decision time, the funding rate of the settlement after the latest one known: a peek at the future,
    read from the native rows."""
    funding = sources["funding"].sort("inst", "available_at")
    ahead = funding.select("inst", "available_at", pl.col("funding_rate").shift(-1).over("inst")).drop_nulls()
    grid = wide.select("t", "inst").sort("t")
    return grid.join_asof(
        ahead.sort("available_at"), left_on="t", right_on="available_at", by="inst", check_sortedness=False
    )


class ToyNeedsNextBar(ToyPairs):
    """Correct, but like many backtest loops it outputs nothing on the last bar it is given."""

    def candidates(self, wide: pl.DataFrame, sources: Sources) -> pl.DataFrame:
        last = max(t.select(pl.col("available_at").max()).item() for t in sources.values())
        return super().candidates(wide.filter(pl.col("t") < last), sources)


class ToyCarry:
    """Holds the top ``k`` pairs by score, equal weight per leg; flow and price accrue per hour after the decision. A
    flow known in ``(t, t + 1h]`` is paid to the positions decided at ``t``."""

    early = flip = fee_free = noisy = peek = same_instant = False

    def __init__(self, k: int = 2, commission_bps: float = 1.0, slippage_bps: float = 1.0) -> None:
        self.k, self.cost = k, (commission_bps + slippage_bps) / 1e4

    def timing(self) -> Timing:
        return Timing(every="1h", note="decide on the hour; fill at the close known then")

    def run(self, sources: Sources, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        close = sources["candles"].pivot(on="inst", index="available_at", values="close").sort("available_at")
        times = close["available_at"].to_list()
        ns = close["available_at"].dt.epoch("ns").to_list()
        flows = sources["funding"]
        cands = ToyPairs().candidates(wide(sources, ns), sources)
        if scores is not None:
            cands = cands.drop("score").join(scores.select("t", "inst", "score"), on=["t", "inst"], how="inner")
        if self.peek:
            cands = cands.drop("score").join(self._future_flow(sources, ns), on=["t", "inst"])
        by_t = {t: g for (t,), g in cands.group_by("t")}
        flow_at: dict[tuple[str, Any], float] = defaultdict(float)
        bucket = bisect_right if self.same_instant else bisect_left
        for e, a, v in flows.select("inst", "available_at", "funding_rate").iter_rows():
            k = bucket(times, a)
            if k < len(times):
                flow_at[(e, times[k])] += v
        prices = {c: close[c].to_list() for c in close.columns if c != "available_at"}
        weights: dict[str, float] = {}
        legs: dict[str, tuple[str, str]] = {}
        net_rows, pos_rows, trade_rows, turnover = [], [], [], 0.0
        for i, t in enumerate(times[:-1]):
            group = by_t.get(t)
            chosen = {} if group is None else self._choose(group)
            for e, (lo, sh) in chosen.items():
                legs[e] = (lo, sh)
            target = dict.fromkeys(chosen, 1.0 / self.k)
            traded = 0.0
            for e in sorted(set(weights) | set(target)):
                before, after = weights.get(e, 0.0), target.get(e, 0.0)
                if before != after:
                    trade_rows.append((t, e, before, after))
                    traded += abs(after - before) * 2
            turnover += traded
            weights = target
            nxt = times[i + 1]
            credit = times[i + 2] if self.early and i + 2 < len(times) else nxt
            flow = price = 0.0
            for e, w in weights.items():
                lo, sh = legs[e]
                f = flow_at.get((sh, credit), 0.0) - flow_at.get((lo, credit), 0.0)
                flow += w * (-f if self.flip else f)
                pl_, ps = prices[lo], prices[sh]
                price += w * ((pl_[i + 1] / pl_[i] - 1) - (ps[i + 1] / ps[i] - 1))
                pos_rows.append((t, e, w, lo, sh))
            fees = 0.0 if self.fee_free else -traded * self.cost
            net = flow + price + fees + (np.random.normal(0, 1e-6) if self.noisy else 0.0)
            net_rows.append((nxt, net, flow, price, fees))
        net = pl.DataFrame(net_rows, schema=["t", "net", "flow", "price", "fees"], orient="row")
        positions = pl.DataFrame(pos_rows, schema=["t", "inst", "weight", "long", "short"], orient="row")
        trades = pl.DataFrame(trade_rows, schema=["t", "inst", "from_weight", "to_weight"], orient="row")
        cast = pl.col("t").cast(TIME)
        return ConsumerResult(
            net=net.with_columns(cast),
            positions=positions.with_columns(cast),
            trades=trades.with_columns(cast),
            turnover=turnover,
            checks={"toy": True},
            policies={},
        )

    def _choose(self, group: pl.DataFrame) -> dict[str, tuple[str, str]]:
        top = group.sort(["score", "inst"], descending=[True, False]).head(self.k)
        return {e: (lo, sh) for e, lo, sh in top.select("inst", "long", "short").iter_rows()}

    def _future_flow(self, sources: Sources, ns: list[int]) -> pl.DataFrame:
        cands = ToyPairs().candidates(wide(sources, ns), sources)
        flows = sources["funding"].select("inst", "available_at", "funding_rate")
        rows = []
        for t, e, lo, sh in cands.select("t", "inst", "long", "short").iter_rows():
            window = flows.filter((pl.col("available_at") > t) & (pl.col("available_at") <= t + 24 * HOUR))
            s = float(window.filter(pl.col("inst") == sh)["funding_rate"].sum()) - float(
                window.filter(pl.col("inst") == lo)["funding_rate"].sum()
            )
            rows.append((t, e, s))
        return pl.DataFrame(rows, schema=["t", "inst", "score"], orient="row").with_columns(pl.col("t").cast(TIME))


class EarlyCredit(ToyCarry):
    early = True


class FlippedFlow(ToyCarry):
    flip = True


class SameInstantCredit(ToyCarry):
    """Pays a flow known exactly at a decision time to the positions that decision opens."""

    same_instant = True


class LateBooking(ToyCarry):
    """Pays the right positions the right amount, but books each period's flow one period late (the last period's
    with its own, so the total is the same)."""

    def run(self, sources: Sources, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        r = super().run(sources, scores)
        flow = r.net["flow"].to_numpy()
        late = np.r_[0.0, flow[:-1]]
        late[-1] += flow[-1]
        net = r.net.with_columns(flow=pl.Series(late)).with_columns(
            net=pl.col("flow") + pl.col("price") + pl.col("fees")
        )
        return ConsumerResult(net, r.positions, r.trades, r.turnover, r.checks, r.policies, r.marked)


class NoFees(ToyCarry):
    fee_free = True


class Noisy(ToyCarry):
    noisy = True


class Peeking(ToyCarry):
    peek = True


class Unbalanced(ToyCarry):
    """Reports fees but leaves them out of net."""

    def run(self, sources: Sources, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        r = super().run(sources, scores)
        net = r.net.with_columns(net=pl.col("flow") + pl.col("price"))
        return ConsumerResult(net, r.positions, r.trades, r.turnover, r.checks, r.policies, r.marked)


class LaggedScores(ToyCarry):
    """Reads the scores it is given one period late, so even its own scores change its result."""

    def run(self, sources: Sources, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        if scores is not None:
            scores = scores.with_columns(pl.col("t") + pl.duration(hours=1, time_unit="ns"))
        return super().run(sources, scores)


class NeverRun(ToyCarry):
    """Gives the toy's decision times but refuses to run: nothing before the algo layer may backtest."""

    def run(self, sources: Sources, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        raise AssertionError("a layer before algo ran the backtest")


class PeekingOutcome(PairCarry):
    """Reads one horizon further than it may."""

    def values(self, sources: Sources, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame:
        return super().values(sources, candidates, 2 * horizon)


class LazyOutcome(PairCarry):
    """Correct code, but its only example has flat prices and no flows, so it tests nothing."""

    def cases(self) -> list[OutcomeCase]:
        rows = [(e, START + h * HOUR, START + h * HOUR, 0, 100.0) for e in ("a", "b") for h in range(6)]
        prices = pl.DataFrame(rows, schema=["inst", "event_time", "available_at", "vintage", "close"], orient="row")
        prices = prices.with_columns(pl.col("event_time").cast(TIME), pl.col("available_at").cast(TIME))
        cands = pl.DataFrame({"t": [START + HOUR], "inst": ["a>b"], "long": ["a"], "short": ["b"]})
        expected = pl.DataFrame({"t": [START + HOUR], "inst": ["a>b"], "y": [0.0], "y_price": [0.0]})
        cast = pl.col("t").cast(TIME)
        return [OutcomeCase({"prices": prices}, cands.with_columns(cast), "2h", expected.with_columns(cast))]


class Backwards(ClusterMean):
    """Reads the scores, but takes the lowest: a planted signal only makes it worse."""

    def reduce(self, rows: pl.DataFrame, horizon: pd.Timedelta, costs: Any) -> pl.DataFrame:
        flipped = rows.with_columns(-pl.col("score")) if "score" in rows.columns else rows
        return super().reduce(flipped, horizon, costs)


REGISTRY = {
    "toy.data": ToyData,
    "toy.pairs": ToyPairs,
    "toy.carry": ToyCarry,
    "toy.timed": ToyTimed,
    "toy.series": ToySeries,
    "toy.events": ToyEvents,
}
OUTCOME = "outcome=Use('ref.carry', {'flow': 'funding_rate'}), horizons=('8h', '24h')"
COSTS = "Use('ref.flat_costs', {'commission_bps': 1, 'slippage_bps': 1, 'legs': 2})"
EFFECT = "Effect(target=Use('ref.mean', {'cluster': 'symbol', 'top': 0.34}))"


def carry_contract(budget: int | dict[str, int] = 3) -> str:
    return (
        "from research_ci.api import *\n\n"
        "CONTRACT = Contract(claim='cross-venue funding spreads persist',\n"
        f"    decisive=({EFFECT},),\n"
        "    alternatives=(Alternative('tail', 'a few symbols carry the result'),),\n"
        f"    used=(), data='toy.data', costs={COSTS}, execution=Execution(every='1h'),\n"
        "    universe=Universe('toy.pairs', clusters='symbol', groups=('venue_pair',)),\n"
        f"    {OUTCOME},\n"
        "    explore_end='2026-01-15T00:00:00Z', seal_end='2026-01-20T00:00:00Z',\n"
        f"    budget={budget!r})\n"
    )


def strategy_contract() -> str:
    """Improving the toy strategy: its backtester is named only for the decision times it keeps."""
    return (
        "from research_ci.api import *\n\n"
        "CONTRACT = Contract(claim='a new score picks pairs that carry more for the toy strategy',\n"
        f"    decisive=({EFFECT},),\n"
        "    alternatives=(Alternative('level', 'only the spread level'),),\n"
        "    used=(Usage('2026-01-01T00:00:00Z', '2026-01-10T00:00:00Z', 'the toy strategy tuned', ('claim',)),),\n"
        f"    data='toy.data', costs={COSTS}, consumer='toy.carry',\n"
        "    universe=Universe('toy.pairs', clusters='symbol'),\n"
        f"    {OUTCOME},\n"
        "    forward=Forward('toy.live'))\n"
    )


LAB_EVALS = '''
"""The toy idea's lab evals: the flow part of the carry, and closes at zero or below."""

from datetime import UTC, datetime

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)


def funding_ic(table):
    """The rank correlation of the score with the flow part of the outcome, at each decision time."""
    rows = table.filter(pl.col("y_flow").is_not_null())
    return rows.group_by("t").agg(pl.corr("score", "y_flow", method="spearman").alias("value")).sort("t")


def closes_at_zero(sources):
    """Closes at zero or below, as cells of the candles."""
    zero = sources["candles"].filter(pl.col("close") <= 0)
    return zero.select(pl.lit("candles").alias("source"), "inst", "event_time", pl.lit("close").alias("column"))


FOUR = pl.DataFrame(
    {
        "t": [T] * 4,
        "inst": list("abcd"),
        "score": [1.0, 2.0, 3.0, 4.0],
        "y": [0.0] * 4,
        "y_flow": [0.1, 0.2, 0.4, 0.3],
    }
)
ROWS = {"candles": pl.DataFrame({"inst": ["a", "a"], "event_time": [T, T], "close": [1.0, 0.0]})}
CASES = [
    Case("one swap in four: 1 - 6·2/(4·15)", funding_ic, 0.8, given=FOUR),
    Case("a close at zero is flagged", closes_at_zero, 1, given=ROWS),
]
'''
DATA_EVALS = """
from research_ci.api import Eval
from evals.toy import closes_at_zero

EVALS = [Eval(closes_at_zero, h0="every close is above zero")]
"""
FEAT_EVALS = """
from research_ci.api import Eval
from evals.toy import funding_ic

EVALS = [Eval(funding_ic, h0="the scores do not rank the flow part of the outcome")]
"""
TOY_NULLS = """
from research_ci.api import Null, PermuteWithin

NULLS = [
    Null("{alternative}", PermuteWithin("time", clusters="symbol", block="horizon"),
         evals=("rank ic without top symbol",), reps=60),
]
"""


def lab(root: Any, alternative: str = "tail", data: str = DATA_EVALS, feat: str = FEAT_EVALS) -> None:
    """The smallest lab an idea on the toy needs: a lab eval for data and one for feat, with their h0 and CASES, and a
    null world for the contract's alternative."""
    from pathlib import Path

    root = Path(root)
    (root / "evals").mkdir(parents=True, exist_ok=True)
    (root / "nulls").mkdir(parents=True, exist_ok=True)
    (root / "evals" / "toy.py").write_text(LAB_EVALS)
    (root / "data.py").write_text(data)
    (root / "feat.py").write_text(feat)
    (root / "nulls" / "feat.py").write_text(TOY_NULLS.replace("{alternative}", alternative))


RETURN = "outcome=Use('ref.return', {'price': 'close'}), horizons=('8h', '24h')"
SHAPED_COSTS = "Use('ref.flat_costs', {'commission_bps': 1, 'slippage_bps': 1})"
SHAPED = {
    "series": ("toy.series", "Use('ref.timing')", "the regime times the asset"),
    "events": ("toy.events", "Use('ref.mean', {'cluster': 'inst'})", "a regime turning up is followed by gains"),
}


def shaped_contract(shape: str, budget: int = 5) -> str:
    """A contract on the timed toy market for a ``series`` (one asset, timed) or ``events`` shape."""
    universe, target, claim = SHAPED[shape]
    return (
        "from research_ci.api import *\n\n"
        f"CONTRACT = Contract(claim={claim!r},\n"
        f"    decisive=(Effect(target={target}),),\n"
        "    alternatives=(Alternative('trend', 'the score only follows the last days'),),\n"
        f"    used=(), data='toy.timed', costs={SHAPED_COSTS}, execution=Execution(every='1h'),\n"
        f"    universe=Universe('{universe}', clusters='inst', shape='{shape}'),\n"
        f"    {RETURN},\n"
        "    explore_end='2026-01-15T00:00:00Z', seal_end='2026-01-20T00:00:00Z',\n"
        f"    budget={budget!r})\n"
    )


SHAPED_EVALS = '''
"""The timed toy's lab evals: whether a score's sign is the outcome's, and the outcome after the events."""

from datetime import UTC, datetime, timedelta

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)


def hits(table):
    """At each decision time, the share of the scored rows whose score has the outcome's sign, less a half."""
    rows = table.filter(pl.col("score").is_finite())
    same = (pl.col("score").sign() == pl.col("y").sign()).cast(pl.Float64)
    return rows.group_by("t").agg((same.mean() - 0.5).alias("value")).sort("t")


def after(table):
    """At each decision time, the mean outcome at the events, the rows with a finite score."""
    rows = table.filter(pl.col("score").is_finite())
    return rows.group_by("t").agg(pl.col("y").mean().alias("value")).sort("t")


def backwards(table):
    """At each decision time, the mean of the outcome against the score's sign: the score read the wrong way round."""
    rows = table.filter(pl.col("score").is_finite())
    return rows.group_by("t").agg((-pl.col("score").sign() * pl.col("y")).mean().alias("value")).sort("t")


FOUR = pl.DataFrame(
    {
        "t": [T + timedelta(hours=h) for h in range(4)],
        "inst": ["a"] * 4,
        "score": [1.0, -1.0, float("nan"), 1.0],
        "y": [0.2, 0.1, 0.5, -0.1],
    }
)
CASES = [
    Case("a hit and two misses: the mean of 0.5, -0.5 and -0.5", lambda: hits(FOUR)["value"].mean(), -1 / 6),
    Case("the outcome after the three events", lambda: after(FOUR)["value"].sum(), 0.2),
    Case("the score read the wrong way round", lambda: backwards(FOUR)["value"][0], -0.2),
]
'''
SHAPED_FEAT = """
from research_ci.api import Eval
from evals.timing import {name}

EVALS = [Eval({name}, h0="{h0}")]
"""
SHAPED_H0 = {
    "series": ("hits", "the score's sign is the outcome's no more often than chance"),
    "events": ("after", "the outcome after an event is no larger than at other times"),
}
SHAPED_NULLS = """
from research_ci.api import Null, ShiftInTime

NULLS = [Null("trend", ShiftInTime(min_shift="72h"), evals=("{name}",), reps=60)]
"""


def shaped_lab(root: Any, shape: str) -> None:
    """The lab an idea of the timed toy needs: the toy's data eval, a lab eval for feat, and a world for the trend."""
    from pathlib import Path

    lab(root)
    root = Path(root)
    name, h0 = SHAPED_H0[shape]
    (root / "evals" / "timing.py").write_text(SHAPED_EVALS)
    (root / "feat.py").write_text(SHAPED_FEAT.format(name=name, h0=h0))
    (root / "nulls" / "feat.py").write_text(SHAPED_NULLS.format(name=name))
