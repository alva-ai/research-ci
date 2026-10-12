"""Small plugins on synthetic data for the engine's own tests, with broken variants the protocol tests must catch."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api import ConsumerResult, Hedge, LabelCase, Levels, Reference, Timing
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import TIME
from research_ci.reference_plugins.labels import PairCarry
from research_ci.reference_plugins.targets import ClusterMean

START = pd.Timestamp("2026-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)
SYMBOLS = ("X", "Y", "Z")
VENUES = ("a", "b", "c")
SETTLE = 8


def toy_fields(hours: int = 24 * 25, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for symbol in SYMBOLS:
        common = np.cumsum(rng.normal(0, 0.004, hours))
        for i, venue in enumerate(VENUES):
            entity = f"{venue}:{symbol}"
            close = 100 * np.exp(common + np.cumsum(rng.normal(0, 0.0005, hours)))
            level = rng.normal(0.0004 * (i + 1), 0.0001)
            for h in range(hours):
                ts = START + (h + 1) * HOUR
                rows.append((entity, "close", ts, ts, float(close[h])))
                if (h + 1) % SETTLE == 0:
                    rows.append((entity, "funding_rate", ts, ts, float(level + rng.normal(0, 0.00005))))
                    rows.append((entity, "funding_interval_hours", ts, ts, float(SETTLE)))
    frame = pd.DataFrame(rows, columns=["entity", "field", "event_time", "available_at", "value"])
    return field_table(frame.assign(vintage=0))


class ToyData:
    """Replays as of any time, and answers reconciliation from the venue, here the same generator run apart; the
    venue keeps no history of funding intervals."""

    fields = ("close", "funding_rate", "funding_interval_hours")
    levels = Levels(fields=("close",), factor=5.0, window=24, mask="24h")

    reference_note = "the toy venue"

    def __init__(self, hours: int = 24 * 25, seed: int = 0) -> None:
        self.hours, self.seed = hours, seed

    def fetch(self, as_of: Any = None, entities: Sequence[str] | None = None) -> pl.DataFrame:
        rows = self.build(as_of)
        if as_of is not None:
            rows = rows.filter(pl.col("available_at") <= pd.Timestamp(as_of))
        return rows if entities is None else rows.filter(pl.col("entity").is_in(list(entities)))

    def build(self, as_of: Any = None) -> pl.DataFrame:
        return toy_fields(self.hours, self.seed)

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(toy_fields(self.hours, self.seed), requests)


def venue_reference(venue: pl.DataFrame, requests: pl.DataFrame) -> Reference:
    """What the toy venue answers from ``venue``; it keeps no history of funding intervals."""
    venue = venue.filter(pl.col("field") != "funding_interval_hours")
    answered = requests.filter(pl.col("field") != "funding_interval_hours")
    rows = venue.join(answered.select("entity", "field", "start", "end"), on=["entity", "field"])
    rows = rows.filter(pl.col("event_time").is_between(pl.col("start"), pl.col("end")))
    return Reference(rows.select("entity", "field", "event_time", "value").unique(), answered.select("rid"))


def timed_fields(hours: int = 24 * 25, seed: int = 0, drift: float = 0.002) -> pl.DataFrame:
    """The toy market with a timing signal planted: each symbol has a regime, +1 or -1 for stretches of one to four
    days, which every entity publishes each hour as the field ``regime`` and whose closes drift with it, by ``drift``
    per hour, over the next hour."""
    rows = toy_fields(hours, seed)
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
    keyed = rows.with_columns(hour, symbol=pl.col("entity").str.split(":").list.get(1)).join(plan, on=["symbol", "h"])
    close = pl.col("field") == "close"
    lifted = keyed.with_columns(value=pl.when(close).then(pl.col("value") * pl.col("lift")).otherwise("value"))
    published = keyed.filter(close).with_columns(field=pl.lit("regime"), value=pl.col("regime"))
    out = pl.concat([lifted, published]).select("entity", "field", "event_time", "available_at", "value", "vintage")
    return field_table(out)


class ToyTimed(ToyData):
    """The toy market with a planted timing signal (``timed_fields``); the venue reports it too."""

    fields = ("close", "funding_rate", "funding_interval_hours", "regime")

    def build(self, as_of: Any = None) -> pl.DataFrame:
        return timed_fields(self.hours, self.seed)

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToySeries:
    """The ``entities`` (one asset, by default) at every decision time once their first close is known."""

    def __init__(self, entities: Sequence[str] = ("a:X",)) -> None:
        self.entities = list(entities)

    def candidates(self, fields: pl.DataFrame, times: Sequence[Any]) -> pl.DataFrame:
        listed = fields.filter((pl.col("field") == "close") & pl.col("entity").is_in(self.entities))
        first = listed.group_by("entity").agg(pl.col("available_at").min().alias("since"))
        grid = pl.DataFrame({"t": list(times)}, schema={"t": TIME})
        out = grid.join(first, how="cross").filter(pl.col("t") >= pl.col("since"))
        return out.select("t", "entity").sort("t", "entity")


class ToyEvents(ToySeries):
    """One asset per symbol, every one a candidate at every decision time; the features score only the events."""

    def __init__(self) -> None:
        super().__init__(("a:X", "a:Y", "a:Z"))


class ToyHindsight(ToyData):
    """Drops a symbol's closes in a window once it knows a jump follows, as an ex-post repair would."""

    def build(self, as_of: Any = None) -> pl.DataFrame:
        rows = super().build()
        lo, hi = START + 200 * HOUR, START + 206 * HOUR
        if as_of is not None and pd.Timestamp(as_of) < hi + HOUR:
            return rows
        window = (pl.col("entity") == "b:Y") & (pl.col("field") == "close") & pl.col("event_time").is_between(lo, hi)
        return rows.filter(~window)


class ToyNoAsOf(ToyData):
    def fetch(self) -> pl.DataFrame:  # type: ignore[override]
        return super().build()


class ToyCorrupt(ToyData):
    """Its closes for one venue are off by a tenth from what the venue reports."""

    def build(self, as_of: Any = None) -> pl.DataFrame:
        rows = super().build()
        hit = (pl.col("entity").str.starts_with("c:")) & (pl.col("field") == "close")
        return rows.with_columns(value=pl.when(hit).then(pl.col("value") * 1.1).otherwise("value"))


class ToySwapped(ToyData):
    """Venue c lists another asset under the ticker Z: its closes follow their own path at the same level."""

    def build(self, as_of: Any = None) -> pl.DataFrame:
        rows = super().build()
        hit = (pl.col("entity") == "c:Z") & (pl.col("field") == "close")
        other = rows.filter(hit).sort("event_time")
        rng = np.random.default_rng(99)
        path = float(other["value"][0]) * np.exp(np.cumsum(rng.normal(0, 0.01, other.height)))
        return pl.concat([rows.filter(~hit), other.with_columns(value=pl.Series(path))]).sort("available_at")

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyRelaunched(ToyData):
    """Venue c delists Z after hour 150 and, twelve hours later, lists another asset under the same ticker at a
    thousand times the level. The venue reports both, so the source is faithful."""

    def build(self, as_of: Any = None) -> pl.DataFrame:
        rows = super().build()
        z, t = pl.col("entity") == "c:Z", pl.col("event_time")
        gone = z & t.is_between(START + 151 * HOUR, START + 161 * HOUR)
        later = z & (pl.col("field") == "close") & (t >= START + 162 * HOUR)
        return rows.filter(~gone).with_columns(value=pl.when(later).then(pl.col("value") * 1000).otherwise("value"))

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyRenamed(ToyRelaunched):
    """The relaunch of ``ToyRelaunched`` under a name of its own, ``c:Z#2``, as splitting it leaves it."""

    def build(self, as_of: Any = None) -> pl.DataFrame:
        rows = super().build()
        later = (pl.col("entity") == "c:Z") & (pl.col("event_time") >= START + 162 * HOUR)
        return rows.with_columns(entity=pl.when(later).then(pl.lit("c:Z#2")).otherwise("entity"))


class ToyZeros(ToyData):
    """Venue b reports a close of zero for X three hours running, and so does the venue's API."""

    def build(self, as_of: Any = None) -> pl.DataFrame:
        rows = super().build()
        hours = pl.col("event_time").is_between(START + 100 * HOUR, START + 102 * HOUR)
        hit = (pl.col("entity") == "b:X") & (pl.col("field") == "close") & hours
        return rows.with_columns(value=pl.when(hit).then(0.0).otherwise("value"))

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyShort(ToyData):
    """Venue c lists Y for twenty hours only, shorter than the longest horizon."""

    def build(self, as_of: Any = None) -> pl.DataFrame:
        rows = super().build()
        brief = pl.col("event_time").is_between(START + 300 * HOUR, START + 319 * HOUR)
        return rows.filter((pl.col("entity") != "c:Y") | brief)

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyPairs:
    """Every venue pair of a symbol; long the venue with the lower latest funding rate per hour."""

    peek = False
    hedge = Hedge(price="close", max_ratio=1.2, min_corr=0.3, window=72, min_obs=48, min_move=1e-3)

    def candidates(self, fields: pl.DataFrame, times: Sequence[Any]) -> pl.DataFrame:
        funding = fields.filter(pl.col("field") == "funding_rate").sort("available_at")
        if self.peek:
            funding = funding.with_columns(pl.col("value").shift(-1).over("entity")).drop_nulls("value")
        rows = []
        for t in times:
            known = funding.filter(pl.col("available_at") <= t).group_by("entity").agg(pl.col("value").last())
            rate = {e: v / SETTLE for e, v in known.iter_rows()}
            for symbol in SYMBOLS:
                for a, b in combinations(VENUES, 2):
                    ea, eb = f"{a}:{symbol}", f"{b}:{symbol}"
                    if ea not in rate or eb not in rate:
                        continue
                    long, short = (ea, eb) if rate[ea] <= rate[eb] else (eb, ea)
                    lv, sv = long.split(":")[0], short.split(":")[0]
                    spread = rate[short] - rate[long]
                    rows.append((t, f"{symbol}:{lv}>{sv}", long, short, symbol, f"{lv}>{sv}", spread, spread, True))
        columns = ["t", "entity", "long", "short", "symbol", "venue_pair", "spread_h", "score", "openable"]
        if not rows:
            return pl.DataFrame(schema={c: pl.String for c in columns})
        out = pl.DataFrame(rows, schema=columns, orient="row")
        return out.with_columns(pl.col("t").cast(TIME))


class ToyPeekPairs(ToyPairs):
    peek = True


class ToyNeedsNextBar(ToyPairs):
    """Correct, but like many backtest loops it outputs nothing on the last bar it is given."""

    def candidates(self, fields: pl.DataFrame, times: Sequence[Any]) -> pl.DataFrame:
        last = fields.select(pl.col("available_at").max()).item()
        return super().candidates(fields, [t for t in times if t < last])


class ToyCarry:
    """Holds the top ``k`` pairs by score, equal weight per leg; flow and price accrue per hour after the decision. A
    flow known in ``(t, t + 1h]`` is paid to the positions decided at ``t``."""

    early = flip = fee_free = noisy = peek = same_instant = False

    def __init__(self, k: int = 2, commission_bps: float = 1.0, slippage_bps: float = 1.0) -> None:
        self.k, self.cost = k, (commission_bps + slippage_bps) / 1e4

    def timing(self) -> Timing:
        return Timing(every="1h", note="decide on the hour; fill at the close known then")

    def run(self, fields: pl.DataFrame, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        close = fields.filter(pl.col("field") == "close").pivot(on="entity", index="available_at", values="value")
        close = close.sort("available_at")
        times = close["available_at"].to_list()
        flows = fields.filter(pl.col("field") == "funding_rate")
        cands = ToyPairs().candidates(fields, times)
        if scores is not None:
            cands = cands.drop("score").join(scores.select("t", "entity", "score"), on=["t", "entity"], how="inner")
        if self.peek:
            cands = cands.drop("score").join(self._future_flow(fields, times), on=["t", "entity"])
        by_t = {t: g for (t,), g in cands.group_by("t")}
        flow_at: dict[tuple[str, Any], float] = defaultdict(float)
        bucket = bisect_right if self.same_instant else bisect_left
        for e, a, v in flows.select("entity", "available_at", "value").iter_rows():
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
        positions = pl.DataFrame(pos_rows, schema=["t", "entity", "weight", "long", "short"], orient="row")
        trades = pl.DataFrame(trade_rows, schema=["t", "entity", "from_weight", "to_weight"], orient="row")
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
        top = group.sort(["score", "entity"], descending=[True, False]).head(self.k)
        return {e: (lo, sh) for e, lo, sh in top.select("entity", "long", "short").iter_rows()}

    def _future_flow(self, fields: pl.DataFrame, times: list[object]) -> pl.DataFrame:
        cands = ToyPairs().candidates(fields, times)
        flows = fields.filter(pl.col("field") == "funding_rate").select("entity", "available_at", "value")
        rows = []
        for t, e, lo, sh in cands.select("t", "entity", "long", "short").iter_rows():
            window = flows.filter((pl.col("available_at") > t) & (pl.col("available_at") <= t + 24 * HOUR))
            s = float(window.filter(pl.col("entity") == sh)["value"].sum()) - float(
                window.filter(pl.col("entity") == lo)["value"].sum()
            )
            rows.append((t, e, s))
        return pl.DataFrame(rows, schema=["t", "entity", "score"], orient="row").with_columns(pl.col("t").cast(TIME))


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

    def run(self, fields: pl.DataFrame, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        r = super().run(fields, scores)
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

    def run(self, fields: pl.DataFrame, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        r = super().run(fields, scores)
        net = r.net.with_columns(net=pl.col("flow") + pl.col("price"))
        return ConsumerResult(net, r.positions, r.trades, r.turnover, r.checks, r.policies, r.marked)


class LaggedScores(ToyCarry):
    """Reads the scores it is given one period late, so even its own scores change its result."""

    def run(self, fields: pl.DataFrame, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        if scores is not None:
            scores = scores.with_columns(pl.col("t") + pl.duration(hours=1, time_unit="ns"))
        return super().run(fields, scores)


class NeverRun(ToyCarry):
    """Gives the toy's decision times but refuses to run: nothing before the algo layer may backtest."""

    def run(self, fields: pl.DataFrame, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult:
        raise AssertionError("a layer before algo ran the backtest")


class PeekingLabel(PairCarry):
    """Reads one horizon further than it may."""

    def values(self, fields: pl.DataFrame, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame:
        return super().values(fields, candidates, 2 * horizon)


class LazyLabel(PairCarry):
    """Correct code, but its only example has flat prices and no flows, so it tests nothing."""

    def cases(self) -> list[LabelCase]:
        rows = [(e, "close", START + h * HOUR, START + h * HOUR, 100.0, 0, "") for e in ("a", "b") for h in range(6)]
        fields = pl.DataFrame(
            rows, schema=["entity", "field", "event_time", "available_at", "value", "vintage", "source"], orient="row"
        ).with_columns(pl.col("event_time").cast(TIME), pl.col("available_at").cast(TIME))
        cands = pl.DataFrame({"t": [START + HOUR], "entity": ["a>b"], "long": ["a"], "short": ["b"]})
        expected = pl.DataFrame({"t": [START + HOUR], "entity": ["a>b"], "value": [0.0], "price": [0.0]})
        cast = pl.col("t").cast(TIME)
        return [LabelCase(fields, cands.with_columns(cast), "2h", expected.with_columns(cast))]


class Backwards(ClusterMean):
    """Reads the scores, but takes the lowest: a planted signal only makes it worse."""

    def reduce(self, labels: pl.DataFrame, horizon: pd.Timedelta, costs: Any) -> pl.DataFrame:
        flipped = labels.with_columns(-pl.col("score")) if "score" in labels.columns else labels
        return super().reduce(flipped, horizon, costs)


REGISTRY = {
    "toy.data": ToyData,
    "toy.pairs": ToyPairs,
    "toy.carry": ToyCarry,
    "toy.timed": ToyTimed,
    "toy.series": ToySeries,
    "toy.events": ToyEvents,
}
CARRY = "Label('carry', Use('ref.carry', {'flow': 'funding_rate'}), horizons=('8h', '24h'))"
COSTS = "Use('ref.flat_costs', {'commission_bps': 1, 'slippage_bps': 1, 'legs': 2})"
EFFECT = "Effect(on='carry', target=Use('ref.mean', {'cluster': 'symbol', 'top': 0.34}))"


def label_contract(budget: int | dict[str, int] = 3) -> str:
    return (
        "from research_ci.api import *\n\n"
        "CONTRACT = Contract(claim='cross-venue funding spreads persist',\n"
        f"    decisive=({EFFECT},),\n"
        "    alternatives=(Alternative('tail', 'a few symbols carry the result'),),\n"
        f"    used=(), data='toy.data', costs={COSTS}, execution=Execution(every='1h'),\n"
        "    universe=Universe('toy.pairs', clusters='symbol', groups=('venue_pair',)),\n"
        f"    labels=({CARRY},),\n"
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
        f"    labels=({CARRY},),\n"
        "    forward=Forward('toy.live'))\n"
    )


LAB_EVALS = '''
"""The toy idea's lab evals: the funding part of the carry, and closes at zero or below."""

from datetime import UTC, datetime

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)


def funding_ic(table):
    """The rank correlation of the score with the funding part of the carry, at each decision time."""
    rows = table.filter(pl.col("y_flow").is_not_null())
    return rows.group_by("t").agg(pl.corr("score", "y_flow", method="spearman").alias("value")).sort("t")


def closes_at_zero(fields):
    """Closes at zero or below."""
    return fields.filter((pl.col("field") == "close") & (pl.col("value") <= 0)).select("entity", "field", "event_time")


FOUR = pl.DataFrame(
    {
        "t": [T] * 4,
        "entity": list("abcd"),
        "score": [1.0, 2.0, 3.0, 4.0],
        "y": [0.0] * 4,
        "y_flow": [0.1, 0.2, 0.4, 0.3],
    }
)
ROWS = pl.DataFrame({"entity": ["a", "a"], "field": ["close"] * 2, "event_time": [T, T], "value": [1.0, 0.0]})
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

EVALS = [Eval(funding_ic, h0="the scores do not rank the funding part of the carry")]
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


RETURN = "Label('return', Use('ref.return', {'price': 'close'}), horizons=('8h', '24h'))"
SHAPED_COSTS = "Use('ref.flat_costs', {'commission_bps': 1, 'slippage_bps': 1})"
SHAPED = {
    "series": ("toy.series", "Use('ref.timing')", "the regime times the asset"),
    "events": ("toy.events", "Use('ref.mean', {'cluster': 'entity'})", "a regime turning up is followed by gains"),
}


def shaped_contract(shape: str, budget: int = 5) -> str:
    """A contract on the timed toy market for a ``series`` (one asset, timed) or ``events`` shape."""
    universe, target, claim = SHAPED[shape]
    return (
        "from research_ci.api import *\n\n"
        f"CONTRACT = Contract(claim={claim!r},\n"
        f"    decisive=(Effect(on='return', target={target}),),\n"
        "    alternatives=(Alternative('trend', 'the score only follows the last days'),),\n"
        f"    used=(), data='toy.timed', costs={SHAPED_COSTS}, execution=Execution(every='1h'),\n"
        f"    universe=Universe('{universe}', clusters='entity', shape='{shape}'),\n"
        f"    labels=({RETURN},),\n"
        "    explore_end='2026-01-15T00:00:00Z', seal_end='2026-01-20T00:00:00Z',\n"
        f"    budget={budget!r})\n"
    )


SHAPED_EVALS = '''
"""The timed toy's lab evals: whether a score's sign is the label's, and the label after the events."""

from datetime import UTC, datetime, timedelta

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)


def hits(table):
    """At each decision time, the share of the scored rows whose score has the label's sign, less a half."""
    rows = table.filter(pl.col("score").is_finite())
    same = (pl.col("score").sign() == pl.col("y").sign()).cast(pl.Float64)
    return rows.group_by("t").agg((same.mean() - 0.5).alias("value")).sort("t")


def after(table):
    """At each decision time, the mean label at the events, the rows with a finite score."""
    rows = table.filter(pl.col("score").is_finite())
    return rows.group_by("t").agg(pl.col("y").mean().alias("value")).sort("t")


def backwards(table):
    """At each decision time, the mean of the label against the score's sign: the score read the wrong way round."""
    rows = table.filter(pl.col("score").is_finite())
    return rows.group_by("t").agg((-pl.col("score").sign() * pl.col("y")).mean().alias("value")).sort("t")


FOUR = pl.DataFrame(
    {
        "t": [T + timedelta(hours=h) for h in range(4)],
        "entity": ["a"] * 4,
        "score": [1.0, -1.0, float("nan"), 1.0],
        "y": [0.2, 0.1, 0.5, -0.1],
    }
)
CASES = [
    Case("a hit and two misses: the mean of 0.5, -0.5 and -0.5", lambda: hits(FOUR)["value"].mean(), -1 / 6),
    Case("the label after the three events", lambda: after(FOUR)["value"].sum(), 0.2),
    Case("the score read the wrong way round", lambda: backwards(FOUR)["value"][0], -0.2),
]
'''
SHAPED_FEAT = """
from research_ci.api import Eval
from evals.timing import {name}

EVALS = [Eval({name}, h0="{h0}")]
"""
SHAPED_H0 = {
    "series": ("hits", "the score's sign is the label's no more often than chance"),
    "events": ("after", "the label after an event is no larger than at other times"),
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
