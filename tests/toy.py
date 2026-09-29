"""Small plugins on synthetic data for the engine's own tests, with broken variants the protocol tests must catch."""

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api import ConsumerResult, LabelCase, Timing
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import TIME
from research_ci.reference_plugins.labels import PairCarry

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
    def __init__(self, hours: int = 24 * 25, seed: int = 0) -> None:
        self.hours, self.seed = hours, seed

    def fetch(self) -> pl.DataFrame:
        return toy_fields(self.hours, self.seed)


class ToyPairs:
    """Every venue pair of a symbol; long the venue with the lower latest funding rate per hour."""

    peek = False

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
    """Holds the top ``k`` pairs by score, equal weight per leg; flow and price accrue per hour after the decision."""

    early = flip = fee_free = noisy = peek = False

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
        flow_at = {(e, a): v for e, a, v in flows.select("entity", "available_at", "value").iter_rows()}
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


REGISTRY = {"toy.data": ToyData, "toy.pairs": ToyPairs, "toy.carry": ToyCarry}
CARRY = "Label('carry', Use('ref.carry', {'flow': 'funding_rate'}), horizons=('8h', '24h'))"
COSTS = "Use('ref.flat_costs', {'commission_bps': 1, 'slippage_bps': 1, 'legs': 2})"


def label_contract(template: str = "factor", budget: int | dict[str, int] = 3, skip: tuple[str, ...] = ()) -> str:
    return (
        "from research_ci.api import *\n\n"
        f"CONTRACT = Contract(template={template!r}, claim='cross-venue funding spreads persist',\n"
        "    decisive=(Effect(on='carry', at_least=0.001, unit='per entry after the round trip'),),\n"
        "    alternatives=(Alternative('tail', 'a few symbols carry the result'),),\n"
        f"    used=(), data='toy.data', costs={COSTS}, execution=Execution(every='1h'),\n"
        "    universe=Universe('toy.pairs', clusters='symbol', groups=('venue_pair',)),\n"
        f"    labels=({CARRY},),\n"
        "    explore_end='2026-01-15T00:00:00Z', seal_end='2026-01-20T00:00:00Z',\n"
        f"    budget={budget!r}, skip={skip!r})\n"
    )


def consumer_contract(consumer: str = "toy.carry") -> str:
    return (
        "from research_ci.api import *\n\n"
        "CONTRACT = Contract(template='factor', skip=('signal', 'algo'),\n"
        "    claim='a new score improves the toy strategy',\n"
        "    decisive=(Effect(on='consumer', at_least=0.05, unit='per year over the baseline',\n"
        "        pilot=Perturb('drop', 0.2, block='24h'), design=Branch(every='24h', horizon='24h', tail='24h')),),\n"
        "    alternatives=(Alternative('level', 'only the spread level'),),\n"
        "    used=(Usage('2026-01-01T00:00:00Z', '2026-01-10T00:00:00Z', 'toy parameters tuned', ('baseline',)),),\n"
        f"    data='toy.data', costs={COSTS}, consumer={consumer!r},\n"
        "    universe=Universe('toy.pairs', clusters='symbol'),\n"
        f"    labels=({CARRY},),\n"
        "    baseline=Baseline('score', config='strategy.toml', code=('strategy.py',), live_config='live.toml'),\n"
        "    forward=Forward('toy.live'), deadline='180 days')\n"
    )
