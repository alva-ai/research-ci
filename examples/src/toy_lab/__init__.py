"""Toy plugins for Research CI, on synthetic data.

- ``toy.data``: hourly closes, and funding rates every 8 hours, for three symbols (X, Y, Z) on three venues (a, b,
  c), from 2026-01-01 for 25 days. Insts are ``<venue>:<symbol>``.
- ``toy.data.relaunch``: the same market, as reported by a venue with a messier listing history.
- ``toy.pairs``: every venue pair of a symbol, long the venue with the lower latest funding rate; ``spread_h`` is the
  latest spread per hour.
"""

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api import Hedge, Levels, Reference
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import TIME

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
            inst = f"{venue}:{symbol}"
            close = 100 * np.exp(common + np.cumsum(rng.normal(0, 0.0005, hours)))
            level = rng.normal(0.0004 * (i + 1), 0.0001)
            for h in range(hours):
                ts = START + (h + 1) * HOUR
                rows.append((inst, "close", ts, ts, float(close[h])))
                if (h + 1) % SETTLE == 0:
                    rows.append((inst, "funding_rate", ts, ts, float(level + rng.normal(0, 5e-05))))
                    rows.append((inst, "funding_interval_hours", ts, ts, float(SETTLE)))
    frame = pd.DataFrame(rows, columns=["inst", "field", "event_time", "available_at", "value"])
    return field_table(frame.assign(vintage=0))


class ToyData:
    """Replays as of any time, and answers reconciliation from the venue, here the same generator run apart; the
    venue keeps no history of funding intervals."""

    fields = ("close", "funding_rate", "funding_interval_hours")
    levels = Levels(fields=("close",), factor=5.0, window=24, mask="24h")
    reference_note = "the toy venue"

    def __init__(self, hours: int = 24 * 25, seed: int = 0) -> None:
        self.hours, self.seed = (hours, seed)

    def fetch(self, as_of: Any = None, insts: Sequence[str] | None = None) -> pl.DataFrame:
        rows = self.build(as_of)
        if as_of is not None:
            rows = rows.filter(pl.col("available_at") <= pd.Timestamp(as_of))
        return rows if insts is None else rows.filter(pl.col("inst").is_in(list(insts)))

    def build(self, as_of: Any = None) -> pl.DataFrame:
        return toy_fields(self.hours, self.seed)

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(toy_fields(self.hours, self.seed), requests)


def venue_reference(venue: pl.DataFrame, requests: pl.DataFrame) -> Reference:
    """What the toy venue answers from ``venue``; it keeps no history of funding intervals."""
    venue = venue.filter(pl.col("field") != "funding_interval_hours")
    answered = requests.filter(pl.col("field") != "funding_interval_hours")
    rows = venue.join(answered.select("inst", "field", "start", "end"), on=["inst", "field"])
    rows = rows.filter(pl.col("event_time").is_between(pl.col("start"), pl.col("end")))
    return Reference(rows.select("inst", "field", "event_time", "value").unique(), answered.select("rid"))


class ToyRelaunched(ToyData):
    def build(self, as_of: Any = None) -> pl.DataFrame:
        rows = super().build()
        z, t = (pl.col("inst") == "c:Z", pl.col("event_time"))
        gone = z & t.is_between(START + 151 * HOUR, START + 161 * HOUR)
        later = z & (pl.col("field") == "close") & (t >= START + 162 * HOUR)
        return rows.filter(~gone).with_columns(value=pl.when(later).then(pl.col("value") * 1000).otherwise("value"))

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyPairs:
    """Every venue pair of a symbol; long the venue with the lower latest funding rate per hour."""

    peek = False
    hedge = Hedge(price="close", max_ratio=1.2, min_corr=0.3, window=72, min_obs=48, min_move=0.001)

    def candidates(self, fields: pl.DataFrame, times: Sequence[Any]) -> pl.DataFrame:
        funding = fields.filter(pl.col("field") == "funding_rate").sort("available_at")
        if self.peek:
            funding = funding.with_columns(pl.col("value").shift(-1).over("inst")).drop_nulls("value")
        rows = []
        for t in times:
            known = funding.filter(pl.col("available_at") <= t).group_by("inst").agg(pl.col("value").last())
            rate = {e: v / SETTLE for e, v in known.iter_rows()}
            for symbol in SYMBOLS:
                for a, b in combinations(VENUES, 2):
                    ea, eb = (f"{a}:{symbol}", f"{b}:{symbol}")
                    if ea not in rate or eb not in rate:
                        continue
                    long, short = (ea, eb) if rate[ea] <= rate[eb] else (eb, ea)
                    lv, sv = (long.split(":")[0], short.split(":")[0])
                    spread = rate[short] - rate[long]
                    rows.append((t, f"{symbol}:{lv}>{sv}", long, short, symbol, f"{lv}>{sv}", spread, spread, True))
        columns = ["t", "inst", "long", "short", "symbol", "venue_pair", "spread_h", "score", "openable"]
        if not rows:
            return pl.DataFrame(schema={c: pl.String for c in columns})
        out = pl.DataFrame(rows, schema=columns, orient="row")
        return out.with_columns(pl.col("t").cast(TIME))
