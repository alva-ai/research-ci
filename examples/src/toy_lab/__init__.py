"""Toy plugins for Research CI, on synthetic data.

- ``toy.data``: two sources for three symbols (X, Y, Z) on three venues (a, b, c), from 2026-01-01 for 25 days:
  ``candles``, an hourly ``close``, and ``funding``, a ``funding_rate`` and its ``funding_interval_hours`` every 8
  hours. Insts are ``<venue>:<symbol>``.
- ``toy.data.relaunch``: the same market, as reported by a venue with a messier listing history.
- ``toy.pairs``: every venue pair of a symbol, long the venue with the lower latest funding rate; ``spread_h`` is the
  latest spread per hour.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api import Hedge, Levels, Reference

START = pd.Timestamp("2026-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)
SYMBOLS = ("X", "Y", "Z")
VENUES = ("a", "b", "c")
SETTLE = 8
NOT_KEPT = "funding_interval_hours"
Sources = Mapping[str, pl.DataFrame]


def toy_sources(hours: int = 24 * 25, seed: int = 0) -> dict[str, pl.DataFrame]:
    """The ``candles`` and ``funding`` sources, each row known at its event."""
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
    return {"candles": _timed(pl.concat(candles)), "funding": _timed(pl.concat(funding))}


def _timed(frame: pl.DataFrame) -> pl.DataFrame:
    at = (pl.lit(START) + pl.duration(hours=pl.col("hour"), time_unit="ns")).cast(pl.Datetime("ns", "UTC"))
    keys = [at.alias("event_time"), at.alias("available_at"), pl.lit(0).alias("vintage")]
    return frame.select("inst", *keys, pl.exclude("inst", "hour"))


class ToyData:
    """Replays as of any time, and answers reconciliation from the venue, here the same generator run apart; the
    venue keeps no history of funding intervals."""

    sources = {"candles": ("close",), "funding": ("funding_rate", NOT_KEPT)}
    levels = Levels(columns=("close",), factor=5.0, window=24, mask="24h")
    reference_note = "the toy venue"

    def __init__(self, hours: int = 24 * 25, seed: int = 0) -> None:
        self.hours, self.seed = (hours, seed)

    def fetch(self, as_of: Any = None, insts: Sequence[str] | None = None) -> dict[str, pl.DataFrame]:
        keep = pl.lit(True)
        if as_of is not None:
            keep = keep & (pl.col("available_at") <= pd.Timestamp(as_of))
        if insts is not None:
            keep = keep & pl.col("inst").is_in(list(insts))
        return {name: table.filter(keep) for name, table in self.build(as_of).items()}

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
        .select(
            "inst",
            "event_time",
            *(c for c in table.columns if c not in ("inst", "event_time", "available_at", "vintage", NOT_KEPT)),
        )
        .unique()
        for name, table in venue.items()
    }
    return Reference(found, answered.select("rid"))


class ToyRelaunched(ToyData):
    def build(self, as_of: Any = None) -> dict[str, pl.DataFrame]:
        z, t = (pl.col("inst") == "c:Z", pl.col("event_time"))
        gone = z & t.is_between(START + 151 * HOUR, START + 161 * HOUR)
        later = z & (t >= START + 162 * HOUR)
        tables = {name: table.filter(~gone) for name, table in super().build().items()}
        relaunched = pl.when(later).then(pl.col("close") * 1000).otherwise("close")
        return {**tables, "candles": tables["candles"].with_columns(close=relaunched)}

    def reference(self, requests: pl.DataFrame) -> Reference:
        return venue_reference(self.build(), requests)


class ToyPairs:
    """Every venue pair of a symbol; long the venue with the lower latest funding rate per hour."""

    hedge = Hedge(price="close", max_ratio=1.2, min_corr=0.3, window=72, min_obs=48, min_move=0.001)

    def candidates(self, wide: pl.DataFrame, sources: Sources) -> pl.DataFrame:
        known = wide.select("t", "inst", (pl.col("funding_rate") / SETTLE).alias("rate")).drop_nulls("rate")
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
