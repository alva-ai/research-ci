"""Synthetic markets the engine builds to check methods: evaluation tables whose information is known."""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl

from research_ci.api.methods import Market
from research_ci.kernel.time.clock import TIME

HOUR_NS = 3_600 * 10**9


def market(m: Market, seed: int) -> pl.DataFrame:
    """Columns ``t``, ``entity``, ``cluster``, ``group``, ``score`` and ``y``, one row per entity and time."""
    rng = np.random.default_rng(seed)
    cluster = np.arange(m.entities) % m.clusters
    group = cluster % m.groups
    shape = (m.times, m.entities)
    own = _noise(rng, shape, m.tail)
    shock = _noise(rng, (m.times, m.clusters), m.tail)[:, cluster]
    level = rng.standard_normal((m.times, m.groups))[:, group]
    own_share = 1 - m.cluster_share - m.group_share
    y = math.sqrt(m.group_share) * level + math.sqrt(m.cluster_share) * shock + math.sqrt(own_share) * own
    rho = 2 * math.sin(math.pi * m.ic / 6)
    score = rho * _standard(y) + math.sqrt(1 - rho * rho) * rng.standard_normal(shape)
    t = np.repeat(np.arange(m.times, dtype=np.int64) * HOUR_NS, m.entities)
    frame = pl.DataFrame(
        {
            "t": t,
            "entity": np.tile(np.array([f"e{i}" for i in range(m.entities)]), m.times),
            "cluster": np.tile(np.array([f"c{c}" for c in cluster]), m.times),
            "group": np.tile(np.array([f"g{g}" for g in group]), m.times),
            "score": score.ravel(),
            "y": y.ravel(),
        }
    )
    return frame.with_columns(pl.col("t").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC").cast(TIME))


def hand(columns: Mapping[str, list[Any]], t: int = 0) -> pl.DataFrame:
    """A small evaluation table worked by hand, all at decision time ``t`` (in hours)."""
    rows = len(next(iter(columns.values())))
    frame = pl.DataFrame({"t": [t * HOUR_NS] * rows, **columns}, strict=False)
    return frame.with_columns(pl.col("t").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC").cast(TIME))


def _noise(rng: np.random.Generator, shape: tuple[int, int], tail: float | None) -> np.ndarray:
    if tail is None:
        return rng.standard_normal(shape)
    draws = rng.standard_t(tail, shape)
    return draws / math.sqrt(tail / (tail - 2)) if tail > 2 else draws


def _standard(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mean) / np.where(sd > 0, sd, 1.0)
