"""Synthetic markets and sources the engine builds to check methods: evaluation tables whose information is
known, and field tables whose faults are known."""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl

from research_ci.api.evals import Market, Source
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import TIME

HOUR_NS = 3_600 * 10**9
START_NS = 1_767_225_600 * 10**9


def market(m: Market, seed: int) -> pl.DataFrame:
    """A synthetic evaluation table, one row per entity and time. Columns ``t``, ``entity``, ``cluster``, ``group``,
    ``score`` and ``y``."""
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
            "entity": np.tile(_named("e", np.arange(m.entities)), m.times),
            "cluster": np.tile(_named("c", cluster), m.times),
            "group": np.tile(_named("g", group), m.times),
            "score": score.ravel(),
            "y": y.ravel(),
        }
    )
    return frame.with_columns(pl.col("t").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC").cast(TIME))


def source(s: Source, seed: int) -> pl.DataFrame:
    """A synthetic field table of hourly prices and settled flows. ``price`` is hourly, known at its event; ``flow``
    comes at each settlement, known ``lag`` hours later, some of it revised one settlement after that as vintage 1."""
    rng = np.random.default_rng(seed)
    names = _named("e", np.arange(s.entities))
    hours = np.arange(1, s.hours + 1, dtype=np.int64)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, (s.entities, s.hours)), axis=1))
    price_t = START_NS + np.tile(hours, s.entities) * HOUR_NS
    rows = [
        pl.DataFrame(
            {
                "entity": np.repeat(names, s.hours),
                "field": "price",
                "event_time": price_t,
                "available_at": price_t,
                "value": walk.ravel(),
                "vintage": 0,
            }
        )
    ]
    settles = hours[hours % s.settle == 0]
    level = rng.normal(0.0002, 0.0001, s.entities)
    flow = level[:, None] + rng.normal(0, 0.00005, (s.entities, len(settles)))
    flow_t = START_NS + np.tile(settles, s.entities) * HOUR_NS
    lag = s.lag * HOUR_NS
    rows.append(
        pl.DataFrame(
            {
                "entity": np.repeat(names, len(settles)),
                "field": "flow",
                "event_time": flow_t,
                "available_at": flow_t + lag,
                "value": flow.ravel(),
                "vintage": 0,
            }
        )
    )
    revised = rng.random(flow.size) < s.revised
    if revised.any():
        first = rows[-1].filter(pl.Series(revised))
        later = pl.col("available_at") + s.settle * HOUR_NS
        rows.append(first.with_columns(available_at=later, value=pl.col("value") * 1.01, vintage=pl.lit(1)))
    frame = pl.concat(rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC"),
        pl.col("available_at").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC"),
        source=pl.lit("synthetic"),
    )
    return field_table(frame)


def hand(columns: Mapping[str, list[Any]], t: int = 0) -> pl.DataFrame:
    """A small evaluation table worked by hand, at one time. All its rows are at decision time ``t`` (in hours)."""
    rows = len(next(iter(columns.values())))
    frame = pl.DataFrame({"t": [t * HOUR_NS] * rows, **columns}, strict=False)
    return frame.with_columns(pl.col("t").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC").cast(TIME))


def _named(prefix: str, numbers: np.ndarray) -> np.ndarray:
    return np.char.add(prefix, numbers.astype(str))


def _noise(rng: np.random.Generator, shape: tuple[int, int], tail: float | None) -> np.ndarray:
    if tail is None:
        return rng.standard_normal(shape)
    draws = rng.standard_t(tail, shape)
    return draws / math.sqrt(tail / (tail - 2)) if tail > 2 else draws


def _standard(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mean) / np.where(sd > 0, sd, 1.0)
