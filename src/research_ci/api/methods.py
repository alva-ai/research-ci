"""What an eval method declares about itself, for the engine to check before it is trusted.

A method's module lists ``CASES``, answers worked by hand, and ``PROPERTIES``, statistical behaviour on synthetic
markets the engine builds. The engine runs both; a method whose obligations fail is not used.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

Statistic = Callable[[pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True)
class Case:
    """``run()`` must give ``expect``: numbers within ``tol``, frames with the same rows and values within ``tol``."""

    name: str
    run: Callable[[], Any]
    expect: Any
    tol: float = 1e-9


@dataclass(frozen=True)
class Market:
    """A synthetic evaluation table: ``entities`` candidates in ``clusters`` (and the clusters in ``groups``) at
    ``times`` hourly decision times. Each label is a group level (share ``group_share`` of its variance), a cluster
    shock (``cluster_share``) and its own noise, Student-t with ``tail`` degrees of freedom (``None``: normal). The
    score's rank correlation with the label is about ``ic``."""

    times: int = 120
    entities: int = 100
    clusters: int = 20
    groups: int = 4
    cluster_share: float = 0.3
    group_share: float = 0.0
    tail: float | None = 4.0
    ic: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.clusters <= self.entities or self.groups < 1 or self.times < 2:
            raise ValueError("a market needs at least two times and no more clusters than entities")
        if self.cluster_share + self.group_share >= 1 or min(self.cluster_share, self.group_share) < 0:
            raise ValueError("the cluster and group shares must leave room for each entity's own noise")
        if not -1 < self.ic < 1:
            raise ValueError("ic is a correlation")


@dataclass(frozen=True)
class Source:
    """A synthetic field table: ``entities`` entities with an hourly ``price`` (a random walk) and a ``flow`` settled
    every ``settle`` hours and known ``lag`` hours after it; a share ``revised`` of the flows gets a second vintage one
    settlement later."""

    entities: int = 12
    hours: int = 24 * 14
    settle: int = 8
    lag: int = 0
    revised: float = 0.0

    def __post_init__(self) -> None:
        if self.entities < 2 or self.hours < 4 * self.settle or self.settle < 1 or self.lag < 0:
            raise ValueError("a source needs two entities and a few settlements")
        if not 0 <= self.revised <= 1:
            raise ValueError("revised is a share")


@dataclass(frozen=True)
class Size:
    """On markets with no information, the statistic must beat the floor world (z ≥ 2) no more often than chance
    allows, over ``markets`` seeds."""

    of: Statistic
    market: Market = field(default_factory=Market)
    markets: int = 100


@dataclass(frozen=True)
class Power:
    """On markets with ``market.ic`` of information, the statistic must beat the floor world at least ``rate`` of
    the time."""

    of: Statistic
    market: Market = field(default_factory=lambda: Market(ic=0.05))
    markets: int = 40
    rate: float = 0.8


@dataclass(frozen=True)
class Recovers:
    """Averaged over ``markets`` seeds and decision times, the statistic must come within ``tol`` of ``expect``."""

    of: Statistic
    market: Market
    expect: float
    tol: float
    markets: int = 20


Property = Size | Power | Recovers
