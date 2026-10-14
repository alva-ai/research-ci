"""Small interface types for external Algo builders and backtesters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .common import ToolInfo


@dataclass(frozen=True, slots=True)
class AlgoBundle:
    """Return an Algo main table plus construction metadata.

    The table shape is declared separately by ``AlgoSet``.  Keeping the bundle
    small lets a Lab implement weights, positions, or trade rows without the
    Engine owning a portfolio model.
    """

    main: pl.DataFrame
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.main, pl.DataFrame):
            raise TypeError("AlgoBundle.main must be a Polars DataFrame")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "artifacts", dict(self.artifacts))


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Standard output returned by an external Lab backtester."""

    timeseries: pl.DataFrame
    summary: pl.DataFrame
    trades: pl.DataFrame | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.timeseries, pl.DataFrame):
            raise TypeError("BacktestResult.timeseries must be a Polars DataFrame")
        if not isinstance(self.summary, pl.DataFrame):
            raise TypeError("BacktestResult.summary must be a Polars DataFrame")
        if self.trades is not None and not isinstance(self.trades, pl.DataFrame):
            raise TypeError("BacktestResult.trades must be a Polars DataFrame or None")
        object.__setattr__(self, "metadata", dict(self.metadata))


def bounded_score(value: pl.Expr, *, scale: float = 1.0) -> pl.Expr:
    """Map an unbounded score to ``[-1, 1]`` with an explicit scale."""

    if scale <= 0:
        raise ValueError("bounded score scale must be positive")
    return (value / scale).tanh()


def dollar_neutral_pair(
    score: pl.Expr,
    *,
    gross: float = 1.0,
) -> tuple[pl.Expr, pl.Expr]:
    """Convert a signed pair score into equal long and short leg weights."""

    if not 0 < gross <= 2:
        raise ValueError("pair gross exposure must be in (0, 2]")
    half = score * (gross / 2.0)
    return half, -half


TOOLS = (
    ToolInfo(
        "kit.algo.bounded_score",
        "algo",
        "expr",
        "bound an approved signal before position sizing",
    ),
    ToolInfo(
        "kit.algo.dollar_neutral_pair",
        "algo",
        "expr",
        "map one pair signal to equal long and short leg weights",
    ),
    ToolInfo(
        "kit.algo.AlgoBundle",
        "algo",
        "bundle",
        "return the Algo main table and explicit construction metadata",
    ),
    ToolInfo(
        "kit.algo.BacktestResult",
        "algo",
        "bundle",
        "return external backtest timeseries, summary, and optional trades",
    ),
)


__all__ = [
    "TOOLS",
    "AlgoBundle",
    "BacktestResult",
    "bounded_score",
    "dollar_neutral_pair",
]
