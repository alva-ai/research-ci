"""Name-free Polars expressions for external Feat builders."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from typing import Literal

import polars as pl

from .common import ToolInfo

ClosedInterval = Literal["left", "right", "both", "none"]


def _positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def lag(value: pl.Expr, periods: int = 1) -> pl.Expr:
    """Return a past value without assigning an output column name."""

    return value.shift(_positive(periods, "periods"))


def difference(value: pl.Expr, periods: int = 1) -> pl.Expr:
    """Return current minus past value."""

    return value - lag(value, periods)


def ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Return a raw ratio without epsilon, clipping, or null filling."""

    return numerator / denominator


def log_return(value: pl.Expr, periods: int = 1) -> pl.Expr:
    """Return the backward log change for an already-positive value."""

    return value.log() - lag(value, periods).log()


def log1p(value: pl.Expr) -> pl.Expr:
    """Apply log(1 + x) without clipping or replacing invalid inputs."""

    return value.log1p()


def tanh(value: pl.Expr, *, scale: float = 1.0) -> pl.Expr:
    """Apply a bounded monotonic transform with an explicit positive scale."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    return (value / scale).tanh()


def rolling_mean_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
) -> pl.Expr:
    """Calculate a row-clock rolling mean."""

    window_size = _positive(window_size, "window_size")
    if min_samples is not None:
        _positive(min_samples, "min_samples")
    return value.rolling_mean(window_size, min_samples=min_samples)


def rolling_std_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
    ddof: int = 1,
) -> pl.Expr:
    """Calculate a row-clock rolling standard deviation."""

    window_size = _positive(window_size, "window_size")
    if min_samples is not None:
        _positive(min_samples, "min_samples")
    if ddof < 0:
        raise ValueError("ddof must not be negative")
    return value.rolling_std(
        window_size,
        min_samples=min_samples,
        ddof=ddof,
    )


def rolling_quantile_rows(
    value: pl.Expr,
    *,
    quantile: float,
    window_size: int,
    min_samples: int | None = None,
    interpolation: Literal[
        "nearest", "higher", "lower", "midpoint", "linear", "equiprobable"
    ] = "linear",
) -> pl.Expr:
    """Calculate one row-clock rolling quantile with explicit semantics."""

    window_size = _positive(window_size, "window_size")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    if min_samples is not None:
        _positive(min_samples, "min_samples")
    return value.rolling_quantile(
        quantile,
        interpolation=interpolation,
        window_size=window_size,
        min_samples=min_samples,
    )


def rolling_zscore_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
    ddof: int = 1,
) -> pl.Expr:
    """Calculate a row-clock z-score without fill, clipping, or epsilon."""

    mean = rolling_mean_rows(
        value,
        window_size=window_size,
        min_samples=min_samples,
    )
    std = rolling_std_rows(
        value,
        window_size=window_size,
        min_samples=min_samples,
        ddof=ddof,
    )
    return (value - mean) / std


def rolling_robust_scale_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
) -> pl.Expr:
    """Scale around the rolling median by the rolling interquartile range."""

    if not 0.0 <= lower_quantile < 0.5 < upper_quantile <= 1.0:
        raise ValueError("quantile bounds must satisfy 0 <= lower < 0.5 < upper <= 1")
    median = rolling_quantile_rows(
        value,
        quantile=0.5,
        window_size=window_size,
        min_samples=min_samples,
    )
    lower = rolling_quantile_rows(
        value,
        quantile=lower_quantile,
        window_size=window_size,
        min_samples=min_samples,
    )
    upper = rolling_quantile_rows(
        value,
        quantile=upper_quantile,
        window_size=window_size,
        min_samples=min_samples,
    )
    return (value - median) / (upper - lower)


def rolling_mean_time(
    value: pl.Expr,
    *,
    by: pl.Expr,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
) -> pl.Expr:
    """Calculate a duration-clock rolling mean with explicit warm-up."""

    _positive(min_samples, "min_samples")
    return value.rolling_mean_by(
        by,
        window_size,
        min_samples=min_samples,
        closed=closed,
    )


def rolling_std_time(
    value: pl.Expr,
    *,
    by: pl.Expr,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
    ddof: int = 1,
) -> pl.Expr:
    """Calculate a duration-clock rolling standard deviation."""

    _positive(min_samples, "min_samples")
    if ddof < 0:
        raise ValueError("ddof must not be negative")
    return value.rolling_std_by(
        by,
        window_size,
        min_samples=min_samples,
        closed=closed,
        ddof=ddof,
    )


def rolling_zscore_time(
    value: pl.Expr,
    *,
    by: pl.Expr,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
    ddof: int = 1,
) -> pl.Expr:
    """Calculate a duration-clock z-score without implicit treatment."""

    mean = rolling_mean_time(
        value,
        by=by,
        window_size=window_size,
        min_samples=min_samples,
        closed=closed,
    )
    std = rolling_std_time(
        value,
        by=by,
        window_size=window_size,
        min_samples=min_samples,
        closed=closed,
        ddof=ddof,
    )
    return (value - mean) / std


def apply_layers(
    table: pl.DataFrame,
    layers: Iterable[Iterable[pl.Expr]],
) -> pl.DataFrame:
    """Apply Expr groups sequentially so later layers can use earlier outputs."""

    result = table
    for layer in layers:
        expressions = tuple(layer)
        if expressions:
            result = result.with_columns(*expressions)
    return result


TOOLS = (
    ToolInfo("kit.feat.lag", "feat", "expr", "lag a value by past rows"),
    ToolInfo("kit.feat.difference", "feat", "expr", "subtract a past value from now"),
    ToolInfo("kit.feat.ratio", "feat", "expr", "divide two expressions directly"),
    ToolInfo("kit.feat.log_return", "feat", "expr", "calculate a backward log change"),
    ToolInfo("kit.feat.log1p", "feat", "expr", "apply a raw log-one-plus transform"),
    ToolInfo(
        "kit.feat.tanh",
        "feat",
        "expr",
        "apply a bounded monotonic transform with explicit scale",
    ),
    ToolInfo(
        "kit.feat.rolling_mean_rows",
        "feat",
        "expr",
        "calculate a trailing mean over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_std_rows",
        "feat",
        "expr",
        "calculate a trailing standard deviation over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_quantile_rows",
        "feat",
        "expr",
        "calculate a trailing quantile over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_zscore_rows",
        "feat",
        "expr",
        "standardize over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_zscore_time",
        "feat",
        "expr",
        "standardize over an explicit duration window",
    ),
    ToolInfo(
        "kit.feat.rolling_robust_scale_rows",
        "feat",
        "expr",
        "scale by a rolling median and interquartile range",
    ),
    ToolInfo(
        "kit.feat.apply_layers",
        "feat",
        "table",
        "apply dependent Polars expression layers in order",
    ),
)

__all__ = [
    "TOOLS",
    "apply_layers",
    "difference",
    "lag",
    "log1p",
    "log_return",
    "ratio",
    "rolling_mean_rows",
    "rolling_mean_time",
    "rolling_quantile_rows",
    "rolling_robust_scale_rows",
    "rolling_std_rows",
    "rolling_std_time",
    "rolling_zscore_rows",
    "rolling_zscore_time",
    "tanh",
]
