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


def second_difference(value: pl.Expr, periods: int = 1) -> pl.Expr:
    """Return the backward difference of a backward difference."""

    first = difference(value, periods)
    return first - lag(first, periods)


def ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Return a raw ratio without epsilon, clipping, or null filling."""

    return numerator / denominator


def finite_ratio_or_null(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Return a ratio only for finite values and a nonzero denominator."""

    numerator = numerator.cast(pl.Float64)
    denominator = denominator.cast(pl.Float64)
    valid = numerator.is_finite() & denominator.is_finite() & (denominator != 0)
    return pl.when(valid).then(numerator / denominator)


def simple_return(value: pl.Expr, periods: int = 1) -> pl.Expr:
    """Return the backward simple return without denominator treatment."""

    return ratio(value, lag(value, periods)) - 1.0


def log_return(value: pl.Expr, periods: int = 1) -> pl.Expr:
    """Return the backward log change for an already-positive value."""

    return value.log() - lag(value, periods).log()


def positive_log_return(value: pl.Expr, periods: int = 1) -> pl.Expr:
    """Return a log change only where current and past values are positive."""

    previous = lag(value, periods)
    return pl.when((value > 0) & (previous > 0)).then((value / previous).log())


def log1p(value: pl.Expr) -> pl.Expr:
    """Apply log(1 + x) without clipping or replacing invalid inputs."""

    return value.log1p()


def signed_log1p(value: pl.Expr) -> pl.Expr:
    """Compress magnitude with sign(x) * log(1 + abs(x))."""

    return value.sign() * value.abs().log1p()


def imbalance(positive: pl.Expr, negative: pl.Expr) -> pl.Expr:
    """Return (positive - negative) / (positive + negative) as a raw ratio."""

    return ratio(positive - negative, positive + negative)


def true_range(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """Return Wilder true range; the initial row reduces to high minus low."""

    previous_close = lag(close)
    return pl.max_horizontal(
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    )


def tanh(value: pl.Expr, *, scale: float = 1.0) -> pl.Expr:
    """Apply a bounded monotonic transform with an explicit positive scale."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    return (value / scale).tanh()


def rolling_sum_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
) -> pl.Expr:
    """Calculate a row-clock rolling sum."""

    window_size = _positive(window_size, "window_size")
    if min_samples is not None:
        _positive(min_samples, "min_samples")
    return value.rolling_sum(window_size, min_samples=min_samples)


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


def relative_to_rolling_mean_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
) -> pl.Expr:
    """Divide the current value by its trailing row-clock mean."""

    mean = rolling_mean_rows(
        value,
        window_size=window_size,
        min_samples=min_samples,
    )
    return ratio(value, mean)


def rolling_mean_ratio_rows(
    value: pl.Expr,
    *,
    short_window_size: int,
    long_window_size: int,
    short_min_samples: int | None = None,
    long_min_samples: int | None = None,
) -> pl.Expr:
    """Divide a short trailing mean by a long trailing mean."""

    short = rolling_mean_rows(
        value,
        window_size=short_window_size,
        min_samples=short_min_samples,
    )
    long = rolling_mean_rows(
        value,
        window_size=long_window_size,
        min_samples=long_min_samples,
    )
    return ratio(short, long)


def average_true_range_rows(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
) -> pl.Expr:
    """Calculate a row-clock simple moving average of Wilder true range."""

    return rolling_mean_rows(
        true_range(high, low, close),
        window_size=window_size,
        min_samples=min_samples,
    )


def rolling_min_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
) -> pl.Expr:
    """Calculate a row-clock rolling minimum."""

    window_size = _positive(window_size, "window_size")
    if min_samples is not None:
        _positive(min_samples, "min_samples")
    return value.rolling_min(window_size, min_samples=min_samples)


def rolling_max_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
) -> pl.Expr:
    """Calculate a row-clock rolling maximum."""

    window_size = _positive(window_size, "window_size")
    if min_samples is not None:
        _positive(min_samples, "min_samples")
    return value.rolling_max(window_size, min_samples=min_samples)


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


def rolling_std_ratio_rows(
    value: pl.Expr,
    *,
    short_window_size: int,
    long_window_size: int,
    short_min_samples: int | None = None,
    long_min_samples: int | None = None,
    ddof: int = 1,
) -> pl.Expr:
    """Divide short-window standard deviation by long-window deviation."""

    short = rolling_std_rows(
        value,
        window_size=short_window_size,
        min_samples=short_min_samples,
        ddof=ddof,
    )
    long = rolling_std_rows(
        value,
        window_size=long_window_size,
        min_samples=long_min_samples,
        ddof=ddof,
    )
    return ratio(short, long)


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


def rolling_correlation_rows(
    left: pl.Expr,
    right: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
) -> pl.Expr:
    """Calculate row-clock Pearson correlation for paired observations."""

    window_size = _positive(window_size, "window_size")
    if min_samples is not None:
        _positive(min_samples, "min_samples")
    return pl.rolling_corr(
        left,
        right,
        window_size=window_size,
        min_samples=min_samples,
    )


def _relative_strength(
    average_gain: pl.Expr,
    average_loss: pl.Expr,
    *,
    scale: float,
    flat_value: float | None,
) -> pl.Expr:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if flat_value is not None and not 0.0 <= flat_value <= scale:
        raise ValueError("flat_value must be between zero and scale")

    total_move = average_gain + average_loss
    strength = ratio(average_gain, total_move) * scale
    if flat_value is None:
        return strength
    observed = average_gain.is_not_null() & average_loss.is_not_null()
    return (
        pl.when(~observed)
        .then(None)
        .when(total_move == 0)
        .then(pl.lit(flat_value))
        .otherwise(strength)
    )


def relative_strength_index_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
    periods: int = 1,
    scale: float = 100.0,
    flat_value: float | None = None,
) -> pl.Expr:
    """Calculate row-clock RSI with optional explicit flat-window treatment."""

    change = difference(value, periods)
    average_gain = rolling_mean_rows(
        change.clip(lower_bound=0),
        window_size=window_size,
        min_samples=min_samples,
    )
    average_loss = rolling_mean_rows(
        (-change).clip(lower_bound=0),
        window_size=window_size,
        min_samples=min_samples,
    )
    return _relative_strength(
        average_gain,
        average_loss,
        scale=scale,
        flat_value=flat_value,
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


def rolling_range_position_rows(
    value: pl.Expr,
    *,
    window_size: int,
    min_samples: int | None = None,
) -> pl.Expr:
    """Locate a value in its trailing [minimum, maximum] row-clock range."""

    low = rolling_min_rows(
        value,
        window_size=window_size,
        min_samples=min_samples,
    )
    high = rolling_max_rows(
        value,
        window_size=window_size,
        min_samples=min_samples,
    )
    return ratio(value - low, high - low)


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


def rolling_sum_time(
    value: pl.Expr,
    *,
    by: pl.Expr,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
) -> pl.Expr:
    """Calculate a duration-clock rolling sum with explicit warm-up."""

    _positive(min_samples, "min_samples")
    return value.rolling_sum_by(
        by,
        window_size,
        min_samples=min_samples,
        closed=closed,
    )


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


def rolling_min_time(
    value: pl.Expr,
    *,
    by: pl.Expr,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
) -> pl.Expr:
    """Calculate a duration-clock rolling minimum with explicit warm-up."""

    _positive(min_samples, "min_samples")
    return value.rolling_min_by(
        by,
        window_size,
        min_samples=min_samples,
        closed=closed,
    )


def rolling_max_time(
    value: pl.Expr,
    *,
    by: pl.Expr,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
) -> pl.Expr:
    """Calculate a duration-clock rolling maximum with explicit warm-up."""

    _positive(min_samples, "min_samples")
    return value.rolling_max_by(
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


def rolling_quantile_time(
    value: pl.Expr,
    *,
    by: pl.Expr,
    quantile: float,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
    interpolation: Literal[
        "nearest", "higher", "lower", "midpoint", "linear", "equiprobable"
    ] = "linear",
) -> pl.Expr:
    """Calculate one duration-clock rolling quantile with explicit semantics."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    _positive(min_samples, "min_samples")
    return value.rolling_quantile_by(
        by,
        window_size,
        quantile=quantile,
        interpolation=interpolation,
        min_samples=min_samples,
        closed=closed,
    )


def rolling_correlation_time(
    left: pl.Expr,
    right: pl.Expr,
    *,
    by: pl.Expr,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
) -> pl.Expr:
    """Calculate pairwise non-null Pearson correlation on a duration clock.

    NaNs remain NaNs and zero-variance windows remain null. The input must be
    ordered by ``by``; sorting and partitioning remain the Lab's responsibility.
    """

    _positive(min_samples, "min_samples")
    paired = left.is_not_null() & right.is_not_null()
    paired_left = pl.when(paired).then(left.cast(pl.Float64))
    paired_right = pl.when(paired).then(right.cast(pl.Float64))
    options = {
        "by": by,
        "window_size": window_size,
        "min_samples": 1,
        "closed": closed,
    }
    count = rolling_sum_time(paired.cast(pl.Float64), **options)
    mean_left = rolling_mean_time(paired_left, **options)
    mean_right = rolling_mean_time(paired_right, **options)
    variance_left = rolling_mean_time(paired_left * paired_left, **options) - (
        mean_left * mean_left
    )
    variance_right = rolling_mean_time(paired_right * paired_right, **options) - (
        mean_right * mean_right
    )
    covariance = rolling_mean_time(paired_left * paired_right, **options) - (
        mean_left * mean_right
    )
    ready = (count >= min_samples) & (variance_left > 0) & (variance_right > 0)
    return pl.when(ready).then(covariance / (variance_left * variance_right).sqrt())


def relative_strength_index_time(
    value: pl.Expr,
    *,
    by: pl.Expr,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
    periods: int = 1,
    scale: float = 100.0,
    flat_value: float | None = None,
) -> pl.Expr:
    """Calculate duration-clock RSI with explicit flat-window treatment."""

    change = difference(value, periods)
    options = {
        "by": by,
        "window_size": window_size,
        "min_samples": min_samples,
        "closed": closed,
    }
    average_gain = rolling_mean_time(change.clip(lower_bound=0), **options)
    average_loss = rolling_mean_time((-change).clip(lower_bound=0), **options)
    return _relative_strength(
        average_gain,
        average_loss,
        scale=scale,
        flat_value=flat_value,
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


def rolling_range_position_time(
    value: pl.Expr,
    *,
    by: pl.Expr,
    window_size: str | timedelta,
    min_samples: int,
    closed: ClosedInterval = "right",
) -> pl.Expr:
    """Locate a value in its trailing duration-clock [minimum, maximum] range."""

    low = rolling_min_time(
        value,
        by=by,
        window_size=window_size,
        min_samples=min_samples,
        closed=closed,
    )
    high = rolling_max_time(
        value,
        by=by,
        window_size=window_size,
        min_samples=min_samples,
        closed=closed,
    )
    return ratio(value - low, high - low)


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
    ToolInfo(
        "kit.feat.second_difference",
        "feat",
        "expr",
        "difference a value twice over the same row lag",
    ),
    ToolInfo("kit.feat.ratio", "feat", "expr", "divide two expressions directly"),
    ToolInfo(
        "kit.feat.finite_ratio_or_null",
        "feat",
        "expr",
        "divide finite expressions and null invalid denominators",
    ),
    ToolInfo(
        "kit.feat.simple_return",
        "feat",
        "expr",
        "calculate a backward simple return",
    ),
    ToolInfo("kit.feat.log_return", "feat", "expr", "calculate a backward log change"),
    ToolInfo(
        "kit.feat.positive_log_return",
        "feat",
        "expr",
        "calculate a backward log change only for positive values",
    ),
    ToolInfo("kit.feat.log1p", "feat", "expr", "apply a raw log-one-plus transform"),
    ToolInfo(
        "kit.feat.signed_log1p",
        "feat",
        "expr",
        "compress magnitude while preserving sign",
    ),
    ToolInfo(
        "kit.feat.imbalance",
        "feat",
        "expr",
        "calculate a raw signed imbalance ratio",
    ),
    ToolInfo(
        "kit.feat.true_range",
        "feat",
        "expr",
        "calculate Wilder true range",
    ),
    ToolInfo(
        "kit.feat.average_true_range_rows",
        "feat",
        "expr",
        "average Wilder true range over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.tanh",
        "feat",
        "expr",
        "apply a bounded monotonic transform with explicit scale",
    ),
    ToolInfo(
        "kit.feat.rolling_sum_rows",
        "feat",
        "expr",
        "calculate a trailing sum over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_mean_rows",
        "feat",
        "expr",
        "calculate a trailing mean over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.relative_to_rolling_mean_rows",
        "feat",
        "expr",
        "divide a value by its trailing row-clock mean",
    ),
    ToolInfo(
        "kit.feat.rolling_mean_ratio_rows",
        "feat",
        "expr",
        "divide short- and long-window trailing means",
    ),
    ToolInfo(
        "kit.feat.rolling_min_rows",
        "feat",
        "expr",
        "calculate a trailing minimum over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_max_rows",
        "feat",
        "expr",
        "calculate a trailing maximum over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_std_rows",
        "feat",
        "expr",
        "calculate a trailing standard deviation over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_std_ratio_rows",
        "feat",
        "expr",
        "divide short- and long-window trailing deviations",
    ),
    ToolInfo(
        "kit.feat.rolling_quantile_rows",
        "feat",
        "expr",
        "calculate a trailing quantile over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_correlation_rows",
        "feat",
        "expr",
        "calculate trailing Pearson correlation over a row window",
    ),
    ToolInfo(
        "kit.feat.relative_strength_index_rows",
        "feat",
        "expr",
        "calculate RSI over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_zscore_rows",
        "feat",
        "expr",
        "standardize over an explicit row window",
    ),
    ToolInfo(
        "kit.feat.rolling_range_position_rows",
        "feat",
        "expr",
        "locate a value in its trailing row-clock range",
    ),
    ToolInfo(
        "kit.feat.rolling_sum_time",
        "feat",
        "expr",
        "calculate a trailing sum over an explicit duration window",
    ),
    ToolInfo(
        "kit.feat.rolling_mean_time",
        "feat",
        "expr",
        "calculate a trailing mean over an explicit duration window",
    ),
    ToolInfo(
        "kit.feat.rolling_min_time",
        "feat",
        "expr",
        "calculate a trailing minimum over an explicit duration window",
    ),
    ToolInfo(
        "kit.feat.rolling_max_time",
        "feat",
        "expr",
        "calculate a trailing maximum over an explicit duration window",
    ),
    ToolInfo(
        "kit.feat.rolling_std_time",
        "feat",
        "expr",
        "calculate a trailing standard deviation over a duration window",
    ),
    ToolInfo(
        "kit.feat.rolling_quantile_time",
        "feat",
        "expr",
        "calculate a trailing quantile over an explicit duration window",
    ),
    ToolInfo(
        "kit.feat.rolling_correlation_time",
        "feat",
        "expr",
        "calculate pairwise correlation over a duration window",
    ),
    ToolInfo(
        "kit.feat.relative_strength_index_time",
        "feat",
        "expr",
        "calculate RSI over an explicit duration window",
    ),
    ToolInfo(
        "kit.feat.rolling_zscore_time",
        "feat",
        "expr",
        "standardize over an explicit duration window",
    ),
    ToolInfo(
        "kit.feat.rolling_range_position_time",
        "feat",
        "expr",
        "locate a value in its trailing duration-clock range",
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
    "average_true_range_rows",
    "apply_layers",
    "difference",
    "finite_ratio_or_null",
    "imbalance",
    "lag",
    "log1p",
    "log_return",
    "positive_log_return",
    "ratio",
    "relative_strength_index_rows",
    "relative_strength_index_time",
    "relative_to_rolling_mean_rows",
    "rolling_correlation_rows",
    "rolling_correlation_time",
    "rolling_max_rows",
    "rolling_max_time",
    "rolling_mean_rows",
    "rolling_mean_ratio_rows",
    "rolling_mean_time",
    "rolling_min_rows",
    "rolling_min_time",
    "rolling_quantile_rows",
    "rolling_quantile_time",
    "rolling_range_position_rows",
    "rolling_range_position_time",
    "rolling_robust_scale_rows",
    "rolling_std_rows",
    "rolling_std_ratio_rows",
    "rolling_std_time",
    "rolling_sum_rows",
    "rolling_sum_time",
    "rolling_zscore_rows",
    "rolling_zscore_time",
    "second_difference",
    "signed_log1p",
    "simple_return",
    "tanh",
    "true_range",
]
