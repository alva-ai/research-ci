"""Stability and temporal-behaviour diagnostics."""

from __future__ import annotations

import math

import polars as pl

from ..protocol import EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo
from ._utils import missing_columns, not_applicable

METHODS = (
    MethodInfo(
        name="library.stability.split_association",
        family="stability",
        stages=("sig",),
        input_kind="evaluation_table",
        purpose="predictive",
        input_roles=("features", "signals", "target"),
        requires=("target", "horizon"),
        description="Compare score/target association across ordered sample halves.",
    ),
    MethodInfo(
        name="library.stability.rank_autocorrelation",
        family="stability",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        purpose="stability",
        input_roles=("features", "signals"),
        description="Measure persistence of a selected column's rank order.",
    ),
    MethodInfo(
        name="library.stability.distribution_shift",
        family="stability",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        purpose="stability",
        input_roles=("features", "signals"),
        description=(
            "Compare early and late selected-column distributions with a standardised "
            "mean shift."
        ),
    ),
)


def split_association(
    score: str,
    target: str,
    *,
    method: str = "spearman",
    max_absolute_change: float | None = None,
    name: str = "library.stability.split_association",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a simple ordered half-sample association stability check."""

    if method not in {"pearson", "spearman"}:
        raise ValueError("split association method must be pearson or spearman")
    if max_absolute_change is not None and max_absolute_change < 0:
        raise ValueError("max_absolute_change cannot be negative")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, (score, target))
        if missing:
            return not_applicable(
                "split association columns are missing", missing_columns=missing
            )
        pairs = _complete_numeric_frame(table, (score, target))
        midpoint = pairs.height // 2
        if midpoint < 2 or pairs.height - midpoint < 2:
            return not_applicable("fewer than four complete rows", rows=pairs.height)
        first = (
            pairs.head(midpoint).select(pl.corr(score, target, method=method)).item()
        )
        second = (
            pairs.slice(midpoint).select(pl.corr(score, target, method=method)).item()
        )
        if (
            first is None
            or second is None
            or not math.isfinite(float(first))
            or not math.isfinite(float(second))
        ):
            return not_applicable(
                "association is undefined in one split", rows=pairs.height
            )
        first = float(first)
        second = float(second)
        absolute_change = abs(second - first)
        unstable = (
            max_absolute_change is not None and absolute_change > max_absolute_change
        )
        return EvalObservation(
            status="failed" if unstable else "passed",
            value=float(second - first),
            output_kind="findings" if unstable else "metric",
            details={
                "score": score,
                "target": target,
                "method": method,
                "first_half": first,
                "second_half": second,
                "absolute_change": absolute_change,
                "max_absolute_change": max_absolute_change,
                "same_sign": first == 0 or second == 0 or (first > 0) == (second > 0),
                "rows": pairs.height,
                "reason_code": "UNSTABLE" if unstable else None,
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_select",
        library_method="library.stability.split_association",
    )


def rank_autocorrelation(
    column: str,
    *,
    lag: int = 1,
    name: str = "library.stability.rank_autocorrelation",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a rank-persistence diagnostic in table order."""

    if lag < 1:
        raise ValueError("rank autocorrelation lag must be positive")

    def measure(table: pl.DataFrame) -> EvalObservation:
        if column not in table.columns:
            return not_applicable(
                "rank autocorrelation column is missing", missing_columns=[column]
            )
        values = _complete_numeric_frame(table, (column,))
        if values.height <= lag:
            return not_applicable(
                "insufficient variation for rank autocorrelation", rows=values.height
            )
        pairs = values.select(
            pl.col(column).alias("__current"),
            pl.col(column).shift(lag).alias("__lag"),
        ).drop_nulls()
        result = pairs.select(pl.corr("__current", "__lag", method="spearman")).item()
        if result is None or not math.isfinite(float(result)):
            return not_applicable(
                "rank autocorrelation is undefined", rows=values.height
            )
        return EvalObservation(
            status="passed",
            value=float(result),
            output_kind="metric",
            details={"column": column, "lag": lag, "rows": values.height},
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_select",
        library_method="library.stability.rank_autocorrelation",
    )


def distribution_shift(
    column: str,
    *,
    max_absolute_shift: float | None = None,
    name: str = "library.stability.distribution_shift",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build an early/late standardised mean-shift diagnostic."""

    if max_absolute_shift is not None and max_absolute_shift < 0:
        raise ValueError("max_absolute_shift cannot be negative")

    def measure(table: pl.DataFrame) -> EvalObservation:
        if column not in table.columns:
            return not_applicable(
                "distribution shift column is missing", missing_columns=[column]
            )
        values = _complete_numeric_frame(table, (column,))
        midpoint = values.height // 2
        if midpoint < 2 or values.height - midpoint < 2:
            return not_applicable("fewer than four usable rows", rows=values.height)
        first = (
            values.head(midpoint)
            .select(
                pl.col(column).mean().alias("mean"),
                pl.col(column).var(ddof=1).alias("variance"),
                pl.len().alias("rows"),
            )
            .row(0, named=True)
        )
        second = (
            values.slice(midpoint)
            .select(
                pl.col(column).mean().alias("mean"),
                pl.col(column).var(ddof=1).alias("variance"),
                pl.len().alias("rows"),
            )
            .row(0, named=True)
        )
        first_mean = float(first["mean"])
        second_mean = float(second["mean"])
        pooled_variance = (
            (first["rows"] - 1) * float(first["variance"])
            + (second["rows"] - 1) * float(second["variance"])
        ) / (first["rows"] + second["rows"] - 2)
        pooled_std = math.sqrt(max(0.0, pooled_variance))
        if pooled_std == 0:
            return not_applicable(
                "distribution has zero pooled variance", rows=values.height
            )
        shift = (second_mean - first_mean) / pooled_std
        unstable = max_absolute_shift is not None and abs(shift) > max_absolute_shift
        return EvalObservation(
            status="failed" if unstable else "passed",
            value=float(shift),
            output_kind="findings" if unstable else "metric",
            details={
                "column": column,
                "first_mean": first_mean,
                "second_mean": second_mean,
                "pooled_std": pooled_std,
                "absolute_shift": abs(shift),
                "max_absolute_shift": max_absolute_shift,
                "rows": values.height,
                "reason_code": "DISTRIBUTION_SHIFT" if unstable else None,
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_select",
        library_method="library.stability.distribution_shift",
    )


def _complete_numeric_frame(
    table: pl.DataFrame, columns: tuple[str, ...]
) -> pl.DataFrame:
    selected = table.select(
        pl.col(column).cast(pl.Float64, strict=False).alias(column)
        for column in columns
    )
    return selected.drop_nulls().filter(
        pl.all_horizontal([pl.col(column).is_finite() for column in columns])
    )


__all__ = ["METHODS", "distribution_shift", "rank_autocorrelation", "split_association"]
