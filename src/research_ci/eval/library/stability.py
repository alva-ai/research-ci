"""Stability and temporal-behaviour diagnostics."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

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
        name="library.stability.walk_forward_association",
        family="stability",
        stages=("sig",),
        input_kind="evaluation_table",
        purpose="predictive",
        input_roles=("signals", "target", "keys"),
        requires=("target", "horizon"),
        description=(
            "Summarise submitted out-of-sample score association across explicit "
            "walk-forward folds."
        ),
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


def walk_forward_association(
    score: str,
    target: str,
    *,
    by: str,
    method: str = "spearman",
    min_fold_size: int = 2,
    min_folds: int = 2,
    expected_direction: str | None = None,
    minimum_abs_mean: float | None = None,
    minimum_sign_consistency: float | None = None,
    maximum_std: float | None = None,
    name: str = "library.stability.walk_forward_association",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Evaluate already out-of-sample scores across an explicit fold column."""

    if not score.strip() or not target.strip() or not by.strip():
        raise ValueError("walk-forward association needs named columns")
    if len({score, target, by}) != 3:
        raise ValueError("walk-forward score, target, and fold columns must differ")
    if method not in {"pearson", "spearman"}:
        raise ValueError("walk-forward method must be pearson or spearman")
    if (
        isinstance(min_fold_size, bool)
        or not isinstance(min_fold_size, int)
        or min_fold_size < 2
    ):
        raise ValueError("walk-forward min_fold_size must be at least 2")
    if isinstance(min_folds, bool) or not isinstance(min_folds, int) or min_folds < 2:
        raise ValueError("walk-forward min_folds must be at least 2")
    if expected_direction not in {None, "positive", "negative"}:
        raise ValueError(
            "walk-forward expected_direction must be positive, negative, or None"
        )
    if minimum_abs_mean is not None and (
        not math.isfinite(minimum_abs_mean) or minimum_abs_mean < 0
    ):
        raise ValueError("walk-forward minimum_abs_mean cannot be negative")
    if minimum_sign_consistency is not None and not (
        0 <= minimum_sign_consistency <= 1
    ):
        raise ValueError(
            "walk-forward minimum_sign_consistency must be between zero and one"
        )
    if maximum_std is not None and (not math.isfinite(maximum_std) or maximum_std < 0):
        raise ValueError("walk-forward maximum_std cannot be negative")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, (score, target, by))
        if missing:
            return not_applicable(
                "walk-forward association columns are missing",
                missing_columns=missing,
            )
        selected = (
            table.select(
                pl.col(by),
                pl.col(score).cast(pl.Float64, strict=False).alias(score),
                pl.col(target).cast(pl.Float64, strict=False).alias(target),
            )
            .drop_nulls()
            .filter(pl.col(score).is_finite() & pl.col(target).is_finite())
        )
        grouped = selected.group_by(by, maintain_order=True).agg(
            pl.len().alias("__rows"),
            pl.col(score).n_unique().alias("__score_unique"),
            pl.col(target).n_unique().alias("__target_unique"),
            pl.corr(score, target, method=method).alias("__association"),
        )
        folds: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for row in grouped.iter_rows(named=True):
            value = row["__association"]
            valid = (
                row["__rows"] >= min_fold_size
                and row["__score_unique"] >= 2
                and row["__target_unique"] >= 2
                and value is not None
                and math.isfinite(float(value))
            )
            item = {
                "fold": row[by],
                "rows": int(row["__rows"]),
                "association": float(value) if valid else None,
            }
            (folds if valid else skipped).append(item)
        if len(folds) < min_folds:
            return not_applicable(
                "fewer than the minimum valid walk-forward folds",
                valid_folds=len(folds),
                min_folds=min_folds,
                skipped_folds=skipped,
                score=score,
                target=target,
                group=by,
            )

        values = [float(fold["association"]) for fold in folds]
        average = sum(values) / len(values)
        dispersion = math.sqrt(
            sum((value - average) ** 2 for value in values) / (len(values) - 1)
        )
        if expected_direction is None:
            reference = average if average != 0 else float(median(values))
            direction = 1 if reference >= 0 else -1
            direction_source = "fold_mean"
        else:
            direction = 1 if expected_direction == "positive" else -1
            direction_source = "expected_direction"
        worst = min(folds, key=lambda fold: direction * float(fold["association"]))
        sign_consistency = sum(direction * value > 0 for value in values) / len(values)

        reasons: list[str] = []
        threshold_failures: list[dict[str, float]] = []
        if minimum_abs_mean is not None and abs(average) < minimum_abs_mean:
            reasons.append("WEAK_ASSOCIATION")
            threshold_failures.append(
                {
                    "metric": "absolute_mean_association",
                    "value": abs(average),
                    "minimum": minimum_abs_mean,
                }
            )
        if (
            minimum_sign_consistency is not None
            and sign_consistency < minimum_sign_consistency
        ):
            reasons.append("WALK_FORWARD_UNSTABLE")
            threshold_failures.append(
                {
                    "metric": "sign_consistency",
                    "value": sign_consistency,
                    "minimum": minimum_sign_consistency,
                }
            )
        if maximum_std is not None and dispersion > maximum_std:
            reasons.append("WALK_FORWARD_UNSTABLE")
            threshold_failures.append(
                {
                    "metric": "association_std",
                    "value": dispersion,
                    "maximum": maximum_std,
                }
            )
        reasons = list(dict.fromkeys(reasons))
        return EvalObservation(
            status="failed" if threshold_failures else "passed",
            value=average,
            output_kind="findings" if threshold_failures else "metric",
            details={
                "score": score,
                "target": target,
                "method": method,
                "group": by,
                "folds": folds,
                "skipped_folds": skipped,
                "valid_folds": len(folds),
                "mean_association": average,
                "median_association": float(median(values)),
                "association_std": dispersion,
                "expected_direction": expected_direction,
                "direction_source": direction_source,
                "sign_consistency": sign_consistency,
                "worst_directional_fold": worst,
                "thresholds": {
                    "minimum_abs_mean": minimum_abs_mean,
                    "minimum_sign_consistency": minimum_sign_consistency,
                    "maximum_std": maximum_std,
                },
                "threshold_failures": threshold_failures,
                "reason_codes": reasons,
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_join_group",
        library_method="library.stability.walk_forward_association",
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


__all__ = [
    "METHODS",
    "distribution_shift",
    "rank_autocorrelation",
    "split_association",
    "walk_forward_association",
]
