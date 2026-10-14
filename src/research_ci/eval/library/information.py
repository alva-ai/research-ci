"""Information-theoretic diagnostics for feature research."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import polars as pl
from sklearn.feature_selection import mutual_info_regression

from ..protocol import EvalContext, EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo
from ._utils import missing_columns, not_applicable

METHODS = (
    MethodInfo(
        name="library.information.mutual_information",
        family="information",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description="Estimate pairwise mutual information with a k-NN estimator.",
    ),
    MethodInfo(
        name="library.information.feature_target_mutual_information",
        family="information",
        stages=("feat", "sig"),
        input_kind="evaluation_table",
        description=(
            "Estimate nonlinear feature/target dependence with a k-NN estimator."
        ),
    ),
    MethodInfo(
        name="library.information.context_feature_target_mutual_information",
        family="information",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description="Estimate feature/target MI from a named target sidecar.",
    ),
)


def mutual_information(
    left: str,
    right: str,
    *,
    bins: int = 10,
    normalized: bool = True,
    min_rows: int = 20,
    minimum: float | None = None,
    n_neighbors: int = 3,
    random_state: int | None = 0,
    name: str = "library.information.mutual_information",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a k-NN mutual-information metric for two numeric columns.

    ``bins`` remains in the public interface for compatibility with the former
    rank-histogram estimator. The scikit-learn estimator does not discretize
    continuous inputs, so the value is recorded but not used in estimation.
    """

    if bins < 2:
        raise ValueError("mutual information requires at least two bins")
    if min_rows < 2:
        raise ValueError("mutual information min_rows must be at least 2")
    if minimum is not None and minimum < 0:
        raise ValueError("mutual information minimum cannot be negative")
    if n_neighbors < 1:
        raise ValueError("mutual information n_neighbors must be positive")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, (left, right))
        if missing:
            return not_applicable(
                "mutual information columns are missing", missing_columns=missing
            )
        matrix = _complete_pair(table, left, right)
        if matrix.shape[0] < min_rows:
            return not_applicable(
                "fewer than the minimum mutual-information rows",
                rows=matrix.shape[0],
                min_rows=min_rows,
            )
        if np.isclose(np.var(matrix[:, 0]), 0.0) or np.isclose(
            np.var(matrix[:, 1]), 0.0
        ):
            return not_applicable("mutual information input has no variation")
        effective_neighbors = min(n_neighbors, matrix.shape[0] - 1)
        value = _estimate_mi(
            matrix[:, 0],
            matrix[:, 1],
            n_neighbors=effective_neighbors,
            random_state=random_state,
        )
        details: dict[str, object] = {"raw_mutual_information": value}
        if normalized:
            left_self_information = _estimate_mi(
                matrix[:, 0],
                matrix[:, 0],
                n_neighbors=effective_neighbors,
                random_state=random_state,
            )
            right_self_information = _estimate_mi(
                matrix[:, 1],
                matrix[:, 1],
                n_neighbors=effective_neighbors,
                random_state=random_state,
            )
            denominator = math.sqrt(left_self_information * right_self_information)
            value = min(1.0, max(0.0, value / denominator)) if denominator else 0.0
            details.update(
                {
                    "left_self_information": left_self_information,
                    "right_self_information": right_self_information,
                }
            )
        details.update(
            {
                "left": left,
                "right": right,
                "rows": matrix.shape[0],
                "bins": bins,
                "bins_used": False,
                "normalized": normalized,
                "execution_mode": "matrix_backend",
                "backend": "scikit-learn",
                "estimator": "sklearn.feature_selection.mutual_info_regression",
                "n_neighbors": effective_neighbors,
                "requested_n_neighbors": n_neighbors,
                "random_state": random_state,
                "n_jobs": 1,
                "matrix_preparation": ("polars_select_cast_drop_nulls_filter_finite"),
            }
        )
        weak = minimum is not None and value < minimum
        details["minimum"] = minimum
        details["reason_code"] = "WEAK_ASSOCIATION" if weak else None
        return EvalObservation(
            status="failed" if weak else "passed",
            value=float(value),
            output_kind="findings" if weak else "metric",
            details=details,
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="matrix_backend",
    )


def feature_target_mutual_information(
    feature: str,
    target: str,
    *,
    bins: int = 10,
    normalized: bool = True,
    min_rows: int = 20,
    minimum: float | None = None,
    n_neighbors: int = 3,
    random_state: int | None = 0,
    name: str = "library.information.feature_target_mutual_information",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Convenience wrapper that names MI as a feature/target diagnostic."""

    return mutual_information(
        feature,
        target,
        bins=bins,
        normalized=normalized,
        min_rows=min_rows,
        minimum=minimum,
        n_neighbors=n_neighbors,
        random_state=random_state,
        name=name,
        failure_policy=failure_policy,
    )


def context_feature_target_mutual_information(
    feature: str,
    target: str,
    *,
    context_table: str = "target",
    join_on: Iterable[str] = ("event_time",),
    bins: int = 10,
    normalized: bool = True,
    min_rows: int = 20,
    minimum: float | None = None,
    n_neighbors: int = 3,
    random_state: int | None = 0,
    name: str = "library.information.context_feature_target_mutual_information",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build target MI while keeping the target in a named sidecar."""

    keys = tuple(dict.fromkeys(join_on))
    metric = mutual_information(
        feature,
        target,
        bins=bins,
        normalized=normalized,
        min_rows=min_rows,
        minimum=minimum,
        n_neighbors=n_neighbors,
        random_state=random_state,
        name=name,
        failure_policy=failure_policy,
    )

    def measure(table: pl.DataFrame, context: EvalContext) -> EvalObservation:
        try:
            sidecar = context.table(context_table)
        except KeyError as exc:
            return not_applicable(str(exc), context_table=context_table)
        required = (*keys, feature)
        missing = missing_columns(table, required) + missing_columns(
            sidecar, (*keys, target)
        )
        if missing:
            return not_applicable(
                "target context is missing required columns",
                missing_columns=sorted(set(missing)),
                context_table=context_table,
            )
        sidecar_keys = sidecar.select(keys)
        duplicate_keys = sidecar_keys.height - sidecar_keys.unique().height
        if duplicate_keys:
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={
                    "reason": "target context has duplicate join keys",
                    "reason_code": "CONTEXT_DUPLICATE_KEYS",
                    "context_table": context_table,
                    "join_on": keys,
                    "duplicate_key_rows": duplicate_keys,
                },
            )
        joined = table.select([*keys, feature]).join(
            sidecar.select([*keys, target]), on=list(keys), how="inner"
        )
        if metric.runner is None:
            raise RuntimeError("mutual-information metric has no runner")
        return metric.runner(joined)

    return EvalSpec(
        name=name,
        context_runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="matrix_backend",
    )


def _complete_pair(table: pl.DataFrame, left: str, right: str) -> np.ndarray:
    """Select finite complete pairs in Polars before crossing into NumPy."""

    if left == right:
        selected = table.select(pl.col(left).cast(pl.Float64, strict=False).alias(left))
        complete = selected.drop_nulls().filter(pl.col(left).is_finite())
        values = complete.to_numpy()
        return np.column_stack((values[:, 0], values[:, 0]))
    selected = table.select(
        pl.col(left).cast(pl.Float64, strict=False).alias(left),
        pl.col(right).cast(pl.Float64, strict=False).alias(right),
    )
    complete = selected.drop_nulls().filter(
        pl.all_horizontal(pl.col(left).is_finite(), pl.col(right).is_finite())
    )
    return complete.to_numpy()


def _estimate_mi(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    n_neighbors: int,
    random_state: int | None,
) -> float:
    value = mutual_info_regression(
        feature.reshape(-1, 1),
        target,
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=1,
    )[0]
    return float(value)


__all__ = [
    "METHODS",
    "context_feature_target_mutual_information",
    "feature_target_mutual_information",
    "mutual_information",
]
