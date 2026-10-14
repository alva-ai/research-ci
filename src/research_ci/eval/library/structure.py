"""General structural diagnostics for feature and signal panels."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..protocol import EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo
from ._utils import (
    column_names,
    missing_columns,
    not_applicable,
)

METHODS = (
    MethodInfo(
        name="library.structure.pairwise_correlation",
        family="structure",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description="Measure dependence between two explicitly selected columns.",
    ),
    MethodInfo(
        name="library.structure.vif",
        family="structure",
        stages=("feat", "sig"),
        input_kind="feature_matrix",
        description="Measure linear redundancy with variance inflation factors.",
    ),
    MethodInfo(
        name="library.structure.pca_diagnostics",
        family="structure",
        stages=("feat", "sig"),
        input_kind="feature_matrix",
        description="Summarise explained variance and effective feature rank.",
    ),
    MethodInfo(
        name="library.structure.autocorrelation",
        family="structure",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description="Measure serial dependence at an explicit lag.",
    ),
)


def pairwise_correlation(
    left: str,
    right: str,
    *,
    method: str = "pearson",
    name: str = "library.structure.pairwise_correlation",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a structural correlation metric for two panel columns."""

    if method not in {"pearson", "spearman"}:
        raise ValueError("pairwise correlation method must be pearson or spearman")
    if not left.strip() or not right.strip():
        raise ValueError("structure evals need two named columns")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, (left, right))
        if missing:
            return not_applicable(
                "correlation columns are missing", missing_columns=missing
            )
        pairs = _complete_numeric_frame(table, (left, right))
        if pairs.height < 2:
            return not_applicable("fewer than two complete rows", rows=pairs.height)
        stats = pairs.select(
            pl.col(left).n_unique().alias("left_unique"),
            pl.col(right).n_unique().alias("right_unique"),
            pl.corr(left, right, method=method).alias("correlation"),
        ).row(0, named=True)
        value = stats["correlation"]
        if (
            stats["left_unique"] < 2
            or stats["right_unique"] < 2
            or value is None
            or not math.isfinite(float(value))
        ):
            return not_applicable("correlation is undefined", rows=pairs.height)
        return EvalObservation(
            status="passed",
            value=float(value),
            output_kind="metric",
            details={
                "left": left,
                "right": right,
                "method": method,
                "rows": pairs.height,
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_select",
    )


def vif(
    columns: Iterable[str],
    *,
    min_rows: int | None = None,
    maximum: float | None = None,
    name: str = "library.structure.vif",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a variance-inflation-factor diagnostic for a feature matrix."""

    selected = column_names(columns)
    required_rows = min_rows or max(3, len(selected) + 1)
    if maximum is not None and maximum <= 0:
        raise ValueError("VIF maximum must be positive")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, selected)
        if missing:
            return not_applicable("VIF columns are missing", missing_columns=missing)
        matrix = _complete_matrix(table, selected)
        if matrix.shape[0] < required_rows:
            return not_applicable(
                "fewer than the minimum complete rows for VIF",
                rows=matrix.shape[0],
                min_rows=required_rows,
            )
        variances = np.var(matrix, axis=0)
        constant_columns = [
            column
            for column, variance in zip(selected, variances, strict=True)
            if np.isclose(variance, 0.0)
        ]
        if constant_columns:
            return not_applicable(
                "VIF input has a constant feature",
                constant_columns=constant_columns,
            )
        values: dict[str, float] = {}
        for index, column in enumerate(selected):
            if len(selected) == 1:
                values[column] = 1.0
                continue
            others = [other for other in range(len(selected)) if other != index]
            predictors = matrix[:, others]
            response = matrix[:, index]
            model = LinearRegression().fit(predictors, response)
            r_squared = min(1.0, max(0.0, float(model.score(predictors, response))))
            values[column] = (
                math.inf if r_squared >= 1.0 - 1e-12 else 1.0 / (1.0 - r_squared)
            )
        max_value = max(values.values())
        over_limit = maximum is not None and max_value > maximum
        return EvalObservation(
            status="failed" if over_limit else "passed",
            value=values,
            output_kind="findings" if over_limit else "metric",
            details={
                "columns": selected,
                "rows": matrix.shape[0],
                "max_vif": max_value,
                "maximum": maximum,
                "singular": any(not math.isfinite(value) for value in values.values()),
                "execution_mode": "matrix_backend",
                "backend": "scikit-learn",
                "estimator": "sklearn.linear_model.LinearRegression",
                "matrix_preparation": ("polars_select_cast_drop_nulls_filter_finite"),
                "interpretation": (
                    "linear redundancy diagnostic; threshold is case-defined"
                ),
                "reason_code": "HIGH_REDUNDANCY" if over_limit else None,
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="matrix_backend",
    )


def pca_diagnostics(
    columns: Iterable[str],
    *,
    standardize: bool = True,
    min_rows: int | None = None,
    max_first_component: float | None = None,
    minimum_effective_rank: float | None = None,
    name: str = "library.structure.pca_diagnostics",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a PCA structure diagnostic with a scikit-learn matrix backend."""

    selected = column_names(columns)
    required_rows = min_rows or max(3, len(selected))
    if max_first_component is not None and not 0 < max_first_component <= 1:
        raise ValueError("PCA max_first_component must be between zero and one")
    if minimum_effective_rank is not None and minimum_effective_rank <= 0:
        raise ValueError("PCA minimum_effective_rank must be positive")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, selected)
        if missing:
            return not_applicable("PCA columns are missing", missing_columns=missing)
        matrix = _complete_matrix(table, selected)
        if matrix.shape[0] < required_rows:
            return not_applicable(
                "fewer than the minimum complete rows for PCA",
                rows=matrix.shape[0],
                min_rows=required_rows,
            )
        if np.allclose(np.var(matrix, axis=0), 0.0):
            return not_applicable("PCA has zero total variance")
        fit_matrix = StandardScaler().fit_transform(matrix) if standardize else matrix
        estimator = PCA(svd_solver="full")
        estimator.fit(fit_matrix)
        eigenvalues = [float(value) for value in estimator.explained_variance_]
        ratios = [float(value) for value in estimator.explained_variance_ratio_]
        if not ratios or not all(math.isfinite(value) for value in ratios):
            return not_applicable("PCA has zero total variance")
        cumulative = [float(value) for value in np.cumsum(ratios)]
        effective_rank = math.exp(
            -sum(ratio * math.log(ratio) for ratio in ratios if ratio > 0)
        )
        concentrated = (
            max_first_component is not None and ratios[0] > max_first_component
        ) or (
            minimum_effective_rank is not None
            and effective_rank < minimum_effective_rank
        )
        return EvalObservation(
            status="failed" if concentrated else "passed",
            value={
                "explained_variance_ratio": ratios,
                "cumulative_explained_variance": cumulative,
                "effective_rank": effective_rank,
            },
            output_kind="findings" if concentrated else "metric",
            details={
                "columns": selected,
                "rows": matrix.shape[0],
                "standardize": standardize,
                "eigenvalues": eigenvalues,
                "effective_rank": effective_rank,
                "execution_mode": "matrix_backend",
                "backend": "scikit-learn",
                "estimator": "sklearn.decomposition.PCA",
                "svd_solver": "full",
                "random_state": None,
                "matrix_preparation": ("polars_select_cast_drop_nulls_filter_finite"),
                "max_first_component": max_first_component,
                "minimum_effective_rank": minimum_effective_rank,
                "reason_code": "STRUCTURE_CONCENTRATED" if concentrated else None,
                "interpretation": (
                    "structure diagnostic; concentration is not an alpha verdict"
                ),
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="matrix_backend",
    )


def autocorrelation(
    column: str,
    *,
    lag: int = 1,
    name: str = "library.structure.autocorrelation",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a time-order serial-correlation diagnostic."""

    if lag < 1:
        raise ValueError("autocorrelation lag must be positive")

    def measure(table: pl.DataFrame) -> EvalObservation:
        if column not in table.columns:
            return not_applicable(
                "autocorrelation column is missing", missing_columns=[column]
            )
        values = _complete_numeric_frame(table, (column,))
        if values.height <= lag:
            return not_applicable(
                "fewer than the required rows for autocorrelation",
                rows=values.height,
                lag=lag,
            )
        pairs = values.select(
            pl.col(column).alias("__current"),
            pl.col(column).shift(lag).alias("__lag"),
        ).drop_nulls()
        stats = pairs.select(
            pl.col("__current").n_unique().alias("current_unique"),
            pl.col("__lag").n_unique().alias("lag_unique"),
            pl.corr("__current", "__lag").alias("correlation"),
        ).row(0, named=True)
        value = stats["correlation"]
        if (
            stats["current_unique"] < 2
            or stats["lag_unique"] < 2
            or value is None
            or not math.isfinite(float(value))
        ):
            return not_applicable(
                "autocorrelation is undefined", rows=values.height, lag=lag
            )
        return EvalObservation(
            status="passed",
            value=float(value),
            output_kind="metric",
            details={"column": column, "lag": lag, "rows": values.height},
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_select",
    )


def _complete_numeric_frame(
    table: pl.DataFrame, columns: tuple[str, ...]
) -> pl.DataFrame:
    """Select finite complete numeric rows with Polars."""

    selected = table.select(
        pl.col(column).cast(pl.Float64, strict=False).alias(column)
        for column in columns
    )
    return selected.drop_nulls().filter(
        pl.all_horizontal(pl.col(column).is_finite() for column in columns)
    )


def _complete_matrix(table: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Select finite complete rows in Polars before crossing into NumPy."""

    return _complete_numeric_frame(table, columns).to_numpy()


__all__ = [
    "METHODS",
    "autocorrelation",
    "pairwise_correlation",
    "pca_diagnostics",
    "vif",
]
