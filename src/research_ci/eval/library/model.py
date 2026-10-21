"""Diagnostics for already-produced out-of-sample model predictions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from ..protocol import EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo
from ._utils import missing_columns, not_applicable

METHODS = (
    MethodInfo(
        name="library.model.regression_diagnostics",
        family="model",
        stages=("sig",),
        input_kind="evaluation_table",
        purpose="predictive",
        input_roles=("signals", "target"),
        requires=("target", "horizon"),
        fit_policy="diagnostic_only",
        description=(
            "Measure error, fit, bias, and sign accuracy for submitted predictions."
        ),
    ),
    MethodInfo(
        name="library.model.classification_probability_diagnostics",
        family="model",
        stages=("sig",),
        input_kind="evaluation_table",
        purpose="predictive",
        input_roles=("signals", "target"),
        requires=("target", "horizon"),
        fit_policy="diagnostic_only",
        description=(
            "Measure discrimination, loss, accuracy, and calibration for submitted "
            "class probabilities."
        ),
    ),
)


def regression_diagnostics(
    prediction: str,
    target: str,
    *,
    min_rows: int = 2,
    maximum_mae: float | None = None,
    maximum_rmse: float | None = None,
    minimum_r2: float | None = None,
    minimum_sign_accuracy: float | None = None,
    name: str = "library.model.regression_diagnostics",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Evaluate submitted numeric predictions without fitting or selecting a model."""

    _named_pair(prediction, target)
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 2:
        raise ValueError("regression diagnostics min_rows must be at least 2")
    _nonnegative(maximum_mae, "maximum_mae")
    _nonnegative(maximum_rmse, "maximum_rmse")
    _finite(minimum_r2, "minimum_r2")
    _unit_interval(minimum_sign_accuracy, "minimum_sign_accuracy")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, (prediction, target))
        if missing:
            return not_applicable(
                "regression prediction or target column is missing",
                missing_columns=missing,
            )
        pairs = _complete_numeric_frame(table, prediction, target)
        if pairs.height < min_rows:
            return not_applicable(
                "fewer than the minimum regression rows",
                rows=pairs.height,
                min_rows=min_rows,
                score=prediction,
                target=target,
            )

        predicted = pairs.get_column(prediction).to_numpy()
        observed = pairs.get_column(target).to_numpy()
        errors = predicted - observed
        metrics = {
            "mae": float(mean_absolute_error(observed, predicted)),
            "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
            "mean_error": float(np.mean(errors)),
            "r2": float(r2_score(observed, predicted, force_finite=True)),
            "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
        }
        failures: list[dict[str, object]] = []
        _maximum_failure(failures, "mae", metrics["mae"], maximum_mae)
        _maximum_failure(failures, "rmse", metrics["rmse"], maximum_rmse)
        _minimum_failure(failures, "r2", metrics["r2"], minimum_r2)
        _minimum_failure(
            failures,
            "sign_accuracy",
            metrics["sign_accuracy"],
            minimum_sign_accuracy,
        )
        failed = bool(failures)
        return EvalObservation(
            status="failed" if failed else "passed",
            value=metrics,
            output_kind="findings" if failed else "metric",
            details={
                "score": prediction,
                "prediction": prediction,
                "target": target,
                "rows": pairs.height,
                "metrics": metrics,
                "thresholds": {
                    "maximum_mae": maximum_mae,
                    "maximum_rmse": maximum_rmse,
                    "minimum_r2": minimum_r2,
                    "minimum_sign_accuracy": minimum_sign_accuracy,
                },
                "threshold_failures": failures,
                "reason_code": (
                    "MODEL_PERFORMANCE_BELOW_THRESHOLD" if failed else None
                ),
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="matrix_backend",
        library_method="library.model.regression_diagnostics",
    )


def classification_probability_diagnostics(
    probability: str,
    target: str,
    *,
    positive_label: Any,
    min_rows: int = 2,
    decision_threshold: float = 0.5,
    calibration_bins: int = 10,
    minimum_roc_auc: float | None = None,
    maximum_brier_score: float | None = None,
    maximum_log_loss: float | None = None,
    minimum_accuracy: float | None = None,
    maximum_expected_calibration_error: float | None = None,
    name: str = "library.model.classification_probability_diagnostics",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Evaluate submitted positive-class probabilities without fitting a model."""

    _named_pair(probability, target)
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 2:
        raise ValueError("classification diagnostics min_rows must be at least 2")
    if positive_label is None:
        raise ValueError("classification diagnostics require a non-null positive_label")
    _unit_interval(decision_threshold, "decision_threshold")
    if (
        isinstance(calibration_bins, bool)
        or not isinstance(calibration_bins, int)
        or calibration_bins < 2
    ):
        raise ValueError("classification diagnostics need at least two bins")
    _unit_interval(minimum_roc_auc, "minimum_roc_auc")
    _unit_interval(maximum_brier_score, "maximum_brier_score")
    _nonnegative(maximum_log_loss, "maximum_log_loss")
    _unit_interval(minimum_accuracy, "minimum_accuracy")
    _unit_interval(
        maximum_expected_calibration_error,
        "maximum_expected_calibration_error",
    )

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, (probability, target))
        if missing:
            return not_applicable(
                "classification probability or target column is missing",
                missing_columns=missing,
            )
        predicted, observed = _complete_probability_rows(
            table,
            probability,
            target,
            positive_label,
        )
        if predicted.size < min_rows:
            return not_applicable(
                "fewer than the minimum classification rows",
                rows=int(predicted.size),
                min_rows=min_rows,
                score=probability,
                target=target,
            )
        if np.any((predicted < 0) | (predicted > 1)):
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={
                    "score": probability,
                    "target": target,
                    "rows": int(predicted.size),
                    "reason_code": "INVALID_PROBABILITY",
                },
            )
        if np.unique(observed).size < 2:
            return not_applicable(
                "classification target has no class variation",
                rows=int(predicted.size),
                positive_label=positive_label,
                score=probability,
                target=target,
            )

        decisions = (predicted >= decision_threshold).astype(np.int8)
        calibration_error, calibration = _expected_calibration_error(
            predicted,
            observed,
            calibration_bins,
        )
        metrics = {
            "roc_auc": float(roc_auc_score(observed, predicted)),
            "brier_score": float(brier_score_loss(observed, predicted)),
            "log_loss": float(log_loss(observed, predicted, labels=[0, 1])),
            "accuracy": float(np.mean(decisions == observed)),
            "expected_calibration_error": calibration_error,
        }
        performance_failures: list[dict[str, object]] = []
        calibration_failures: list[dict[str, object]] = []
        _minimum_failure(
            performance_failures,
            "roc_auc",
            metrics["roc_auc"],
            minimum_roc_auc,
        )
        _maximum_failure(
            performance_failures,
            "brier_score",
            metrics["brier_score"],
            maximum_brier_score,
        )
        _maximum_failure(
            performance_failures,
            "log_loss",
            metrics["log_loss"],
            maximum_log_loss,
        )
        _minimum_failure(
            performance_failures,
            "accuracy",
            metrics["accuracy"],
            minimum_accuracy,
        )
        _maximum_failure(
            calibration_failures,
            "expected_calibration_error",
            metrics["expected_calibration_error"],
            maximum_expected_calibration_error,
        )
        failures = [*performance_failures, *calibration_failures]
        reasons = []
        if performance_failures:
            reasons.append("MODEL_PERFORMANCE_BELOW_THRESHOLD")
        if calibration_failures:
            reasons.append("MODEL_CALIBRATION_BELOW_THRESHOLD")
        return EvalObservation(
            status="failed" if failures else "passed",
            value=metrics,
            output_kind="findings" if failures else "metric",
            details={
                "score": probability,
                "probability": probability,
                "target": target,
                "positive_label": positive_label,
                "rows": int(predicted.size),
                "positive_rows": int(observed.sum()),
                "negative_rows": int(observed.size - observed.sum()),
                "decision_threshold": decision_threshold,
                "calibration_bins": calibration,
                "metrics": metrics,
                "thresholds": {
                    "minimum_roc_auc": minimum_roc_auc,
                    "maximum_brier_score": maximum_brier_score,
                    "maximum_log_loss": maximum_log_loss,
                    "minimum_accuracy": minimum_accuracy,
                    "maximum_expected_calibration_error": (
                        maximum_expected_calibration_error
                    ),
                },
                "threshold_failures": failures,
                "reason_codes": reasons,
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="matrix_backend",
        library_method="library.model.classification_probability_diagnostics",
    )


def _complete_numeric_frame(
    table: pl.DataFrame,
    prediction: str,
    target: str,
) -> pl.DataFrame:
    return (
        table.select(
            pl.col(prediction).cast(pl.Float64, strict=False).alias(prediction),
            pl.col(target).cast(pl.Float64, strict=False).alias(target),
        )
        .drop_nulls()
        .filter(pl.col(prediction).is_finite() & pl.col(target).is_finite())
    )


def _complete_probability_rows(
    table: pl.DataFrame,
    probability: str,
    target: str,
    positive_label: Any,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[float] = []
    labels: list[int] = []
    for raw_probability, raw_target in table.select(
        probability,
        target,
    ).iter_rows():
        if raw_probability is None or raw_target is None:
            continue
        try:
            value = float(raw_probability)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        probabilities.append(value)
        labels.append(int(raw_target == positive_label))
    return np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=np.int8)


def _expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int,
) -> tuple[float, list[dict[str, float | int]]]:
    indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows = probabilities.size
    error = 0.0
    summaries: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(probabilities[mask].mean())
        observed_rate = float(labels[mask].mean())
        gap = abs(confidence - observed_rate)
        error += gap * count / rows
        summaries.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "rows": count,
                "mean_probability": confidence,
                "observed_rate": observed_rate,
                "absolute_gap": gap,
            }
        )
    return float(error), summaries


def _named_pair(left: str, right: str) -> None:
    if not left.strip() or not right.strip():
        raise ValueError("model diagnostics need named prediction and target columns")
    if left == right:
        raise ValueError("model prediction and target columns must differ")


def _finite(value: float | None, label: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _nonnegative(value: float | None, label: str) -> None:
    _finite(value, label)
    if value is not None and value < 0:
        raise ValueError(f"{label} cannot be negative")


def _unit_interval(value: float | None, label: str) -> None:
    _finite(value, label)
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{label} must be between zero and one")


def _maximum_failure(
    failures: list[dict[str, object]],
    metric: str,
    value: float,
    maximum: float | None,
) -> None:
    if maximum is not None and value > maximum:
        failures.append({"metric": metric, "value": value, "maximum": maximum})


def _minimum_failure(
    failures: list[dict[str, object]],
    metric: str,
    value: float,
    minimum: float | None,
) -> None:
    if minimum is not None and value < minimum:
        failures.append({"metric": metric, "value": value, "minimum": minimum})


__all__ = [
    "METHODS",
    "classification_probability_diagnostics",
    "regression_diagnostics",
]
