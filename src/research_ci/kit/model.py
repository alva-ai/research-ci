"""Explicit sklearn model adapters for external Sig builders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import ToolInfo

ModelTask = Literal["regression", "classification"]
ModelFamily = Literal[
    "ridge",
    "logistic",
    "hist_gradient_boosting_regressor",
    "hist_gradient_boosting_classifier",
]
Scaling = Literal["none", "standard"]
Imputation = Literal["none", "mean", "median", "most_frequent", "constant"]
ScoreMethod = Literal["predict", "predict_proba", "decision_function"]

_FAMILY_TASK: dict[ModelFamily, ModelTask] = {
    "ridge": "regression",
    "logistic": "classification",
    "hist_gradient_boosting_regressor": "regression",
    "hist_gradient_boosting_classifier": "classification",
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Declare every preprocessing, estimator, and scoring choice."""

    task: ModelTask
    family: ModelFamily
    feature_columns: tuple[str, ...]
    target_column: str
    scaling: Scaling
    imputation: Imputation
    seed: int
    score_method: ScoreMethod
    class_label: Any | None = None
    imputation_fill_value: float | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in {"regression", "classification"}:
            raise ValueError("ModelSpec.task must be regression or classification")
        if self.family not in _FAMILY_TASK:
            raise ValueError(f"unknown ModelSpec.family: {self.family!r}")
        if _FAMILY_TASK[self.family] != self.task:
            raise ValueError(
                f"model family {self.family!r} does not support task {self.task!r}"
            )
        features = tuple(self.feature_columns)
        if not features or any(not column.strip() for column in features):
            raise ValueError("ModelSpec.feature_columns needs named columns")
        if len(features) != len(set(features)):
            raise ValueError("ModelSpec.feature_columns must not contain duplicates")
        if not self.target_column.strip():
            raise ValueError("ModelSpec.target_column must not be empty")
        if self.target_column in features:
            raise ValueError("ModelSpec.target_column must not be a feature")
        if self.scaling not in {"none", "standard"}:
            raise ValueError("ModelSpec.scaling must be none or standard")
        if self.imputation not in {
            "none",
            "mean",
            "median",
            "most_frequent",
            "constant",
        }:
            raise ValueError("unknown ModelSpec.imputation strategy")
        if self.imputation == "constant":
            if self.imputation_fill_value is None:
                raise ValueError("constant imputation requires imputation_fill_value")
        elif self.imputation_fill_value is not None:
            raise ValueError(
                "imputation_fill_value is only valid for constant imputation"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("ModelSpec.seed must be an integer")
        if self.score_method not in {
            "predict",
            "predict_proba",
            "decision_function",
        }:
            raise ValueError("unknown ModelSpec.score_method")
        if self.task == "regression":
            if self.score_method != "predict":
                raise ValueError("regression models must use predict scoring")
            if self.class_label is not None:
                raise ValueError("regression models cannot declare class_label")
        elif self.score_method == "predict":
            if self.class_label is not None:
                raise ValueError("predict scoring does not use class_label")
        elif self.class_label is None:
            raise ValueError(
                "classification probability or decision scoring requires class_label"
            )
        if self.score_method == "decision_function" and self.family != "logistic":
            raise ValueError(
                "decision_function scoring is supported only by logistic models"
            )

        parameters = dict(self.parameters)
        if any(not isinstance(name, str) or not name.strip() for name in parameters):
            raise ValueError("ModelSpec.parameters must use non-empty string keys")
        if "random_state" in parameters:
            raise ValueError("use ModelSpec.seed instead of parameters['random_state']")
        object.__setattr__(self, "feature_columns", features)
        object.__setattr__(self, "parameters", parameters)

    def as_dict(self) -> dict[str, Any]:
        """Return the complete recipe for lineage or artifact metadata."""

        return {
            "task": self.task,
            "family": self.family,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "scaling": self.scaling,
            "imputation": self.imputation,
            "imputation_fill_value": self.imputation_fill_value,
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "score_method": self.score_method,
            "class_label": self.class_label,
        }


@dataclass(frozen=True, slots=True)
class FittedModel:
    """Keep a fitted pipeline together with its complete immutable recipe."""

    spec: ModelSpec
    estimator: Pipeline
    training_rows: int
    classes: tuple[Any, ...] = ()


def make_model(spec: ModelSpec) -> Pipeline:
    """Build an unfitted sklearn pipeline exactly from ``spec``."""

    parameters = dict(spec.parameters)
    parameters["random_state"] = spec.seed
    if spec.family == "ridge":
        estimator = Ridge(**parameters)
    elif spec.family == "logistic":
        estimator = LogisticRegression(**parameters)
    elif spec.family == "hist_gradient_boosting_regressor":
        estimator = HistGradientBoostingRegressor(**parameters)
    else:
        estimator = HistGradientBoostingClassifier(**parameters)

    steps: list[tuple[str, Any]] = []
    if spec.imputation != "none":
        options: dict[str, Any] = {
            "strategy": spec.imputation,
            "keep_empty_features": True,
        }
        if spec.imputation == "constant":
            options["fill_value"] = spec.imputation_fill_value
        steps.append(("imputer", SimpleImputer(**options)))
    if spec.scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _require_columns(table: pl.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"model table is missing columns: {missing}")


def _feature_matrix(table: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    _require_columns(table, columns)
    non_numeric = [
        column for column in columns if not table.schema[column].is_numeric()
    ]
    if non_numeric:
        raise TypeError(f"model features must be numeric: {non_numeric}")
    matrix = table.select(
        pl.col(column).cast(pl.Float64) for column in columns
    ).to_numpy()
    if np.isinf(matrix).any():
        raise ValueError(
            "model features must not contain positive or negative infinity"
        )
    return matrix


def _target_array(table: pl.DataFrame, spec: ModelSpec) -> np.ndarray:
    _require_columns(table, (spec.target_column,))
    target = table.get_column(spec.target_column)
    if target.null_count():
        raise ValueError("model target must not contain null values")
    values = target.to_numpy()
    if target.dtype.is_numeric():
        numeric = np.asarray(values, dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("model target must contain only finite values")
        if spec.task == "regression":
            return numeric
    elif spec.task == "regression":
        raise TypeError("regression target must be numeric")
    return values


def fit_model(spec: ModelSpec, table: pl.DataFrame) -> FittedModel:
    """Fit preprocessing and estimator state using only the supplied table."""

    if table.is_empty():
        raise ValueError("model training table must not be empty")
    features = _feature_matrix(table, spec.feature_columns)
    if spec.imputation not in {"none", "constant"}:
        empty = np.isnan(features).all(axis=0)
        if empty.any():
            columns = [
                column
                for column, is_empty in zip(spec.feature_columns, empty)
                if is_empty
            ]
            raise ValueError(
                "non-constant imputation needs an observed value for features: "
                f"{columns}"
            )
    target = _target_array(table, spec)
    if spec.task == "classification" and table[spec.target_column].n_unique() < 2:
        raise ValueError("classification training target needs at least two classes")
    estimator = make_model(spec)
    estimator.fit(features, target)

    model = estimator.named_steps["model"]
    classes = tuple(getattr(model, "classes_", ()))
    if spec.class_label is not None and spec.class_label not in classes:
        raise ValueError(
            f"class_label {spec.class_label!r} was not observed in training data"
        )
    return FittedModel(
        spec=spec,
        estimator=estimator,
        training_rows=table.height,
        classes=classes,
    )


def score_model(
    fitted: FittedModel,
    table: pl.DataFrame,
    *,
    output_column: str,
) -> pl.DataFrame:
    """Append scores without refitting any preprocessing or estimator state."""

    if not output_column.strip():
        raise ValueError("output_column must not be empty")
    if output_column in table.columns:
        raise ValueError(f"output_column already exists: {output_column!r}")
    if table.is_empty():
        raise ValueError("model scoring table must not be empty")

    features = _feature_matrix(table, fitted.spec.feature_columns)
    method = fitted.spec.score_method
    if method == "predict":
        scores = fitted.estimator.predict(features)
    elif method == "predict_proba":
        class_index = fitted.classes.index(fitted.spec.class_label)
        scores = fitted.estimator.predict_proba(features)[:, class_index]
    else:
        raw = np.asarray(fitted.estimator.decision_function(features))
        class_index = fitted.classes.index(fitted.spec.class_label)
        if raw.ndim == 1:
            if len(fitted.classes) != 2:
                raise ValueError("one-dimensional decision scores require two classes")
            scores = raw if class_index == 1 else -raw
        else:
            scores = raw[:, class_index]
    return table.with_columns(pl.Series(output_column, np.asarray(scores)))


TOOLS = (
    ToolInfo(
        "kit.model.ModelSpec",
        "sig",
        "model",
        "declare explicit preprocessing, estimator, and scoring choices",
    ),
    ToolInfo(
        "kit.model.make_model",
        "sig",
        "model",
        "construct an unfitted sklearn pipeline from a complete model spec",
    ),
    ToolInfo(
        "kit.model.fit_model",
        "sig",
        "model",
        "fit one model pipeline on an explicitly supplied training table",
    ),
    ToolInfo(
        "kit.model.score_model",
        "sig",
        "model",
        "score a table without refitting preprocessing or estimator state",
    ),
)

__all__ = [
    "TOOLS",
    "FittedModel",
    "Imputation",
    "ModelFamily",
    "ModelSpec",
    "ModelTask",
    "Scaling",
    "ScoreMethod",
    "fit_model",
    "make_model",
    "score_model",
]
