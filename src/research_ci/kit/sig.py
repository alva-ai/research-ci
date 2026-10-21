"""Small construction helpers for external Sig builders."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import reduce
from operator import add
from typing import Any, Literal

import polars as pl

from .common import ToolInfo

FeatureAction = Literal[
    "keep_direct",
    "keep_as_context",
    "representative_of_cluster",
    "input_to_component",
    "reject_quality",
    "reject_redundant",
    "reject_unstable",
    "reject_weak",
]
SignalRole = Literal["candidate", "final"]
QuantileInterpolation = Literal[
    "nearest", "higher", "lower", "midpoint", "linear", "equiprobable"
]

FEATURE_ACTIONS = frozenset(
    {
        "keep_direct",
        "keep_as_context",
        "representative_of_cluster",
        "input_to_component",
        "reject_quality",
        "reject_redundant",
        "reject_unstable",
        "reject_weak",
    }
)


def _positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def forward_return(price: pl.Expr, periods: int = 1) -> pl.Expr:
    """Build a future simple return for an explicit label sidecar."""

    periods = _positive(periods, "periods")
    return price.shift(-periods) / price - 1.0


def forward_log_return(price: pl.Expr, periods: int = 1) -> pl.Expr:
    """Build a future log return for an explicit label sidecar."""

    periods = _positive(periods, "periods")
    return price.shift(-periods).log() - price.log()


def forward_difference(value: pl.Expr, periods: int = 1) -> pl.Expr:
    """Build a future change for a level or already-log-transformed series."""

    periods = _positive(periods, "periods")
    return value.shift(-periods) - value


def direction(value: pl.Expr, *, threshold: float = 0.0) -> pl.Expr:
    """Map an expression to its signed direction around an explicit threshold."""

    return (value - threshold).sign()


def weighted_sum(values: Sequence[pl.Expr], weights: Sequence[float]) -> pl.Expr:
    """Combine expressions with explicit fixed weights."""

    if not values:
        raise ValueError("values must contain at least one expression")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError("weights must be finite")
    return reduce(add, (value * weight for value, weight in zip(values, weights)))


def mean_signal(
    values: Sequence[pl.Expr],
    *,
    ignore_nulls: bool = False,
) -> pl.Expr:
    """Combine expressions with equal weight and an explicit null policy."""

    if not values:
        raise ValueError("values must contain at least one expression")
    return pl.mean_horizontal(*values, ignore_nulls=ignore_nulls)


def clip(value: pl.Expr, *, lower: float, upper: float) -> pl.Expr:
    """Clip a signal using explicit lower and upper bounds."""

    if lower > upper:
        raise ValueError("lower must not be greater than upper")
    return value.clip(lower_bound=lower, upper_bound=upper)


@dataclass(frozen=True, slots=True)
class TimeSplit:
    """Non-overlapping chronological train, validation, and test tables."""

    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """Declare one contiguous train, calibration, and evaluation fold."""

    name: str
    train_start: Any
    train_end: Any
    calibration_end: Any
    evaluation_end: Any

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("WalkForwardWindow.name must not be empty")
        if not (
            self.train_start
            < self.train_end
            < self.calibration_end
            < self.evaluation_end
        ):
            raise ValueError(
                "walk-forward boundaries must satisfy "
                "train_start < train_end < calibration_end < evaluation_end"
            )


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """Hold the three non-overlapping tables for one declared fold."""

    window: WalkForwardWindow
    train: pl.DataFrame
    calibration: pl.DataFrame
    evaluation: pl.DataFrame


@dataclass(frozen=True, slots=True)
class ZScoreCalibration:
    """Store group statistics fitted on one explicit calibration table."""

    value_column: str
    group_keys: tuple[str, ...]
    ddof: int
    statistics: pl.DataFrame


@dataclass(frozen=True, slots=True)
class AbsoluteQuantileCalibration:
    """Store absolute-value thresholds fitted on one calibration table."""

    value_column: str
    group_keys: tuple[str, ...]
    quantile: float
    interpolation: QuantileInterpolation
    thresholds: pl.DataFrame


def _column_names(value: str | Sequence[str], name: str) -> tuple[str, ...]:
    columns = (value,) if isinstance(value, str) else tuple(value)
    if not columns or any(not column.strip() for column in columns):
        raise ValueError(f"{name} must contain at least one named column")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{name} must not contain duplicates")
    return columns


def _require_columns(table: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"signal table is missing columns: {missing}")


def _temporary_name(columns: Sequence[str], stem: str) -> str:
    existing = set(columns)
    candidate = stem
    suffix = 1
    while candidate in existing:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    return candidate


def split_walk_forward(
    table: pl.DataFrame,
    *,
    time_key: str,
    windows: Sequence[WalkForwardWindow],
) -> tuple[WalkForwardSplit, ...]:
    """Materialize explicit half-open walk-forward folds without fitting state."""

    if not time_key.strip():
        raise ValueError("time_key must not be empty")
    _require_columns(table, (time_key,))
    declared = tuple(windows)
    if not declared:
        raise ValueError("windows must contain at least one WalkForwardWindow")
    if any(not isinstance(window, WalkForwardWindow) for window in declared):
        raise TypeError("windows must contain WalkForwardWindow values")
    names = [window.name for window in declared]
    if len(names) != len(set(names)):
        raise ValueError("walk-forward window names must be unique")

    time = pl.col(time_key)
    return tuple(
        WalkForwardSplit(
            window=window,
            train=table.filter(
                (time >= pl.lit(window.train_start)) & (time < pl.lit(window.train_end))
            ),
            calibration=table.filter(
                (time >= pl.lit(window.train_end))
                & (time < pl.lit(window.calibration_end))
            ),
            evaluation=table.filter(
                (time >= pl.lit(window.calibration_end))
                & (time < pl.lit(window.evaluation_end))
            ),
        )
        for window in declared
    )


def purge_group_tail(
    table: pl.DataFrame,
    *,
    group_keys: str | Sequence[str],
    order_keys: str | Sequence[str],
    rows: int,
) -> pl.DataFrame:
    """Remove the final ordered rows per group while preserving input order."""

    groups = _column_names(group_keys, "group_keys")
    order = _column_names(order_keys, "order_keys")
    if set(groups) & set(order):
        raise ValueError("group_keys and order_keys must not overlap")
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise TypeError("rows must be an integer")
    if rows < 0:
        raise ValueError("rows must not be negative")
    _require_columns(table, (*groups, *order))
    if rows == 0 or table.is_empty():
        return table.clone()
    if table.select(*groups, *order).is_duplicated().any():
        raise ValueError(
            "group_keys plus order_keys must uniquely order rows before purging"
        )

    input_order = _temporary_name(table.columns, "__kit_input_order")
    within_group = _temporary_name((*table.columns, input_order), "__kit_group_order")
    group_size = _temporary_name(
        (*table.columns, input_order, within_group), "__kit_group_size"
    )
    ordered = (
        table.with_row_index(input_order)
        .sort([*groups, *order])
        .with_columns(
            pl.int_range(pl.len()).over(list(groups)).alias(within_group),
            pl.len().over(list(groups)).cast(pl.Int64).alias(group_size),
        )
    )
    return (
        ordered.filter(pl.col(within_group) < pl.col(group_size) - rows)
        .sort(input_order)
        .drop(input_order, within_group, group_size)
    )


def _group_statistics(
    table: pl.DataFrame,
    group_keys: tuple[str, ...],
    expressions: Sequence[pl.Expr],
) -> pl.DataFrame:
    if group_keys:
        return table.group_by(*group_keys, maintain_order=True).agg(*expressions)
    return table.select(*expressions)


def fit_zscore_calibration(
    table: pl.DataFrame,
    *,
    value_column: str,
    group_keys: str | Sequence[str] | None = None,
    ddof: int = 0,
) -> ZScoreCalibration:
    """Fit means and scales only on the supplied calibration rows.

    Null observations follow Polars aggregate semantics and are excluded. A
    group with a null, non-finite, or non-positive fitted scale is rejected.
    """

    if not value_column.strip():
        raise ValueError("value_column must not be empty")
    groups = () if group_keys is None else _column_names(group_keys, "group_keys")
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError("ddof must be an integer")
    if ddof < 0:
        raise ValueError("ddof must not be negative")
    if table.is_empty():
        raise ValueError("z-score calibration table must not be empty")
    _require_columns(table, (*groups, value_column))

    statistics = _group_statistics(
        table,
        groups,
        (
            pl.col(value_column).mean().alias("mean"),
            pl.col(value_column).std(ddof=ddof).alias("scale"),
        ),
    )
    invalid = statistics.filter(
        pl.col("mean").is_null()
        | ~pl.col("mean").is_finite()
        | pl.col("scale").is_null()
        | ~pl.col("scale").is_finite()
        | (pl.col("scale") <= 0)
    )
    if not invalid.is_empty():
        raise ValueError("every z-score calibration group needs finite positive scale")
    return ZScoreCalibration(value_column, groups, ddof, statistics)


def _attach_fitted_columns(
    table: pl.DataFrame,
    fitted: pl.DataFrame,
    group_keys: tuple[str, ...],
    columns: Sequence[str],
) -> tuple[pl.DataFrame, dict[str, str]]:
    names = list(table.columns)
    renames: dict[str, str] = {}
    for column in columns:
        temporary = _temporary_name(names, f"__kit_{column}")
        renames[column] = temporary
        names.append(temporary)
    if group_keys:
        attached = table.join(
            fitted.rename(renames),
            on=list(group_keys),
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    else:
        if fitted.height != 1:
            raise ValueError(
                "global calibration must contain exactly one statistics row"
            )
        attached = table.with_columns(
            pl.lit(fitted.get_column(column)[0]).alias(temporary)
            for column, temporary in renames.items()
        )
    return attached, renames


def apply_zscore_calibration(
    table: pl.DataFrame,
    calibration: ZScoreCalibration,
    *,
    output_column: str,
) -> pl.DataFrame:
    """Apply stored calibration statistics without recomputing them."""

    if not output_column.strip():
        raise ValueError("output_column must not be empty")
    if output_column in table.columns:
        raise ValueError(f"output_column already exists: {output_column!r}")
    _require_columns(table, (*calibration.group_keys, calibration.value_column))
    attached, names = _attach_fitted_columns(
        table,
        calibration.statistics,
        calibration.group_keys,
        ("mean", "scale"),
    )
    return attached.with_columns(
        (
            (pl.col(calibration.value_column) - pl.col(names["mean"]))
            / pl.col(names["scale"])
        ).alias(output_column)
    ).drop(*names.values())


def fit_absolute_quantile_calibration(
    table: pl.DataFrame,
    *,
    value_column: str,
    quantile: float,
    group_keys: str | Sequence[str] | None = None,
    interpolation: QuantileInterpolation = "linear",
) -> AbsoluteQuantileCalibration:
    """Fit absolute-value quantile thresholds on calibration rows only."""

    if not value_column.strip():
        raise ValueError("value_column must not be empty")
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be finite and between zero and one")
    if interpolation not in {
        "nearest",
        "higher",
        "lower",
        "midpoint",
        "linear",
        "equiprobable",
    }:
        raise ValueError("unknown quantile interpolation")
    groups = () if group_keys is None else _column_names(group_keys, "group_keys")
    if table.is_empty():
        raise ValueError("quantile calibration table must not be empty")
    _require_columns(table, (*groups, value_column))

    thresholds = _group_statistics(
        table,
        groups,
        (
            pl.col(value_column)
            .abs()
            .quantile(quantile, interpolation=interpolation)
            .alias("threshold"),
        ),
    )
    invalid = thresholds.filter(
        pl.col("threshold").is_null()
        | ~pl.col("threshold").is_finite()
        | (pl.col("threshold") < 0)
    )
    if not invalid.is_empty():
        raise ValueError("every quantile calibration group needs a finite threshold")
    return AbsoluteQuantileCalibration(
        value_column,
        groups,
        quantile,
        interpolation,
        thresholds,
    )


def apply_absolute_quantile_calibration(
    table: pl.DataFrame,
    calibration: AbsoluteQuantileCalibration,
    *,
    output_column: str,
) -> pl.DataFrame:
    """Attach stored group thresholds without inspecting evaluation values."""

    if not output_column.strip():
        raise ValueError("output_column must not be empty")
    if output_column in table.columns:
        raise ValueError(f"output_column already exists: {output_column!r}")
    _require_columns(table, calibration.group_keys)
    attached, names = _attach_fitted_columns(
        table,
        calibration.thresholds,
        calibration.group_keys,
        ("threshold",),
    )
    return attached.rename({names["threshold"]: output_column})


@dataclass(frozen=True, slots=True)
class FeatureDisposition:
    """Record one explicit Sig-stage decision about one approved feature."""

    feature: str
    action: FeatureAction
    reason: str
    evidence: tuple[str, ...]
    selected_for: tuple[str, ...] = ()
    cluster: str | None = None

    def __post_init__(self) -> None:
        if not self.feature.strip():
            raise ValueError("FeatureDisposition.feature must not be empty")
        if self.action not in FEATURE_ACTIONS:
            raise ValueError(f"unknown FeatureDisposition.action: {self.action!r}")
        if not self.reason.strip():
            raise ValueError("FeatureDisposition.reason must not be empty")
        evidence = tuple(dict.fromkeys(self.evidence))
        selected_for = tuple(dict.fromkeys(self.selected_for))
        if not evidence or any(not item.strip() for item in evidence):
            raise ValueError("FeatureDisposition.evidence needs named evidence")
        if any(not item.strip() for item in selected_for):
            raise ValueError("FeatureDisposition.selected_for contains an empty name")
        if self.cluster is not None and not self.cluster.strip():
            raise ValueError("FeatureDisposition.cluster must not be empty")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "selected_for", selected_for)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "action": self.action,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "selected_for": list(self.selected_for),
            "cluster": self.cluster,
        }


@dataclass(frozen=True, slots=True)
class SignalLineage:
    """Describe how one candidate or final signal was constructed."""

    signal: str
    role: SignalRole
    method: str
    inputs: tuple[str, ...]
    fit_policy: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.signal.strip():
            raise ValueError("SignalLineage.signal must not be empty")
        if self.role not in {"candidate", "final"}:
            raise ValueError("SignalLineage.role must be candidate or final")
        if not self.method.strip():
            raise ValueError("SignalLineage.method must not be empty")
        inputs = tuple(dict.fromkeys(self.inputs))
        if not inputs or any(not item.strip() for item in inputs):
            raise ValueError("SignalLineage.inputs needs at least one named input")
        if not self.fit_policy.strip():
            raise ValueError("SignalLineage.fit_policy must not be empty")
        evidence = tuple(dict.fromkeys(self.evidence))
        if any(not item.strip() for item in evidence):
            raise ValueError("SignalLineage.evidence contains an empty name")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "evidence", evidence)

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "role": self.role,
            "method": self.method,
            "inputs": list(self.inputs),
            "fit_policy": self.fit_policy,
            "parameters": dict(self.parameters),
            "evidence": list(self.evidence),
        }


def split_by_time(
    table: pl.DataFrame,
    *,
    time_key: str,
    train_end: Any,
    validation_end: Any,
) -> TimeSplit:
    """Split with half-open boundaries: train, validation, then test."""

    if train_end >= validation_end:
        raise ValueError("train_end must be earlier than validation_end")
    time = pl.col(time_key)
    return TimeSplit(
        train=table.filter(time < pl.lit(train_end)),
        validation=table.filter(
            (time >= pl.lit(train_end)) & (time < pl.lit(validation_end))
        ),
        test=table.filter(time >= pl.lit(validation_end)),
    )


@dataclass(frozen=True, slots=True)
class SignalBundle:
    """Return final signals plus auditable Sig research products."""

    main: pl.DataFrame
    candidates: pl.DataFrame
    dispositions: tuple[FeatureDisposition, ...]
    lineage: tuple[SignalLineage, ...]
    sidecars: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    research: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.main, pl.DataFrame):
            raise TypeError("SignalBundle.main must be a Polars DataFrame")
        if not isinstance(self.candidates, pl.DataFrame):
            raise TypeError("SignalBundle.candidates must be a Polars DataFrame")
        dispositions = tuple(self.dispositions)
        lineage = tuple(self.lineage)
        if any(not isinstance(item, FeatureDisposition) for item in dispositions):
            raise TypeError(
                "SignalBundle.dispositions must contain FeatureDisposition values"
            )
        if any(not isinstance(item, SignalLineage) for item in lineage):
            raise TypeError("SignalBundle.lineage must contain SignalLineage values")
        disposition_features = [item.feature for item in dispositions]
        if len(disposition_features) != len(set(disposition_features)):
            raise ValueError("SignalBundle.dispositions contains duplicate features")
        lineage_signals = [item.signal for item in lineage]
        if len(lineage_signals) != len(set(lineage_signals)):
            raise ValueError("SignalBundle.lineage contains duplicate signals")
        sidecars = dict(self.sidecars)
        if any(not name.strip() for name in sidecars):
            raise ValueError("SignalBundle sidecar names must not be empty")
        if any(not isinstance(table, pl.DataFrame) for table in sidecars.values()):
            raise TypeError("SignalBundle sidecars must be Polars DataFrames")
        research = dict(self.research)
        if any(not name.strip() for name in research):
            raise ValueError("SignalBundle research table names must not be empty")
        if any(not isinstance(table, pl.DataFrame) for table in research.values()):
            raise TypeError("SignalBundle research values must be Polars DataFrames")
        object.__setattr__(self, "dispositions", dispositions)
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "sidecars", sidecars)
        object.__setattr__(self, "research", research)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "artifacts", dict(self.artifacts))

    def disposition_frame(self) -> pl.DataFrame:
        """Return the feature decisions as a stable, reviewable table."""

        return pl.DataFrame(
            [item.as_dict() for item in self.dispositions],
            schema={
                "feature": pl.String,
                "action": pl.String,
                "reason": pl.String,
                "evidence": pl.List(pl.String),
                "selected_for": pl.List(pl.String),
                "cluster": pl.String,
            },
        )

    def lineage_records(self) -> list[dict[str, Any]]:
        """Return JSON-compatible construction lineage in dependency order."""

        return [item.as_dict() for item in self.lineage]


TOOLS = (
    ToolInfo(
        "kit.sig.forward_return",
        "sig",
        "expr",
        "build an explicit future-return label sidecar column",
    ),
    ToolInfo(
        "kit.sig.forward_log_return",
        "sig",
        "expr",
        "build an explicit future log-return label sidecar column",
    ),
    ToolInfo(
        "kit.sig.forward_difference",
        "sig",
        "expr",
        "build an explicit future-change label sidecar column",
    ),
    ToolInfo(
        "kit.sig.weighted_sum",
        "sig",
        "expr",
        "combine aligned feature or model outputs with fixed weights",
    ),
    ToolInfo(
        "kit.sig.mean_signal",
        "sig",
        "expr",
        "combine aligned outputs with equal weight and explicit null handling",
    ),
    ToolInfo(
        "kit.sig.split_by_time",
        "sig",
        "split",
        "create non-overlapping chronological train, validation, and test tables",
    ),
    ToolInfo(
        "kit.sig.split_walk_forward",
        "sig",
        "split",
        "materialize declared train, calibration, and evaluation folds",
    ),
    ToolInfo(
        "kit.sig.purge_group_tail",
        "sig",
        "split",
        "purge the final explicitly ordered label-horizon rows per group",
    ),
    ToolInfo(
        "kit.sig.fit_zscore_calibration",
        "sig",
        "table",
        "fit reusable z-score statistics on an explicit calibration table",
    ),
    ToolInfo(
        "kit.sig.apply_zscore_calibration",
        "sig",
        "table",
        "apply stored z-score statistics without refitting",
    ),
    ToolInfo(
        "kit.sig.fit_absolute_quantile_calibration",
        "sig",
        "table",
        "fit absolute score thresholds on an explicit calibration table",
    ),
    ToolInfo(
        "kit.sig.apply_absolute_quantile_calibration",
        "sig",
        "table",
        "attach stored absolute score thresholds without refitting",
    ),
    ToolInfo(
        "kit.sig.SignalBundle",
        "sig",
        "bundle",
        "return final signals with candidates, dispositions, lineage, and sidecars",
    ),
)

__all__ = [
    "AbsoluteQuantileCalibration",
    "FEATURE_ACTIONS",
    "QuantileInterpolation",
    "TOOLS",
    "FeatureAction",
    "FeatureDisposition",
    "SignalBundle",
    "SignalLineage",
    "SignalRole",
    "TimeSplit",
    "WalkForwardSplit",
    "WalkForwardWindow",
    "ZScoreCalibration",
    "apply_absolute_quantile_calibration",
    "apply_zscore_calibration",
    "clip",
    "direction",
    "fit_absolute_quantile_calibration",
    "fit_zscore_calibration",
    "forward_difference",
    "forward_log_return",
    "forward_return",
    "mean_signal",
    "purge_group_tail",
    "split_by_time",
    "split_walk_forward",
    "weighted_sum",
]
