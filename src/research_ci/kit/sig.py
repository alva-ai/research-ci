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
        "kit.sig.SignalBundle",
        "sig",
        "bundle",
        "return final signals with candidates, dispositions, lineage, and sidecars",
    ),
)

__all__ = [
    "FEATURE_ACTIONS",
    "TOOLS",
    "FeatureAction",
    "FeatureDisposition",
    "SignalBundle",
    "SignalLineage",
    "SignalRole",
    "TimeSplit",
    "clip",
    "direction",
    "forward_difference",
    "forward_log_return",
    "forward_return",
    "mean_signal",
    "split_by_time",
    "weighted_sum",
]
