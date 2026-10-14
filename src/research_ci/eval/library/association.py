"""General score/target association evals for Feat and Sig tables."""

from __future__ import annotations

import polars as pl

from ..protocol import EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo

METHODS = (
    MethodInfo(
        name="library.association.pearson_ic",
        family="association",
        stages=("feat", "sig"),
        input_kind="evaluation_table",
        description=(
            "Measure Pearson score/target association when a target is available."
        ),
    ),
    MethodInfo(
        name="library.association.spearman_ic",
        family="association",
        stages=("feat", "sig"),
        input_kind="evaluation_table",
        description=(
            "Measure rank based score/target association when a target is available."
        ),
    ),
)


def pearson_ic(
    score: str,
    target: str,
    *,
    name: str = "library.association.pearson_ic",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a Pearson association metric over explicit score and target columns."""

    return _correlation_spec(
        score,
        target,
        method="pearson",
        name=name,
        failure_policy=failure_policy,
    )


def spearman_ic(
    score: str,
    target: str,
    *,
    name: str = "library.association.spearman_ic",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a rank based association metric over explicit columns."""

    return _correlation_spec(
        score,
        target,
        method="spearman",
        name=name,
        failure_policy=failure_policy,
    )


def _correlation_spec(
    score: str,
    target: str,
    *,
    method: str,
    name: str,
    failure_policy: FailurePolicy,
) -> EvalSpec:
    if not score.strip() or not target.strip():
        raise ValueError("association evals need named score and target columns")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = [column for column in (score, target) if column not in table.columns]
        if missing:
            return EvalObservation(
                status="not_applicable",
                output_kind="none",
                details={"missing_columns": missing},
            )

        clean = table.select(score, target).drop_nulls()
        if clean.height < 2:
            return EvalObservation(
                status="not_applicable",
                output_kind="none",
                details={"reason": "fewer than two complete rows"},
            )
        if (
            clean.get_column(score).n_unique() < 2
            or clean.get_column(target).n_unique() < 2
        ):
            return EvalObservation(
                status="not_applicable",
                output_kind="none",
                details={"reason": "score or target has no variation"},
            )

        value = clean.select(pl.corr(score, target, method=method)).item()
        if value is None:
            return EvalObservation(
                status="not_applicable",
                output_kind="none",
                details={"reason": "correlation is undefined"},
            )
        return EvalObservation(
            status="passed",
            value=float(value),
            output_kind="metric",
            details={"score": score, "target": target, "rows": clean.height},
        )

    return EvalSpec(name=name, runner=measure, failure_policy=failure_policy)


__all__ = ["METHODS", "pearson_ic", "spearman_ic"]
