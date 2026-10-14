"""General structural evals for feature and signal panels."""

from __future__ import annotations

import polars as pl

from ..protocol import EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo

METHODS = (
    MethodInfo(
        name="library.structure.pairwise_correlation",
        family="structure",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description="Measure dependence between two explicitly selected columns.",
    ),
)


def pairwise_correlation(
    left: str,
    right: str,
    *,
    name: str = "library.structure.pairwise_correlation",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a structural correlation metric for two panel columns."""

    if not left.strip() or not right.strip():
        raise ValueError("structure evals need two named columns")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = [column for column in (left, right) if column not in table.columns]
        if missing:
            return EvalObservation(
                status="not_applicable",
                output_kind="none",
                details={"missing_columns": missing},
            )
        clean = table.select(left, right).drop_nulls()
        if clean.height < 2:
            return EvalObservation(
                status="not_applicable",
                output_kind="none",
                details={"reason": "fewer than two complete rows"},
            )
        value = clean.select(pl.corr(left, right)).item()
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
            details={"left": left, "right": right, "rows": clean.height},
        )

    return EvalSpec(name=name, runner=measure, failure_policy=failure_policy)


__all__ = ["METHODS", "pairwise_correlation"]
