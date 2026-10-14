"""General data and table-quality evals.

These methods report observations over caller-selected columns. They do not
mutate the main table and do not decide which columns a case must contain.
That decision remains in the Contract or the Lab's case plan.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from ..protocol import EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo

METHODS = (
    MethodInfo(
        name="library.quality.required_columns",
        family="quality",
        stages=("data", "feat", "sig", "algo"),
        input_kind="main_table",
        description=(
            "Check that caller-declared columns exist and the table is non-empty."
        ),
    ),
    MethodInfo(
        name="library.quality.finite_columns",
        family="quality",
        stages=("data", "feat", "sig", "algo"),
        input_kind="main_table",
        description="Report non-finite values in caller-declared numeric columns.",
    ),
)


def required_columns(
    columns: Iterable[str],
    *,
    name: str = "library.quality.required_columns",
    failure_policy: FailurePolicy = "stop",
) -> EvalSpec:
    """Build a general check for a declared set of required columns."""

    required = _column_names(columns)

    def check(table: pl.DataFrame) -> bool:
        return table.height > 0 and all(column in table.columns for column in required)

    return EvalSpec(name=name, runner=check, failure_policy=failure_policy)


def finite_columns(
    columns: Iterable[str],
    *,
    name: str = "library.quality.finite_columns",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a general check for finite values in numeric columns."""

    selected = _column_names(columns)

    def check(table: pl.DataFrame) -> EvalObservation:
        missing = [column for column in selected if column not in table.columns]
        if missing:
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={"missing_columns": missing},
            )

        non_numeric = [
            column
            for column in selected
            if not table.schema[column].is_numeric()
        ]
        if non_numeric:
            return EvalObservation(
                status="not_applicable",
                output_kind="none",
                details={"non_numeric_columns": non_numeric},
            )

        findings = {
            column: int((~table.get_column(column).is_finite()).sum())
            for column in selected
        }
        findings = {column: count for column, count in findings.items() if count}
        return EvalObservation(
            status="failed" if findings else "passed",
            value=findings,
            output_kind="findings" if findings else "summary",
            details={"columns": selected, "rows": table.height},
        )

    return EvalSpec(name=name, runner=check, failure_policy=failure_policy)


def _column_names(columns: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(columns))
    if not selected or any(not column.strip() for column in selected):
        raise ValueError("library quality evals need at least one named column")
    return selected


__all__ = ["METHODS", "finite_columns", "required_columns"]
