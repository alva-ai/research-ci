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
from ._utils import column_names, missing_columns, not_applicable

METHODS = (
    MethodInfo(
        name="library.quality.required_columns",
        family="quality",
        stages=("data", "feat", "sig", "algo"),
        input_kind="main_table",
        purpose="quality",
        input_roles=("table",),
        description=(
            "Check that caller-declared columns exist and the table is non-empty."
        ),
    ),
    MethodInfo(
        name="library.quality.finite_columns",
        family="quality",
        stages=("data", "feat", "sig", "algo"),
        input_kind="main_table",
        purpose="quality",
        input_roles=("table",),
        description="Report non-finite values in caller-declared numeric columns.",
    ),
    MethodInfo(
        name="library.quality.coverage",
        family="quality",
        stages=("data", "feat", "sig", "algo"),
        input_kind="main_table",
        purpose="quality",
        input_roles=("table",),
        description="Report usable finite-value coverage for declared columns.",
    ),
    MethodInfo(
        name="library.quality.dispersion",
        family="quality",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        purpose="quality",
        input_roles=("features", "signals"),
        description="Report variation and quantile range for feature columns.",
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

    def check(table: pl.DataFrame) -> EvalObservation | bool:
        missing = missing_columns(table, required)
        if missing:
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={
                    "missing_columns": missing,
                    "reason_code": "MISSING_COLUMNS",
                },
            )
        if table.height == 0:
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={"reason": "table is empty", "reason_code": "EMPTY_TABLE"},
            )
        return True

    return EvalSpec(
        name=name,
        runner=check,
        failure_policy=failure_policy,
        execution_mode="polars_select",
        library_method="library.quality.required_columns",
    )


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
                details={
                    "missing_columns": missing,
                    "reason_code": "MISSING_COLUMNS",
                },
            )

        non_numeric = [
            column for column in selected if not table.schema[column].is_numeric()
        ]
        if non_numeric:
            return EvalObservation(
                status="not_applicable",
                output_kind="none",
                details={"non_numeric_columns": non_numeric},
            )

        aliases = {
            column: f"__rci_non_finite_{index}" for index, column in enumerate(selected)
        }
        aggregate = table.select(
            (~pl.col(column).is_finite()).sum().alias(aliases[column])
            for column in selected
        ).row(0, named=True)
        findings = {column: int(aggregate[aliases[column]]) for column in selected}
        findings = {column: count for column, count in findings.items() if count}
        details = {"columns": selected, "rows": table.height}
        return EvalObservation(
            status="failed" if findings else "passed",
            value=findings,
            output_kind="findings" if findings else "summary",
            details=details,
        )

    return EvalSpec(
        name=name,
        runner=check,
        failure_policy=failure_policy,
        execution_mode="polars_select",
        library_method="library.quality.finite_columns",
    )


def coverage(
    columns: Iterable[str],
    *,
    minimum: float | None = None,
    name: str = "library.quality.coverage",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a usable finite-value coverage diagnostic.

    ``minimum`` is optional by design.  Without it the method reports evidence
    and leaves the research decision to the case plan; with it, the case has
    explicitly chosen a completeness requirement.
    """

    selected = column_names(columns)
    if minimum is not None and not 0 <= minimum <= 1:
        raise ValueError("coverage minimum must be between zero and one")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, selected)
        if missing:
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={"missing_columns": missing, "reason_code": "MISSING_COLUMNS"},
            )
        rows = table.height
        aliases = {
            column: f"__rci_usable_{index}" for index, column in enumerate(selected)
        }
        aggregate = table.select(
            _numeric_expr(table, column)
            .is_finite()
            .fill_null(False)
            .sum()
            .alias(aliases[column])
            for column in selected
        ).row(0, named=True)
        counts = {column: int(aggregate[aliases[column]]) for column in selected}
        values = {
            column: usable / rows if rows else 0.0 for column, usable in counts.items()
        }
        failures = {
            column: value
            for column, value in values.items()
            if minimum is not None and value < minimum
        }
        if failures:
            status = "failed"
            details = {
                "reason_code": "LOW_COVERAGE",
                "minimum": minimum,
                "below_minimum": failures,
            }
        else:
            status = "passed"
            details = {"minimum": minimum}
        details.update({"rows": rows, "usable_rows": counts, "coverage": values})
        return EvalObservation(
            status=status,
            value=values,
            output_kind="findings" if failures else "metric",
            details=details,
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_select",
        library_method="library.quality.coverage",
    )


def dispersion(
    columns: Iterable[str],
    *,
    name: str = "library.quality.dispersion",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a distribution screen for constants and near-constants."""

    selected = column_names(columns)

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, selected)
        if missing:
            return not_applicable(
                "dispersion columns are missing", missing_columns=missing
            )
        aliases: dict[str, dict[str, str]] = {}
        expressions: list[pl.Expr] = []
        for index, column in enumerate(selected):
            values = _numeric_expr(table, column)
            finite_values = values.filter(values.is_finite())
            column_aliases = {
                metric: f"__rci_{index}_{metric}"
                for metric in ("rows", "unique", "std", "min", "max")
            }
            aliases[column] = column_aliases
            expressions.extend(
                (
                    finite_values.count().alias(column_aliases["rows"]),
                    finite_values.n_unique().alias(column_aliases["unique"]),
                    finite_values.std(ddof=1).alias(column_aliases["std"]),
                    finite_values.min().alias(column_aliases["min"]),
                    finite_values.max().alias(column_aliases["max"]),
                )
            )

        aggregate = table.select(expressions).row(0, named=True)
        summaries: dict[str, dict[str, float | int | None]] = {}
        constants: list[str] = []
        for column in selected:
            column_aliases = aliases[column]
            usable_rows = int(aggregate[column_aliases["rows"]])
            unique = int(aggregate[column_aliases["unique"]])
            if usable_rows == 0:
                summaries[column] = {"rows": 0, "unique": 0, "std": None}
                continue
            # The previous implementation defined the sample standard
            # deviation of a single usable value as zero. Polars correctly
            # returns null for ddof=1, so preserve the public result here.
            raw_std = aggregate[column_aliases["std"]]
            std = 0.0 if usable_rows == 1 else float(raw_std)
            summaries[column] = {
                "rows": usable_rows,
                "unique": unique,
                "std": std,
                "min": float(aggregate[column_aliases["min"]]),
                "max": float(aggregate[column_aliases["max"]]),
            }
            if unique < 2 or std == 0:
                constants.append(column)
        return EvalObservation(
            status="passed",
            value=summaries,
            output_kind="findings" if constants else "metric",
            details={
                "columns": selected,
                "constant_columns": constants,
                "reason_code": "NO_VARIATION" if constants else None,
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_select",
        library_method="library.quality.dispersion",
    )


def _column_names(columns: Iterable[str]) -> tuple[str, ...]:
    return column_names(columns)


def _numeric_expr(table: pl.DataFrame, column: str) -> pl.Expr:
    """Return a columnar equivalent of ``float(value)`` with null on failure."""

    dtype = table.schema[column]
    value = pl.col(column)
    if dtype.is_numeric() or dtype == pl.Boolean:
        return value.cast(pl.Float64, strict=False)
    return (
        value.cast(pl.String, strict=False)
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
    )


__all__ = [
    "METHODS",
    "coverage",
    "dispersion",
    "finite_columns",
    "required_columns",
]
