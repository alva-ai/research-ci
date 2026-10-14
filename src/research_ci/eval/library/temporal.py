"""General key-grid and temporal-alignment diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta

import polars as pl

from ..protocol import EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo
from ._utils import column_names, missing_columns

METHODS = (
    MethodInfo(
        name="library.temporal.key_grid",
        family="temporal",
        stages=("data",),
        input_kind="main_table",
        description=(
            "Check key uniqueness, ordering, and an optional regular time grid."
        ),
    ),
)


def key_grid(
    keys: Iterable[str],
    *,
    time_key: str | None = None,
    frequency: timedelta | None = None,
    name: str = "library.temporal.key_grid",
    failure_policy: FailurePolicy = "stop",
) -> EvalSpec:
    """Build a columnar key-grid check for a declared stage table."""

    selected = column_names(keys)
    if time_key is not None and time_key not in selected:
        raise ValueError("key_grid time_key must be one of keys")
    if frequency is not None and time_key is None:
        raise ValueError("key_grid frequency requires time_key")
    if frequency is not None and frequency <= timedelta(0):
        raise ValueError("key_grid frequency must be positive")

    def check(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, selected)
        if missing:
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={"missing_columns": missing, "reason_code": "MISSING_COLUMNS"},
            )
        if table.is_empty():
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={"reason_code": "EMPTY_TABLE"},
            )

        keys_table = table.select(selected)
        duplicate_rows = keys_table.height - keys_table.unique().height
        sorted_keys = keys_table.equals(keys_table.sort(list(selected)))
        irregular_gaps = 0
        if time_key is not None and frequency is not None:
            entity_keys = tuple(column for column in selected if column != time_key)
            ordered = table.select(selected).sort([*entity_keys, time_key])
            if entity_keys:
                gap = pl.col(time_key).diff().over(*entity_keys)
            else:
                gap = pl.col(time_key).diff()
            irregular_gaps = int(
                ordered.select(
                    gap.drop_nulls().ne(frequency).sum().alias("irregular_gaps")
                ).item()
            )

        reasons: list[str] = []
        if duplicate_rows:
            reasons.append("DUPLICATE_KEYS")
        if not sorted_keys:
            reasons.append("UNSORTED_KEYS")
        if irregular_gaps:
            reasons.append("IRREGULAR_TIME_GRID")
        details = {
            "keys": selected,
            "rows": table.height,
            "duplicate_key_rows": duplicate_rows,
            "sorted": sorted_keys,
            "time_key": time_key,
            "frequency_seconds": frequency.total_seconds() if frequency else None,
            "irregular_gaps": irregular_gaps,
            "reason_codes": reasons,
        }
        return EvalObservation(
            status="failed" if reasons else "passed",
            value=not reasons,
            output_kind="findings" if reasons else "summary",
            details=details,
        )

    return EvalSpec(
        name=name,
        runner=check,
        failure_policy=failure_policy,
        output_kind="summary",
        execution_mode="polars_select",
    )


__all__ = ["METHODS", "key_grid"]
