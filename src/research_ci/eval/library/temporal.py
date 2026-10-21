"""General key-grid and temporal-alignment diagnostics."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
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
        purpose="temporal",
        input_roles=("keys",),
        requires=("time_key",),
        description=(
            "Check key uniqueness, ordering, and an optional regular time grid."
        ),
    ),
    MethodInfo(
        name="library.temporal.cross_section_completeness",
        family="temporal",
        stages=("data",),
        input_kind="main_table",
        purpose="temporal",
        input_roles=("table", "keys"),
        requires=("time_key",),
        description=(
            "Audit each declared time section against an explicit entity universe."
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
        library_method="library.temporal.key_grid",
    )


def cross_section_completeness(
    time_keys: Iterable[str],
    entity_key: str,
    expected_entities: Iterable[Hashable],
    required_columns: Iterable[str],
    *,
    minimum_complete_ratio: float | None = None,
    report_limit: int = 20,
    name: str = "library.temporal.cross_section_completeness",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Audit observed sections against an explicitly declared entity universe.

    Missing expected entities are evidence unless ``minimum_complete_ratio`` is
    supplied. Duplicate entities, unexpected entities, and null required values
    are contract violations regardless of that optional research threshold.
    """

    section_keys = column_names(time_keys)
    if not entity_key.strip():
        raise ValueError("cross-section entity_key must not be empty")
    if entity_key in section_keys:
        raise ValueError("cross-section entity_key must differ from time_keys")
    required = column_names(required_columns)
    if isinstance(expected_entities, (str, bytes)):
        raise TypeError("cross-section expected_entities must be an iterable of IDs")
    expected = tuple(expected_entities)
    if not expected:
        raise ValueError("cross-section expected_entities must not be empty")
    try:
        expected_set = set(expected)
    except TypeError as exc:
        raise TypeError("cross-section expected_entities must be hashable") from exc
    if len(expected_set) != len(expected):
        raise ValueError("cross-section expected_entities must be unique")
    if minimum_complete_ratio is not None and (
        not math.isfinite(minimum_complete_ratio)
        or not 0 <= minimum_complete_ratio <= 1
    ):
        raise ValueError(
            "cross-section minimum_complete_ratio must be between zero and one"
        )
    if (
        isinstance(report_limit, bool)
        or not isinstance(report_limit, int)
        or report_limit < 1
    ):
        raise ValueError("cross-section report_limit must be positive")

    selected = tuple(dict.fromkeys((*section_keys, entity_key, *required)))

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

        missing_required = pl.any_horizontal(
            [pl.col(column).is_null() for column in required]
        )
        sections = (
            table.select(selected)
            .with_columns(missing_required.alias("__missing_required"))
            .group_by(list(section_keys), maintain_order=True)
            .agg(
                pl.col(entity_key).alias("__entities"),
                pl.col("__missing_required").sum().alias("__missing_required_rows"),
            )
        )

        complete_sections = 0
        duplicate_entity_rows = 0
        unexpected_entity_rows = 0
        missing_required_rows = 0
        missing_entity_slots = 0
        unexpected_entities: list[Hashable | None] = []
        findings: list[dict[str, object]] = []

        for row in sections.iter_rows(named=True):
            entities = row["__entities"]
            counts: dict[Hashable | None, int] = {}
            for entity in entities:
                counts[entity] = counts.get(entity, 0) + 1
            observed = set(counts)
            missing_entities = [entity for entity in expected if entity not in observed]
            unexpected = [entity for entity in counts if entity not in expected_set]
            duplicates = sum(count - 1 for count in counts.values() if count > 1)
            unexpected_rows = sum(counts[entity] for entity in unexpected)
            section_missing_values = int(row["__missing_required_rows"])
            complete = not (
                missing_entities or unexpected or duplicates or section_missing_values
            )
            if complete:
                complete_sections += 1
            else:
                finding = {key: row[key] for key in section_keys}
                finding.update(
                    {
                        "missing_entities": missing_entities,
                        "unexpected_entities": unexpected,
                        "duplicate_entity_rows": duplicates,
                        "missing_required_rows": section_missing_values,
                    }
                )
                if len(findings) < report_limit:
                    findings.append(finding)
            duplicate_entity_rows += duplicates
            unexpected_entity_rows += unexpected_rows
            missing_required_rows += section_missing_values
            missing_entity_slots += len(missing_entities)
            for entity in unexpected:
                if entity not in unexpected_entities:
                    unexpected_entities.append(entity)

        total_sections = sections.height
        complete_ratio = complete_sections / total_sections
        threshold_failed = (
            minimum_complete_ratio is not None
            and complete_ratio < minimum_complete_ratio
        )
        reasons: list[str] = []
        if duplicate_entity_rows:
            reasons.append("DUPLICATE_KEYS")
        if unexpected_entity_rows:
            reasons.append("UNEXPECTED_ENTITIES")
        if missing_required_rows:
            reasons.append("MISSING_REQUIRED_VALUES")
        if missing_entity_slots:
            reasons.append("INCOMPLETE_CROSS_SECTION")
        if threshold_failed:
            reasons.append("LOW_CROSS_SECTION_COMPLETENESS")
        failed = bool(
            duplicate_entity_rows
            or unexpected_entity_rows
            or missing_required_rows
            or threshold_failed
        )
        details = {
            "time_keys": section_keys,
            "entity_key": entity_key,
            "expected_entities": expected,
            "required_columns": required,
            "rows": table.height,
            "sections": total_sections,
            "complete_sections": complete_sections,
            "incomplete_sections": total_sections - complete_sections,
            "complete_section_ratio": complete_ratio,
            "minimum_complete_ratio": minimum_complete_ratio,
            "missing_entity_slots": missing_entity_slots,
            "duplicate_entity_rows": duplicate_entity_rows,
            "unexpected_entity_rows": unexpected_entity_rows,
            "unexpected_entities": unexpected_entities,
            "missing_required_rows": missing_required_rows,
            "section_findings": findings,
            "findings_truncated": total_sections - complete_sections > len(findings),
            "reason_codes": reasons,
        }
        return EvalObservation(
            status="failed" if failed else "passed",
            value=complete_ratio,
            output_kind="findings" if reasons else "metric",
            details=details,
        )

    return EvalSpec(
        name=name,
        runner=check,
        failure_policy=failure_policy,
        output_kind="metric",
        execution_mode="polars_select",
        library_method="library.temporal.cross_section_completeness",
    )


__all__ = ["METHODS", "cross_section_completeness", "key_grid"]
