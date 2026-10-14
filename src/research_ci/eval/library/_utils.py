"""Small, dependency-free numerical helpers for the eval library."""

from __future__ import annotations

import math
from collections.abc import Iterable

import polars as pl

from ..protocol import EvalObservation


def column_names(columns: Iterable[str]) -> tuple[str, ...]:
    """Return unique, non-empty column names while preserving order."""

    selected = tuple(dict.fromkeys(columns))
    if not selected or any(not column.strip() for column in selected):
        raise ValueError("an eval method needs at least one named column")
    return selected


def paired_values(
    table: pl.DataFrame,
    left: str,
    right: str,
) -> tuple[list[float], list[float]]:
    """Return finite numeric pairs, dropping null and non-finite rows."""

    if left not in table.columns or right not in table.columns:
        return [], []
    values_left = table.get_column(left).to_list()
    values_right = table.get_column(right).to_list()
    paired_left: list[float] = []
    paired_right: list[float] = []
    for raw_left, raw_right in zip(values_left, values_right, strict=True):
        if raw_left is None or raw_right is None:
            continue
        try:
            value_left = float(raw_left)
            value_right = float(raw_right)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_left) and math.isfinite(value_right):
            paired_left.append(value_left)
            paired_right.append(value_right)
    return paired_left, paired_right


def numeric_values(table: pl.DataFrame, column: str) -> list[float]:
    """Return finite numeric values from one named column."""

    if column not in table.columns:
        return []
    values: list[float] = []
    for raw in table.get_column(column).to_list():
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def missing_columns(table: pl.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return requested columns that are absent from a table."""

    return [column for column in columns if column not in table.columns]


def not_applicable(reason: str, **details: object) -> EvalObservation:
    """Build a consistently annotated inapplicable observation."""

    payload = {"reason": reason, "reason_code": _reason_code(reason)}
    payload.update(details)
    return EvalObservation(status="not_applicable", details=payload)


def _reason_code(reason: str) -> str:
    normalized = reason.lower().replace(" ", "_")
    if "target" in normalized or "context" in normalized:
        return "MISSING_CONTEXT"
    if "variation" in normalized or "constant" in normalized:
        return "NO_VARIATION"
    if "sample" in normalized or "row" in normalized:
        return "INSUFFICIENT_SAMPLE"
    if "group" in normalized:
        return "INSUFFICIENT_GROUPS"
    return "NOT_APPLICABLE"


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def sample_std(values: Iterable[float]) -> float | None:
    values = list(values)
    if len(values) < 2:
        return None
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def correlation(left: list[float], right: list[float]) -> float | None:
    """Compute Pearson correlation without importing a second numeric stack."""

    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator_left = math.sqrt(sum(value * value for value in centered_left))
    denominator_right = math.sqrt(sum(value * value for value in centered_right))
    if denominator_left == 0 or denominator_right == 0:
        return None
    numerator = sum(a * b for a, b in zip(centered_left, centered_right, strict=True))
    result = numerator / (denominator_left * denominator_right)
    if abs(result - 1.0) < 1e-12:
        return 1.0
    if abs(result + 1.0) < 1e-12:
        return -1.0
    return result


def rank(values: list[float]) -> list[float]:
    """Average-rank values, retaining deterministic ordering for ties."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        for original_index, _ in indexed[index:end]:
            result[original_index] = average_rank
        index = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    """Compute rank correlation using the same finite pairs as Pearson."""

    return correlation(rank(left), rank(right))


__all__ = [
    "column_names",
    "correlation",
    "mean",
    "missing_columns",
    "not_applicable",
    "numeric_values",
    "paired_values",
    "rank",
    "sample_std",
    "spearman",
]
