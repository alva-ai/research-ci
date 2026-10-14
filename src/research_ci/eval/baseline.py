"""Engine-owned floor checks that every stage receives automatically."""

from __future__ import annotations

import polars as pl

from .protocol import EvalObservation


def general_missing_values(table: pl.DataFrame) -> EvalObservation:
    """Report null, NaN, non-finite, and string missing sentinels.

    A null can be valid for a particular research source, so this common
    floor records findings and leaves the stage decision to the review card.
    Stage-specific plans can still choose a stop policy when a field is
    required.
    """

    try:
        findings: dict[str, dict[str, int]] = {}
        totals = {
            "null": 0,
            "nan": 0,
            "non_finite": 0,
            "string_sentinel": 0,
        }
        sentinel_values = {"nan", "none", "null"}

        for name, dtype in table.schema.items():
            series = table.get_column(name)
            counts = {
                "null": series.null_count(),
                "nan": 0,
                "non_finite": 0,
                "string_sentinel": 0,
            }
            if dtype.is_float():
                counts["nan"] = int(series.is_nan().sum())
                counts["non_finite"] = int((~series.is_finite()).sum())
            if dtype == pl.String:
                counts["string_sentinel"] = int(
                    series.str.strip_chars()
                    .str.to_lowercase()
                    .is_in(sentinel_values)
                    .sum()
                )
            for key, value in counts.items():
                totals[key] += value
            if any(counts.values()):
                findings[name] = counts

        details = {
            "rows": table.height,
            "columns": table.width,
            "totals": totals,
            "columns_with_findings": findings,
            "string_sentinels": sorted(sentinel_values),
        }
        if findings:
            return EvalObservation(
                status="failed",
                value=totals,
                output_kind="findings",
                details=details,
            )
        return EvalObservation(
            status="passed",
            value=totals,
            output_kind="summary",
            details=details,
        )
    except Exception as exc:  # noqa: BLE001 - preserve a baseline audit record
        return EvalObservation(
            status="error",
            output_kind="none",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )


__all__ = ["general_missing_values"]
