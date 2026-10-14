"""Engine-owned floor checks that every stage receives automatically."""

from __future__ import annotations

from collections.abc import Iterable

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
        sentinel_values = {"nan", "none", "null"}
        expressions: list[pl.Expr] = []
        aliases: dict[str, dict[str, str]] = {}

        # Build one aggregate projection for the complete table. The small
        # Python loop below only assembles expressions and the scalar result;
        # Polars performs the row scan for every column in one select.
        for index, (name, dtype) in enumerate(table.schema.items()):
            column_aliases = {"null": f"__rci_{index}_null"}
            expressions.append(pl.col(name).null_count().alias(column_aliases["null"]))
            if dtype.is_float():
                column_aliases.update(
                    {
                        "nan": f"__rci_{index}_nan",
                        "non_finite": f"__rci_{index}_non_finite",
                    }
                )
                expressions.extend(
                    (
                        pl.col(name).is_nan().sum().alias(column_aliases["nan"]),
                        (~pl.col(name).is_finite())
                        .sum()
                        .alias(column_aliases["non_finite"]),
                    )
                )
            if dtype == pl.String:
                column_aliases["string_sentinel"] = f"__rci_{index}_string_sentinel"
                expressions.append(
                    pl.col(name)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .is_in(sorted(sentinel_values))
                    .sum()
                    .alias(column_aliases["string_sentinel"])
                )
            aliases[name] = column_aliases

        aggregate = table.select(expressions).row(0, named=True) if expressions else {}
        findings: dict[str, dict[str, int]] = {}
        totals = {
            "null": 0,
            "nan": 0,
            "non_finite": 0,
            "string_sentinel": 0,
        }

        for name, column_aliases in aliases.items():
            counts = {
                "null": int(aggregate[column_aliases["null"]]),
                "nan": 0,
                "non_finite": 0,
                "string_sentinel": 0,
            }
            for metric in ("nan", "non_finite", "string_sentinel"):
                alias = column_aliases.get(metric)
                if alias is not None:
                    counts[metric] = int(aggregate[alias])
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
            "execution_mode": "polars_select",
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


def feature_output_shape(
    input_table: pl.DataFrame,
    feature_table: pl.DataFrame,
    keys: Iterable[str],
) -> EvalObservation:
    """Check that a Feat output preserves the input key grid.

    Warm-up rows may contain null feature values, but a feature builder must
    not silently drop, duplicate, or invent key rows. The feature table may
    contain only keys plus feature columns.
    """

    selected = tuple(dict.fromkeys(keys))
    missing_key_columns = [
        column for column in selected if column not in input_table.columns
    ]
    missing_output_key_columns = [
        column for column in selected if column not in feature_table.columns
    ]
    feature_columns = [
        column for column in feature_table.columns if column not in selected
    ]
    details: dict[str, object] = {
        "input_rows": input_table.height,
        "output_rows": feature_table.height,
        "keys": selected,
        "feature_columns": feature_columns,
        "missing_input_key_columns": missing_key_columns,
        "missing_output_key_columns": missing_output_key_columns,
    }

    if missing_key_columns or missing_output_key_columns or not selected:
        details["reason_code"] = "MISSING_KEY_COLUMNS"
        return EvalObservation(
            status="failed",
            output_kind="findings",
            details=details,
        )

    input_keys = input_table.select(selected)
    output_keys = feature_table.select(selected)
    missing_keys = input_keys.join(output_keys, on=list(selected), how="anti")
    extra_keys = output_keys.join(input_keys, on=list(selected), how="anti")
    unique_rows = output_keys.unique(maintain_order=True).height
    duplicate_key_rows = output_keys.height - unique_rows
    details.update(
        {
            "missing_key_rows": missing_keys.height,
            "extra_key_rows": extra_keys.height,
            "duplicate_key_rows": duplicate_key_rows,
        }
    )

    valid = (
        input_table.height == feature_table.height
        and missing_keys.is_empty()
        and extra_keys.is_empty()
        and duplicate_key_rows == 0
        and bool(feature_columns)
    )
    if not valid:
        if duplicate_key_rows:
            details["reason_code"] = "DUPLICATE_KEYS"
        elif missing_keys.height or extra_keys.height:
            details["reason_code"] = "KEY_GRID_MISMATCH"
        elif input_table.height != feature_table.height:
            details["reason_code"] = "ROW_COUNT_MISMATCH"
        elif not feature_columns:
            details["reason_code"] = "NO_FEATURE_COLUMNS"
    return EvalObservation(
        status="passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


__all__ = ["feature_output_shape", "general_missing_values"]
