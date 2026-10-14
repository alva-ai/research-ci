"""Engine-owned floor checks that every stage receives automatically."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

import polars as pl

from research_ci.kit.sig import FeatureDisposition, SignalLineage

from .declarations import LibrarySelection, SignalSet
from .protocol import EvalObservation

_TARGET_SCREENING_METHODS = frozenset(
    {
        "library.association.pearson_ic",
        "library.association.spearman_ic",
        "library.information.feature_target_mutual_information",
    }
)


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


def sidecar_output_shape(
    input_table: pl.DataFrame,
    sidecar: pl.DataFrame,
    keys: Iterable[str],
    required: Iterable[str],
) -> EvalObservation:
    """Check that a named stage sidecar preserves the input key grid."""

    selected = tuple(dict.fromkeys(keys))
    required_columns = tuple(dict.fromkeys((*selected, *required)))
    missing_input = sorted(set(selected) - set(input_table.columns))
    missing_sidecar = sorted(set(required_columns) - set(sidecar.columns))
    details: dict[str, object] = {
        "input_rows": input_table.height,
        "sidecar_rows": sidecar.height,
        "keys": selected,
        "required_columns": required_columns,
        "missing_input_key_columns": missing_input,
        "missing_sidecar_columns": missing_sidecar,
    }
    if not selected or missing_input or missing_sidecar:
        details["reason_code"] = "MISSING_CONTEXT"
        return EvalObservation("failed", output_kind="findings", details=details)

    input_keys = input_table.select(selected)
    sidecar_keys = sidecar.select(selected)
    missing_keys = input_keys.join(sidecar_keys, on=list(selected), how="anti")
    extra_keys = sidecar_keys.join(input_keys, on=list(selected), how="anti")
    duplicate_key_rows = sidecar_keys.height - sidecar_keys.unique().height
    details.update(
        {
            "missing_key_rows": missing_keys.height,
            "extra_key_rows": extra_keys.height,
            "duplicate_key_rows": duplicate_key_rows,
        }
    )
    valid = (
        input_table.height == sidecar.height
        and missing_keys.is_empty()
        and extra_keys.is_empty()
        and duplicate_key_rows == 0
    )
    if not valid:
        details["reason_code"] = (
            "CONTEXT_DUPLICATE_KEYS"
            if duplicate_key_rows
            else "CONTEXT_KEY_GRID_MISMATCH"
        )
    return EvalObservation(
        "passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


def signal_feature_universe(
    feature_table: pl.DataFrame,
    keys: Iterable[str],
    declared_features: Iterable[str],
) -> EvalObservation:
    """Require Sig to declare every feature in the approved Feat table."""

    key_names = tuple(dict.fromkeys(keys))
    actual = tuple(
        column for column in feature_table.columns if column not in key_names
    )
    raw_declared = tuple(declared_features)
    declared = tuple(dict.fromkeys(raw_declared))
    missing = sorted(set(actual) - set(declared))
    unknown = sorted(set(declared) - set(actual))
    duplicates = len(raw_declared) - len(declared)
    valid = not missing and not unknown and duplicates == 0
    details: dict[str, object] = {
        "feature_count": len(actual),
        "declared_count": len(declared),
        "missing_from_declaration": missing,
        "unknown_in_declaration": unknown,
        "duplicate_declarations": duplicates,
        "coverage": len(set(actual) & set(declared)) / len(actual) if actual else 0.0,
    }
    if not valid:
        details["reason_code"] = "INCOMPLETE_FEATURE_UNIVERSE"
    return EvalObservation(
        "passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


def signal_screening_plan_coverage(
    declaration: SignalSet,
    selection: LibrarySelection,
) -> EvalObservation:
    """Require target-aware Sig plans to screen every feature and horizon."""

    expected = {
        (feature, horizon)
        for feature in declaration.features
        for horizon in declaration.targets
    }
    covered: set[tuple[str, str]] = set()
    selected_methods: list[str] = []
    for selected in selection.methods:
        if selected.name not in _TARGET_SCREENING_METHODS:
            continue
        columns = set(declaration.columns(selected.on)) & set(declaration.features)
        horizons = set(selected.targets or tuple(declaration.targets))
        covered.update((column, horizon) for column in columns for horizon in horizons)
        selected_methods.append(selected.name)
    missing = sorted(expected - covered)
    valid = not expected or not missing
    details: dict[str, object] = {
        "expected_feature_target_pairs": len(expected),
        "covered_feature_target_pairs": len(expected & covered),
        "coverage": len(expected & covered) / len(expected) if expected else 1.0,
        "screening_methods": sorted(set(selected_methods)),
        "missing_pairs": [
            {"feature": feature, "horizon": horizon}
            for feature, horizon in missing
        ],
    }
    if not valid:
        details["reason_code"] = "INCOMPLETE_TARGET_SCREENING"
    return EvalObservation(
        "passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


def signal_screening_artifact_coverage(
    screening: pl.DataFrame | None,
    features: Iterable[str],
    targets: Mapping[str, str],
) -> EvalObservation:
    """Check that the reusable screening table covers the complete universe."""

    feature_names = tuple(dict.fromkeys(features))
    expected = {(feature, horizon) for feature in feature_names for horizon in targets}
    details: dict[str, object] = {
        "expected_feature_target_pairs": len(expected),
        "required_columns": ["feature", "horizon"],
    }
    if not targets:
        details.update(
            {
                "required": False,
                "covered_feature_target_pairs": 0,
                "coverage": 1.0,
                "missing_pairs": [],
                "unknown_pairs": [],
            }
        )
        return EvalObservation(
            "passed",
            value=True,
            output_kind="summary",
            details=details,
        )
    if screening is None:
        details.update(
            {
                "reason_code": "MISSING_SCREENING_ARTIFACT",
                "missing_pairs": len(expected),
            }
        )
        return EvalObservation("failed", output_kind="findings", details=details)
    missing_columns = sorted({"feature", "horizon"} - set(screening.columns))
    if missing_columns:
        details.update(
            {
                "reason_code": "INVALID_SCREENING_ARTIFACT",
                "missing_columns": missing_columns,
            }
        )
        return EvalObservation("failed", output_kind="findings", details=details)
    actual_rows = screening.select("feature", "horizon").unique()
    actual = {
        (str(feature), str(horizon)) for feature, horizon in actual_rows.iter_rows()
    }
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    valid = not missing and not unknown
    details.update(
        {
            "rows": screening.height,
            "covered_feature_target_pairs": len(expected & actual),
            "coverage": len(expected & actual) / len(expected) if expected else 1.0,
            "missing_pairs": [
                {"feature": feature, "horizon": horizon}
                for feature, horizon in missing
            ],
            "unknown_pairs": [
                {"feature": feature, "horizon": horizon}
                for feature, horizon in unknown
            ],
        }
    )
    if not valid:
        details["reason_code"] = "INCOMPLETE_SCREENING_ARTIFACT"
    return EvalObservation(
        "passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


def signal_disposition_coverage(
    feature_table: pl.DataFrame,
    keys: Iterable[str],
    dispositions: Iterable[FeatureDisposition],
    outputs: Iterable[str],
) -> EvalObservation:
    """Require one explained decision for every approved feature."""

    key_names = set(keys)
    expected = {column for column in feature_table.columns if column not in key_names}
    items = tuple(dispositions)
    actual = [item.feature for item in items]
    missing = sorted(expected - set(actual))
    unknown = sorted(set(actual) - expected)
    duplicates = sorted(
        feature for feature, count in Counter(actual).items() if count > 1
    )
    allowed_outputs = set(outputs)
    unknown_selected_for = sorted(
        {
            output
            for item in items
            for output in item.selected_for
            if output not in allowed_outputs
        }
    )
    rejected_with_outputs = sorted(
        item.feature
        for item in items
        if item.action.startswith("reject_") and item.selected_for
    )
    selected_without_outputs = sorted(
        item.feature
        for item in items
        if item.action
        in {"keep_direct", "representative_of_cluster", "input_to_component"}
        and not item.selected_for
    )
    valid = not (
        missing
        or unknown
        or duplicates
        or unknown_selected_for
        or rejected_with_outputs
        or selected_without_outputs
    )
    action_counts: dict[str, int] = {}
    for item in items:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
    details: dict[str, object] = {
        "feature_count": len(expected),
        "disposition_count": len(items),
        "coverage": len(expected & set(actual)) / len(expected) if expected else 0.0,
        "action_counts": dict(sorted(action_counts.items())),
        "missing_features": missing,
        "unknown_features": unknown,
        "duplicate_features": duplicates,
        "unknown_selected_for": unknown_selected_for,
        "rejected_with_outputs": rejected_with_outputs,
        "selected_without_outputs": selected_without_outputs,
    }
    if not valid:
        details["reason_code"] = "INCOMPLETE_FEATURE_DISPOSITION"
    return EvalObservation(
        "passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


def signal_lineage_coverage(
    features: Iterable[str],
    candidates: Iterable[str],
    signals: Iterable[str],
    lineage: Iterable[SignalLineage],
) -> EvalObservation:
    """Require every candidate and final signal to have acyclic lineage."""

    feature_names = set(features)
    candidate_names = tuple(dict.fromkeys(candidates))
    signal_names = tuple(dict.fromkeys(signals))
    expected_roles = {
        **{name: "candidate" for name in candidate_names},
        **{name: "final" for name in signal_names},
    }
    items = tuple(lineage)
    actual = [item.signal for item in items]
    missing = sorted(set(expected_roles) - set(actual))
    unknown = sorted(set(actual) - set(expected_roles))
    duplicates = sorted(
        name for name, count in Counter(actual).items() if count > 1
    )
    wrong_roles = sorted(
        item.signal
        for item in items
        if expected_roles.get(item.signal) not in {None, item.role}
    )
    candidate_depends_on_final = sorted(
        item.signal
        for item in items
        if item.role == "candidate" and set(item.inputs) & set(signal_names)
    )
    available = set(feature_names)
    unresolved: dict[str, list[str]] = {}
    for item in items:
        missing_inputs = sorted(set(item.inputs) - available)
        if missing_inputs:
            unresolved[item.signal] = missing_inputs
        available.add(item.signal)
    valid = not (
        missing
        or unknown
        or duplicates
        or wrong_roles
        or candidate_depends_on_final
        or unresolved
    )
    details: dict[str, object] = {
        "expected_outputs": len(expected_roles),
        "lineage_records": len(items),
        "missing_outputs": missing,
        "unknown_outputs": unknown,
        "duplicate_outputs": duplicates,
        "wrong_roles": wrong_roles,
        "candidate_depends_on_final": candidate_depends_on_final,
        "unresolved_inputs": unresolved,
    }
    if not valid:
        details["reason_code"] = "INCOMPLETE_SIGNAL_LINEAGE"
    return EvalObservation(
        "passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


__all__ = [
    "feature_output_shape",
    "general_missing_values",
    "sidecar_output_shape",
    "signal_disposition_coverage",
    "signal_feature_universe",
    "signal_lineage_coverage",
    "signal_screening_artifact_coverage",
    "signal_screening_plan_coverage",
]
