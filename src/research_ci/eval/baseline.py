"""Engine-owned floor checks that every stage receives automatically."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

import polars as pl

from research_ci.kit.sig import FeatureDisposition, SignalLineage

from .declarations import AlgoSet, LibrarySelection, SignalSet
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


def signal_main_output_shape(
    signal_table: pl.DataFrame,
    main_table: pl.DataFrame,
    keys: Iterable[str],
    signals: Iterable[str],
) -> EvalObservation:
    """Validate a fixed-instrument, wide Sig main table for Algo."""

    key_names = tuple(dict.fromkeys(keys))
    signal_names = tuple(dict.fromkeys(signals))
    grid = sidecar_output_shape(
        signal_table,
        main_table,
        key_names,
        signal_names,
    )
    inst_columns = tuple(
        column for column in main_table.columns if column.endswith("_inst")
    )
    price_columns = tuple(
        column for column in main_table.columns if column.endswith("_price")
    )
    inst_prefixes = {column.removesuffix("_inst") for column in inst_columns}
    price_prefixes = {column.removesuffix("_price") for column in price_columns}
    unmatched_inst = sorted(inst_prefixes - price_prefixes)
    unmatched_price = sorted(price_prefixes - inst_prefixes)
    invalid_inst_columns = [
        column
        for column in inst_columns
        if main_table.schema[column] != pl.String
        or main_table.get_column(column).null_count()
    ]
    invalid_price_columns: list[str] = []
    for column in price_columns:
        series = main_table.get_column(column)
        if not series.dtype.is_numeric():
            invalid_price_columns.append(column)
            continue
        invalid = series.is_null() | ~series.is_finite() | (series <= 0)
        if bool(invalid.any()):
            invalid_price_columns.append(column)
    details = {
        **grid.details,
        "layout": "wide",
        "inst_columns": inst_columns,
        "price_columns": price_columns,
        "unmatched_inst_prefixes": unmatched_inst,
        "unmatched_price_prefixes": unmatched_price,
        "invalid_inst_columns": invalid_inst_columns,
        "invalid_price_columns": invalid_price_columns,
    }
    valid = (
        grid.status == "passed"
        and bool(inst_columns)
        and inst_prefixes == price_prefixes
        and not invalid_inst_columns
        and not invalid_price_columns
    )
    if not valid and "reason_code" not in details:
        details["reason_code"] = "INVALID_WIDE_SIGNAL_MAIN"
    return EvalObservation(
        "passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


def algo_main_output_shape(
    signal_table: pl.DataFrame,
    main_table: pl.DataFrame,
    declaration: AlgoSet,
) -> EvalObservation:
    """Validate declared prices and positions in the Algo main table."""

    required = declaration.required
    missing = sorted(set(required) - set(main_table.columns))
    extra = sorted(set(main_table.columns) - set(required))
    source_columns = (
        *declaration.context,
        *declaration.prices,
        *declaration.signals,
    )
    source_keys = (
        declaration.keys
        if declaration.representation == "weights"
        else (declaration.time_key,)
    )
    missing_source = sorted(
        set((*source_keys, *source_columns)) - set(signal_table.columns)
    )
    details: dict[str, object] = {
        "representation": declaration.representation,
        "keys": declaration.keys,
        "required_columns": required,
        "missing_columns": missing,
        "extra_columns": extra,
        "missing_source_columns": missing_source,
    }
    if missing or extra or missing_source:
        details["reason_code"] = "INVALID_ALGO_SCHEMA"
        return EvalObservation("failed", output_kind="findings", details=details)

    if declaration.representation == "weights":
        grid = sidecar_output_shape(
            signal_table,
            main_table,
            declaration.keys,
            (*source_columns, *declaration.positions),
        )
        grid_valid = grid.status == "passed"
        details.update(grid.details)
    else:
        output_keys = main_table.select(declaration.keys)
        duplicate_keys = output_keys.height - output_keys.unique().height
        source_times = signal_table.select(declaration.time_key).unique()
        unknown_times = (
            main_table.select(declaration.time_key)
            .join(
                source_times,
                on=declaration.time_key,
                how="anti",
            )
            .height
        )
        grid_valid = duplicate_keys == 0 and unknown_times == 0
        details.update(
            {
                "duplicate_key_rows": duplicate_keys,
                "unknown_source_times": unknown_times,
            }
        )

    source_equal = False
    if declaration.representation == "weights":
        expected = signal_table.select(*declaration.keys, *source_columns).sort(
            declaration.keys
        )
        actual = main_table.select(*declaration.keys, *source_columns).sort(
            declaration.keys
        )
        source_equal = actual.equals(expected)
    else:
        output_times = main_table.select(declaration.time_key).unique()
        expected = (
            signal_table.select(declaration.time_key, *source_columns)
            .join(output_times, on=declaration.time_key, how="semi")
            .sort(declaration.time_key)
        )
        actual = (
            main_table.select(declaration.time_key, *source_columns)
            .unique()
            .sort(declaration.time_key)
        )
        source_equal = actual.equals(expected)

    invalid_prices: list[str] = []
    for column in declaration.prices:
        series = main_table.get_column(column)
        if not series.dtype.is_numeric() or bool(
            (series.is_null() | ~series.is_finite() | (series <= 0)).any()
        ):
            invalid_prices.append(column)

    invalid_positions: list[str] = []
    position_limits: dict[str, float] = {}
    for column in declaration.positions:
        series = main_table.get_column(column)
        if not series.dtype.is_numeric() or bool(
            (series.is_null() | ~series.is_finite()).any()
        ):
            invalid_positions.append(column)
            continue
        maximum = float(series.abs().max()) if main_table.height else 0.0
        position_limits[column] = maximum
        if (
            declaration.max_abs_position is not None
            and maximum > declaration.max_abs_position + 1e-12
        ):
            invalid_positions.append(column)

    gross_exposure: dict[str, float] = {}
    invalid_gross: list[str] = []
    if declaration.representation == "weights":
        for strategy, columns in declaration.strategies.items():
            gross = main_table.select(
                pl.sum_horizontal(*(pl.col(column).abs() for column in columns)).max()
            ).item()
            maximum = float(gross) if gross is not None else 0.0
            gross_exposure[strategy] = maximum
            if (
                declaration.max_gross is not None
                and maximum > declaration.max_gross + 1e-12
            ):
                invalid_gross.append(strategy)

    details.update(
        {
            "source_columns_preserved": source_equal,
            "invalid_price_columns": invalid_prices,
            "invalid_position_columns": invalid_positions,
            "maximum_absolute_position": position_limits,
            "maximum_gross_exposure": gross_exposure,
            "strategies_above_maximum_gross": invalid_gross,
        }
    )
    valid = (
        grid_valid
        and source_equal
        and not invalid_prices
        and not invalid_positions
        and not invalid_gross
    )
    if not valid and "reason_code" not in details:
        details["reason_code"] = "INVALID_ALGO_MAIN"
    return EvalObservation(
        "passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


def algo_backtest_output_shape(
    signal_table: pl.DataFrame,
    backtest_table: pl.DataFrame,
    declaration: AlgoSet,
) -> EvalObservation:
    """Validate a standardized external backtest timeseries sidecar."""

    grid = sidecar_output_shape(
        signal_table,
        backtest_table,
        declaration.backtest_keys,
        declaration.backtest_required,
    )
    invalid_returns: list[str] = []
    invalid_equity: list[str] = []
    invalid_turnover: list[str] = []
    for column in declaration.return_columns.values():
        if column not in backtest_table.columns:
            invalid_returns.append(column)
            continue
        series = backtest_table.get_column(column)
        if not series.dtype.is_numeric() or bool(
            (series.is_null() | ~series.is_finite()).any()
        ):
            invalid_returns.append(column)
    for column in declaration.equity_columns.values():
        if column not in backtest_table.columns:
            invalid_equity.append(column)
            continue
        series = backtest_table.get_column(column)
        if not series.dtype.is_numeric() or bool(
            (series.is_null() | ~series.is_finite() | (series <= 0)).any()
        ):
            invalid_equity.append(column)
    for column in declaration.turnover_columns.values():
        if column not in backtest_table.columns:
            invalid_turnover.append(column)
            continue
        series = backtest_table.get_column(column)
        if not series.dtype.is_numeric() or bool(
            (series.is_null() | ~series.is_finite() | (series < 0)).any()
        ):
            invalid_turnover.append(column)
    details = {
        **grid.details,
        "context_table": declaration.backtest_table,
        "invalid_return_columns": invalid_returns,
        "invalid_equity_columns": invalid_equity,
        "invalid_turnover_columns": invalid_turnover,
    }
    valid = (
        grid.status == "passed"
        and not invalid_returns
        and not invalid_equity
        and not invalid_turnover
    )
    if not valid and "reason_code" not in details:
        details["reason_code"] = "INVALID_BACKTEST_OUTPUT"
    return EvalObservation(
        "passed" if valid else "failed",
        value=valid,
        output_kind="summary" if valid else "findings",
        details=details,
    )


def algo_backtest_summary_shape(
    summary: pl.DataFrame,
    declaration: AlgoSet,
) -> EvalObservation:
    """Require one external backtest summary row per declared strategy."""

    missing_columns = sorted({"strategy"} - set(summary.columns))
    actual: list[str] = []
    if not missing_columns:
        actual = [str(value) for value in summary.get_column("strategy")]
    expected = sorted(declaration.strategies)
    duplicates = sorted(
        strategy for strategy, count in Counter(actual).items() if count > 1
    )
    missing_strategies = sorted(set(expected) - set(actual))
    unknown_strategies = sorted(set(actual) - set(expected))
    valid = not (
        missing_columns or duplicates or missing_strategies or unknown_strategies
    )
    details: dict[str, object] = {
        "summary_rows": summary.height,
        "expected_strategies": expected,
        "missing_columns": missing_columns,
        "missing_strategies": missing_strategies,
        "unknown_strategies": unknown_strategies,
        "duplicate_strategies": duplicates,
    }
    if not valid:
        details["reason_code"] = "INVALID_BACKTEST_SUMMARY"
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
            {"feature": feature, "horizon": horizon} for feature, horizon in missing
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
                {"feature": feature, "horizon": horizon} for feature, horizon in missing
            ],
            "unknown_pairs": [
                {"feature": feature, "horizon": horizon} for feature, horizon in unknown
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
    duplicates = sorted(name for name, count in Counter(actual).items() if count > 1)
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
