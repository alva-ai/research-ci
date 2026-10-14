"""Expand agent-selected library methods over declared stage columns."""

from __future__ import annotations

import re
from dataclasses import replace
from itertools import combinations, product
from typing import Any, Callable

import polars as pl

from .declarations import (
    AlgoSet,
    DataSet,
    FeatureSet,
    LibrarySelection,
    SelectedMethod,
    SignalSet,
    StageDeclaration,
)
from .library.association import (
    bucket_spread,
    cumulative_ic,
    icir,
    pearson_ic,
    spearman_ic,
)
from .library.economic import backtest_summary
from .library.information import (
    feature_pair_mutual_information,
    feature_target_mutual_information,
)
from .library.quality import coverage, dispersion, finite_columns, required_columns
from .library.stability import (
    distribution_shift,
    rank_autocorrelation,
    split_association,
)
from .library.structure import (
    autocorrelation,
    pairwise_correlation,
    pca_diagnostics,
    vif,
)
from .library.temporal import key_grid
from .menu import require_selectable_method
from .protocol import EvalContext, EvalObservation, EvalSpec


class LibrarySelectionError(ValueError):
    """Raised when a selected method is incompatible with its declaration."""


def expand_library(
    declaration: StageDeclaration,
    selection: LibrarySelection,
) -> tuple[EvalSpec, ...]:
    """Expand only methods selected by the external case agent."""

    specs: list[EvalSpec] = []
    for selected in selection.methods:
        try:
            require_selectable_method(declaration.stage, selected.name, declaration)
        except ValueError as exc:
            raise LibrarySelectionError(str(exc)) from exc
        specs.extend(_expand_one(declaration, selected))
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise LibrarySelectionError(f"expanded eval names are not unique: {duplicates}")
    return tuple(specs)


def _expand_one(
    declaration: StageDeclaration,
    selected: SelectedMethod,
) -> list[EvalSpec]:
    method = selected.name
    parameters = dict(selected.parameters)
    base_name = _base_name(declaration, selected)

    if method == "library.economic.backtest_summary":
        if not isinstance(declaration, AlgoSet):
            raise LibrarySelectionError(f"{method} requires AlgoSet")
        annualization_periods = parameters.pop("annualization_periods", None)
        if annualization_periods is None:
            raise LibrarySelectionError(
                f"{method} requires parameter 'annualization_periods'"
            )
        if parameters:
            unknown = ", ".join(sorted(parameters))
            raise LibrarySelectionError(f"{method} has unknown parameters: {unknown}")
        return [
            _selected(
                backtest_summary(
                    table_name=declaration.backtest_table,
                    return_columns=declaration.return_columns,
                    equity_columns=declaration.equity_columns,
                    turnover_columns=declaration.turnover_columns,
                    annualization_periods=float(annualization_periods),
                    name=base_name,
                ),
                selected,
                "polars_select",
            )
        ]

    columns = declaration.columns(selected.on)

    if method == "library.quality.required_columns":
        required = declaration.required if selected.on is None else columns
        return [
            _selected(
                required_columns(required, name=base_name, **parameters),
                selected,
                "polars_select",
            )
        ]
    if method == "library.quality.finite_columns":
        return [
            _selected(
                finite_columns(columns, name=base_name, **parameters),
                selected,
                "polars_select",
            )
        ]
    if method == "library.quality.coverage":
        return [
            _selected(
                coverage(columns, name=base_name, **parameters),
                selected,
                "polars_select",
            )
        ]
    if method == "library.quality.dispersion":
        return [
            _selected(
                dispersion(columns, name=base_name, **parameters),
                selected,
                "polars_select",
            )
        ]
    if method == "library.temporal.key_grid":
        if not isinstance(declaration, DataSet):
            raise LibrarySelectionError("temporal.key_grid currently requires DataSet")
        return [
            _selected(
                key_grid(
                    declaration.keys,
                    time_key=declaration.time_key,
                    frequency=declaration.frequency,
                    name=base_name,
                    **parameters,
                ),
                selected,
                "polars_select",
            )
        ]
    if method == "library.structure.pairwise_correlation":
        return [
            _selected(
                pairwise_correlation(
                    left,
                    right,
                    name=f"{base_name}.{_slug(left)}__{_slug(right)}",
                    **parameters,
                ),
                selected,
                "polars_select",
            )
            for left, right in _pairs(columns, method)
        ]
    if method == "library.structure.vif":
        return [
            _selected(
                vif(columns, name=base_name, **parameters), selected, "matrix_backend"
            )
        ]
    if method == "library.structure.pca_diagnostics":
        return [
            _selected(
                pca_diagnostics(columns, name=base_name, **parameters),
                selected,
                "matrix_backend",
            )
        ]
    if method == "library.structure.autocorrelation":
        return [
            _selected(
                autocorrelation(
                    column,
                    name=f"{base_name}.{_slug(column)}",
                    **parameters,
                ),
                selected,
                "polars_select",
            )
            for column in columns
        ]
    if method == "library.information.feature_pair_mutual_information":
        return [
            _selected(
                feature_pair_mutual_information(
                    left,
                    right,
                    name=f"{base_name}.{_slug(left)}__{_slug(right)}",
                    **parameters,
                ),
                selected,
                "matrix_backend",
            )
            for left, right in _pairs(columns, method)
        ]
    if method == "library.stability.rank_autocorrelation":
        return [
            _selected(
                rank_autocorrelation(
                    column,
                    name=f"{base_name}.{_slug(column)}",
                    **parameters,
                ),
                selected,
                "polars_select",
            )
            for column in columns
        ]
    if method == "library.stability.distribution_shift":
        return [
            _selected(
                distribution_shift(
                    column,
                    name=f"{base_name}.{_slug(column)}",
                    **parameters,
                ),
                selected,
                "polars_select",
            )
            for column in columns
        ]
    if method in _TARGET_FACTORIES:
        if not isinstance(declaration, SignalSet):
            raise LibrarySelectionError(f"{method} requires SignalSet targets")
        return _expand_targets(declaration, selected, columns, base_name, parameters)
    raise LibrarySelectionError(
        f"{method} is registered but has no declaration expander"
    )


TargetFactory = Callable[..., EvalSpec]
_TARGET_FACTORIES: dict[str, TargetFactory] = {
    "library.association.pearson_ic": pearson_ic,
    "library.association.spearman_ic": spearman_ic,
    "library.association.icir": icir,
    "library.association.cumulative_ic": cumulative_ic,
    "library.association.bucket_spread": bucket_spread,
    "library.information.feature_target_mutual_information": (
        feature_target_mutual_information
    ),
    "library.stability.split_association": split_association,
}


def _expand_targets(
    declaration: SignalSet,
    selected: SelectedMethod,
    features: tuple[str, ...],
    base_name: str,
    parameters: dict[str, Any],
) -> list[EvalSpec]:
    if not declaration.targets:
        raise LibrarySelectionError(
            f"{selected.name} was selected but SignalSet.targets is empty"
        )
    horizons = selected.targets or tuple(declaration.targets)
    unknown = sorted(set(horizons) - set(declaration.targets))
    if unknown:
        raise LibrarySelectionError(
            f"selected target horizons are not declared: {unknown}"
        )
    factory = _TARGET_FACTORIES[selected.name]
    specs: list[EvalSpec] = []
    for feature, horizon in product(features, horizons):
        target = declaration.targets[horizon]
        options = dict(parameters)
        if selected.name in {
            "library.association.icir",
            "library.association.cumulative_ic",
        }:
            by = options.pop(
                "by",
                declaration.evaluation_by or declaration.cross_section_by,
            )
            if by is None:
                raise LibrarySelectionError(
                    f"{selected.name} requires SignalSet.evaluation_by, "
                    "SignalSet.cross_section_by, or parameter 'by'"
                )
            options["by"] = by
        name = f"{base_name}.{_slug(feature)}__{_slug(horizon)}"
        base = factory(feature, target, name=name, **options)
        context_columns = ()
        group_column = options.get("by")
        if isinstance(group_column, str) and group_column:
            context_columns = (group_column,)
        contextual = _from_label_sidecar(
            base,
            declaration,
            feature,
            target,
            context_columns=context_columns,
        )
        specs.append(
            _selected(
                contextual,
                selected,
                "matrix_backend"
                if "mutual_information" in selected.name
                else "polars_join_group",
            )
        )
    return specs


def _from_label_sidecar(
    base: EvalSpec,
    declaration: SignalSet,
    feature: str,
    target: str,
    *,
    context_columns: tuple[str, ...] = (),
) -> EvalSpec:
    """Join declared labels in scratch before running a target-aware method."""

    keys = declaration.keys

    def run(table: pl.DataFrame, context: EvalContext) -> EvalObservation:
        try:
            sidecar = context.table(declaration.label_table)
        except KeyError as exc:
            return EvalObservation(
                status="not_applicable",
                details={
                    "reason": str(exc),
                    "reason_code": "MISSING_CONTEXT",
                    "context_table": declaration.label_table,
                },
            )
        main_columns = tuple(dict.fromkeys((*keys, feature)))
        target_columns = tuple(dict.fromkeys((*keys, target, *context_columns)))
        missing_main = sorted(set(main_columns) - set(table.columns))
        missing_target = sorted(set(target_columns) - set(sidecar.columns))
        if missing_main or missing_target:
            return EvalObservation(
                status="not_applicable",
                details={
                    "reason": "target join columns are missing",
                    "reason_code": "MISSING_CONTEXT",
                    "missing_main_columns": missing_main,
                    "missing_target_columns": missing_target,
                },
            )
        duplicate_keys = (
            sidecar.select(keys).height - sidecar.select(keys).unique().height
        )
        if duplicate_keys:
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={
                    "reason_code": "CONTEXT_DUPLICATE_KEYS",
                    "duplicate_key_rows": duplicate_keys,
                    "context_table": declaration.label_table,
                },
            )
        joined = table.select(main_columns).join(
            sidecar.select(target_columns), on=list(keys), how="inner"
        )
        if base.runner is None:
            raise RuntimeError(f"target method {base.name!r} has no table runner")
        observation = base.runner(joined)
        if not isinstance(observation, EvalObservation):
            raise TypeError(f"target method {base.name!r} returned no observation")
        details = dict(observation.details)
        details.update(
            {
                "context_table": declaration.label_table,
                "join_on": keys,
                "joined_rows": joined.height,
            }
        )
        return EvalObservation(
            status=observation.status,
            value=observation.value,
            output_kind=observation.output_kind,
            details=details,
        )

    return replace(base, runner=None, context_runner=run)


def _selected(
    spec: EvalSpec,
    selected: SelectedMethod,
    execution_mode: str,
) -> EvalSpec:
    return replace(
        spec,
        failure_policy=selected.failure_policy,
        execution_mode=execution_mode,
        selection_reason=selected.reason,
        library_method=selected.name,
    )


def _pairs(columns: tuple[str, ...], method: str) -> tuple[tuple[str, str], ...]:
    pairs = tuple(combinations(columns, 2))
    if not pairs:
        raise LibrarySelectionError(f"{method} requires at least two selected columns")
    return pairs


def _base_name(declaration: StageDeclaration, selected: SelectedMethod) -> str:
    if selected.alias:
        suffix = _slug(selected.alias)
    elif isinstance(selected.on, str):
        suffix = _slug(selected.on)
    elif isinstance(selected.on, tuple):
        suffix = "selected"
    else:
        suffix = (
            "features"
            if isinstance(declaration, (FeatureSet, SignalSet))
            else "default"
        )
    return f"{declaration.stage}.{selected.name}.{suffix}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


__all__ = ["LibrarySelectionError", "expand_library"]
