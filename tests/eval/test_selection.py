from datetime import timedelta

import polars as pl
import pytest

from research_ci import (
    CaseEvalTest,
    DataSet,
    EvalContext,
    EvalPlan,
    EvalSpec,
    FeatureSet,
    LibrarySelection,
    LibrarySelectionError,
    SelectedMethod,
    expand_library,
    run_eval_plan,
)


def _plan(stage: str, library: tuple[EvalSpec, ...]) -> EvalPlan:
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=library,
        case=(EvalSpec("case.positive", expr=pl.col("x") > 0),),
        case_eval=(
            CaseEvalTest.from_rows(
                name="case.positive.counterexample",
                case="case.positive",
                valid={"x": [1]},
                invalid={"x": [-1]},
            ),
        ),
    )


def test_data_agent_selects_methods_and_engine_expands_declared_groups() -> None:
    declaration = DataSet(
        keys=("event_time",),
        required=("event_time", "x", "y"),
        numeric=("x", "y"),
        groups={"prices": ("x", "y")},
        time_key="event_time",
        frequency=timedelta(hours=1),
    )
    selection = LibrarySelection(
        (
            SelectedMethod(
                "library.quality.finite_columns",
                on="prices",
                reason="prices must be finite before feature construction",
            ),
            SelectedMethod(
                "library.temporal.key_grid",
                reason="the source contract declares an hourly key grid",
                failure_policy="stop",
            ),
        )
    )

    specs = expand_library(declaration, selection)

    assert [spec.name for spec in specs] == [
        "data.library.quality.finite_columns.prices",
        "data.library.temporal.key_grid.default",
    ]
    assert all(spec.selection_reason for spec in specs)
    assert all(spec.execution_mode == "polars_select" for spec in specs)


def test_feature_pair_and_target_methods_expand_from_agent_selection() -> None:
    declaration = FeatureSet(
        keys=("event_time",),
        features=("x", "y"),
        targets={"1h": "return_1h", "6h": "return_6h"},
    )
    selection = LibrarySelection(
        (
            SelectedMethod(
                "library.structure.pairwise_correlation",
                reason="screen linear redundancy",
            ),
            SelectedMethod(
                "library.association.pearson_ic",
                reason="measure linear feature/target association",
            ),
        )
    )

    specs = expand_library(declaration, selection)

    assert len(specs) == 5
    assert sum("pairwise_correlation" in spec.name for spec in specs) == 1
    assert sum("pearson_ic" in spec.name for spec in specs) == 4
    assert {spec.execution_mode for spec in specs} == {
        "polars_select",
        "polars_join_group",
    }


def test_target_selection_reads_sidecar_without_mutating_feature_table() -> None:
    features = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    labels = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "return_1h": [0.1, 0.2, 0.3, 0.4],
        }
    )
    declaration = FeatureSet(
        keys=("event_time",),
        features=("x",),
        targets={"1h": "return_1h"},
        target_table="labels",
    )
    specs = expand_library(
        declaration,
        LibrarySelection(
            (
                SelectedMethod(
                    "library.association.pearson_ic",
                    reason="evaluate the declared one-hour label",
                ),
            )
        ),
    )

    result = run_eval_plan(
        features,
        _plan("feat", specs),
        EvalContext(tables={"labels": labels}),
    )

    record = next(record for record in result.records if "pearson_ic" in record.name)
    assert record.status == "passed"
    assert record.value == 1.0
    assert record.execution_mode == "polars_join_group"
    assert record.selection_reason == "evaluate the declared one-hour label"
    assert features.columns == ["event_time", "x"]


def test_agent_cannot_select_target_method_without_declared_targets() -> None:
    declaration = FeatureSet(keys=("event_time",), features=("x",), targets={})
    selection = LibrarySelection(
        (
            SelectedMethod(
                "library.association.pearson_ic",
                reason="try a target method",
            ),
        )
    )

    with pytest.raises(LibrarySelectionError, match="targets is empty"):
        expand_library(declaration, selection)


def test_catalog_context_method_expands_through_declared_target_sidecar() -> None:
    declaration = FeatureSet(
        keys=("event_time",),
        features=("x",),
        targets={"1h": "return_1h"},
        target_table="labels",
    )
    selection = LibrarySelection(
        (
            SelectedMethod(
                "library.association.context_spearman_ic",
                reason="evaluate the explicit label sidecar",
            ),
        )
    )

    specs = expand_library(declaration, selection)

    assert len(specs) == 1
    assert specs[0].context_runner is not None
    assert specs[0].execution_mode == "polars_join_group"
