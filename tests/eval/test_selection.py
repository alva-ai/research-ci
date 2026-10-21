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
    SignalSet,
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


def test_data_cross_section_selection_requires_an_explicit_entity_universe() -> None:
    declaration = DataSet(
        keys=("event_time", "symbol"),
        required=("event_time", "symbol", "x"),
        numeric=("x",),
        time_key="event_time",
    )

    with pytest.raises(LibrarySelectionError, match="expected_entities"):
        expand_library(
            declaration,
            LibrarySelection(
                (
                    SelectedMethod(
                        "library.temporal.cross_section_completeness",
                        reason="audit the declared daily symbol universe",
                    ),
                )
            ),
        )

    specs = expand_library(
        declaration,
        LibrarySelection(
            (
                SelectedMethod(
                    "library.temporal.cross_section_completeness",
                    reason="audit the declared daily symbol universe",
                    parameters={
                        "expected_entities": ("A", "B"),
                        "minimum_complete_ratio": 1.0,
                    },
                ),
            )
        ),
    )
    result = run_eval_plan(
        pl.DataFrame(
            {
                "event_time": [1, 1, 2],
                "symbol": ["A", "B", "A"],
                "x": [1.0, 2.0, 3.0],
            }
        ),
        _plan("data", specs),
    )
    record = next(
        record
        for record in result.records
        if record.library_method.endswith("cross_section_completeness")
    )

    assert record.status == "failed"
    assert record.value == 0.5
    assert record.details["entity_key"] == "symbol"


def test_feat_expands_only_structural_methods() -> None:
    declaration = FeatureSet(
        keys=("event_time",),
        features=("x", "y"),
    )
    selection = LibrarySelection(
        (
            SelectedMethod(
                "library.structure.pairwise_correlation",
                reason="screen linear redundancy",
            ),
        )
    )

    specs = expand_library(declaration, selection)

    assert len(specs) == 1
    assert specs[0].library_method == "library.structure.pairwise_correlation"


def test_sig_expands_target_methods_across_features_and_horizons() -> None:
    declaration = SignalSet(
        keys=("event_time",),
        features=("x", "y"),
        targets={"1h": "return_1h", "6h": "return_6h"},
    )
    specs = expand_library(
        declaration,
        LibrarySelection(
            (
                SelectedMethod(
                    "library.association.pearson_ic",
                    reason="measure linear feature/target association",
                ),
            )
        ),
    )

    assert len(specs) == 4
    assert all(
        spec.library_method == "library.association.pearson_ic" for spec in specs
    )
    assert {spec.execution_mode for spec in specs} == {"polars_join_group"}


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
    declaration = SignalSet(
        keys=("event_time",),
        features=("x",),
        targets={"1h": "return_1h"},
        label_table="labels",
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
        _plan("sig", specs),
        EvalContext(tables={"labels": labels}),
    )

    record = next(record for record in result.records if "pearson_ic" in record.name)
    assert record.status == "passed"
    assert record.value == 1.0
    assert record.execution_mode == "polars_join_group"
    assert record.selection_reason == "evaluate the declared one-hour label"
    assert features.columns == ["event_time", "x"]


def test_assessment_target_selection_uses_approved_feature_grid() -> None:
    assessment = pl.DataFrame(
        {
            "thesis_id": [1001],
            "author_version_id": [2001],
            "as_of": ["2026-10-15T09:16:39Z"],
            "assessment_id": ["1001:2001:2026-10-15T09:16:39Z"],
            "overall_status": ["insufficient"],
        }
    )
    assessment_before = assessment.clone()
    features = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    labels = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "proxy_target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    declaration = SignalSet(
        keys=("thesis_id", "author_version_id", "as_of"),
        features=("x",),
        targets={"P1": "proxy_target"},
        label_table="labels",
    )
    specs = expand_library(
        declaration,
        LibrarySelection(
            (
                SelectedMethod(
                    "library.association.pearson_ic",
                    reason="evaluate the declared premise proxy",
                ),
            )
        ),
    )
    status_case = EvalSpec(
        "case.assessment_status",
        expr=pl.col("overall_status") == "insufficient",
    )

    result = run_eval_plan(
        assessment,
        EvalPlan(
            stage="sig",
            baseline=(),
            library=specs,
            case=(status_case,),
            case_eval=(
                CaseEvalTest.from_rows(
                    name="case.assessment_status.counterexample",
                    case=status_case.name,
                    valid={"overall_status": ["insufficient"]},
                    invalid={"overall_status": ["buy"]},
                ),
            ),
        ),
        EvalContext(
            tables={
                "labels": labels,
                "__rci_feature_table": features.select("event_time", "x"),
            }
        ),
    )

    record = next(
        record
        for record in result.records
        if record.library_method.endswith("pearson_ic")
    )
    assert record.status == "passed"
    assert record.value == 1.0
    assert record.details["context_source"] == "__rci_feature_table"
    assert record.details["join_on"] == ("event_time",)
    assert record.details["joined_rows"] == 4
    assert assessment.equals(assessment_before)


def test_agent_cannot_select_target_method_in_targetless_sig() -> None:
    declaration = SignalSet(
        keys=("event_time",),
        features=("x",),
        hypothesis="extreme values should mean revert",
    )
    selection = LibrarySelection(
        (
            SelectedMethod(
                "library.association.pearson_ic",
                reason="try a target method",
            ),
        )
    )

    with pytest.raises(LibrarySelectionError, match="requires SignalSet.targets"):
        expand_library(declaration, selection)


def test_agent_cannot_select_predictive_method_in_feat() -> None:
    declaration = FeatureSet(keys=("event_time",), features=("x",))
    selection = LibrarySelection(
        (
            SelectedMethod(
                "library.association.pearson_ic",
                reason="this belongs to Sig",
            ),
        )
    )

    with pytest.raises(LibrarySelectionError, match="unavailable in stage 'feat'"):
        expand_library(declaration, selection)


def test_catalog_target_method_expands_through_declared_label_sidecar() -> None:
    declaration = SignalSet(
        keys=("event_time",),
        features=("x",),
        targets={"1h": "return_1h"},
        label_table="labels",
    )
    selection = LibrarySelection(
        (
            SelectedMethod(
                "library.association.spearman_ic",
                reason="evaluate the explicit label sidecar",
            ),
        )
    )

    specs = expand_library(declaration, selection)

    assert len(specs) == 1
    assert specs[0].context_runner is not None
    assert specs[0].execution_mode == "polars_join_group"


def test_grouped_sig_method_reads_evaluation_group_from_label_sidecar() -> None:
    scores = pl.DataFrame(
        {
            "event_time": list(range(12)),
            "signal": [float(value % 4) for value in range(12)],
        }
    )
    labels = pl.DataFrame(
        {
            "event_time": list(range(12)),
            "block": [value // 4 for value in range(12)],
            "return_1h": [
                0.0,
                1.0,
                2.0,
                3.0,
                3.0,
                2.0,
                1.0,
                0.0,
                0.0,
                1.0,
                3.0,
                2.0,
            ],
        }
    )
    declaration = SignalSet(
        keys=("event_time",),
        features=("x",),
        signals=("signal",),
        targets={"1h": "return_1h"},
        label_table="labels",
        evaluation_by="block",
    )
    specs = expand_library(
        declaration,
        LibrarySelection(
            (
                SelectedMethod(
                    "library.association.icir",
                    on="signals",
                    reason="summarise IC over explicit time blocks",
                    parameters={"min_group_size": 3},
                ),
            )
        ),
    )

    result = run_eval_plan(
        scores,
        _plan("sig", specs),
        EvalContext(tables={"labels": labels}),
    )

    record = next(record for record in result.records if "icir" in record.name)
    assert record.status == "passed"
    assert record.details["joined_rows"] == 12
    assert record.details["group"] == "block"


def test_walk_forward_selection_reads_fold_from_label_sidecar() -> None:
    scores = pl.DataFrame(
        {
            "event_time": list(range(9)),
            "x": [1.0] * 9,
            "signal": [1.0, 2.0, 3.0] * 3,
        }
    )
    labels = pl.DataFrame(
        {
            "event_time": list(range(9)),
            "fold": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "return_1h": [1.0, 2.0, 3.0] * 3,
        }
    )
    declaration = SignalSet(
        keys=("event_time",),
        features=("x",),
        signals=("signal",),
        targets={"1h": "return_1h"},
        label_table="labels",
        evaluation_by="fold",
    )
    specs = expand_library(
        declaration,
        LibrarySelection(
            (
                SelectedMethod(
                    "library.stability.walk_forward_association",
                    on="signals",
                    reason="audit frozen out-of-sample fold evidence",
                    parameters={"min_fold_size": 3},
                ),
            )
        ),
    )

    result = run_eval_plan(
        scores,
        _plan("sig", specs),
        EvalContext(tables={"labels": labels}),
    )
    record = next(
        record
        for record in result.records
        if record.library_method.endswith("walk_forward_association")
    )

    assert record.status == "passed"
    assert record.details["valid_folds"] == 3
    assert record.details["context_table"] == "labels"
    assert record.details["joined_rows"] == 9
    assert record.execution_mode == "polars_join_group"
    assert scores.columns == ["event_time", "x", "signal"]


def test_model_selection_uses_label_sidecar_and_explicit_class_semantics() -> None:
    declaration = SignalSet(
        keys=("event_time",),
        features=("x",),
        signals=("probability",),
        targets={"1h": "direction"},
        label_table="labels",
    )
    missing_label = LibrarySelection(
        (
            SelectedMethod(
                "library.model.classification_probability_diagnostics",
                on="signals",
                reason="audit submitted class probabilities",
            ),
        )
    )
    with pytest.raises(LibrarySelectionError, match="positive_label"):
        expand_library(declaration, missing_label)

    specs = expand_library(
        declaration,
        LibrarySelection(
            (
                SelectedMethod(
                    "library.model.classification_probability_diagnostics",
                    on="signals",
                    reason="audit submitted class probabilities",
                    parameters={"positive_label": "up", "calibration_bins": 2},
                ),
            )
        ),
    )
    scores = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "x": [1.0] * 4,
            "probability": [0.1, 0.2, 0.8, 0.9],
        }
    )
    labels = pl.DataFrame(
        {"event_time": [1, 2, 3, 4], "direction": ["down", "down", "up", "up"]}
    )
    result = run_eval_plan(
        scores,
        _plan("sig", specs),
        EvalContext(tables={"labels": labels}),
    )
    record = next(
        record
        for record in result.records
        if record.library_method.endswith("classification_probability_diagnostics")
    )

    assert record.status == "passed"
    assert record.value["roc_auc"] == 1.0
    assert record.details["context_table"] == "labels"
    assert record.execution_mode == "matrix_backend"
    assert scores.columns == ["event_time", "x", "probability"]
