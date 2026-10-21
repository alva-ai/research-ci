import polars as pl
import pytest

from research_ci import CaseEvalTest, EvalPlan, EvalSpec
from research_ci.eval.defaults import DEFAULT_METHODS
from research_ci.eval.library import METHODS
from research_ci.eval.library.association import pearson_ic
from research_ci.eval.library.model import (
    classification_probability_diagnostics,
    regression_diagnostics,
)
from research_ci.eval.library.quality import finite_columns, required_columns
from research_ci.eval.library.stability import walk_forward_association
from research_ci.eval.library.structure import pairwise_correlation
from research_ci.eval.library.temporal import cross_section_completeness
from research_ci.eval.runner import run_eval_plan


def _plan(*specs: EvalSpec, stage: str = "feat") -> EvalPlan:
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=specs or (EvalSpec.noop("library.placeholder"),),
        case=(EvalSpec("case.ok", expr=pl.lit(True)),),
        case_eval=(
            CaseEvalTest.from_rows(
                name="case.ok.test",
                case="case.ok",
                valid={"x": [1]},
                invalid={"x": [-1]},
            ),
        ),
    )


def test_quality_library_reports_required_and_finite_values() -> None:
    table = pl.DataFrame({"x": [1.0, float("inf")], "y": [1.0, 2.0]})

    result = run_eval_plan(
        table,
        _plan(
            required_columns(["x", "y"]),
            finite_columns(["x", "y"]),
        ),
    )

    assert result.records[1].status == "passed"
    assert result.records[2].status == "failed"
    assert result.records[2].details == {"columns": ("x", "y"), "rows": 2}


def test_association_and_structure_methods_are_metrics() -> None:
    table = pl.DataFrame(
        {"score": [1.0, 2.0, 3.0], "target": [2.0, 4.0, 6.0], "other": [3.0, 2.0, 1.0]}
    )

    association = run_eval_plan(
        table,
        _plan(pearson_ic("score", "target"), stage="sig"),
    )
    structure = run_eval_plan(
        table,
        _plan(pairwise_correlation("score", "other")),
    )

    assert association.records[1].value == 1.0
    assert association.records[1].output_kind == "metric"
    assert structure.records[1].value == -1.0


def test_pairwise_correlation_can_record_a_redundancy_threshold() -> None:
    table = pl.DataFrame(
        {
            "x": [1.0, 1.0, 1.0],
            "left": [1.0, 2.0, 3.0],
            "right": [-1.0, -2.0, -3.0],
        }
    )

    case = EvalSpec("case.positive", expr=pl.col("x") > 0)
    plan = EvalPlan(
        stage="feat",
        baseline=(),
        library=(pairwise_correlation("left", "right", maximum_absolute=0.8),),
        case=(case,),
        case_eval=(
            CaseEvalTest.from_rows(
                name="case.positive.counterexample",
                case=case.name,
                valid={"x": [1]},
                invalid={"x": [-1]},
            ),
        ),
    )
    result = run_eval_plan(table, plan)

    record = result.records[1]
    assert record.status == "failed"
    assert record.failure_policy == "record"
    assert record.value == -1.0
    assert record.output_kind == "findings"
    assert record.details["reason_code"] == "HIGH_REDUNDANCY"
    assert result.passed


def test_association_is_explicitly_not_applicable_without_a_target() -> None:
    result = run_eval_plan(
        pl.DataFrame({"score": [1.0, 2.0]}),
        _plan(pearson_ic("score", "target"), stage="sig"),
    )

    assert result.records[1].status == "not_applicable"
    assert result.records[1].details == {"missing_columns": ["target"]}


def test_cross_section_completeness_reports_contract_level_findings() -> None:
    table = pl.DataFrame(
        {
            "event_time": [1, 1, 2, 2, 2],
            "symbol": ["A", "B", "A", "A", "C"],
            "close": [10.0, 20.0, 11.0, None, 30.0],
            "x": [1, 1, 1, 1, 1],
        }
    )

    result = run_eval_plan(
        table,
        _plan(
            cross_section_completeness(
                ("event_time",),
                "symbol",
                ("A", "B"),
                ("close",),
                minimum_complete_ratio=1.0,
            ),
            stage="data",
        ),
    )
    record = result.records[1]

    assert record.status == "failed"
    assert record.value == 0.5
    assert record.details["complete_sections"] == 1
    assert record.details["missing_entity_slots"] == 1
    assert record.details["duplicate_entity_rows"] == 1
    assert record.details["unexpected_entities"] == ["C"]
    assert record.details["missing_required_rows"] == 1
    assert set(record.details["reason_codes"]) == {
        "DUPLICATE_KEYS",
        "INCOMPLETE_CROSS_SECTION",
        "LOW_CROSS_SECTION_COMPLETENESS",
        "MISSING_REQUIRED_VALUES",
        "UNEXPECTED_ENTITIES",
    }


def test_walk_forward_association_reports_fold_stability() -> None:
    table = pl.DataFrame(
        {
            "fold": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "score": [1.0, 2.0, 3.0] * 3,
            "target": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "x": [1] * 9,
        }
    )

    result = run_eval_plan(
        table,
        _plan(
            walk_forward_association(
                "score",
                "target",
                by="fold",
                minimum_sign_consistency=0.8,
            ),
            stage="sig",
        ),
    )
    record = result.records[1]

    assert record.status == "failed"
    assert record.value == pytest.approx(1 / 3)
    assert record.details["median_association"] == 1.0
    assert record.details["valid_folds"] == 3
    assert record.details["sign_consistency"] == pytest.approx(2 / 3)
    assert record.details["worst_directional_fold"]["fold"] == 3
    assert record.details["reason_codes"] == ["WALK_FORWARD_UNSTABLE"]


def test_regression_diagnostics_measure_submitted_predictions() -> None:
    table = pl.DataFrame(
        {
            "prediction": [0.1, -0.2, 0.5, -0.3],
            "target": [0.0, -0.1, 0.4, -0.4],
            "x": [1, 1, 1, 1],
        }
    )

    result = run_eval_plan(
        table,
        _plan(
            regression_diagnostics(
                "prediction",
                "target",
                maximum_mae=0.05,
            ),
            stage="sig",
        ),
    )
    record = result.records[1]

    assert record.status == "failed"
    assert record.value["mae"] == pytest.approx(0.1)
    assert record.value["rmse"] == pytest.approx(0.1)
    assert record.value["mean_error"] == pytest.approx(0.05)
    assert record.value["sign_accuracy"] == 0.75
    assert record.details["reason_code"] == "MODEL_PERFORMANCE_BELOW_THRESHOLD"


def test_classification_probability_diagnostics_include_calibration() -> None:
    table = pl.DataFrame(
        {
            "probability": [0.1, 0.2, 0.8, 0.9],
            "target": ["no", "no", "yes", "yes"],
            "x": [1, 1, 1, 1],
        }
    )

    result = run_eval_plan(
        table,
        _plan(
            classification_probability_diagnostics(
                "probability",
                "target",
                positive_label="yes",
                calibration_bins=2,
            ),
            stage="sig",
        ),
    )
    record = result.records[1]

    assert record.status == "passed"
    assert record.value["roc_auc"] == 1.0
    assert record.value["brier_score"] == pytest.approx(0.025)
    assert record.value["accuracy"] == 1.0
    assert record.value["expected_calibration_error"] == pytest.approx(0.15)
    assert len(record.details["calibration_bins"]) == 2


def test_new_methodologies_reject_ambiguous_or_non_finite_parameters() -> None:
    with pytest.raises(TypeError, match="iterable of IDs"):
        cross_section_completeness(
            ("event_time",),
            "symbol",
            "AB",
            ("close",),
        )
    with pytest.raises(ValueError, match="maximum_std"):
        walk_forward_association(
            "score",
            "target",
            by="fold",
            maximum_std=float("nan"),
        )
    with pytest.raises(ValueError, match="non-null positive_label"):
        classification_probability_diagnostics(
            "probability",
            "target",
            positive_label=None,
        )
    with pytest.raises(ValueError, match="min_rows"):
        regression_diagnostics("prediction", "target", min_rows=2.5)


def test_stage_proposals_reference_registered_methods() -> None:
    registered = {method.name for method in METHODS}

    assert set().union(*DEFAULT_METHODS.values()) <= registered
    assert {
        "library.model.regression_diagnostics",
        "library.model.classification_probability_diagnostics",
    } <= registered
