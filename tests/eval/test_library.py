import polars as pl

from research_ci import CaseEvalTest, EvalPlan, EvalSpec
from research_ci.eval.defaults import DEFAULT_METHODS
from research_ci.eval.library import METHODS
from research_ci.eval.library.association import pearson_ic
from research_ci.eval.library.quality import finite_columns, required_columns
from research_ci.eval.library.structure import pairwise_correlation
from research_ci.eval.runner import run_eval_plan


def _plan(*specs: EvalSpec) -> EvalPlan:
    return EvalPlan(
        stage="feat",
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

    result = run_eval_plan(
        table,
        _plan(
            pearson_ic("score", "target"),
            pairwise_correlation("score", "other"),
        ),
    )

    assert result.records[1].value == 1.0
    assert result.records[1].output_kind == "metric"
    assert result.records[2].value == -1.0


def test_association_is_explicitly_not_applicable_without_a_target() -> None:
    result = run_eval_plan(
        pl.DataFrame({"score": [1.0, 2.0]}),
        _plan(pearson_ic("score", "target")),
    )

    assert result.records[1].status == "not_applicable"
    assert result.records[1].details == {"missing_columns": ["target"]}


def test_stage_proposals_reference_registered_methods() -> None:
    registered = {method.name for method in METHODS}

    assert set().union(*DEFAULT_METHODS.values()) <= registered
