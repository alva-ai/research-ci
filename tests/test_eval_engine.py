import polars as pl
import pytest

from research_ci import CaseEvalTest, EvalPlan, EvalSpec, Slot, run_eval_plan


def _plan() -> EvalPlan:
    return EvalPlan(
        stage="data",
        baseline=(
            EvalSpec(
                "required_inst",
                expr=pl.col("inst").is_not_null(),
                failure_policy="stop",
            ),
        ),
        library=(
            EvalSpec(
                "prices_finite",
                expr=pl.all_horizontal(
                    pl.col("long_price").is_finite(),
                    pl.col("short_price").is_finite(),
                ),
                failure_policy="stop",
            ),
        ),
        case=(
            EvalSpec(
                "spread_positive",
                expr=pl.col("long_price") > pl.col("short_price"),
                failure_policy="stop",
            ),
        ),
        case_eval=(
            CaseEvalTest(
                name="spread_rule_has_a_counterexample",
                case="spread_positive",
                valid=lambda _: pl.DataFrame(
                    {"long_price": [101.0], "short_price": [100.0]}
                ),
                invalid=lambda _: pl.DataFrame(
                    {"long_price": [99.0], "short_price": [100.0]}
                ),
            ),
        ),
    )


def test_all_four_slots_run_and_main_table_is_unchanged() -> None:
    table = pl.DataFrame(
        {
            "inst": ["a", "b"],
            "long_price": [101.0, 99.0],
            "short_price": [100.0, 100.0],
        }
    )
    before = table.clone()

    result = run_eval_plan(table, _plan())

    assert [record.slot for record in result.records] == [
        Slot.BASELINE.value,
        Slot.BASELINE.value,
        Slot.LIBRARY.value,
        Slot.CASE.value,
        Slot.CASE_EVAL.value,
    ]
    assert result.records[0].name == "engine.baseline.general_missing_values"
    assert result.passed is False
    assert result.records[-1].status == "passed"
    assert table.equals(before)
    assert table.columns == ["inst", "long_price", "short_price"]


def test_empty_slot_is_rejected_even_when_other_slots_are_valid() -> None:
    plan = _plan()
    incomplete = EvalPlan(
        stage=plan.stage,
        baseline=plan.baseline,
        library=(),
        case=plan.case,
        case_eval=plan.case_eval,
    )

    with pytest.raises(ValueError, match="library"):
        run_eval_plan(pl.DataFrame({"inst": ["a"]}), incomplete)


def test_slot_can_be_filled_without_a_business_value() -> None:
    plan = EvalPlan(
        stage="data",
        baseline=(EvalSpec.noop("baseline.not_applicable"),),
        library=(EvalSpec.noop("library.not_selected"),),
        case=(
            EvalSpec(
                "case_rule",
                expr=pl.col("x") > 0,
                output_kind="none",
            ),
        ),
        case_eval=(
            CaseEvalTest.from_rows(
                name="case_rule_test",
                case="case_rule",
                valid={"x": [1]},
                invalid={"x": [-1]},
            ),
        ),
    )

    result = run_eval_plan(pl.DataFrame({"inst": ["a"], "x": [1]}), plan)

    assert result.passed
    assert result.records[1].value is None
    assert result.records[1].status == "not_applicable"


def test_engine_baseline_reports_null_nan_and_string_sentinels() -> None:
    table = pl.DataFrame(
        {
            "null_value": [None, 1.0],
            "nan_value": [float("nan"), 2.0],
            "infinite_value": [float("inf"), 3.0],
            "text_value": ["NAN", "None"],
        }
    )

    result = run_eval_plan(table, _plan_for_missing_values())
    record = result.records[0]

    assert record.name == "engine.baseline.general_missing_values"
    assert record.status == "failed"
    assert record.failure_policy == "record"
    assert record.details["totals"] == {
        "nan": 1,
        "non_finite": 2,
        "null": 1,
        "string_sentinel": 2,
    }
    assert table.columns == [
        "null_value",
        "nan_value",
        "infinite_value",
        "text_value",
    ]


def _plan_for_missing_values() -> EvalPlan:
    return EvalPlan(
        stage="data",
        baseline=(),
        library=(EvalSpec.noop("library.ok"),),
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
