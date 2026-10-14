import math

import polars as pl
import pytest

from research_ci.eval.baseline import general_missing_values
from research_ci.eval.library.quality import coverage, dispersion, finite_columns


def _run(spec: object, table: pl.DataFrame) -> object:
    runner = getattr(spec, "runner")
    assert runner is not None
    return runner(table)


def test_baseline_aggregates_all_missing_value_kinds() -> None:
    table = pl.DataFrame(
        {
            "number": [1.0, None, math.nan, math.inf, -math.inf],
            "text": ["value", " NAN ", "none", "NULL", None],
            "integer": [1, 2, None, 4, 5],
        }
    )

    observation = general_missing_values(table)

    assert observation.status == "failed"
    assert observation.details["execution_mode"] == "polars_select"
    assert observation.details["totals"] == {
        "null": 3,
        "nan": 1,
        "non_finite": 3,
        "string_sentinel": 3,
    }
    assert observation.details["columns_with_findings"]["number"] == {
        "null": 1,
        "nan": 1,
        "non_finite": 3,
        "string_sentinel": 0,
    }


def test_quality_metrics_do_not_materialize_series_as_python_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_to_list(_: pl.Series) -> list[object]:
        raise AssertionError("quality evals must stay on the Polars columnar path")

    monkeypatch.setattr(pl.Series, "to_list", reject_to_list)
    table = pl.DataFrame(
        {
            "varying": [1.0, 2.0, None, math.nan, math.inf],
            "constant": [3.0, 3.0, 3.0, None, 3.0],
        }
    )

    finite = _run(finite_columns(("varying", "constant")), table)
    covered = _run(coverage(("varying", "constant")), table)
    spread = _run(dispersion(("varying", "constant")), table)

    assert finite.status == "failed"
    assert finite.value == {"varying": 2}
    assert covered.value == {"varying": 0.4, "constant": 0.8}
    assert covered.details["usable_rows"] == {"varying": 2, "constant": 4}
    assert spread.value["varying"] == {
        "rows": 2,
        "unique": 2,
        "std": pytest.approx(math.sqrt(0.5)),
        "min": 1.0,
        "max": 2.0,
    }
    assert spread.value["constant"] == {
        "rows": 4,
        "unique": 1,
        "std": 0.0,
        "min": 3.0,
        "max": 3.0,
    }
    assert spread.details["constant_columns"] == ["constant"]


def test_coverage_keeps_convertible_value_compatibility() -> None:
    table = pl.DataFrame({"numeric_text": ["1.5", "bad", None, "2"]})

    observation = _run(coverage(("numeric_text",), minimum=0.5), table)

    assert observation.status == "passed"
    assert observation.value == {"numeric_text": 0.5}
