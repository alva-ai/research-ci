"""Every eval method's obligations hold, and the obligations can fail."""

import importlib
import pkgutil
from datetime import UTC, datetime
from types import ModuleType

import polars as pl
import pytest

import research_ci.eval
from research_ci.api.methods import Case, Market, Power, Recovers
from research_ci.eval.obligations import differs, obligations, run_tests
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.statistics.spread import quantile_spread

DECLARING = ("data", "statistics", "structure", "calibration")


def _modules() -> list[ModuleType]:
    found = pkgutil.walk_packages(research_ci.eval.__path__, prefix="research_ci.eval.")
    return [importlib.import_module(m.name) for m in found if not m.ispkg]


METHODS = [m for m in _modules() if m.__name__.split(".")[2] in DECLARING]


@pytest.mark.parametrize("module", METHODS, ids=lambda m: m.__name__.removeprefix("research_ci.eval."))
def test_every_method_declares_and_keeps_its_obligations(module: ModuleType) -> None:
    assert getattr(module, "CASES", None), f"{module.__name__} declares no worked examples"
    failed = [c for c in obligations(module) if c.status != "pass"]
    assert not failed, failed


def test_obligations_fail_when_the_method_is_wrong() -> None:
    def ignores_the_score(table: pl.DataFrame) -> pl.DataFrame:
        return table.group_by("t").agg(pl.col("y").mean().alias("value"))

    def doubled(table: pl.DataFrame) -> pl.DataFrame:
        return rank_ic(table).with_columns(pl.col("value") * 2)

    wrong = ModuleType("wrong")
    properties = [
        Power(ignores_the_score, markets=10),
        Recovers(doubled, Market(ic=0.1, tail=None), expect=0.1, tol=0.01, markets=5),
    ]
    vars(wrong).update(CASES=[Case("a wrong answer", lambda: 1.0, 2.0)], PROPERTIES=properties)
    assert [c.status for c in obligations(wrong)] == ["fail", "fail", "fail"]


SIX = pl.DataFrame(
    {
        "t": [datetime(2026, 1, 1)] * 6,
        "entity": list("abcdef"),
        "score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "y": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    }
)


def _thirds(table: pl.DataFrame) -> pl.DataFrame:
    return quantile_spread(table, share=1 / 3)


def test_a_case_takes_frames_in_and_frames_or_numbers_out() -> None:
    """``given`` is the eval's input; an expected frame matches in any row order, on its own columns, with times as
    instants; an expected number reads a one-row frame of t and value as its value, and any other frame as its rows."""
    aware = pl.DataFrame({"t": [datetime(2026, 1, 1, tzinfo=UTC)], "value": [0.4]})
    naive = pl.DataFrame({"value": [0.4], "t": [datetime(2026, 1, 1)]})
    flagged = pl.DataFrame({"entity": ["b", "a"], "n": [2, 1]})
    cases = [
        Case("a number from a one-row frame", _thirds, 0.4, given=SIX),
        Case("a frame, times naive or aware, columns in any order", _thirds, naive, given=SIX),
        Case("rows in any order, other columns ignored", lambda f: f.with_columns(x=1), flagged, given=flagged[::-1]),
        Case("a number of rows", lambda f: f, 2, given=flagged),
    ]
    assert [c.status for c in run_tests("thirds", cases, [])] == ["pass"] * 4
    assert differs(_thirds(SIX), aware, 1e-9) == ""


def test_a_failed_case_says_what_it_expected_and_what_it_got() -> None:
    wrong = pl.DataFrame({"t": [datetime(2026, 1, 1)], "value": [0.3]})
    found = run_tests(
        "thirds",
        [
            Case("a wrong frame", _thirds, wrong, given=SIX),
            Case("a wrong number", _thirds, 0.3, given=SIX),
            Case("a missing column", _thirds, pl.DataFrame({"t": [1], "level": [0.3]}), given=SIX),
            Case("an error", _thirds, 0.3, given=SIX.drop("y")),
        ],
        [],
    )
    assert [c.status for c in found] == ["fail"] * 4
    assert found[0].detail == "1 of 1 rows differ; at t=2026-01-01 00:00:00: value: expected 0.3, got 0.4"
    assert found[1].detail == "expected 0.3, got 0.4"
    assert found[2].detail == "expected columns t, level, got t, value"
    assert found[3].detail.startswith("expected 0.3, got ColumnNotFoundError")
