"""Every eval method's obligations hold, and the obligations can fail."""

import importlib
import pkgutil
from types import ModuleType

import polars as pl
import pytest

import research_ci.eval
from research_ci.api.methods import Case, Market, Power, Recovers
from research_ci.eval.obligations import obligations
from research_ci.eval.statistics.ic import rank_ic

DECLARING = ("statistics", "structure", "calibration")


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
