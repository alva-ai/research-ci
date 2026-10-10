"""Running a method's obligations: its worked examples, and its behaviour on synthetic markets judged the way the
engine judges a submission (block means against the floor world, as a z score)."""

import math
from types import ModuleType
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from research_ci.api.methods import Case, Power, Property, Recovers, Size, Statistic
from research_ci.eval.synthetic import HOUR_NS, market
from research_ci.kernel.verdict.check import Check
from research_ci.kernel.verdict.judge import judge, per_t
from research_ci.kernel.verdict.permute import Permutation
from research_ci.kernel.verdict.power import Z_BASE

REPS = 60


def obligations(module: ModuleType) -> list[Check]:
    """One check per case and property the module declares."""
    name = module.__name__.rsplit(".", 1)[-1]
    out = [_case(name, c) for c in getattr(module, "CASES", [])]
    return out + [_property(name, p) for p in getattr(module, "PROPERTIES", [])]


def _case(module: str, case: Case) -> Check:
    label = f"{module}: {case.name}"
    try:
        got = case.run()
    except Exception as exc:  # noqa: BLE001
        return Check(label, "fail", f"{type(exc).__name__}: {exc}")
    problem = _differs(got, case.expect, case.tol)
    return Check(label, "fail" if problem else "pass", problem or "as worked by hand")


def _property(module: str, prop: Property) -> Check:
    label = f"{module}: {type(prop).__name__.lower()} of {prop.of.__name__}"
    if isinstance(prop, Recovers):
        values = [_mean(prop.of, market(prop.market, seed)) for seed in range(prop.markets)]
        got = float(np.nanmean(values))
        ok = abs(got - prop.expect) <= prop.tol
        return Check(label, "pass" if ok else "fail", f"mean {got:.4f}, expected {prop.expect} ± {prop.tol}")
    passed = [_z(prop.of, market(prop.market, seed), seed) >= Z_BASE for seed in range(prop.markets)]
    rate = float(np.mean(passed))
    beats = f"beats the floor world on {rate:.0%} of {prop.markets} markets"
    if isinstance(prop, Size):
        chance = _chance(REPS)
        allowed = chance + 3 * math.sqrt(chance * (1 - chance) / prop.markets)
        detail = f"{beats} without information (at most {allowed:.0%})"
        return Check(label, "pass" if rate <= allowed else "fail", detail)
    assert isinstance(prop, Power)
    detail = f"{beats} with ic {prop.market.ic} (at least {prop.rate:.0%})"
    return Check(label, "pass" if rate >= prop.rate else "fail", detail)


def _z(of: Statistic, table: pl.DataFrame, seed: int) -> float:
    shuffle = Permutation(table, None, "cluster", HOUR_NS).index
    result = judge(of, {"1h": (table, HOUR_NS)}, lambda h: lambda r: shuffle(seed * REPS + r), REPS, 5, "max")
    z = result["z"]
    return z if math.isfinite(z) else -math.inf


def _mean(of: Statistic, table: pl.DataFrame) -> float:
    return float(per_t(of, table)["value"].mean())  # type: ignore[arg-type]


def _chance(reps: int) -> float:
    return float(stats.t.sf(Z_BASE / math.sqrt(1 + 1 / reps), reps - 1))


def _differs(got: Any, expect: Any, tol: float) -> str:
    if isinstance(expect, pl.DataFrame):
        if not isinstance(got, pl.DataFrame):
            return f"expected a frame, got {type(got).__name__}"
        if got.columns != expect.columns or got.height != expect.height:
            return f"expected columns {expect.columns} and {expect.height} rows, got {got.columns} and {got.height}"
        for column in expect.columns:
            a, b = got[column], expect[column]
            if a.dtype.is_numeric() and b.dtype.is_numeric():
                if not np.allclose(a.to_numpy(), b.to_numpy(), atol=tol, rtol=0, equal_nan=True):
                    return f"{column}: got {a.to_list()}, expected {b.to_list()}"
            elif a.to_list() != b.to_list():
                return f"{column}: got {a.to_list()}, expected {b.to_list()}"
        return ""
    if isinstance(expect, float) and isinstance(got, int | float):
        return "" if abs(got - expect) <= tol else f"got {got}, expected {expect}"
    return "" if got == expect else f"got {got!r}, expected {expect!r}"
