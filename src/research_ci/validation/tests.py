"""Running an eval's tests: its worked examples, and its behaviour on synthetic markets judged the way the engine
judges a submission (block means against the floor world, as a z score). The engine's own methods are held to theirs
in its CI; an idea's case evals are held to theirs before the engine runs them."""

import math
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from research_ci.api.evals import Case, Power, Property, Recovers, Size, Statistic
from research_ci.eval.synthetic import HOUR_NS, market
from research_ci.kernel.verdict.check import Check
from research_ci.kernel.verdict.judge import judge, per_t
from research_ci.kernel.verdict.permute import Permutation
from research_ci.kernel.verdict.power import Z_BASE

REPS = 60
SHOWN = 3


def obligations(module: ModuleType) -> list[Check]:
    """One check per case and property the module declares."""
    name = module.__name__.rsplit(".", 1)[-1]
    return run_tests(name, getattr(module, "CASES", []), getattr(module, "PROPERTIES", []))


def run_tests(name: str, cases: Sequence[Case], properties: Sequence[Property]) -> list[Check]:
    """One check per case and property, each named after ``name``. A failed one says what it expected and what it
    got."""
    return [_case(name, c) for c in cases] + [_property(name, p) for p in properties]


def label(name: str, test: Case | Property) -> str:
    """How a test is named in its check. It is the case's name, or the kind of property and its statistic."""
    if isinstance(test, Case):
        return f"{name}: {test.name}"
    return f"{name}: {type(test).__name__.lower()} of {getattr(test.of, '__name__', 'the eval')}"


def _case(module: str, case: Case) -> Check:
    named = label(module, case)
    try:
        got = case.run() if case.given is None else case.run(case.given)
    except Exception as exc:  # noqa: BLE001
        return Check(named, "fail", f"expected {_short(case.expect)}, got {type(exc).__name__}: {exc}")
    problem = differs(got, case.expect, case.tol)
    return Check(named, "fail" if problem else "pass", problem or "as worked by hand")


def _property(module: str, prop: Property) -> Check:
    named = label(module, prop)
    try:
        return _judged(named, prop)
    except Exception as exc:  # noqa: BLE001
        return Check(named, "fail", f"the statistic failed on a synthetic market: {type(exc).__name__}: {exc}")


def _judged(named: str, prop: Property) -> Check:
    if isinstance(prop, Recovers):
        values = [_mean(prop.of, market(prop.market, seed)) for seed in range(prop.markets)]
        got = float(np.nanmean(values))
        ok = abs(got - prop.expect) <= prop.tol
        return Check(named, "pass" if ok else "fail", f"mean {got:.4f}, expected {prop.expect} ± {prop.tol}")
    passed = [_z(prop.of, market(prop.market, seed), seed) >= Z_BASE for seed in range(prop.markets)]
    rate = float(np.mean(passed))
    beats = f"beats the floor world on {rate:.0%} of {prop.markets} markets"
    if isinstance(prop, Size):
        chance = _chance(REPS)
        allowed = chance + 3 * math.sqrt(chance * (1 - chance) / prop.markets)
        detail = f"{beats} without information (at most {allowed:.0%})"
        return Check(named, "pass" if rate <= allowed else "fail", detail)
    assert isinstance(prop, Power)
    detail = f"{beats} with ic {prop.market.ic} (at least {prop.rate:.0%})"
    return Check(named, "pass" if rate >= prop.rate else "fail", detail)


def _z(of: Statistic, table: pl.DataFrame, seed: int) -> float:
    shuffle = Permutation(table, None, "cluster", HOUR_NS).index
    result = judge(of, {"1h": (table, HOUR_NS)}, lambda h: lambda r: shuffle(seed * REPS + r), REPS, 5, "max")
    z = result["z"]
    return z if math.isfinite(z) else -math.inf


def _mean(of: Statistic, table: pl.DataFrame) -> float:
    return float(per_t(of, table)["value"].mean())  # type: ignore[arg-type]


def _chance(reps: int) -> float:
    return float(stats.t.sf(Z_BASE / math.sqrt(1 + 1 / reps), reps - 1))


def differs(got: Any, expect: Any, tol: float) -> str:
    """How ``got`` differs from ``expect``; empty when it does not. The text says what was expected and what came."""
    if isinstance(got, pl.LazyFrame):
        got = got.collect()
    if isinstance(expect, pl.DataFrame):
        return _frames(got, expect, tol)
    if isinstance(expect, int | float) and not isinstance(expect, bool):
        if isinstance(got, pl.DataFrame):
            got = _number(got)
        if isinstance(got, int | float | np.integer | np.floating) and not isinstance(got, bool):
            ok = abs(float(got) - expect) <= tol or (math.isnan(float(got)) and math.isnan(expect))
            return "" if ok else f"expected {expect:.6g}, got {float(got):.6g}"
        return f"expected the number {expect!r}, got {_short(got)}"
    return "" if got == expect else f"expected {_short(expect)}, got {_short(got)}"


def _number(frame: pl.DataFrame) -> Any:
    """A frame as a number: the value of a one-row frame of ``t`` and ``value``, else its number of rows."""
    if {"t", "value"} <= set(frame.columns) and frame.height == 1:
        return frame["value"][0]
    return frame.height


def _frames(got: Any, expect: pl.DataFrame, tol: float) -> str:
    if not isinstance(got, pl.DataFrame):
        return f"expected a frame with columns {', '.join(expect.columns)}, got {_short(got)}"
    missing = [c for c in expect.columns if c not in got.columns]
    if missing:
        return f"expected columns {', '.join(expect.columns)}, got {', '.join(got.columns) or 'none'}"
    if got.height != expect.height:
        return f"expected {expect.height} rows, got {got.height}: {_rows(got, expect.columns)}"
    shown = got.select(expect.columns)
    a, b = _comparable(shown), _comparable(expect)
    keys = [c for c in a.columns if not a.schema[c].is_float()]
    order = [*keys, *(c for c in a.columns if c not in keys)]
    a = a.with_row_index("_i").sort(order, nulls_last=True)
    b = b.with_row_index("_i").sort(order, nulls_last=True)
    same = pl.DataFrame([_same(a[c], b[c], tol).alias(c) for c in expect.columns])
    wrong = ~same.select(pl.all_horizontal(pl.all())).to_series()
    if not wrong.any():
        return ""
    i = int(wrong.arg_true()[0])
    mine, theirs = shown.row(int(a["_i"][i]), named=True), expect.row(int(b["_i"][i]), named=True)
    where = ", ".join(f"{k}={theirs[k]}" for k in keys) or f"row {i}"
    said = "; ".join(f"{c}: expected {theirs[c]}, got {mine[c]}" for c in expect.columns if not same[c][i])
    return f"{int(wrong.sum())} of {a.height} rows differ; at {where}: {said}"


def _comparable(frame: pl.DataFrame) -> pl.DataFrame:
    """Times as instants in ns (a naive time read as UTC), categories as text."""
    return frame.select(
        pl.col(c).dt.epoch("ns")
        if isinstance(dtype, pl.Datetime)
        else pl.col(c).cast(pl.String)
        if isinstance(dtype, pl.Categorical | pl.Enum)
        else pl.col(c)
        for c, dtype in frame.schema.items()
    )


def _same(a: pl.Series, b: pl.Series, tol: float) -> pl.Series:
    """Whether each pair of values agrees: integers exactly, other numbers within ``tol``, anything else as text."""
    if a.dtype.is_integer() and b.dtype.is_integer():
        return a.eq_missing(b)
    if a.dtype.is_numeric() and b.dtype.is_numeric():
        x, y = a.cast(pl.Float64), b.cast(pl.Float64)
        close = ((x - y).abs() <= tol) | (x.is_nan() & y.is_nan())
        return (close | (x.is_null() & y.is_null())).fill_null(False)
    if a.dtype == b.dtype:
        return a.eq_missing(b)
    return a.cast(pl.String).eq_missing(b.cast(pl.String))


def _rows(frame: pl.DataFrame, columns: Sequence[str]) -> str:
    kept = [c for c in columns if c in frame.columns]
    shown = frame.select(kept).head(SHOWN).rows()
    more = f" and {frame.height - SHOWN} more" if frame.height > SHOWN else ""
    return ("; ".join(", ".join(str(v) for v in row) for row in shown) + more) if shown else "none"


def _short(x: Any) -> str:
    if isinstance(x, pl.DataFrame):
        return f"a frame of {x.height} rows with columns {', '.join(x.columns) or 'none'}"
    text = repr(x)
    return text if len(text) <= 80 else text[:77] + "..."
