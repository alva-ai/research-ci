from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.api.typing import SeriesGroupBy

from alpha_ci.kernel.tables import field_table
from alpha_ci.kernel.time.stepper import PanelFormat, _compare, check_leaks, step

HOUR = pd.Timedelta(hours=1)
START = pd.Timestamp("2026-01-01", tz="UTC")
ENTITIES = ["a", "b", "c"]
DECISIONS = pd.DatetimeIndex([START + i * HOUR for i in range(60, 200, 7)])


def _fields() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for entity in ENTITIES:
        close = 100 + np.cumsum(rng.normal(loc=0.3, size=200))
        for i, value in enumerate(close):
            ts = START + i * HOUR
            rows.append((entity, "close", ts, ts, value, 0))
        for i in range(0, 200, 24):
            ts = START + i * HOUR
            rows.append((entity, "inventory", ts, ts + 30 * HOUR, float(rng.integers(50, 150)), 0))
    return field_table(
        pd.DataFrame(rows, columns=["entity", "field", "event_time", "available_at", "value", "vintage"])
    )


def _pd(panel: pd.DataFrame, column: str) -> SeriesGroupBy:
    return panel[column].groupby(level="entity")


def _pl(expr: pl.Expr) -> Callable[[pl.DataFrame], pl.DataFrame]:
    return lambda p: p.select("time", "entity", expr.alias("value"))


LOOKAHEAD: dict[PanelFormat, dict[str, Callable[[Any], Any]]] = {
    "pandas": {
        "shift_minus_one": lambda p: _pd(p, "close").shift(-1),
        "full_sample_zscore": lambda p: (p["close"] - p["close"].mean()) / p["close"].std(),
        "centered_rolling": lambda p: _pd(p, "close").transform(lambda s: s.rolling(5, center=True).mean()),
        "backfill_sparse_field": lambda p: _pd(p, "inventory").bfill(),
        "normalize_by_future_max": lambda p: p["close"] / _pd(p, "close").transform("max"),
    },
    "polars": {
        "shift_minus_one": _pl(pl.col("close").shift(-1).over("entity")),
        "full_sample_zscore": _pl((pl.col("close") - pl.col("close").mean()) / pl.col("close").std()),
        "centered_rolling": _pl(pl.col("close").rolling_mean(5, center=True).over("entity")),
        "backfill_sparse_field": _pl(pl.col("inventory").backward_fill().over("entity")),
        "normalize_by_future_max": _pl(pl.col("close") / pl.col("close").max().over("entity")),
    },
}

CAUSAL: dict[PanelFormat, dict[str, Callable[[Any], Any]]] = {
    "pandas": {
        "trailing_rolling": lambda p: _pd(p, "close").transform(lambda s: s.rolling(10).mean()),
        "expanding_mean": lambda p: _pd(p, "close").transform(lambda s: s.expanding().mean()),
        "ewm": lambda p: _pd(p, "close").transform(lambda s: s.ewm(halflife=5).mean()),
        "pct_change": lambda p: _pd(p, "close").pct_change(),
        "forward_fill_sparse_field": lambda p: _pd(p, "inventory").ffill(),
    },
    "polars": {
        "trailing_rolling": _pl(pl.col("close").rolling_mean(10).over("entity")),
        "expanding_mean": _pl((pl.col("close").cum_sum() / pl.col("close").cum_count()).over("entity")),
        "ewm": _pl(pl.col("close").ewm_mean(half_life=5).over("entity")),
        "pct_change": _pl(pl.col("close").pct_change().over("entity")),
        "forward_fill_sparse_field": _pl(pl.col("inventory").forward_fill().over("entity")),
    },
}

FORMATS: list[PanelFormat] = ["pandas", "polars"]
CASES = [(fmt, name) for fmt in FORMATS for name in sorted(LOOKAHEAD["pandas"])]
CLEAN = [(fmt, name) for fmt in FORMATS for name in sorted(CAUSAL["pandas"])]


@pytest.mark.parametrize(("fmt", "name"), CASES)
def test_lookahead_is_caught(fmt: PanelFormat, name: str) -> None:
    report = check_leaks(LOOKAHEAD[fmt][name], _fields(), DECISIONS, panel_format=fmt)
    assert report.kind == "lookahead"
    assert report.mismatches > 0
    assert report.first is not None


@pytest.mark.parametrize(("fmt", "name"), CLEAN)
def test_causal_features_are_clean(fmt: PanelFormat, name: str) -> None:
    report = check_leaks(CAUSAL[fmt][name], _fields(), DECISIONS, panel_format=fmt)
    assert report.kind == "none", report
    assert report.mismatches == 0


@pytest.mark.parametrize("name", ["trailing_rolling", "expanding_mean", "pct_change", "forward_fill_sparse_field"])
def test_pandas_and_polars_give_the_same_values(name: str) -> None:
    fields = _fields()
    from_pandas = step(CAUSAL["pandas"][name], fields, DECISIONS, panel_format="pandas")
    from_polars = step(CAUSAL["polars"][name], fields, DECISIONS, panel_format="polars")
    assert _compare(from_pandas, from_polars.rename({"value": from_pandas.columns[-1]})) == []


@pytest.mark.parametrize("fmt", FORMATS)
def test_short_lookback_is_not_called_a_leak(fmt: PanelFormat) -> None:
    report = check_leaks(CAUSAL[fmt]["trailing_rolling"], _fields(), DECISIONS, lookback=5 * HOUR, panel_format=fmt)
    assert report.kind == "lookback"


@pytest.mark.parametrize("fmt", FORMATS)
def test_lookahead_still_caught_with_short_lookback(fmt: PanelFormat) -> None:
    report = check_leaks(LOOKAHEAD[fmt]["shift_minus_one"], _fields(), DECISIONS, lookback=5 * HOUR, panel_format=fmt)
    assert report.kind == "lookahead"
