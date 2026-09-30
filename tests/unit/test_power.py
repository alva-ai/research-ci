import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from research_ci.kernel.judge import power as pw
from research_ci.kernel.time.clock import TIME

HOUR_NS = 3_600 * 10**9


def _values(n: int, sd: float, seed: int = 0) -> pl.DataFrame:
    t = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"t": t, "value": rng.normal(0.0, sd, n)}).with_columns(pl.col("t").cast(TIME))


def test_the_grid_starts_after_the_first_data_and_keeps_its_offset() -> None:
    start = int(pd.Timestamp("2026-01-01T00:30Z").value)
    end = int(pd.Timestamp("2026-01-02T00:00Z").value)
    grid = pw.decision_grid(8 * HOUR_NS, 5 * 60 * 10**9, start, end)
    assert [pd.Timestamp(t, tz="UTC").strftime("%H:%M") for t in grid] == ["08:05", "16:05"]


def test_block_standard_error_matches_theory_for_independent_values() -> None:
    ses = [pw.block_se(_values(2_000, 1.0, seed), HOUR_NS) for seed in range(20)]
    assert np.mean(ses) == pytest.approx(1 / math.sqrt(2_000), rel=0.05)
    assert pw.block_se(_values(2_000, 1.0), 10 * HOUR_NS) == pytest.approx(math.sqrt(0.1 / 200), rel=0.2)


def test_power_needs_more_new_data_for_smaller_effects() -> None:
    span, h = 1_000 * HOUR_NS, HOUR_NS
    p = pw.power(0.01, span, h, min_effect=0.05)
    assert p.blocks == 1_000
    assert p.mde == pytest.approx((pw.Z_BASE + pw.Z_BETA) * 0.01, rel=0.01)
    assert pw.power(0.01, span, h, min_effect=0.025).forward_days > 3.5 * p.forward_days


def test_noise_scaling_tells_whether_waiting_helps() -> None:
    assert pw.scaling({40: 1.0, 160: 2.0}) == pytest.approx(0.5)
    assert pw.scaling({80: 0.7, 160: 1.4}) == pytest.approx(1.0)
    assert pw.consumer_power(0.0147, 160, 1.05, 0.05).forward_months is None
    walk = pw.consumer_power(0.0147, 160, 0.5, 0.05)
    assert walk.history_mde == pytest.approx((pw.Z_BASE + pw.Z_BETA) * 0.0147 / (160 / 365))
    assert walk.forward_months == pytest.approx(160 / 365 * (walk.history_mde / 0.05) ** 2 * 12)


def test_thresholds_rise_with_trials() -> None:
    assert pw.z_need(1) == pw.Z_BASE
    assert pw.z_need(10) == pytest.approx(2.0 * math.sqrt(1 + math.log(10)))
