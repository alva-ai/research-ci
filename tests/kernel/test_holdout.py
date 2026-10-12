import pandas as pd
import polars as pl
import pytest

from research_ci.kernel.holdout.tiers import Tier, TierError, Tiers, decision_window, require_in_window, visible
from research_ci.kernel.tables import source_tables
from research_ci.kernel.time.asof import wide
from research_ci.kernel.time.clock import to_ns

T0 = pd.Timestamp("2026-01-01", tz="UTC")
H = pd.Timedelta(hours=1)
TIERS = Tiers(explore_end=to_ns(T0 + 10 * H), seal_end=to_ns(T0 + 20 * H))


def _prices(n: int = 30) -> dict[str, pl.DataFrame]:
    rows = [("a", T0 + i * H, T0 + i * H, 0, 100.0 + i) for i in range(n)]
    rows += [("b", T0 + i * H, T0 + i * H, 0, 200.0 - i) for i in range(n)]
    columns = ["inst", "event_time", "available_at", "vintage", "close"]
    return source_tables({"candles": pd.DataFrame(rows, columns=columns)})


def test_explore_tier_never_shows_rows_after_its_cutoff() -> None:
    explore = visible(_prices(), Tier.EXPLORE, TIERS)
    assert explore["candles"]["available_at"].max() == T0 + 10 * H
    late = wide(explore, [to_ns(T0 + 25 * H)])
    assert late["close"].to_list() == [110.0, 190.0] and late["age_candles"].max() == 15 * H


def test_seal_tier_stops_at_the_seal_end() -> None:
    sealed = visible(_prices(), Tier.SEAL, TIERS)
    assert sealed["candles"]["available_at"].max() == T0 + 20 * H


def test_forward_tier_needs_a_freeze_time() -> None:
    with pytest.raises(TierError):
        visible(_prices(), Tier.FORWARD, TIERS)


def test_decision_windows_do_not_overlap() -> None:
    frozen = to_ns(T0 + 25 * H)
    assert decision_window(Tier.EXPLORE, TIERS) == (None, to_ns(T0 + 10 * H))
    assert decision_window(Tier.SEAL, TIERS) == (to_ns(T0 + 10 * H), to_ns(T0 + 20 * H))
    assert decision_window(Tier.FORWARD, TIERS, frozen_at=frozen) == (frozen, None)


def test_decision_times_outside_the_tier_are_rejected() -> None:
    require_in_window([to_ns(T0 + 5 * H)], Tier.EXPLORE, TIERS)
    with pytest.raises(TierError):
        require_in_window([to_ns(T0 + 11 * H)], Tier.EXPLORE, TIERS)
    with pytest.raises(TierError):
        require_in_window([to_ns(T0 + 10 * H)], Tier.SEAL, TIERS)
