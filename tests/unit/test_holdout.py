import pandas as pd
import polars as pl
import pytest

from alpha_ci.kernel.holdout.labels import holding_return, markout
from alpha_ci.kernel.holdout.tiers import Tier, TierError, Tiers, decision_window, require_in_window, visible
from alpha_ci.kernel.tables import field_table
from alpha_ci.kernel.time.asof import asof
from alpha_ci.kernel.time.clock import TIME, to_ns

T0 = pd.Timestamp("2026-01-01", tz="UTC")
H = pd.Timedelta(hours=1)
TIERS = Tiers(explore_end=to_ns(T0 + 10 * H), seal_end=to_ns(T0 + 20 * H))


def _prices(n: int = 30) -> pl.DataFrame:
    rows = [("a", "close", T0 + i * H, T0 + i * H, 100.0 + i, 0) for i in range(n)]
    rows += [("b", "close", T0 + i * H, T0 + i * H, 200.0 - i, 0) for i in range(n)]
    return field_table(
        pd.DataFrame(rows, columns=["entity", "field", "event_time", "available_at", "value", "vintage"])
    )


def test_explore_tier_never_shows_rows_after_its_cutoff() -> None:
    explore = visible(_prices(), Tier.EXPLORE, TIERS)
    assert explore["available_at"].max() == T0 + 10 * H
    late = asof(explore, T0 + 25 * H)
    assert late["time"].max() == T0 + 10 * H


def test_seal_tier_stops_at_the_seal_end() -> None:
    sealed = visible(_prices(), Tier.SEAL, TIERS)
    assert sealed["available_at"].max() == T0 + 20 * H


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


def test_holding_return_uses_prices_known_at_t_and_t_plus_h() -> None:
    out = holding_return(_prices(), "close", [T0 + 5 * H], [2 * H, 4 * H])
    got = {(e, h): v for e, h, v in zip(out["entity"], out["horizon"], out["value"], strict=True)}
    assert got[("a", 2 * H)] == pytest.approx(107 / 105 - 1)
    assert got[("a", 4 * H)] == pytest.approx(109 / 105 - 1)
    assert got[("b", 2 * H)] == pytest.approx(193 / 195 - 1)
    assert out.schema["t"] == TIME


def test_holding_return_is_null_when_the_horizon_is_not_yet_known() -> None:
    out = holding_return(_prices(n=8), "close", [T0 + 5 * H], [4 * H])
    assert out.filter(pl.col("entity") == "a")["value"].to_list() == [None]


def test_holding_return_waits_for_publication() -> None:
    rows = [("a", "px", T0, T0, 100.0, 0), ("a", "px", T0 + H, T0 + 3 * H, 110.0, 0)]
    fields = field_table(
        pd.DataFrame(rows, columns=["entity", "field", "event_time", "available_at", "value", "vintage"])
    )
    out = holding_return(fields, "px", [T0], [2 * H, 3 * H])
    assert out.sort("horizon")["value"].to_list() == [pytest.approx(0.0), pytest.approx(0.1)]


def test_markout_is_signed_by_side_and_in_basis_points() -> None:
    events = pl.DataFrame({"t": [T0 + 5 * H, T0 + 5 * H], "entity": ["a", "b"], "side": [1.0, -1.0]}).with_columns(
        pl.col("t").cast(TIME)
    )
    out = markout(_prices(), "close", events, [H])
    got = dict(zip(out["entity"], out["value"], strict=True))
    assert got["a"] == pytest.approx((106 / 105 - 1) * 1e4)
    assert got["b"] == pytest.approx(-(194 / 195 - 1) * 1e4)


def test_labels_on_explore_data_never_reach_past_the_cutoff() -> None:
    explore = visible(_prices(), Tier.EXPLORE, TIERS)
    out = holding_return(explore, "close", [T0 + 8 * H], [H, 2 * H, 3 * H])
    assert out.filter(pl.col("entity") == "a").sort("horizon")["value"].to_list() == [
        pytest.approx(109 / 108 - 1),
        pytest.approx(110 / 108 - 1),
        None,
    ]
