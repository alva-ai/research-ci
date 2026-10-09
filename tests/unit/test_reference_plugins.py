import pandas as pd
import polars as pl
import pytest

from research_ci.kernel.holdout.tiers import Tier, Tiers, visible
from research_ci.kernel.judge import protocol as pr
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import TIME, to_ns
from research_ci.reference_plugins.costs import FlatCosts
from research_ci.reference_plugins.labels import HoldingReturn, Markout, PairCarry
from research_ci.reference_plugins.targets import ClusterMean

T0 = pd.Timestamp("2026-01-01", tz="UTC")
H = pd.Timedelta(hours=1)


def _prices(n: int = 30) -> pl.DataFrame:
    rows = [("a", "close", T0 + i * H, T0 + i * H, 100.0 + i, 0) for i in range(n)]
    rows += [("b", "close", T0 + i * H, T0 + i * H, 200.0 - i, 0) for i in range(n)]
    return field_table(
        pd.DataFrame(rows, columns=["entity", "field", "event_time", "available_at", "value", "vintage"])
    )


def _at(*rows: tuple) -> pl.DataFrame:
    names = ["t", "entity", "side"][: len(rows[0])]
    return pl.DataFrame([(T0 + h * H, *rest) for h, *rest in rows], schema=names, orient="row").with_columns(
        pl.col("t").cast(TIME)
    )


def _by_entity(out: pl.DataFrame) -> dict[str, float]:
    return dict(zip(out["entity"], out["value"], strict=True))


def test_holding_return_uses_prices_known_at_t_and_t_plus_h() -> None:
    label = HoldingReturn()
    two = _by_entity(label.values(_prices(), _at((5, "a"), (5, "b")), 2 * H))
    assert two == {"a": pytest.approx(107 / 105 - 1), "b": pytest.approx(193 / 195 - 1)}
    assert _by_entity(label.values(_prices(), _at((5, "a")), 4 * H)) == {"a": pytest.approx(109 / 105 - 1)}


def test_holding_return_has_no_row_until_the_horizon_is_known_and_waits_for_publication() -> None:
    assert HoldingReturn().values(_prices(n=8), _at((5, "a")), 4 * H).is_empty()
    rows = [("a", "px", T0, T0, 100.0, 0), ("a", "px", T0 + H, T0 + 3 * H, 110.0, 0)]
    late = field_table(
        pd.DataFrame(rows, columns=["entity", "field", "event_time", "available_at", "value", "vintage"])
    )
    label = HoldingReturn("px")
    assert _by_entity(label.values(late, _at((0, "a")), 2 * H)) == {"a": pytest.approx(0.0)}
    assert _by_entity(label.values(late, _at((0, "a")), 3 * H)) == {"a": pytest.approx(0.1)}


def test_a_price_is_current_until_the_next_is_due() -> None:
    label = HoldingReturn()
    half = _at((5, "a")).with_columns(pl.col("t") + pl.duration(minutes=30))
    assert _by_entity(label.values(_prices(), half, 2 * H)) == {"a": pytest.approx(107 / 105 - 1)}
    gap = _prices().filter(~((pl.col("entity") == "a") & pl.col("event_time").is_between(T0 + 10 * H, T0 + 14 * H)))
    out = label.values(gap, _at((7, "a"), (9, "a"), (11, "a"), (15, "a")), 2 * H)
    assert out["t"].to_list() == [T0 + 7 * H, T0 + 15 * H]
    assert out["value"].to_list() == pytest.approx([109 / 107 - 1, 117 / 115 - 1])


def test_markout_is_signed_by_side() -> None:
    out = _by_entity(Markout().values(_prices(), _at((5, "a", 1.0), (5, "b", -1.0)), H))
    assert out == {"a": pytest.approx(106 / 105 - 1), "b": pytest.approx(-(194 / 195 - 1))}


def test_labels_on_explore_data_never_reach_past_the_cutoff() -> None:
    explore = visible(_prices(), Tier.EXPLORE, Tiers(explore_end=to_ns(T0 + 10 * H)))
    label = HoldingReturn()
    assert _by_entity(label.values(explore, _at((8, "a")), 2 * H)) == {"a": pytest.approx(110 / 108 - 1)}
    assert label.values(explore, _at((8, "a")), 3 * H).is_empty()


@pytest.mark.parametrize("plugin", [HoldingReturn(), Markout(), PairCarry(flow="funding")])
def test_reference_labels_pass_the_checks_every_label_must_pass(plugin) -> None:
    assert {c.status for c in pr.label_cases(type(plugin).__name__, plugin.values, plugin.cases())} == {"pass"}


def test_cluster_mean_weights_clusters_equally_after_costs() -> None:
    labels = pl.DataFrame(
        {"t": [T0, T0, T0], "entity": ["x1", "x2", "y"], "symbol": ["x", "x", "y"], "value": [0.01, 0.03, 0.05]}
    ).with_columns(pl.col("t").cast(TIME))
    out = ClusterMean(cluster="symbol").reduce(labels, H, FlatCosts(5, 5, legs=2))
    assert out["value"].to_list() == [pytest.approx((0.02 + 0.05) / 2 - 0.004)]


def test_flat_costs_charge_every_leg_of_every_weight_change() -> None:
    trades = pl.DataFrame({"t": [T0, T0], "entity": ["p", "q"], "from_weight": [0.0, 0.5], "to_weight": [0.25, 0.0]})
    assert FlatCosts(5, 5, legs=2).fees(trades) == pytest.approx((0.25 + 0.5) * 2 * 0.001)
