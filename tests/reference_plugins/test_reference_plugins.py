import pandas as pd
import polars as pl
import pytest

from research_ci.api.evals import Source
from research_ci.eval import protocol as pr
from research_ci.eval.synthetic import START_NS, sources
from research_ci.kernel.holdout.tiers import Tier, Tiers, visible
from research_ci.kernel.tables import source_tables
from research_ci.kernel.time.clock import TIME, to_ns
from research_ci.reference_plugins.costs import FlatCosts
from research_ci.reference_plugins.outcomes import HoldingReturn, Markout, PairCarry
from research_ci.reference_plugins.targets import ClusterMean

T0 = pd.Timestamp("2026-01-01", tz="UTC")
H = pd.Timedelta(hours=1)
COLUMNS = ["inst", "event_time", "available_at", "vintage"]


def _prices(n: int = 30) -> dict[str, pl.DataFrame]:
    rows = [("a", T0 + i * H, T0 + i * H, 0, 100.0 + i) for i in range(n)]
    rows += [("b", T0 + i * H, T0 + i * H, 0, 200.0 - i) for i in range(n)]
    return source_tables({"candles": pd.DataFrame(rows, columns=[*COLUMNS, "close"])})


def _at(*rows: tuple) -> pl.DataFrame:
    names = ["t", "inst", "side"][: len(rows[0])]
    return pl.DataFrame([(T0 + h * H, *rest) for h, *rest in rows], schema=names, orient="row").with_columns(
        pl.col("t").cast(TIME)
    )


def _by_inst(out: pl.DataFrame) -> dict[str, float]:
    return dict(zip(out["inst"], out["y"], strict=True))


def test_holding_return_uses_prices_known_at_t_and_t_plus_h() -> None:
    outcome = HoldingReturn()
    found = outcome.values(_prices(), _at((5, "a"), (5, "b")), 2 * H)
    assert _by_inst(found) == {"a": pytest.approx(107 / 105 - 1), "b": pytest.approx(193 / 195 - 1)}
    assert found["y_end"].to_list() == [T0 + 7 * H] * 2, "the last time it reads, t + h"
    assert _by_inst(outcome.values(_prices(), _at((5, "a")), 4 * H)) == {"a": pytest.approx(109 / 105 - 1)}


def test_holding_return_has_no_row_until_the_horizon_is_known_and_waits_for_publication() -> None:
    assert HoldingReturn().values(_prices(n=8), _at((5, "a")), 4 * H).is_empty()
    rows = [("a", T0, T0, 0, 100.0), ("a", T0 + H, T0 + 3 * H, 0, 110.0)]
    late = source_tables({"marks": pd.DataFrame(rows, columns=[*COLUMNS, "px"])})
    outcome = HoldingReturn("px")
    assert _by_inst(outcome.values(late, _at((0, "a")), 2 * H)) == {"a": pytest.approx(0.0)}
    assert _by_inst(outcome.values(late, _at((0, "a")), 3 * H)) == {"a": pytest.approx(0.1)}


def test_a_price_is_current_until_the_next_is_due() -> None:
    outcome = HoldingReturn()
    half = _at((5, "a")).with_columns(pl.col("t") + pl.duration(minutes=30))
    assert _by_inst(outcome.values(_prices(), half, 2 * H)) == {"a": pytest.approx(107 / 105 - 1)}
    hole = (pl.col("inst") == "a") & pl.col("event_time").is_between(T0 + 10 * H, T0 + 14 * H)
    gap = {"candles": _prices()["candles"].filter(~hole)}
    out = outcome.values(gap, _at((7, "a"), (9, "a"), (11, "a"), (15, "a")), 2 * H)
    assert out["t"].to_list() == [T0 + 7 * H, T0 + 15 * H]
    assert out["y"].to_list() == pytest.approx([109 / 107 - 1, 117 / 115 - 1])


def test_a_masked_price_is_no_price() -> None:
    """A price set to null, as a mask leaves it, is not a price: the outcomes that would read it give no row."""
    masked = _prices()["candles"].with_columns(
        close=pl.when((pl.col("inst") == "a") & (pl.col("event_time") == T0 + 7 * H)).then(None).otherwise("close")
    )
    out = HoldingReturn().values({"candles": masked}, _at((5, "a"), (6, "a")), 2 * H)
    assert out["t"].to_list() == [T0 + 6 * H]


def test_markout_is_signed_by_side() -> None:
    out = _by_inst(Markout().values(_prices(), _at((5, "a", 1.0), (5, "b", -1.0)), H))
    assert out == {"a": pytest.approx(106 / 105 - 1), "b": pytest.approx(-(194 / 195 - 1))}


def test_outcomes_on_explore_data_never_reach_past_the_cutoff() -> None:
    explore = visible(_prices(), Tier.EXPLORE, Tiers(explore_end=to_ns(T0 + 10 * H)))
    outcome = HoldingReturn()
    assert _by_inst(outcome.values(explore, _at((8, "a")), 2 * H)) == {"a": pytest.approx(110 / 108 - 1)}
    assert outcome.values(explore, _at((8, "a")), 3 * H).is_empty()


@pytest.mark.parametrize("plugin", [HoldingReturn(), Markout(), PairCarry(flow="funding")])
def test_reference_outcomes_pass_the_checks_every_outcome_must_pass(plugin) -> None:
    assert {c.status for c in pr.outcome_cases(type(plugin).__name__, plugin.values, plugin.cases())} == {"pass"}


def test_cluster_mean_weights_clusters_equally_after_costs() -> None:
    rows = pl.DataFrame(
        {"t": [T0, T0, T0], "inst": ["x1", "x2", "y"], "symbol": ["x", "x", "y"], "y": [0.01, 0.03, 0.05]}
    ).with_columns(pl.col("t").cast(TIME))
    out = ClusterMean(cluster="symbol").reduce(rows, H, FlatCosts(5, 5, legs=2))
    assert out["value"].to_list() == [pytest.approx((0.02 + 0.05) / 2 - 0.004)]


def test_cluster_mean_with_top_counts_only_the_highest_scores() -> None:
    rows = pl.DataFrame(
        {"t": [T0] * 4, "inst": list("abcd"), "symbol": list("abcd"), "y": [0.4, 0.3, 0.2, 0.1]}
    ).with_columns(pl.col("t").cast(TIME))
    top = ClusterMean(cluster="symbol", net=False, top=0.5)
    scored = rows.with_columns(score=pl.Series([1.0, 2.0, 3.0, 4.0]))
    assert top.reduce(scored, H, FlatCosts(0, 0))["value"].to_list() == [pytest.approx(0.15)]
    assert top.reduce(rows, H, FlatCosts(0, 0))["value"].to_list() == [pytest.approx(0.25)]


def test_flat_costs_charge_every_leg_of_every_weight_change() -> None:
    trades = pl.DataFrame({"t": [T0, T0], "inst": ["p", "q"], "from_weight": [0.0, 0.5], "to_weight": [0.25, 0.0]})
    assert FlatCosts(5, 5, legs=2).fees(trades) == pytest.approx((0.25 + 0.5) * 2 * 0.001)


def test_pair_carry_gives_every_horizon_at_once_as_it_gives_each_alone() -> None:
    """``values_each`` shares the levels and running flows at ``t`` between horizons and changes no number."""

    def pairs(tables: dict[str, pl.DataFrame], every: int) -> pl.DataFrame:
        names = sorted(tables["prices"]["inst"].unique().to_list())
        rows = [(START_NS + h * 3_600 * 10**9, f"{a}>{b}", a, b) for h in range(5, 200, every) for a, b in [names[:2]]]
        frame = pl.DataFrame(rows, schema=["t", "inst", "long", "short"], orient="row")
        return frame.with_columns(pl.col("t").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC").cast(TIME))

    horizons = [4 * H, 24 * H, 72 * H]
    for seed, flow in ((0, "flow"), (1, None)):
        tables = sources(Source(insts=4, hours=240, lag=1, revised=0.3), seed)
        outcome, cands = PairCarry("price", flow), pairs(tables, 3 + seed)
        each = list(outcome.values_each(tables, cands, horizons))
        assert len(each) == 3 and all(
            e.equals(outcome.values(tables, cands, h)) for e, h in zip(each, horizons, strict=True)
        )
