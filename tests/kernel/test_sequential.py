"""The decisive test's sequential rule against a fixed run of the most shuffles, on synthetic markets: it must decide
the same way except near the bar, so its false-pass rate without information and its power with some are those of the
fixed run within simulation error, while it draws far fewer shuffles away from the bar."""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl

from research_ci.api.evals import Market
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.statistics.shuffled import RankCorrelation
from research_ci.eval.synthetic import HOUR_NS, market
from research_ci.kernel.verdict.judge import (
    Lane,
    Request,
    Sequential,
    Statistic,
    blocks,
    judge_all,
    null_values,
    per_t,
    settled,
    summary,
)
from research_ci.kernel.verdict.permute import Permutation
from research_ci.kernel.verdict.power import Z_BASE

STAT = Statistic(rank_ic, RankCorrelation())
HORIZONS = {"1h": HOUR_NS, "4h": 4 * HOUR_NS}
MOST, BATCH, MIN_BLOCKS = 200, 50, 5
SMALL = Market(times=48, entities=30, clusters=10)


def _decisions(table: pl.DataFrame) -> tuple[float, float, int]:
    """The z of a fixed run of ``MOST`` shuffles, and the sequential rule's z and shuffles on the same shuffles."""
    reals, nulls = {}, {}
    prepared = RankCorrelation()(table)
    for h, h_ns in HORIZONS.items():
        perm = Permutation(table, None, "cluster", h_ns)
        nulls[h] = null_values(prepared, np.stack([perm.index(s) for s in range(MOST)]), h_ns)
        reals[h] = blocks(per_t(STAT, table), h_ns)
    full = summary(reals, nulls, MIN_BLOCKS, "max")["z"]
    n = BATCH
    while True:
        part = {h: v[:n] for h, v in nulls.items()}
        result = summary(reals, part, MIN_BLOCKS, "max")
        if n >= MOST or settled(result, reals, part, "max", Z_BASE):
            return full, result["z"], n
        n += BATCH


def _runs(m: Market, markets: int) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=4) as pool:
        return np.array(list(pool.map(lambda seed: _decisions(market(m, seed)), range(markets))))


def _passes(z: np.ndarray) -> np.ndarray:
    return np.nan_to_num(z, nan=-math.inf) >= Z_BASE


def _check(runs: np.ndarray) -> tuple[float, float, float]:
    """Decisions differ only near the bar; returns the two pass rates and their binomial standard error."""
    full, seq = _passes(runs[:, 0]), _passes(runs[:, 1])
    differ = runs[full != seq, 0]
    assert np.all(np.abs(differ - Z_BASE) < 0.5), differ
    assert len(differ) <= max(1, 0.02 * len(runs))
    far = np.abs(runs[:, 0] - Z_BASE) > 1
    assert not np.any(full[far] != seq[far])
    rate = float(full.mean())
    return rate, float(seq.mean()), math.sqrt(max(rate, 1 / len(runs)) * (1 - rate) / len(runs))


def test_without_information_it_passes_no_more_often_than_the_fixed_run_and_draws_few_shuffles() -> None:
    runs = _runs(SMALL, 200)
    fixed, seq, se = _check(runs)
    assert seq <= fixed + 3 * se
    assert runs[:, 2].mean() < 100


def test_with_information_it_passes_as_often_as_the_fixed_run() -> None:
    runs = _runs(Market(times=48, entities=30, clusters=10, ic=0.05), 120)
    fixed, seq, se = _check(runs)
    assert 0.2 < fixed < 0.8
    assert seq >= fixed - 3 * se


def test_the_judge_stops_where_the_rule_says_on_the_same_shuffles() -> None:
    for seed, ic in ((0, 0.0), (3, 0.05), (5, 0.05), (8, 0.08)):
        table = market(Market(times=48, entities=30, clusters=10, ic=ic), seed)
        full, z, n = _decisions(table)
        lanes = {h: Lane(table, h_ns, h, _shuffle(table, h_ns)) for h, h_ns in HORIZONS.items()}
        fixed = Request(STAT, lanes, MOST, MIN_BLOCKS, "max")
        rule = Request(STAT, lanes, MOST, MIN_BLOCKS, "max", Sequential(BATCH, Z_BASE))
        a, b = judge_all([fixed, rule], workers=2)
        assert isinstance(a, dict) and isinstance(b, dict)
        assert a["shuffles"] == MOST and b["shuffles"] == n
        assert math.isclose(a["z"], full, rel_tol=1e-9) and math.isclose(b["z"], z, rel_tol=1e-9)


def _shuffle(table: pl.DataFrame, h_ns: int):  # noqa: ANN202
    return lambda: Permutation(table, None, "cluster", h_ns).index


def test_a_rule_that_decides_horizon_by_horizon_waits_until_every_horizon_is_settled() -> None:
    """The family's z is far above the bar, so the family's decision is settled; another horizon sits at the bar
    against the family's null, so whether it passes is not, and a rule horizon by horizon draws on."""
    rng = np.random.default_rng(0)
    nulls = {"1h": rng.standard_normal(BATCH), "4h": rng.standard_normal(BATCH)}
    mu, sd = {h: float(v.mean()) for h, v in nulls.items()}, {h: float(v.std(ddof=1)) for h, v in nulls.items()}
    top = np.maximum(*[(v - mu[h]) / sd[h] for h, v in nulls.items()])
    at_bar = mu["4h"] + sd["4h"] * (float(top.mean()) + Z_BASE * float(top.std(ddof=1)))
    reals = {"1h": (mu["1h"] + 10 * sd["1h"], 0.1, 20), "4h": (at_bar, 0.1, 20)}
    result = summary(reals, nulls, MIN_BLOCKS, "max")
    near = next(r for r in result["horizons"] if r["h"] == "4h")
    assert abs(near["adjusted"] - Z_BASE) < 0.5 < result["z"] - Z_BASE
    assert settled(result, reals, nulls, "max", Z_BASE)
    assert not settled(result, reals, nulls, "max", Z_BASE, horizons=True)
