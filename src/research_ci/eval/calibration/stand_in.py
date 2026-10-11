"""The test of a test: before a criterion and a null world judge anything, they must kill an output that carries only
what the world keeps, and pass one that adds a known amount of real information.

A world claims to keep the label's level in each value of a column (``keeps``) at each decision time, or only the
level at each time. The stand-in is that level of the label plus noise; the planted output adds the label's residual
around it, sized so the criterion moves by 1.5 times its smallest detectable effect. Both use the labels on purpose
and are never features.

Even in a world that keeps what it claims, a stand-in beats ``z_base`` by chance at the rate a t distribution with as
many degrees of freedom as null shuffles gives; it must not live more often than that rate plus three binomial
standard errors over ``reps``. The planted output must live at least 80% of the time.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from research_ci.api.methods import Case, Market
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.synthetic import HOUR_NS, market
from research_ci.kernel.verdict.judge import Evaluate, blocks, per_t, with_scores
from research_ci.kernel.verdict.permute import Permutation

Z_BETA = 0.84


def calibrate(
    evaluate: Evaluate,
    table: pl.DataFrame,
    h_ns: int,
    keeps: str | None,
    shuffle: Callable[[int], np.ndarray],
    reps: int,
    nulls: int,
    z_base: float,
    seed: int = 0,
) -> dict[str, Any]:
    keys = ["t"] if keeps is None or keeps == "time" else ["t", keeps]
    level = table.select(pl.col("y").mean().over(keys))["y"].to_numpy()
    y = table["y"].to_numpy()
    level_z = _standard(level)
    resid = _standard(y - level)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((reps, len(y)))
    shuffles = [shuffle(10_000 + k) for k in range(nulls)]

    def value(scores: np.ndarray) -> float:
        return blocks(per_t(evaluate, with_scores(table, scores)), h_ns)[0]

    def z(scores: np.ndarray) -> float:
        null = np.array([value(scores[i]) for i in shuffles])
        sd = float(np.nanstd(null, ddof=1))
        return (value(scores) - float(np.nanmean(null))) / sd if sd > 0 else math.nan

    stand = level_z + noise[0]
    null0 = np.array([value(stand[i]) for i in shuffles])
    sd0 = float(np.nanstd(null0, ddof=1))
    mde = (z_base + Z_BETA) * sd0
    lam = _size(lambda k: value(stand + k * resid) - value(stand), 1.5 * mde)
    die = [z(level_z + n) < z_base for n in noise]
    live = [z(level_z + n + lam * resid) >= z_base for n in noise]
    chance = float(stats.t.sf(z_base / math.sqrt(1 + 1 / nulls), nulls - 1))
    allowed = chance + 3 * math.sqrt(chance * (1 - chance) / reps)
    out = {
        "keeps": keeps or "time",
        "reps": reps,
        "nulls": nulls,
        "mde": mde,
        "lambda": lam,
        "stand_in_dies": float(np.mean(die)),
        "planted_lives": float(np.mean(live)),
        "dies_at_least": 1 - allowed,
    }
    out["ok"] = out["stand_in_dies"] >= 1 - allowed and out["planted_lives"] >= 0.80
    return out


def _size(gain: Callable[[float], float], target: float) -> float:
    """The smallest weight whose gain reaches ``target``, by doubling then bisection."""
    if not math.isfinite(target) or target <= 0:
        return math.nan
    hi = 1.0
    for _ in range(12):
        if gain(hi) >= target:
            break
        hi *= 2
    else:
        return math.nan
    lo = 0.0
    for _ in range(18):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gain(mid) < target else (lo, mid)
    return hi


def _standard(x: np.ndarray) -> np.ndarray:
    sd = float(np.nanstd(x))
    return (x - float(np.nanmean(x))) / sd if sd > 0 else np.zeros_like(x)


def _calibrated(bucket: str | None) -> bool:
    """On a market whose labels rest partly on the group, a world shuffling within ``bucket`` claims to keep it."""
    table = market(Market(group_share=0.4, tail=None), seed=0)
    shuffle = Permutation(table, bucket, "cluster", HOUR_NS).index
    return calibrate(rank_ic, table, HOUR_NS, "group", shuffle, reps=40, nulls=20, z_base=2.0)["ok"]


CASES = [
    Case("a world that keeps the group passes", lambda: _calibrated("group"), True),
    Case(
        "a world shuffling only within time does not keep the group: returned",
        lambda: _calibrated(None),
        False,
    ),
]
