"""The test of a test: before an eval and a null world judge anything, they must kill an output that carries only
what the world keeps, and pass one that adds a known amount of real information.

A world claims to keep the label's level in each value of a column (``keeps``) at each decision time, or only the
level at each time; a world that shifts the scores in time keeps each inst's own level over time instead. The
stand-in is that level of the label plus noise; the planted output adds the label's residual around it, a signal of
what the world destroys (which candidates do better at a time, or when an inst does better), sized so the eval
moves by 1.5 times its smallest detectable effect. For events, both are kept only at the share ``events`` of the rows
where they are highest, as events, and missing elsewhere, so a planted output marks the rows that do better. Which rows
those are varies from draw to draw, so its move is the mean over ``SIZED`` draws; and events that follow the label
bunch in time where the label does, which spreads their null wider than a stand-in's, so the move is sized in the
planted output's own MDE. Both use the labels on purpose and are never features.

Even in a world that keeps what it claims, a stand-in beats ``z_base`` by chance at the rate a t distribution with as
many degrees of freedom as null shuffles gives; it must not live more often than that rate plus three binomial
standard errors over ``reps``. The planted output must live at least 80% of the time.
"""

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import polars as pl
from scipy import stats

from research_ci.api.evals import Case, Market
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.synthetic import HOUR_NS, market
from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.judge import (
    Evaluate,
    Prepared,
    blocks,
    checked,
    null_values,
    per_t,
    stacked,
    with_scores,
)
from research_ci.kernel.verdict.permute import Permutation
from research_ci.kernel.verdict.shift import Shift

EVENTS = 0.1
SIZED = 10


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
    shifts: bool = False,
    events: float = 0.0,
) -> dict[str, Any]:
    """Stand-ins die as often as chance allows; planted signals live. A stand-in carrying only what the world keeps
    must die as often as chance allows, and a planted signal 1.5 MDE strong must live. ``shifts``: the world shifts
    the scores in time; ``events``: the share of the rows the outputs mark as events, or 0 for scores on every row."""
    keys = _kept(keeps, shifts)
    level = table.select(pl.col("y").mean().over(keys))["y"].to_numpy()
    y = table["y"].to_numpy()
    level_z = _standard(level)
    resid = _standard(y - level)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((reps, len(y)))
    shuffles = np.stack([shuffle(10_000 + k) for k in range(nulls)])
    prepared = _batched(evaluate, with_scores(table, level_z), not shifts, shuffles[0] if shifts else None)

    def mark(scores: np.ndarray) -> np.ndarray:
        return _events(scores, events) if events else scores

    def value(scores: np.ndarray) -> float:
        return blocks(per_t(evaluate, with_scores(table, scores)), h_ns)[0]

    def shuffled(scores: np.ndarray) -> np.ndarray:
        if prepared is None:
            return np.array([value(scores[i]) for i in shuffles])
        return null_values(prepared.with_scores(scores), shuffles, h_ns)

    def z(scores: np.ndarray) -> float:
        null = shuffled(scores)
        sd = float(np.nanstd(null, ddof=1))
        return (value(scores) - float(np.nanmean(null))) / sd if sd > 0 else math.nan

    stand = mark(level_z + noise[0])
    null0 = shuffled(stand)
    mde = pw.mde(float(np.nanstd(null0, ddof=1)), z_base)
    sized = noise[: SIZED if events else 1]
    base = float(np.mean([value(mark(level_z + n)) for n in sized]))

    def gain(k: float) -> float:
        return float(np.mean([value(mark(level_z + n + k * resid)) for n in sized])) - base

    def resolved(k: float) -> float:
        own = float(np.nanstd(shuffled(mark(level_z + noise[0] + k * resid)), ddof=1))
        return gain(k) / pw.mde(own, z_base)

    lam = _size(resolved, 1.5) if events else _size(gain, 1.5 * mde)
    mde = gain(lam) / 1.5 if events and math.isfinite(lam) else mde
    die = [z(mark(level_z + n)) < z_base for n in noise]
    live = [z(mark(level_z + n + lam * resid)) >= z_base for n in noise]
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


def _batched(evaluate: Evaluate, table: pl.DataFrame, within: bool = True, other: np.ndarray | None = None) -> Any:
    """The eval on many shuffles at once, by its batched form when the world shuffles ``within`` each decision time
    and the form can take other scores and gives the eval's numbers, or else stacked when that gives them, checked
    beside ``other``, one of the world's shuffles; the stand-ins and planted outputs then share everything that does
    not read the scores."""
    values = per_t(evaluate, table)
    form = getattr(evaluate, "batched", None) if within else None
    prepared = checked(cast(Prepared, form(table)), evaluate, table, values) if callable(form) else None
    if prepared is None or not callable(getattr(prepared, "with_scores", None)):
        prepared = stacked(evaluate, table, values, other)
    return prepared


def _kept(keeps: str | None, shifts: bool) -> list[str]:
    """The columns of the label's level a world keeps: each decision time's (and each value of ``keeps`` there), or,
    for a shift in time, each inst's."""
    if shifts:
        return ["inst"] + ([keeps] if keeps not in (None, "time", "inst") else [])
    return ["t"] if keeps is None or keeps == "time" else ["t", keeps]


def _events(scores: np.ndarray, share: float) -> np.ndarray:
    """The scores at the share of the rows where they are highest, the events; NaN elsewhere."""
    return np.where(scores >= np.quantile(scores, 1 - share), scores, np.nan)


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


def _shifted(events: float) -> bool:
    """On a series of three insts, a world shifting the scores in time, with scores on every row or at events."""
    table = market(Market(times=240, insts=3, clusters=3, shape="series", tail=None), seed=0).drop("score")
    shuffle = Shift(table, HOUR_NS).index
    found = calibrate(_signed, table, HOUR_NS, "inst", shuffle, 40, 20, 2.0, shifts=True, events=events)
    return found["ok"]


def _signed(table: pl.DataFrame) -> pl.DataFrame:
    """The mean of sign(score) × label at each time, over the rows with a score."""
    rows = table.filter(pl.col("score").is_finite())
    return rows.group_by("t").agg((pl.col("score").sign() * pl.col("y")).mean().alias("value")).sort("t")


CASES = [
    Case("a world that keeps the group passes", lambda: _calibrated("group"), True),
    Case(
        "a world shuffling only within time does not keep the group: returned",
        lambda: _calibrated(None),
        False,
    ),
    Case("a world shifting the scores in time passes on a series", lambda: _shifted(0.0), True),
    Case("and with the scores only at events", lambda: _shifted(EVENTS), True),
]
