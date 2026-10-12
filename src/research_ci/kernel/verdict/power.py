"""How much a target's estimate can move by chance, and what that allows to be detected.

Everything is computed from the dispersion of the target across independent time blocks, never from where it sits,
so a planning number never reveals a result.
"""

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

Z_BASE = 2.0
Z_BETA = float(stats.norm.ppf(0.8))
DAY_NS = 86_400 * 10**9
YEAR_DAYS = 365.0


def z_need(n_trials: int, z_base: float = Z_BASE) -> float:
    """The z an eval needs on the ``n``-th submission of a layer; it grows like the best of ``n`` z scores."""
    return z_base * math.sqrt(1 + math.log(max(n_trials, 1)))


def decision_grid(every_ns: int, offset_ns: int, start_ns: int, end_ns: int) -> list[int]:
    """Grid points ``midnight + offset + k * every`` in ``(start, end]``, midnight being that of ``start`` in UTC."""
    midnight = start_ns - start_ns % DAY_NS
    first = midnight + offset_ns
    if first <= start_ns:
        first += ((start_ns - first) // every_ns + 1) * every_ns
    return list(range(first, end_ns + 1, every_ns))


def block_means(values: pl.DataFrame, h_ns: int) -> pl.DataFrame:
    """Mean of ``value`` per block of length ``h`` counted from the first ``t``; overlapping labels share a block."""
    t0 = values.select(pl.col("t").min().dt.epoch("ns")).item()
    block = (pl.col("t").dt.epoch("ns") - t0) // h_ns
    return values.group_by(block.alias("block")).agg(pl.col("value").mean()).sort("block")


def block_se(values: pl.DataFrame, h_ns: int) -> float:
    means = block_means(values, h_ns)["value"].drop_nulls()
    if means.len() < 2:
        return math.nan
    return float(np.std(means.to_numpy(), ddof=1) / math.sqrt(means.len()))


@dataclass(frozen=True)
class Power:
    """For one horizon: independent blocks in the span, the standard error of the target, and the smallest gain over
    random the data can detect at 80% power."""

    blocks: float
    se: float
    mde: float


def power(se: float, span_ns: int, h_ns: int, z: float = Z_BASE) -> Power:
    """``se`` NaN (too few blocks to measure dispersion) gives NaN for everything that needs it."""
    blocks = span_ns / h_ns
    t_a = float(stats.t.ppf(stats.norm.cdf(z), max(blocks - 1, 1)))
    return Power(blocks=blocks, se=se, mde=(t_a + Z_BETA) * se)


def scaling(sd_by_days: dict[float, float]) -> float:
    """Exponent ``beta`` of ``sd ~ days ** beta`` for a paired difference: 0.5 when noise averages out like a random
    walk, 1 or more when paths keep diverging and waiting longer does not help."""
    points = [(d, s) for d, s in sd_by_days.items() if s > 0 and math.isfinite(s)]
    if len(points) < 2:
        return math.nan
    x, y = np.log([d for d, _ in points]), np.log([s for _, s in points])
    return float(np.polyfit(x, y, 1)[0])


@dataclass(frozen=True)
class ConsumerPower:
    """For a per-year effect through the consumer: the noise per year over the history, the smallest effect the
    history can detect, and the months of new data needed for ``min_effect`` (``None`` when it never gets there)."""

    sd_per_year: float
    history_mde: float
    forward_months: float | None


def consumer_power(sd: float, days: float, beta: float, min_effect: float, z: float = Z_BASE) -> ConsumerPower:
    years = days / YEAR_DAYS
    per_year = sd / years
    mde = (z + Z_BETA) * per_year
    if not math.isfinite(beta) or beta >= 1:
        return ConsumerPower(per_year, mde, None)
    ratio = mde / min_effect
    months = years * ratio ** (1 / (1 - beta)) * 12
    return ConsumerPower(per_year, mde, months)


def branch_power(pilot: dict[str, float], min_effect: float, z: float = Z_BASE) -> ConsumerPower:
    """Per-year power of a branched comparison: each branch measures the change over its horizon; branches closer
    than their measurement window share data, so they count as ``independence`` of a branch each."""
    horizon_years = pilot["horizon_days"] / YEAR_DAYS
    n_eff = pilot["branches"] * pilot["independence"]
    if n_eff < 2 or not math.isfinite(pilot["sd"]):
        return ConsumerPower(math.nan, math.nan, None)
    per_year = pilot["sd"] / horizon_years
    mde = (z + Z_BETA) * per_year / math.sqrt(n_eff)
    needed = ((z + Z_BETA) * per_year / min_effect) ** 2 / pilot["independence"]
    return ConsumerPower(per_year, mde, needed * pilot["every_days"] / 30.4)
