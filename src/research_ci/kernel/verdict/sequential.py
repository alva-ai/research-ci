"""When the decisive test has drawn enough shuffles.

The decisive test draws its shuffles a batch at a time, up to a most. After each batch it has a z and a measure of how
much that z owes to which shuffles happened to be drawn: its standard error over resamples of them (a bootstrap: the
same z on ``RESAMPLES`` draws, with replacement, of the shuffles so far, the null values of one shuffle at every horizon
kept together, so the mean, the spread and the best-of-horizons correction all vary as they would with other
shuffles). It stops once z is further from the bar than ``MARGIN`` standard errors, a gap the shuffles still to come
are unlikely to bridge, or when no z can be computed, or at the most. Near the bar it draws them all, so it decides as
a run of the most does, except where the shuffles themselves cannot tell.
"""

import math

import numpy as np

RESAMPLES = 400
MARGIN = 3.0


def family_z(real: np.ndarray, nulls: np.ndarray, combine: str) -> np.ndarray:
    """The z of a family of horizons for every set of null values: ``real`` per horizon, ``nulls`` shaped
    (..., shuffles, horizons); the horizons are the counted ones."""
    mu = np.nanmean(nulls, axis=-2)
    sd = np.nanstd(nulls, axis=-2, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (real - mu) / sd, np.nan)
    if combine == "each":
        return np.min(z, axis=-1)
    if z.shape[-1] == 1:
        return z[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        standard = np.where(sd[..., None, :] > 0, (nulls - mu[..., None, :]) / sd[..., None, :], np.nan)
        top = np.nanmax(standard, axis=-1)
        spread = np.nanstd(top, axis=-1, ddof=1)
        return np.where(spread > 0, (np.max(z, axis=-1) - np.nanmean(top, axis=-1)) / spread, np.nan)


def standard_error(real: np.ndarray, nulls: np.ndarray, combine: str, seed: int = 0) -> float:
    """How much the family's z moves over resamples of its shuffles (``nulls``: shuffles by counted horizons)."""
    n = nulls.shape[0]
    if n < 2 or not nulls.shape[1]:
        return math.nan
    pick = np.random.default_rng(seed).integers(0, n, (RESAMPLES, n))
    with np.errstate(invalid="ignore"):
        z = family_z(real, nulls[pick], combine)
    finite = z[np.isfinite(z)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan


def settled(z: float, se: float, need: float) -> bool:
    """Whether more shuffles could still change the decision ``z ≥ need``."""
    if not math.isfinite(z):
        return True
    return math.isfinite(se) and abs(z - need) > MARGIN * se
