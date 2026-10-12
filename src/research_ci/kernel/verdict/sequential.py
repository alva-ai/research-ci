"""When a judgement has drawn enough shuffles.

The decisive test, and every lab eval and null world that can veto, draws its shuffles a batch at a time, up to a most.
After each batch it has a z and a measure of how much that z owes to which shuffles happened to be drawn: its standard
error over resamples of them (a bootstrap: the same z on ``RESAMPLES`` draws, with replacement, of the shuffles so far,
the null values of one shuffle at every horizon kept together, so the mean, the spread and the best-of-horizons
correction all vary as they would with other shuffles). It stops once z is further from the bar than ``MARGIN``
standard errors, a gap the shuffles still to come are unlikely to bridge, or when no z can be computed, or at the most.
A judgement that decides horizon by horizon, such as the decisive test's passing horizons, stops once every counted
horizon's z is that far from the bar. Near the bar it draws them all, so it decides as a run of the most does, except
where the shuffles themselves cannot tell.
"""

import math

import numpy as np

RESAMPLES = 400
MARGIN = 3.0


def horizon_z(real: np.ndarray, nulls: np.ndarray, combine: str) -> np.ndarray:
    """Each horizon's z for every set of null values: ``real`` per horizon, ``nulls`` shaped (..., shuffles,
    horizons); the horizons are the counted ones. With ``max`` and more than one horizon, a horizon's z is set against
    the best standardized horizon of each shuffle (a single-step max-T), so the best horizon's is the family's;
    otherwise it is the horizon's own z."""
    mu = np.nanmean(nulls, axis=-2)
    sd = np.nanstd(nulls, axis=-2, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (real - mu) / sd, np.nan)
    if combine == "each" or z.shape[-1] == 1:
        return z
    with np.errstate(divide="ignore", invalid="ignore"):
        standard = np.where(sd[..., None, :] > 0, (nulls - mu[..., None, :]) / sd[..., None, :], np.nan)
        top = np.nanmax(standard, axis=-1)
        spread = np.nanstd(top, axis=-1, ddof=1)[..., None]
        return np.where(spread > 0, (z - np.nanmean(top, axis=-1)[..., None]) / spread, np.nan)


def family_z(real: np.ndarray, nulls: np.ndarray, combine: str) -> np.ndarray:
    """The z of a family of horizons for every set of null values: the best horizon's (``max``), or the weakest's
    (``each``)."""
    z = horizon_z(real, nulls, combine)
    return np.min(z, axis=-1) if combine == "each" else np.max(z, axis=-1)


def standard_error(real: np.ndarray, nulls: np.ndarray, combine: str, seed: int = 0) -> float:
    """How much the family's z moves over resamples of its shuffles (``nulls``: shuffles by counted horizons)."""
    z = _resampled(real, nulls, combine, seed)
    if z is None:
        return math.nan
    family = np.min(z, axis=-1) if combine == "each" else np.max(z, axis=-1)
    return _spread(family)


def standard_errors(real: np.ndarray, nulls: np.ndarray, combine: str, seed: int = 0) -> np.ndarray:
    """How much each counted horizon's z (``horizon_z``) moves over resamples of the shuffles."""
    z = _resampled(real, nulls, combine, seed)
    if z is None:
        return np.full(nulls.shape[1], math.nan)
    return np.array([_spread(z[:, k]) for k in range(z.shape[1])])


def _resampled(real: np.ndarray, nulls: np.ndarray, combine: str, seed: int) -> np.ndarray | None:
    n = nulls.shape[0]
    if n < 2 or not nulls.shape[1]:
        return None
    pick = np.random.default_rng(seed).integers(0, n, (RESAMPLES, n))
    with np.errstate(invalid="ignore"):
        return horizon_z(real, nulls[pick], combine)


def _spread(z: np.ndarray) -> float:
    finite = z[np.isfinite(z)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan


def settled(z: float, se: float, need: float) -> bool:
    """Whether more shuffles could still change the decision ``z ≥ need``."""
    if not math.isfinite(z):
        return True
    return math.isfinite(se) and abs(z - need) > MARGIN * se
