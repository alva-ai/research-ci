from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from research_ci.kernel.time.asof import known


class Tier(StrEnum):
    EXPLORE = "explore"
    SEAL = "seal"
    FORWARD = "forward"


class TierError(ValueError):
    pass


@dataclass(frozen=True)
class Tiers:
    """Cutoffs in ns since epoch (UTC): explore data ends at ``explore_end``, the sealed segment at ``seal_end``."""

    explore_end: int
    seal_end: int | None = None

    def __post_init__(self) -> None:
        if self.seal_end is not None and self.seal_end <= self.explore_end:
            raise TierError("seal_end must be after explore_end")


def visible(
    sources: Mapping[str, pl.DataFrame], tier: Tier, tiers: Tiers, frozen_at: int | None = None
) -> dict[str, pl.DataFrame]:
    """The rows of each source the given tier may read. Explore and seal are cut by ``available_at``; forward sees
    everything."""
    if tier is Tier.FORWARD:
        if frozen_at is None:
            raise TierError("the forward tier needs the freeze time")
        return dict(sources)
    cutoff = tiers.explore_end if tier is Tier.EXPLORE else _seal_end(tiers)
    return known(sources, cutoff)


def decision_window(tier: Tier, tiers: Tiers, frozen_at: int | None = None) -> tuple[int | None, int | None]:
    """Half-open window ``(start, end]`` of decision times allowed in a tier; ``None`` means unbounded."""
    if tier is Tier.EXPLORE:
        return (None, tiers.explore_end)
    if tier is Tier.SEAL:
        return (tiers.explore_end, _seal_end(tiers))
    if frozen_at is None:
        raise TierError("the forward tier needs the freeze time")
    return (frozen_at, None)


def require_in_window(times: list[int], tier: Tier, tiers: Tiers, frozen_at: int | None = None) -> None:
    start, end = decision_window(tier, tiers, frozen_at)
    outside = [t for t in times if (start is not None and t <= start) or (end is not None and t > end)]
    if outside:
        raise TierError(f"{len(outside)} decision times fall outside the {tier} window ({start}, {end}]")


def _seal_end(tiers: Tiers) -> int:
    if tiers.seal_end is None:
        raise TierError("no sealed segment is defined")
    return tiers.seal_end
