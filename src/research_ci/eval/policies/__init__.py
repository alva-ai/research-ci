"""Stage policies that constrain the shared methodology library."""

from __future__ import annotations

from ..registry import StageName, StagePolicy
from .algo import POLICY as ALGO_POLICY
from .data import POLICY as DATA_POLICY
from .feat import POLICY as FEAT_POLICY
from .sig import POLICY as SIG_POLICY

POLICIES: dict[StageName, StagePolicy] = {
    "data": DATA_POLICY,
    "feat": FEAT_POLICY,
    "sig": SIG_POLICY,
    "algo": ALGO_POLICY,
}


def stage_policy(stage: str) -> StagePolicy:
    """Return the hard policy for a known research stage."""

    try:
        return POLICIES[stage]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"unknown stage: {stage!r}") from exc


__all__ = [
    "ALGO_POLICY",
    "DATA_POLICY",
    "FEAT_POLICY",
    "POLICIES",
    "SIG_POLICY",
    "stage_policy",
]
