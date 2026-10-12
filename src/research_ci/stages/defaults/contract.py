"""The checks derive runs on every contract by default, before its floor (``stages/floor/contract.py``): one list, to
adjust from experience. The contract is not judged by the engine hash, and the engine hash leaves this list out."""

from research_ci.stages.defaults.data import freshness
from research_ci.stages.floor.contract import Seen
from research_ci.stages.steps import Found, Step


def fresh(seen: Seen) -> Found:
    """Data whose newest row is older than ``STALE_AFTER`` is a warning: collect live data first. The data check
    warns too."""
    return freshness(seen.latest, seen.now)


CHECKS = [Step("data is current", "warns", fresh, "Warns when the newest row is too old")]
