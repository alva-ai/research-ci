"""The checks derive runs on every contract by default, before its floor (``stages/contract/floor.py``): one list, to
adjust from experience. The contract is not judged by the engine hash, and the engine hash leaves this list out."""

from research_ci.stages.contract.floor import Seen
from research_ci.stages.data.defaults import freshness
from research_ci.stages.skeleton import Found, Step


def fresh(seen: Seen) -> Found:
    """Data whose newest row is older than ``STALE_AFTER`` is a warning: collect live data first. The data check
    warns too."""
    return freshness(seen.latest, seen.now)


CHECKS = [Step("data is current", "warns", fresh, "Warns when the newest row is too old")]
