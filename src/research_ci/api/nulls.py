"""Null worlds, declared rather than run. Each keeps one alternative explanation and destroys the claimed information.

The challenger writes them into ``nulls/<layer>.py`` as ``NULLS = [Null(...), ...]``; the engine runs them.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PermuteWithin:
    """Shuffle scores across entities inside each bucket of ``bucket_by`` at each decision time (``"time"``: one
    bucket per decision time); the labels stay.

    ``clusters`` names a column whose members move together, so dependence inside a cluster survives. ``block``
    repeats the same shuffle for every decision time inside a block, which keeps overlapping label windows honest;
    ``"horizon"`` uses each label horizon as its own block length.
    """

    bucket_by: str
    clusters: str | None = None
    block: str | None = None


@dataclass(frozen=True)
class Null:
    """One alternative explanation ("it is only X") and the world that keeps X.

    ``explains`` is the id of an alternative in the contract. ``keeps`` names the column X lives in (``"time"``: only
    when a decision happens), the world's own bucket unless told otherwise. In feat a world judges the layer's
    criteria, which are shown next to the decisive effect and do not decide; ``criteria`` limits it to some of them by
    name, and empty means every criterion of the layer.
    """

    explains: str
    world: PermuteWithin
    keeps: str | None = None
    criteria: tuple[str, ...] = ()
    reps: int = 200
    note: str = ""

    def __post_init__(self) -> None:
        if not self.explains.strip():
            raise ValueError("say which alternative explanation this null world keeps")
        if not isinstance(self.world, PermuteWithin):
            raise ValueError("a null world is a PermuteWithin")
        if self.reps < 20:
            raise ValueError("a null world needs at least 20 repetitions")
        object.__setattr__(self, "criteria", tuple(self.criteria))
        if self.keeps is None:
            object.__setattr__(self, "keeps", self.world.bucket_by)


def nulls_of(module: Any) -> list[Null]:
    found = getattr(module, "NULLS", None)
    if found is None:
        return []
    if not isinstance(found, list) or not all(isinstance(n, Null) for n in found):
        raise ValueError("nulls/<layer>.py must define NULLS = [Null(...), ...]")
    return found
