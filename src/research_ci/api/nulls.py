"""Null worlds, declared rather than run. Each keeps one alternative explanation and destroys the claimed information.

The lab writes them into ``nulls/feat.py`` as ``NULLS = [Null(...), ...]``, one for every alternative explanation in the
contract; the engine runs them, and each can veto the claim where what it judges does not reach ``z ≥ 2``.
"""

from dataclasses import dataclass
from typing import Any

from research_ci.api.contract import span


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
class ShiftInTime:
    """Shift every entity's scores along its own decision times, circularly, by one offset per shuffle, the same for
    every entity; the labels stay. The scores' own autocorrelation and what they share across entities survive; only
    their alignment with the labels goes. H0: the scores carry no timing information.

    The offset is at least ``min_shift`` and at most the length less ``min_shift``, so a shifted score never lands
    within ``min_shift`` of where it was, either way round; ``"horizon"`` uses each label horizon, so a score's own
    label window never overlaps the one it is judged against. With fewer distinct offsets than shuffles asked for,
    each is drawn once and the card says so.
    """

    min_shift: str = "horizon"

    def __post_init__(self) -> None:
        if self.min_shift == "horizon":
            return
        try:
            positive = span(self.min_shift).value > 0
        except ValueError:
            positive = False
        if not positive:
            raise ValueError(f'min_shift is "horizon" or a positive duration such as "24h", not {self.min_shift!r}')


Shuffle = PermuteWithin | ShiftInTime


@dataclass(frozen=True)
class Null:
    """One alternative explanation ("it is only X") and the world that keeps X.

    ``explains`` is the id of an alternative in the contract. ``keeps`` names the column X lives in (``"time"``: only
    when a decision happens), the world's own bucket unless told otherwise, or ``entity`` for a ``ShiftInTime``. In feat
    a world judges the evals it names in ``evals``, any eval of the layer (a lab eval, a library eval, or a default
    diagnostic), or, when it names none, the decisive effect itself; what it judges must reach ``z ≥ 2`` at every
    horizon, or the submission fails there. It draws its shuffles a batch at a time, up to ``reps``, until each of its
    votes is settled, as the decisive test does.
    """

    explains: str
    world: Shuffle
    keeps: str | None = None
    evals: tuple[str, ...] = ()
    reps: int = 200
    note: str = ""

    def __post_init__(self) -> None:
        if not self.explains.strip():
            raise ValueError("say which alternative explanation this null world keeps")
        if not isinstance(self.world, PermuteWithin | ShiftInTime):
            raise ValueError("a null world is a PermuteWithin or a ShiftInTime")
        if self.reps < 20:
            raise ValueError("a null world needs at least 20 repetitions")
        object.__setattr__(self, "evals", tuple(self.evals))
        if self.keeps is None:
            kept = self.world.bucket_by if isinstance(self.world, PermuteWithin) else "entity"
            object.__setattr__(self, "keeps", kept)


def nulls_of(module: Any) -> list[Null]:
    found = getattr(module, "NULLS", None)
    if found is None:
        return []
    if not isinstance(found, list) or not all(isinstance(n, Null) for n in found):
        raise ValueError("nulls/<layer>.py must define NULLS = [Null(...), ...]")
    return found
