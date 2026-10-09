"""Null worlds, declared rather than run. Each keeps one alternative explanation and destroys the claimed information.

The challenger writes them into ``nulls/<layer>.py`` as ``NULLS = [Null(...), ...]``; the engine runs them.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BlockPermute:
    """Permute labels across time blocks of ``block`` (for example a horizon), keeping autocorrelation."""

    block: str


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
class ShiftTime:
    """Shift outputs against labels by a random offset between ``min_shift`` and ``max_shift``, or by whole years."""

    min_shift: str
    max_shift: str | None = None
    whole_years: bool = False


@dataclass(frozen=True)
class ReplaceField:
    """Scramble the named input fields and recompute the output: the claimed mechanism is removed."""

    fields: tuple[str, ...]


@dataclass(frozen=True)
class RandomEvents:
    """Replace selected events with unselected ones from the same pool, same entity and density."""

    pool: str


@dataclass(frozen=True)
class RandomizeTiming:
    """Random decisions at the same rate as the real ones (algo layer)."""

    keep: str = "rate"


World = BlockPermute | PermuteWithin | ShiftTime | ReplaceField | RandomEvents | RandomizeTiming


@dataclass(frozen=True)
class Null:
    """One alternative explanation ("it is only X") and the world that keeps X.

    ``explains`` is the id of an alternative in the contract. ``keeps`` names the column X lives in (``"time"``: only
    when a decision happens); a ``PermuteWithin`` world keeps its own bucket unless told otherwise. The engine checks
    the claim: an output that carries only X must not beat this world. ``criteria`` limits it to some criteria by
    name; empty means every criterion of the layer.
    """

    explains: str
    world: World
    keeps: str | None = None
    criteria: tuple[str, ...] = ()
    reps: int = 200
    note: str = ""

    def __post_init__(self) -> None:
        if not self.explains.strip():
            raise ValueError("say which alternative explanation this null world keeps")
        if self.reps < 20:
            raise ValueError("a null world needs at least 20 repetitions")
        object.__setattr__(self, "criteria", tuple(self.criteria))
        if self.keeps is None and isinstance(self.world, PermuteWithin):
            object.__setattr__(self, "keeps", self.world.bucket_by)


def nulls_of(module: Any) -> list[Null]:
    found = getattr(module, "NULLS", None)
    if found is None:
        return []
    if not isinstance(found, list) or not all(isinstance(n, Null) for n in found):
        raise ValueError("nulls/<layer>.py must define NULLS = [Null(...), ...]")
    return found
