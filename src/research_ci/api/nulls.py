"""Null worlds, declared rather than run. Each keeps one alternative explanation and destroys the claimed information.

The challenger writes them into ``nulls/<layer>.py`` as ``NULLS = [Null(...), ...]``; the kernel runs them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockPermute:
    """Permute labels across time blocks of ``block`` (for example a horizon), keeping autocorrelation."""

    block: str


@dataclass(frozen=True)
class PermuteWithin:
    """Shuffle labels across entities inside each bucket of ``bucket_by`` (``"time"`` for each decision time).

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
    """One alternative explanation ("it is only X") and the world that keeps X. ``criteria`` limits it to some
    criteria by name; empty means every stat criterion of the layer."""

    explains: str
    world: World
    criteria: tuple[str, ...] = ()
    reps: int = 200

    def __post_init__(self) -> None:
        if not self.explains.strip():
            raise ValueError("say which alternative explanation this null world keeps")
        if self.reps < 20:
            raise ValueError("a null world needs at least 20 repetitions")
