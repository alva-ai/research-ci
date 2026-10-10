"""What the feat layer reads from an idea.

The contract's decisive effect alone decides a feat submission. ``ideas/<idea>/feat.py``, if it exists, adds what is
shown next to it, read at every submission and recorded with it by hash:

- ``CRITERIA``: criteria, each a function of the evaluation table, judged against the null worlds and shown on the
  card; they never decide the verdict.

Each submission names its own file of features, which may change from one submission to the next:

- ``features(fields, candidates)``: one score per candidate row, ``t``, ``entity``, ``score``, for every decision
  time at once. ``fields`` is the frozen data and ``candidates`` the universe table; labels never reach it. The engine
  computes it again on the data cut at a few decision times, and the rows at or before each cut must not change.

The evaluation table has fixed columns for one horizon: ``t``, ``entity``, ``score``, ``y`` (the label at that
horizon) and its components (``y_flow``, ``y_price``, ...), the universe's cluster column, its group columns, and any
other universe column.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

Evaluator = Callable[[pl.DataFrame], pl.DataFrame]
TABLE_KEYS = ("t", "entity", "score", "y")


@dataclass(frozen=True)
class Criterion:
    """One way of looking at the claim, shown next to the decisive effect.

    ``evaluate(table)`` returns one value per decision time (``t``, ``value``) from the evaluation table at one
    horizon; any metric works, and any search inside it is repeated in every null world. The engine averages the
    values by blocks of the horizon and compares the real average with the same average in each null world, as a z
    score. ``horizons`` picks label horizons from the contract (``None`` means all); with ``combine="max"`` the whole
    family counts once and each null world keeps its own maximum, after each horizon is standardized. ``label`` names
    the contract's label (its first by default).
    """

    name: str
    evaluate: Evaluator
    label: str | None = None
    horizons: tuple[str, ...] | None = None
    combine: str = "max"
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name the criterion")
        if self.combine not in ("max", "each"):
            raise ValueError("combine is max or each")
        if self.horizons is not None:
            object.__setattr__(self, "horizons", tuple(self.horizons))


def criteria_of(module: Any) -> list[Criterion]:
    found = getattr(module, "CRITERIA", [])
    if not isinstance(found, list) or not all(isinstance(c, Criterion) for c in found):
        raise ValueError("CRITERIA in feat.py must be a list of Criterion")
    names = [c.name for c in found]
    if len(names) != len(set(names)):
        raise ValueError(f"criterion names must be unique: {names}")
    return found
