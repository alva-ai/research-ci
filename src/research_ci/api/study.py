"""What the feat layer reads from an idea.

``ideas/<idea>/feat.py`` declares how the layer is judged, frozen when a person approves it:

- ``CRITERIA``: the pre-registered criteria, each a function of the evaluation table.

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
    """One pre-registered way of testing the claim.

    ``evaluate(table)`` returns one value per decision time (``t``, ``value``) from the evaluation table at one
    horizon; any metric works, and any search inside it is repeated in every null world. The engine averages the
    values by blocks of the horizon and compares the real average with the same average in each null world, as a z
    score. ``horizons`` picks label horizons from the contract (``None`` means all); with ``combine="max"`` the whole
    family counts once and each null world keeps its own maximum, after each horizon is standardized. ``label`` names
    the contract's label (its first by default). ``required=False`` records a reference criterion that appears on the
    card but does not decide the verdict.
    """

    name: str
    evaluate: Evaluator
    label: str | None = None
    horizons: tuple[str, ...] | None = None
    combine: str = "max"
    required: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name the criterion")
        if self.combine not in ("max", "each"):
            raise ValueError("combine is max or each")
        if self.horizons is not None:
            object.__setattr__(self, "horizons", tuple(self.horizons))


def criteria_of(module: Any) -> list[Criterion]:
    found = getattr(module, "CRITERIA", None)
    if not isinstance(found, list) or not all(isinstance(c, Criterion) for c in found) or not found:
        raise ValueError("feat.py must define CRITERIA = [Criterion(...), ...]")
    names = [c.name for c in found]
    if len(names) != len(set(names)):
        raise ValueError(f"criterion names must be unique: {names}")
    return found
