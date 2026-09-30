"""What a layer file declares. The engine imports ``ideas/<idea>/<layer>.py`` and reads these names:

- ``features(panel, candidates)``: the output at one decision time, from the panel known at that time, one row per
  candidate. ``candidates`` is the universe table at that time (``None`` when the contract names no universe, and
  the output is then one row per entity). Labels never reach it. Code that learns from labels defines ``fit(X, y)``
  and ``predict(X)`` instead, and the engine runs walk-forward.
- ``LOOKBACK``: how far back ``features`` reads, for example ``"168h"``, or ``None`` for all history.
- ``PANEL_FORMAT``: ``"pandas"`` (default) or ``"polars"``.
- ``CRITERIA``: the pre-registered criteria the claim must pass.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import pandas as pd


class Context(Protocol):
    """Helpers the engine hands to each evaluator. Everything is restricted to the current tier and world."""

    horizon: str

    def label(self, name: str) -> pd.Series: ...
    def baseline(self) -> pd.Series: ...
    def rank_ic(self, x: pd.Series, y: pd.Series) -> float: ...
    def residualize(self, x: pd.Series, on: pd.Series) -> pd.Series: ...
    def consume(self, scores: pd.Series) -> "Result": ...


class Result(Protocol):
    net: pd.Series
    fees: pd.Series


Evaluator = Callable[[pd.DataFrame, Context], float]


@dataclass(frozen=True)
class Criterion:
    """One pre-registered way of testing the claim.

    ``evaluate(outputs, ctx)`` returns one number from the layer's outputs; it may be any metric, and any search
    inside it is repeated in every null world. A ``stat`` criterion is compared with its null worlds as a z score; a
    ``limit`` criterion is compared with ``limit`` directly. ``horizons`` picks label horizons from the contract
    (``None`` means all); with ``combine="max"`` the whole family counts once and each null world keeps its own
    maximum (GridMax), after each horizon is standardized. ``required=False`` records a reference criterion that
    appears on the card but does not decide the verdict.
    """

    name: str
    evaluate: Evaluator
    kind: Literal["stat", "limit"] = "stat"
    horizons: tuple[str, ...] | None = None
    combine: Literal["max", "each"] = "max"
    limit: float | None = None
    direction: Literal["greater", "less"] = "greater"
    required: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if (self.kind == "limit") != (self.limit is not None):
            raise ValueError("a limit criterion needs a limit, and only a limit criterion has one")


def criteria_of(module: Any) -> list[Criterion]:
    found = getattr(module, "CRITERIA", None)
    if not isinstance(found, list) or not all(isinstance(c, Criterion) for c in found) or not found:
        raise ValueError("a layer file must define CRITERIA = [Criterion(...), ...]")
    names = [c.name for c in found]
    if len(names) != len(set(names)):
        raise ValueError(f"criterion names must be unique: {names}")
    return found
