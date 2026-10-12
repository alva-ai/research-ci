"""What a data source, a universe, and an idea's pits declare for the data check. The idea's own checks are evals
(``EVALS`` in ``data.py``, ``api/evals.py``); ``PITS`` in a data eval's module, optional, adds pits to the built-in
ones that the eval's ``covers`` is checked against, next to its ``CASES``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

OnFail = Literal["mask", "drop_insts", "ask", "block"]
Plant = Callable[[Mapping[str, pl.DataFrame], str, np.random.Generator], Mapping[str, pl.DataFrame]]


@dataclass(frozen=True)
class Pit:
    """A planted problem of ``category`` in a value column: ``plant(sources, column, rng)`` returns copies of the
    sources with it planted in the source that holds the column, in one inst where the column varies. A data eval's
    module lists them in ``PITS``, next to its ``CASES``: a test of every eval of the module that covers the
    category."""

    category: str
    plant: Plant
    note: str = ""


@dataclass(frozen=True)
class Levels:
    """Value columns that are levels, such as prices, as a data plugin declares them in a class attribute
    ``levels``. A move of ``factor`` or more that lasts over the next ``window`` events of the column's source is a
    level shift: admission asks the independent source about it, and the data check masks the ones it did not confirm
    for ``mask`` on each side. A source that declares no levels is not checked for level shifts."""

    columns: tuple[str, ...]
    factor: float
    window: int
    mask: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        if not self.columns or self.factor <= 1 or self.window < 1:
            raise ValueError("levels need columns, a factor above 1 and a window of at least one event")


@dataclass(frozen=True)
class Hedge:
    """Legs that are one asset on two venues, as a universe plugin declares them in a class attribute ``hedge``. At
    every decision time the column ``price`` of ``wide`` must put the legs within ``max_ratio`` of each other and, when
    both legs' returns from one decision to the next vary by at least ``min_move``, those returns over the pair's last
    ``window`` decisions (at least ``min_obs``) must correlate by ``min_corr`` or more; the data check drops the
    candidate rows where they do not. A universe that declares no hedge is not checked."""

    price: str
    max_ratio: float
    min_corr: float
    window: int
    min_obs: int
    min_move: float

    def __post_init__(self) -> None:
        if self.max_ratio <= 1 or not 0 < self.min_obs <= self.window or self.min_move < 0:
            raise ValueError("a hedge needs a ratio above 1, 0 < min_obs <= window and min_move >= 0")


def pits_of(module: Any) -> list[Pit]:
    return _listed(module, "PITS", Pit)


def _listed(module: Any, name: str, kind: type) -> list[Any]:
    found = getattr(module, name, None)
    if found is None:
        return []
    if not isinstance(found, list) or not all(isinstance(x, kind) for x in found):
        raise ValueError(f"{name} must be a list of {kind.__name__}")
    return found
