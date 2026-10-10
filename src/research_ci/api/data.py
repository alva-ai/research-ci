"""What an idea may add to its data checks, for risks the engine's default checks do not cover, such as a holiday that
merges two weekly releases. The engine reads ``TESTS`` from ``ideas/<idea>/data.py`` and ``PITS`` from
``ideas/<idea>/nulls/data.py``; both are optional.

A test returns the rows it flags (``entity`` and ``event_time``, optionally ``field``); no rows means it passed. It
names the (category, field) pairs it claims to catch in ``covers``. Before the test counts, the engine plants pits of
those categories in clean copies of the data, built in (the categories ``PITS`` lists in
``research_ci/stages/defaults/data.py``) or from the idea's ``PITS``, and the test must flag every pit that moves the
contract's decisive number by more than the materiality threshold. Tests only add checks; they cannot waive one.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

OnFail = Literal["mask", "drop_entities", "ask"]


@dataclass(frozen=True)
class DataTest:
    """``on_fail`` is what the engine does with the flagged rows: drop them (``mask``), drop their entities
    (``drop_entities``), or stop for a person (``ask``)."""

    name: str
    check: Callable[[pl.DataFrame], pl.DataFrame]
    covers: Sequence[tuple[str, str]]
    on_fail: OnFail
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "covers", tuple(tuple(c) for c in self.covers))
        if not self.covers:
            raise ValueError(f"data test {self.name!r} must say what it covers")
        if self.on_fail not in ("mask", "drop_entities", "ask"):
            raise ValueError(f"data test {self.name!r}: on_fail must be mask, drop_entities or ask")


@dataclass(frozen=True)
class Pit:
    """A planted problem of ``category`` in ``field``: ``plant(fields, field, rng)`` returns a copy with it planted
    in one entity where the field varies. Written by someone other than the author of the tests it is used against."""

    category: str
    plant: Callable[[pl.DataFrame, str, np.random.Generator], pl.DataFrame]
    note: str = ""


@dataclass(frozen=True)
class Levels:
    """Fields whose values are levels, such as prices, as a data plugin declares them in a class attribute
    ``levels``. A move of ``factor`` or more that lasts over the next ``window`` events is a level shift: admission asks
    the independent source about it, and the data check masks the ones it did not confirm for ``mask`` on each side.
    A source that declares no levels is not checked for level shifts."""

    fields: tuple[str, ...]
    factor: float
    window: int
    mask: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.fields or self.factor <= 1 or self.window < 1:
            raise ValueError("levels need fields, a factor above 1 and a window of at least one event")


@dataclass(frozen=True)
class Hedge:
    """Legs that are one asset on two venues, as a universe plugin declares them in a class attribute ``hedge``. At
    every decision time ``price`` must put the legs within ``max_ratio`` of each other and, when both legs' returns vary
    by at least ``min_move``, their returns over the pair's last ``window`` decisions (at least ``min_obs``) must
    correlate by ``min_corr`` or more; the data check drops the candidate rows where they do not. A universe that
    declares no hedge is not checked."""

    price: str
    max_ratio: float
    min_corr: float
    window: int
    min_obs: int
    min_move: float

    def __post_init__(self) -> None:
        if self.max_ratio <= 1 or not 0 < self.min_obs <= self.window or self.min_move < 0:
            raise ValueError("a hedge needs a ratio above 1, 0 < min_obs <= window and min_move >= 0")


def tests_of(module: Any) -> list[DataTest]:
    return _listed(module, "TESTS", DataTest)


def pits_of(module: Any) -> list[Pit]:
    return _listed(module, "PITS", Pit)


def _listed(module: Any, name: str, kind: type) -> list[Any]:
    found = getattr(module, name, None)
    if found is None:
        return []
    if not isinstance(found, list) or not all(isinstance(x, kind) for x in found):
        raise ValueError(f"{name} must be a list of {kind.__name__}")
    return found
