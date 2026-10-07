"""What an idea may add to its data checks, for risks the engine cannot derive from the contract or the plugins, such
as a holiday that merges two weekly releases. The engine reads ``TESTS`` from ``ideas/<idea>/data.py`` and ``PITS``
from ``ideas/<idea>/nulls/data.py``; both are optional.

A test returns the rows it flags (``entity`` and ``event_time``, optionally ``field``); no rows means it passed. It
names the (category, field) pairs it claims to catch in ``covers``. Before the test counts, the engine plants pits of
those categories in clean copies of the data, built in or from ``PITS``, and the test must flag every pit that moves
the contract's decisive number by more than the materiality threshold. Tests only add checks; they cannot waive one.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

OnFail = Literal["mask", "drop_entities", "ask"]
BUILTIN_PITS = ("unit", "level_break", "gap", "duplicate", "shift", "leak", "spike", "swap")


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
