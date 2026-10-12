"""What an idea runs as its own evals: ``EVALS = [...]`` in ``ideas/<idea>/data.py`` and ``ideas/<idea>/feat.py``.

An entry is an eval function, or one wrapped in ``Eval`` where the layer needs to know more about it. The function is a
general eval from the engine's library (``research_ci.eval``; ``rci methods`` lists them), proven by the engine's CI and
trusted, or a case eval: a function of the idea's own, such as one in ``ideas/<idea>/evals/<name>.py``, whose module
holds its tests the way the library's modules hold theirs: ``CASES`` (required) and ``PROPERTIES`` (optional)
(below). Before it runs anything, the engine runs the tests of every case eval an idea uses, and a case eval runs
only when they all pass; ``rci eval check <file>`` runs them any time, for free.

In ``data.py`` an eval gets the explore data, a FieldTable, and returns the rows it flags (``entity``,
``event_time``, optionally ``field``); no rows means it found nothing. ``on_fail`` is what the engine does with them:
masks them (``mask``, the default), drops their entities (``drop_entities``), or asks a person (``ask``). ``covers``
names the (category, field) pairs it claims to catch: the engine plants pits of those categories in clean copies of
the data, the built-in ones and those of ``PITS`` in ``nulls/data.py``, and the eval must flag every pit that moves the
contract's decisive number by the materiality threshold. Evals only add to the data check; they cannot waive a step.

In ``feat.py`` an eval gets the evaluation table of one horizon (``t``, ``entity``, ``score``, ``y`` and its
components ``y_<part>``, the candidate columns) and returns one value per decision time (``t``, ``value``); the
engine averages the values by blocks of the horizon and compares the average with the same in the floor world and the
idea's null worlds, as a z score, and shows it next to the decisive test without deciding anything. ``label`` names
the contract's label (its first by default), ``horizons`` picks its horizons (all by default), and ``combine`` says
how they count: the best, standardized, once for the family (``max``, the default), or the weakest (``each``). An eval
named like a default diagnostic of the engine is judged in its place.

What an eval declares about itself, for the engine to check before it is trusted: its tests. An eval's module lists
``CASES``, answers worked by hand, and may list ``PROPERTIES``, statistical behaviour on synthetic markets the engine
builds. The engine's library is held to them in its CI; an idea's case evals are held to theirs before the engine runs
them, and one whose tests fail is not used.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Literal

import polars as pl

from research_ci.api.data import OnFail

Layer = Literal["data", "feat"]
Combine = Literal["max", "each"]
TABLE_KEYS = ("t", "entity", "score", "y")
ON_FAIL = ("mask", "drop_entities", "ask")
COMBINE = ("max", "each")
FORMAT: dict[str, str] = {
    "data": (
        "data.py defines EVALS = [fn, Eval(fn, on_fail=..., covers=[(category, field)]), ...], where fn(fields) "
        "returns the rows it flags (entity, event_time, optionally field), on_fail is mask (the default), "
        "drop_entities, or ask, and covers names the pits it claims to catch; fn is an eval of the engine's library "
        "(`rci methods`; give it its options with functools.partial) or a case eval of the idea's own with CASES in "
        "its module (`rci eval check <file>`)"
    ),
    "feat": (
        "feat.py defines EVALS = [fn, Eval(fn, label=..., horizons=(...), combine='max'), ...], where fn(table) "
        "returns t, value for each decision time from the evaluation table (t, entity, score, y, y_<component>, and "
        "the candidate columns); fn is an eval of the engine's library (`rci methods`; give it its options with "
        "functools.partial) or a case eval of the idea's own with CASES in its module (`rci eval check <file>`)"
    ),
}
FOR_DATA = ("on_fail", "covers")
FOR_FEAT = ("label", "horizons", "combine")


@dataclass(frozen=True)
class Eval:
    """An eval with what its layer needs to know: ``of`` is the function; ``name`` defaults to its name. For
    ``data.py``: ``on_fail`` and ``covers``; for ``feat.py``: ``label``, ``horizons`` and ``combine``. ``note`` is
    free text kept with the result."""

    of: Callable[[pl.DataFrame], Any]
    name: str = ""
    on_fail: OnFail | None = None
    covers: Sequence[tuple[str, str]] = ()
    label: str | None = None
    horizons: Sequence[str] | None = None
    combine: Combine | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not callable(self.of):
            raise ValueError(f"an eval wraps a function, Eval(fn, ...); got {type(self.of).__name__}")
        name = self.name or name_of(self.of)
        if not name.strip() or name == "<lambda>":
            raise ValueError("name an eval that is a lambda: Eval(lambda ..., name='...')")
        object.__setattr__(self, "name", name)
        covers = tuple(tuple(c) for c in self.covers)
        if not all(len(c) == 2 and all(isinstance(x, str) for x in c) for c in covers):
            raise ValueError(f"eval {name!r}: covers is a list of (category, field) pairs")
        object.__setattr__(self, "covers", covers)
        if self.on_fail is not None and self.on_fail not in ON_FAIL:
            raise ValueError(f"eval {name!r}: on_fail is mask, drop_entities or ask")
        if self.combine is not None and self.combine not in COMBINE:
            raise ValueError(f"eval {name!r}: combine is max or each")
        if self.horizons is not None:
            object.__setattr__(self, "horizons", tuple(self.horizons))


def name_of(fn: Any) -> str:
    """A function's own name; a ``partial`` takes the name of the function it wraps."""
    return str(getattr(unwrapped(fn), "__name__", ""))


def unwrapped(fn: Any) -> Any:
    """The function under any ``functools.partial``."""
    while isinstance(fn, partial):
        fn = fn.func
    return fn


def evals_of(module: Any, layer: Layer) -> list[Eval]:
    """The ``EVALS`` a layer file declares, each an ``Eval`` with its layer's defaults; a bare function is wrapped.
    Raises ``ValueError`` saying what is wrong."""
    found = module.EVALS
    if not isinstance(found, list | tuple):
        raise ValueError(f"EVALS must be a list, not {type(found).__name__}")
    out = []
    for i, item in enumerate(found, start=1):
        if not isinstance(item, Eval) and not callable(item):
            raise ValueError(f"EVALS[{i - 1}] is a {type(item).__name__}: an entry is a function or Eval(fn, ...)")
        e = item if isinstance(item, Eval) else Eval(item)
        other = FOR_FEAT if layer == "data" else FOR_DATA
        wrong = [f for f in other if getattr(e, f) not in (None, ())]
        if wrong:
            home = "feat.py" if layer == "data" else "data.py"
            raise ValueError(f"eval {e.name!r} sets {', '.join(wrong)}, which only {home} reads")
        out.append(
            replace(e, on_fail=e.on_fail or "mask") if layer == "data" else replace(e, combine=e.combine or "max")
        )
    names = [e.name for e in out]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"eval names must be unique; give these a name=: {', '.join(twice)}")
    return out


Statistic = Callable[[pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True)
class Case:
    """``run()`` must give ``expect``; with ``given``, ``run(given)`` must, so a case can name the eval and the frame
    it reads: ``Case("the top two of six", thirds, 0.4, given=SIX)``. Numbers must agree within ``tol``. An expected
    frame must come back with its columns (others are ignored) and the same rows in any order, numbers within ``tol``
    and times compared as instants. An expected number against a returned frame: a frame with ``t`` and ``value`` and
    one row gives that value, any other frame its number of rows."""

    name: str
    run: Callable[..., Any]
    expect: Any
    tol: float = 1e-9
    given: Any = None


@dataclass(frozen=True)
class Market:
    """A synthetic evaluation table: ``entities`` candidates in ``clusters`` (and the clusters in ``groups``) at
    ``times`` hourly decision times. Each label is a group level (share ``group_share`` of its variance), a cluster
    shock (``cluster_share``) and its own noise, Student-t with ``tail`` degrees of freedom (``None``: normal). The
    score's rank correlation with the label is about ``ic``."""

    times: int = 120
    entities: int = 100
    clusters: int = 20
    groups: int = 4
    cluster_share: float = 0.3
    group_share: float = 0.0
    tail: float | None = 4.0
    ic: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.clusters <= self.entities or self.groups < 1 or self.times < 2:
            raise ValueError("a market needs at least two times and no more clusters than entities")
        if self.cluster_share + self.group_share >= 1 or min(self.cluster_share, self.group_share) < 0:
            raise ValueError("the cluster and group shares must leave room for each entity's own noise")
        if not -1 < self.ic < 1:
            raise ValueError("ic is a correlation")


@dataclass(frozen=True)
class Source:
    """A synthetic field table: ``entities`` entities with an hourly ``price`` (a random walk) and a ``flow`` settled
    every ``settle`` hours and known ``lag`` hours after it; a share ``revised`` of the flows gets a second vintage one
    settlement later."""

    entities: int = 12
    hours: int = 24 * 14
    settle: int = 8
    lag: int = 0
    revised: float = 0.0

    def __post_init__(self) -> None:
        if self.entities < 2 or self.hours < 4 * self.settle or self.settle < 1 or self.lag < 0:
            raise ValueError("a source needs two entities and a few settlements")
        if not 0 <= self.revised <= 1:
            raise ValueError("revised is a share")


@dataclass(frozen=True)
class Size:
    """On markets with no information, the statistic must beat the floor world (z ≥ 2) no more often than chance
    allows, over ``markets`` seeds."""

    of: Statistic
    market: Market = field(default_factory=Market)
    markets: int = 100


@dataclass(frozen=True)
class Power:
    """On markets with ``market.ic`` of information, the statistic must beat the floor world at least ``rate`` of
    the time."""

    of: Statistic
    market: Market = field(default_factory=lambda: Market(ic=0.05))
    markets: int = 40
    rate: float = 0.8


@dataclass(frozen=True)
class Recovers:
    """Averaged over ``markets`` seeds and decision times, the statistic must come within ``tol`` of ``expect``."""

    of: Statistic
    market: Market
    expect: float
    tol: float
    markets: int = 20


Property = Size | Power | Recovers
