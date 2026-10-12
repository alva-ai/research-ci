"""What an idea runs as its own evals: ``EVALS = [...]`` in ``ideas/<idea>/data.py`` and ``ideas/<idea>/feat.py``.

An entry is an eval function, or one wrapped in ``Eval`` where the layer needs to know more about it. The function is a
general eval from the engine's library (``research_ci.eval``; ``rci methods`` lists them), proven by the engine's CI and
trusted, or a lab eval: a function of the idea's own, such as one in ``ideas/<idea>/evals/<name>.py``, whose module
holds its tests the way the library's modules hold theirs: ``CASES`` (required), ``PROPERTIES`` (optional) and, for a
data eval, ``PITS`` (optional) (below). A library eval tests a generic H0, that the scores are random or the data
generic, and is shown, or treats the data like the default checks; it sets no ``h0``. A lab eval is a test of the claim:
why a case is wrong is specific to the case, so every layer with ``EVALS`` needs at least one, each names the H0 it
rejects in one sentence (``h0``), and it can veto. It can never pass what the decisive test failed. Before it runs
anything, the engine runs the tests of every lab eval an idea lists, and a layer whose lab evals do not all pass them is
refused; ``rci eval check <file>`` runs them any time, for free.

In ``data.py`` an eval gets the explore data, a FieldTable, and returns the rows it flags (``entity``,
``event_time``, optionally ``field``); no rows means it found nothing. ``on_fail`` is what the engine does with them:
masks them (``mask``, the default), drops their entities (``drop_entities``), asks a person (``ask``), or, for a lab
eval, blocks the data check (``block``): it judges the data after the treatments found before it, and any row it
flags there is a veto. ``covers`` names the (category, field) pairs it
claims to catch: the engine plants pits of those categories in clean copies of the data, the built-in ones and those of
``PITS`` in the eval's module, and the eval must flag every pit that moves the contract's decisive number by the
materiality threshold. Evals only add to the data check; they cannot waive a step.

In ``feat.py`` an eval gets the evaluation table of one horizon (``t``, ``entity``, ``score``, ``y`` and its
components ``y_<part>``, the candidate columns) and returns one value per decision time (``t``, ``value``); the
engine averages the values by blocks of the horizon and compares the average with the same in the floor world and the
idea's null worlds that name it, as a z score. A library eval is shown next to the decisive test; a lab eval must reach
``z ≥ 2`` at every horizon it covers, or the submission fails there. ``label`` names the contract's label (its first by
default), ``horizons`` picks its horizons (all by default), and ``combine`` says how they count: the best,
standardized, one vote for all its horizons (``max``, the default), or each horizon on its own (``each``). An eval
named like a default diagnostic of the engine is judged in its place.

What an eval declares about itself, for the engine to check before it is trusted: its tests. An eval's module lists
``CASES``, answers worked by hand, and may list ``PROPERTIES``, statistical behaviour on synthetic markets the engine
builds. The engine's library is held to them in its CI; an idea's lab evals are held to theirs before the engine runs
them, and a layer with one whose tests fail is refused.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Literal

import polars as pl

from research_ci.api.data import OnFail

Layer = Literal["data", "feat"]
Combine = Literal["max", "each"]
Origin = Literal["library", "lab"]
LIBRARY = "research_ci.eval"
TABLE_KEYS = ("t", "entity", "score", "y")
ON_FAIL = ("mask", "drop_entities", "ask", "block")
COMBINE = ("max", "each")
FORMAT: dict[str, str] = {
    "data": (
        "data.py defines EVALS = [Eval(fn, h0='<the H0 it rejects>', on_fail=..., covers=[(category, field)]), ...], "
        "with at least one lab eval: fn(fields) returns the rows it flags (entity, event_time, optionally field), "
        "on_fail is mask (the default), drop_entities, ask, or block (a lab eval only: any row it flags in the data "
        "after the treatments before it blocks the data), and covers names the pits it claims to catch; a lab eval "
        "is a function of the idea's own in evals/<name>.py with CASES in its module (`rci eval check <file>`), and "
        "h0 names in one sentence the H0 it rejects; an eval of the engine's library (`rci methods`; its options with "
        "functools.partial) sets no h0"
    ),
    "feat": (
        "feat.py defines EVALS = [Eval(fn, h0='<the H0 it rejects>', label=..., horizons=(...), combine='max'), ...], "
        "with at least one lab eval: fn(table) returns t, value for each decision time from the evaluation table "
        "(t, entity, score, y, y_<component>, and the candidate columns); a lab eval is a function of the idea's own "
        "in evals/<name>.py with CASES in its module (`rci eval check <file>`), h0 names in one sentence the H0 it "
        "rejects, and it must reach z ≥ 2 at every horizon it covers; an eval of the engine's library "
        "(`rci methods`; its options with functools.partial) sets no h0 and is shown"
    ),
}
FOR_DATA = ("on_fail", "covers")
FOR_FEAT = ("label", "horizons", "combine")
H0_EXAMPLE = {
    "data": "every close is a price the venue traded at",
    "feat": "the scores do not rank the funding part of the carry",
}


@dataclass(frozen=True)
class Eval:
    """An eval with what its layer needs to know: ``of`` is the function; ``name`` defaults to its name; ``h0``, for a
    lab eval and only for one, is the H0 it rejects, in one sentence. For ``data.py``: ``on_fail`` and ``covers``; for
    ``feat.py``: ``label``, ``horizons`` and ``combine``. ``note`` is free text kept with the result."""

    of: Callable[[pl.DataFrame], Any]
    name: str = ""
    h0: str = ""
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
        if not isinstance(self.h0, str):
            raise ValueError(f"eval {name!r}: h0 is one sentence, the H0 the eval rejects")
        object.__setattr__(self, "h0", " ".join(self.h0.split()))
        covers = tuple(tuple(c) for c in self.covers)
        if not all(len(c) == 2 and all(isinstance(x, str) for x in c) for c in covers):
            raise ValueError(f"eval {name!r}: covers is a list of (category, field) pairs")
        object.__setattr__(self, "covers", covers)
        if self.on_fail is not None and self.on_fail not in ON_FAIL:
            raise ValueError(f"eval {name!r}: on_fail is mask, drop_entities, ask or block")
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


def source_of(fn: Callable[..., Any]) -> Origin:
    """An eval of the engine's library, or a lab eval: any other function."""
    module = str(getattr(unwrapped(fn), "__module__", "") or "")
    return "library" if module == LIBRARY or module.startswith(f"{LIBRARY}.") else "lab"


def evals_of(module: Any, layer: Layer) -> list[Eval]:
    """The ``EVALS`` a layer file declares, each an ``Eval`` with its layer's defaults; a bare function is wrapped.
    A lab eval must name its H0 (``h0``) and a library eval must not; only a lab eval blocks. Raises ``ValueError``
    saying what is wrong and what to write."""
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
        _check_h0(e, layer)
        out.append(
            replace(e, on_fail=e.on_fail or "mask") if layer == "data" else replace(e, combine=e.combine or "max")
        )
    names = [e.name for e in out]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"eval names must be unique; give these a name=: {', '.join(twice)}")
    return out


def _check_h0(e: Eval, layer: Layer) -> None:
    fn = getattr(unwrapped(e.of), "__name__", "fn")
    if source_of(e.of) == "lab" and not e.h0:
        raise ValueError(
            f"the lab eval {e.name!r} names no h0: a lab eval tests the claim and can veto it, so it names in one "
            f"sentence the H0 it rejects; write Eval({fn}, h0='<the H0>', ...) in EVALS, such as "
            f"h0={H0_EXAMPLE[layer]!r}"
        )
    if source_of(e.of) == "library" and e.h0:
        raise ValueError(
            f"the library eval {e.name!r} sets h0: an eval of the engine's library tests a generic H0, is shown, and "
            "never vetoes; drop h0=, or write a lab eval of your own in evals/<name>.py with its CASES"
        )
    if source_of(e.of) == "library" and e.on_fail == "block":
        raise ValueError(
            f"the library eval {e.name!r} sets on_fail='block': only a lab eval, with its h0, can block the data; "
            "use mask, drop_entities, or ask, or write a lab eval of your own in evals/<name>.py"
        )


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
    score's rank correlation with the label is about ``ic``: across the entities at each time (``shape="cross"``), or
    over time within each entity (``"series"``, judged against a world that shifts the scores in time)."""

    times: int = 120
    entities: int = 100
    clusters: int = 20
    groups: int = 4
    cluster_share: float = 0.3
    group_share: float = 0.0
    tail: float | None = 4.0
    ic: float = 0.0
    shape: Literal["cross", "series"] = "cross"

    def __post_init__(self) -> None:
        if not 0 < self.clusters <= self.entities or self.groups < 1 or self.times < 2:
            raise ValueError("a market needs at least two times and no more clusters than entities")
        if self.shape not in ("cross", "series"):
            raise ValueError("a market's shape is cross or series")
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
