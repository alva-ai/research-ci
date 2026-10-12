"""What an idea runs as its own evals: ``EVALS = [...]`` in ``ideas/<idea>/data.py`` and ``ideas/<idea>/feat.py``.

An entry is an eval function, or one wrapped in ``Eval`` where the layer needs to know more about it. The function is a
general eval from the engine's library (``research_ci.eval``; ``rci methods`` lists them), proven by the engine's CI and
trusted, or a case eval: a function of the idea's own, such as one in ``ideas/<idea>/evals/<name>.py``, whose module
holds its tests the way the library's modules hold theirs: ``CASES`` (required) and ``PROPERTIES`` (optional)
(``api/methods.py``). Before it runs anything, the engine runs the tests of every case eval an idea uses, and a case
eval runs only when they all pass; ``rci eval check <file>`` runs them any time, for free.

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
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
