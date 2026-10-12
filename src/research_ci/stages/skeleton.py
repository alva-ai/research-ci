"""What a stage's two lists are made of.

A stage runs two lists of steps. Its floor list (``stages/<layer>/floor.py``) holds the steps every run takes; it is
part of the engine hash, and no idea can change it. Its default list (``stages/<layer>/defaults.py``) holds the checks
to adjust from experience; the engine hash leaves it out. A floor list marks with ``DEFAULTS`` where the default list
runs.

A step names the eval methods it runs through its function, the role it plays, and one short line on what it checks;
its function's docstring says the rest. Every step reads one context object of its stage and returns what it found:
checks, warnings, treatments, items a person must decide, and what it shows. The stage builds the context, runs both
lists in order, and combines what they found into its state.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, TypeVar

from research_ci.kernel.verdict.check import Check
from research_ci.stages.common import Treatment

Role = Literal["blocks", "decides", "treats", "warns", "shown"]
ROLES: dict[str, str] = {
    "blocks": "stops the stage when it fails: BLOCKED, refused, TAINTED, or FAILED",
    "decides": "PASS or FAIL; only the decisive test",
    "treats": "changes the data by default, weighed by its influence",
    "warns": "a warning on the card, or an item a person decides",
    "shown": "on the card, decides nothing",
}
S = TypeVar("S")


@dataclass
class Found:
    """What a step found: ``checks`` and ``warnings`` for the card, ``treatments`` the stage applies by default,
    ``decide`` the items a person must look at, and ``shown``, what goes into the stage's record by key."""

    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    treatments: list[Treatment] = field(default_factory=list)
    decide: list[str] = field(default_factory=list)
    shown: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Step(Generic[S]):
    """One entry of a list: ``run`` reads the stage's context; ``needs`` names what it runs only with, such as a
    declaration of a plugin or a file of the idea."""

    name: str
    role: Role
    run: Callable[[S], Found]
    what: str
    needs: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"step {self.name!r}: the role is one of {', '.join(ROLES)}")


class _Defaults:
    def __repr__(self) -> str:
        return "DEFAULTS"


DEFAULTS = _Defaults()


def ordered(floor: Sequence[Step[S] | _Defaults], defaults: Iterable[Step[S]]) -> list[Step[S]]:
    """The floor list with the default list where it marks it."""
    out: list[Step[S]] = []
    for item in floor:
        out += list(defaults) if isinstance(item, _Defaults) else [item]
    return out


def tagged(found: Found, by: str) -> Found:
    """``found`` with its checks marked as found by ``by`` (the ``floor`` or the ``default`` list), unless a check
    says whose it is."""
    found.checks = [c if c.by else replace(c, by=by) for c in found.checks]
    return found


def run(step: Step[S], seen: S, defaults: Iterable[Step[S]] = ()) -> Found:
    """One step on its stage's context, its checks marked with the list it comes from."""
    return tagged(step.run(seen), "default" if any(step is d for d in defaults) else "floor")


def combined(found: Iterable[Found]) -> Found:
    """What a run of steps found, in order; a later step's ``shown`` replaces an earlier one's under the same key."""
    out = Found()
    for f in found:
        out.checks += f.checks
        out.warnings += f.warnings
        out.treatments += f.treatments
        out.decide += f.decide
        out.shown |= f.shown
    return out
