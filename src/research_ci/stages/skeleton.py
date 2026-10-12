"""What every layer is made of: four slots, filled by two lists of steps.

Slot 1, the engine's floor: the premises and the decisive test, which passes when the claim beats random by
``z ≥ z_need``. Slot 2, the engine's lists: the default list and the library evals an idea lists, which test generic
H0s and are shown, or treat the data. Slot 3, the lab's evals and null worlds: tests of the claim the lab writes, which
can veto it but never pass what slot 1 failed. Slot 4, the lab evals' own tests, which the engine runs before a lab
eval judges anything.

A stage runs two lists of steps. Its floor list (``stages/<layer>/floor.py``) holds the steps every run takes, those of
slots 1, 3 and 4 and the steps that run the library evals an idea lists; it is part of the engine hash, and no idea can
change it. Its default list (``stages/<layer>/defaults.py``) holds the checks to adjust from experience, slot 2; the
engine hash leaves it out. A floor list marks with ``DEFAULTS`` where the default list runs.

A step names the eval methods it runs through its function, the role it plays, its slot, and one short line on what it
checks; its function's docstring says the rest. Every step reads one context object of its stage and returns what it
found: checks, warnings, treatments, items a person must decide, and what it shows. The stage builds the context, runs
both lists in order, and combines what they found into its state.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, TypeVar

from research_ci.kernel.verdict.check import Check
from research_ci.stages.common import Treatment

Role = Literal["blocks", "decides", "vetoes", "treats", "warns", "shown"]
ROLES: dict[str, str] = {
    "blocks": "stops the stage when it fails: BLOCKED, refused, TAINTED, or FAILED",
    "decides": "PASS or FAIL; only the decisive test",
    "vetoes": "rejects the claim where it does not hold; never passes what the decisive test failed",
    "treats": "changes the data by default, weighed by its influence",
    "warns": "a warning on the card, or an item a person decides",
    "shown": "on the card, decides nothing",
}
SLOTS: dict[int, str] = {
    1: "floor",
    2: "engine lists",
    3: "lab evals",
    4: "tests of lab evals",
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
    declaration of a plugin or a file of the idea; ``slot`` is the slot it fills (``SLOTS``), by default that of its
    list: 1 for the floor list, 2 for the default list."""

    name: str
    role: Role
    run: Callable[[S], Found]
    what: str
    needs: str = ""
    slot: int | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"step {self.name!r}: the role is one of {', '.join(ROLES)}")
        if self.slot is not None and self.slot not in SLOTS:
            raise ValueError(f"step {self.name!r}: the slot is one of {', '.join(map(str, SLOTS))}")


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
