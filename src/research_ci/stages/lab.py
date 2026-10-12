"""What every layer with ``EVALS`` checks of the lab's evals before anything heavy runs: that the layer file lists at
least one lab eval (slot 3), and that every lab eval it lists passes its own tests (slot 4). A layer that fails either
is refused, in feat before a submission is counted and in data before the data is fetched, with the slot, the eval, the
test it failed with what it expected and got, and how to fix it. Listing a lab eval whose tests fail does not set its
veto aside: it refuses the layer."""

from collections.abc import Mapping
from dataclasses import dataclass

from research_ci.api.evals import FORMAT, Eval, Layer, source_of
from research_ci.kernel.verdict.check import Check
from research_ci.stages import skeleton
from research_ci.stages.skeleton import Found, Step
from research_ci.validation import load as ev


@dataclass(frozen=True)
class Lab:
    """What the checks of the lab's evals read: the layer, its file and whether it exists, the evals it lists, and
    their validations."""

    layer: Layer
    file: str
    present: bool
    evals: tuple[Eval, ...]
    validated: Mapping[str, ev.Validation]


def listed(lab: Lab) -> Found:
    """The layer file lists at least one lab eval: a function of the idea's own, with the H0 it rejects and its
    tests."""
    names = [e.name for e in lab.evals if source_of(e.of) == "lab"]
    if names:
        return Found([Check("lab evals listed", "pass", f"{lab.file} lists {', '.join(names)}", "lab")])
    where = f"{lab.file} lists no lab eval" if lab.present else f"there is no {lab.file}"
    detail = (
        f"slot 3, lab evals: {where}, and the {lab.layer} layer needs at least one, a test of the claim written for "
        "this idea, which can veto it: write a function of your own in evals/<name>.py with its CASES next to it "
        "(`rci eval check evals/<name>.py` runs them for free) and list it with the H0 it rejects; "
        f"{FORMAT[lab.layer]}"
    )
    return Found([Check("lab evals listed", "fail", detail, "lab")])


def tested(lab: Lab) -> Found:
    """Every lab eval the layer lists passes its ``CASES`` and ``PROPERTIES`` (kept by the hash of its files, so an
    unchanged one is not run again); one that fails refuses the layer, saying which test, what it expected and got,
    and how to fix it."""
    checks = []
    for e in lab.evals:
        v = lab.validated.get(e.name)
        if v is None or source_of(e.of) != "lab":
            continue
        if v.status == "invalid":
            detail = (
                f"slot 4, tests of lab evals: the lab eval {e.name!r} ({v.file}) failed its tests, so it cannot judge: "
                f"{v.summary()}; {v.fix()}"
            )
            checks.append(Check(e.name, "fail", detail, "lab"))
        else:
            checks.append(Check(e.name, "pass", v.summary(), "lab"))
    return Found(checks)


READY = (
    Step(
        "lab evals listed", "blocks", listed, "The layer file lists at least one lab eval", "data.py, feat.py", slot=3
    ),
    Step(
        "lab eval tests", "blocks", tested, "Each lab eval passes its CASES and PROPERTIES", "data.py, feat.py", slot=4
    ),
)


def refusal(lab: Lab) -> str:
    """Why the layer is refused before anything heavy runs; empty when it is not."""
    found = skeleton.combined(step.run(lab) for step in READY)
    return "; ".join(c.detail for c in found.checks if c.status == "fail")
