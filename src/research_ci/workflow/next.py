from dataclasses import dataclass
from typing import Literal

from research_ci.workflow.machine import Kind, State

Who = Literal["agent", "human", "challenger", "scheduler"]
CHECKPOINT_RUN: dict[str, tuple[Who, str]] = {
    "seal": ("human", "rci seal run"),
    "forward": ("scheduler", "rci forward --all"),
    "capital": ("human", "rci capital run"),
    "monitor": ("scheduler", "rci monitor --all"),
}
AGENT: dict[str, tuple[str, str, str]] = {
    "contract": (
        "derive",
        "rci contract derive",
        "fill in contract.py (START.md says how; `rci plugins` lists the plugins, their options and fields), then "
        "derive the contract",
    ),
    "data": (
        "check",
        "rci data check",
        "check the data against the contract; tests for risks the default checks miss go in data.py "
        "(`rci methods` lists what they can call)",
    ),
    "feat": (
        "submit",
        "rci feat submit <file>",
        "write research/<name>.py from research/example.py (polars expressions; row loops run with a warning), try "
        "it with `rci feat explore <file>`, then submit it; give each alternative of contract.py a world in "
        'nulls/feat.py, such as NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", '
        'block="horizon"))], or the ResultCard marks it uncovered; diagnostics in feat.py are optional; '
        "`rci methods` lists what diagnostics can call and the fields of a null world",
    ),
}
BUILT = frozenset({"contract", "data", "feat"})
AGAIN: dict[str, str] = {
    "contract": "fix contract.py and derive it again",
    "data": "fix what it names and check again",
    "feat": "write a new features file in research/ and submit it",
}
CALIBRATION = "derive the changed contract with `rci contract derive`, then check the data again"
CARD = {"contract": "ContractCard", "data": "DataCard"}
SHOW = {"contract": "rci contract show", "data": "rci data show", "feat": "rci feat show"}


@dataclass(frozen=True)
class StageStatus:
    stage: str
    kind: Kind
    state: State
    version: int
    trials: int = 0
    budget: int | None = None
    note: str = ""
    approved: bool = False

    @property
    def trials_left(self) -> int | None:
        return None if self.budget is None else max(self.budget - self.trials, 0)


@dataclass(frozen=True)
class IdeaStatus:
    idea: str
    stages: tuple[StageStatus, ...]

    def states(self) -> dict[str, State]:
        return {s.stage: s.state for s in self.stages}

    def stage(self, name: str) -> StageStatus:
        for s in self.stages:
            if s.stage == name:
                return s
        raise KeyError(name)


@dataclass(frozen=True)
class Action:
    who: Who
    stage: str
    kind: str
    command: str
    reason: str


def next_actions(status: IdeaStatus) -> list[Action]:
    """What should happen next and who should do it. Every driver, human or not, reads only this."""
    for s in status.stages:
        if s.state is not State.PASSED:
            return [_checkpoint(s) if s.kind is Kind.CHECKPOINT else _stage(s)]
    return []


def _stage(s: StageStatus) -> Action:
    """OPEN and STALE are the agent's turn; REVIEW and PAUSED a person's."""
    name = s.stage
    if s.state is State.REVIEW:
        what = f"{s.note}; review" if s.note and name not in CARD else "review"
        reason = f"{what} the {CARD.get(name, 'ResultCard')} and approve it"
        if name in SHOW:
            reason += f"; `{SHOW[name]}` prints it with the card id that a relayed approval quotes"
        return Action("human", name, "approve", f"rci approve --stage {name}", reason)
    if s.state is State.PAUSED:
        reason = f"all {s.budget} submissions used; extend the budget or drop the idea"
        return Action("human", name, "extend", f"rci approve --stage {name} --extend N", reason)
    if s.kind is Kind.LAYER and name not in BUILT and s.state is State.OPEN:
        reason = (
            f"the {name} layer is not built yet, so this idea is done for now: every stage before {name} passed; "
            "nothing to run until Research CI adds it"
        )
        return Action("agent", name, "wait", "", reason)
    kind, command, default = AGENT.get(name, ("submit", f"rci {name} submit <file>", "submit the research code"))
    reason = default
    if s.note:
        again = CALIBRATION if "calibrat" in s.note else AGAIN.get(name, "")
        reason = f"{s.note}; {again}" if again else s.note
    if s.state is State.STALE:
        reason = f"{s.note}; {kind} again to start v{s.version + 1}"
    if s.budget is not None:
        reason = f"{reason} ({s.trials_left} of {s.budget} submissions left)"
    return Action("agent", name, kind, command, reason)


def _checkpoint(s: StageStatus) -> Action:
    name, state = s.stage, s.state
    if state is State.STALE:
        reason = f"{s.note}; register the plan again to start v{s.version + 1}"
        return Action("agent", name, "register", f"rci {name} register", reason)
    if not s.approved:
        if state is State.REVIEW:
            return Action("human", name, "approve", f"rci approve --stage {name}", "review the plan and approve it")
        return Action("agent", name, "register", f"rci {name} register", "declare the plan before the checkpoint runs")
    if state is State.OPEN:
        who, command = CHECKPOINT_RUN[name]
        return Action(who, name, "run", command, "run the checkpoint once")
    reason = f"{name} ran once and did not pass; reopen a layer, which makes the checkpoints STALE"
    return Action("human", name, "reopen", "rci <layer> submit <file>", reason)
