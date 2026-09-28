from dataclasses import dataclass
from typing import Literal

from alpha_ci.workflow.machine import DONE, Kind, State

Who = Literal["agent", "human", "challenger", "scheduler"]
GATE_RUN: dict[str, tuple[Who, str]] = {
    "seal": ("human", "aci seal run"),
    "forward": ("scheduler", "aci forward --all"),
    "capital": ("human", "aci capital run"),
    "monitor": ("scheduler", "aci monitor --all"),
}


@dataclass(frozen=True)
class StageStatus:
    stage: str
    kind: Kind
    state: State
    version: int
    trials: int = 0
    budget: int | None = None
    note: str = ""

    @property
    def trials_left(self) -> int | None:
        return None if self.budget is None else max(self.budget - self.trials, 0)


@dataclass(frozen=True)
class IdeaStatus:
    idea: str
    template: str
    stages: tuple[StageStatus, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.stage for s in self.stages)

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
        if s.state not in DONE:
            if s.kind is Kind.CONTRACT:
                return [_contract(s)]
            return [_gate(s) if s.kind is Kind.GATE else _layer(s)]
    return []


def _contract(s: StageStatus) -> Action:
    if s.state is State.REGISTERED:
        return Action("human", s.stage, "approve", "aci approve --stage contract", "review the contract and freeze it")
    return Action("agent", s.stage, "register", "aci contract derive", "write contract.py, then derive the contract")


def _layer(s: StageStatus) -> Action:
    name, state = s.stage, s.state
    left = f"{s.trials_left} of {s.budget} submissions left"
    if state is State.OPEN:
        return Action("agent", name, "register", f"aci {name} register", "declare how this stage will be judged")
    if state is State.STALE:
        return Action(
            "agent", name, "register", f"aci {name} register", f"{s.note}; register again to start v{s.version + 1}"
        )
    if state is State.REGISTERED:
        return Action("challenger", name, "challenge", f"aci {name} challenge", "write the nulls and pitfalls")
    if state is State.CHALLENGED:
        reason = s.note or "calibrate: stand-ins must fail, planted signals must pass, mutants must be caught"
        return Action("agent", name, "check", f"aci {name} check", reason)
    if state is State.CHECKED:
        return Action("human", name, "approve", f"aci approve --stage {name}", "review the tests and freeze them")
    if state is State.PAUSED:
        reason = f"all {s.budget} submissions used; extend the budget or drop the idea"
        return Action("human", name, "extend", f"aci approve --stage {name} --extend N", reason)
    if state is State.FROZEN:
        return Action("agent", name, "submit", f"aci {name} submit", f"submit the research code ({left})")
    reason = f"{state}: {s.note}; " if s.note else f"{state}; "
    return Action("agent", name, "submit", f"aci {name} submit", f"{reason}change the research code ({left})")


def _gate(s: StageStatus) -> Action:
    name, state = s.stage, s.state
    if state in (State.OPEN, State.STALE):
        return Action("agent", name, "register", f"aci {name} register", "declare the plan before the gate runs")
    if state is State.REGISTERED:
        return Action("human", name, "approve", f"aci approve --stage {name}", "review the plan and freeze it")
    if state is State.FROZEN:
        who, command = GATE_RUN[name]
        return Action(who, name, "submit", command, "run the gate once")
    reason = f"{name} ran once and ended {state}; reopen the layer to rework, which restarts the gates"
    return Action("human", name, "reopen", "aci <layer> register", reason)
