from dataclasses import dataclass
from typing import Literal

from research_ci.workflow.machine import DONE, Kind, State

Who = Literal["agent", "human", "challenger", "scheduler"]
CHECKPOINT_RUN: dict[str, tuple[Who, str]] = {
    "seal": ("human", "rci seal run"),
    "forward": ("scheduler", "rci forward --all"),
    "capital": ("human", "rci capital run"),
    "monitor": ("scheduler", "rci monitor --all"),
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
            if s.stage == "data":
                return [_data(s)]
            return [_checkpoint(s) if s.kind is Kind.CHECKPOINT else _layer(s)]
    return []


def _contract(s: StageStatus) -> Action:
    if s.state is State.REGISTERED:
        return Action("human", s.stage, "approve", "rci approve --stage contract", "review the contract and freeze it")
    return Action("agent", s.stage, "register", "rci contract derive", "write contract.py, then derive the contract")


def _data(s: StageStatus) -> Action:
    """The engine derives the data checks, so one command registers, challenges and checks them."""
    state = s.state
    if state is State.CHECKED:
        return Action("human", s.stage, "approve", "rci approve --stage data", "review the data card and freeze it")
    if state is State.FROZEN:
        return Action("agent", s.stage, "submit", "rci data submit", "record the frozen data as what feat reads")
    if state is State.PAUSED:
        return _layer(s)
    if "waits for a person" in s.note:
        return Action("human", s.stage, "approve", "rci source approve", s.note)
    reason = s.note or "check the data against the contract"
    return Action("agent", s.stage, "check", "rci data check", reason)


def _layer(s: StageStatus) -> Action:
    name, state = s.stage, s.state
    left = f"{s.trials_left} of {s.budget} submissions left"
    if state is State.OPEN:
        return Action("agent", name, "register", f"rci {name} register", "declare how this stage will be judged")
    if state is State.STALE:
        return Action(
            "agent", name, "register", f"rci {name} register", f"{s.note}; register again to start v{s.version + 1}"
        )
    if state in (State.REGISTERED, State.CHALLENGED):
        reason = s.note or "register again: it reads the nulls and calibrates every criterion"
        return Action("agent", name, "register", f"rci {name} register", reason)
    if state is State.CHECKED:
        return Action("human", name, "approve", f"rci approve --stage {name}", "review the tests and freeze them")
    if state is State.PAUSED:
        reason = f"all {s.budget} submissions used; extend the budget or drop the idea"
        return Action("human", name, "extend", f"rci approve --stage {name} --extend N", reason)
    if state is State.FROZEN:
        return Action("agent", name, "submit", f"rci {name} submit <file>", f"submit the research code ({left})")
    if state is State.INSUFFICIENT:
        reason = f"the data cannot decide yet: wait for new data, submit other research code ({left}), or drop it"
        return Action("human", name, "decide", f"rci {name} show", reason)
    reason = f"{state}: {s.note}; " if s.note else f"{state}; "
    return Action("agent", name, "submit", f"rci {name} submit <file>", f"{reason}change the research code ({left})")


def _checkpoint(s: StageStatus) -> Action:
    name, state = s.stage, s.state
    if state in (State.OPEN, State.STALE):
        return Action("agent", name, "register", f"rci {name} register", "declare the plan before the checkpoint runs")
    if state is State.REGISTERED:
        return Action("human", name, "approve", f"rci approve --stage {name}", "review the plan and freeze it")
    if state is State.FROZEN:
        who, command = CHECKPOINT_RUN[name]
        return Action(who, name, "submit", command, "run the checkpoint once")
    reason = f"{name} ran once and ended {state}; reopen the layer to rework, which restarts the checkpoints"
    return Action("human", name, "reopen", "rci <layer> register", reason)
