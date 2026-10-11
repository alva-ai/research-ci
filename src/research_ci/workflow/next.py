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
    "contract": ("derive", "rci contract derive", "write contract.py, then derive the contract"),
    "data": ("check", "rci data check", "check the data against the contract"),
    "feat": ("submit", "rci feat submit <file>", "submit a features file"),
}
CARD = {"contract": "ContractCard", "data": "DataCard"}


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
        return Action("human", name, "approve", f"rci approve --stage {name}", reason)
    if s.state is State.PAUSED:
        reason = f"all {s.budget} submissions used; extend the budget or drop the idea"
        return Action("human", name, "extend", f"rci approve --stage {name} --extend N", reason)
    kind, command, default = AGENT.get(name, ("submit", f"rci {name} submit <file>", "submit the research code"))
    reason = s.note or default
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
