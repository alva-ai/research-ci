from enum import StrEnum


class Kind(StrEnum):
    CONTRACT = "contract"
    LAYER = "layer"
    CHECKPOINT = "checkpoint"


class State(StrEnum):
    OPEN = "OPEN"
    REVIEW = "REVIEW"
    PASSED = "PASSED"
    PAUSED = "PAUSED"
    STALE = "STALE"


class Verdict(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    TAINTED = "TAINTED"


class Command(StrEnum):
    DERIVE = "derive"
    CHECK = "check"
    SUBMIT = "submit"
    REGISTER = "register"
    RUN = "run"
    APPROVE = "approve"
    EXTEND = "extend"


class WorkflowError(Exception):
    pass


S = State
C = Command
WORK = frozenset({C.DERIVE, C.CHECK, C.SUBMIT, C.REGISTER})
AGAIN = frozenset({S.OPEN, S.REVIEW, S.PASSED, S.STALE})

ALLOWED: dict[Kind, dict[Command, frozenset[State]]] = {
    Kind.CONTRACT: {
        C.DERIVE: frozenset({S.OPEN, S.REVIEW, S.PASSED}),
        C.APPROVE: frozenset({S.REVIEW}),
    },
    Kind.LAYER: {
        C.CHECK: AGAIN,
        C.SUBMIT: AGAIN,
        C.APPROVE: frozenset({S.REVIEW}),
        C.EXTEND: frozenset({S.PAUSED}),
    },
    Kind.CHECKPOINT: {
        C.REGISTER: frozenset({S.OPEN, S.REVIEW, S.STALE}),
        C.APPROVE: frozenset({S.REVIEW}),
        C.RUN: frozenset({S.OPEN}),
    },
}


def transition(kind: Kind, state: State, command: Command, ready: bool = False, approved: bool = False) -> State:
    """State after ``command``. ``ready`` says whether the agent's work has a result for a person, or whether a
    checkpoint's run passed. ``approved`` says whether a checkpoint's plan was approved in this version: it then runs
    once and takes nothing else."""
    allowed = ALLOWED[kind].get(command, frozenset())
    if kind is Kind.CHECKPOINT and approved and state is S.REVIEW:
        raise WorkflowError(f"cannot {command} the checkpoint: it ran once; reopen a layer")
    if kind is Kind.CHECKPOINT and state is not S.STALE and approved is not (command is C.RUN):
        if command is C.RUN:
            raise WorkflowError("a checkpoint runs only after a person approved its plan")
        raise WorkflowError(f"cannot {command} the checkpoint: its plan is approved and it runs once")
    if state not in allowed:
        expected = " or ".join(sorted(allowed)) if allowed else "never"
        raise WorkflowError(f"cannot {command} a {kind} stage that is {state} (allowed from: {expected})")
    if command in WORK:
        return S.REVIEW if ready else S.OPEN
    if command is C.APPROVE:
        return S.OPEN if kind is Kind.CHECKPOINT else S.PASSED
    if command is C.RUN:
        return S.PASSED if ready else S.REVIEW
    return S.OPEN


def starts_new_version(state: State, command: Command) -> bool:
    return command in WORK and state in (S.PASSED, S.STALE)
