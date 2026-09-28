from enum import StrEnum


class Kind(StrEnum):
    CONTRACT = "contract"
    LAYER = "layer"
    GATE = "gate"


class State(StrEnum):
    OPEN = "OPEN"
    REGISTERED = "REGISTERED"
    CHALLENGED = "CHALLENGED"
    CHECKED = "CHECKED"
    FROZEN = "FROZEN"
    PASSED = "PASSED"
    FAILED = "FAILED"
    TAINTED = "TAINTED"
    PAUSED = "PAUSED"
    STALE = "STALE"
    SKIPPED = "SKIPPED"


class Command(StrEnum):
    REGISTER = "register"
    CHALLENGE = "challenge"
    CHECK = "check"
    APPROVE = "approve"
    SUBMIT = "submit"
    EXTEND = "extend"


class WorkflowError(Exception):
    pass


S = State
C = Command
VERDICTS = frozenset({S.PASSED, S.FAILED, S.TAINTED})
DONE = frozenset({S.PASSED, S.SKIPPED})
BEFORE_FREEZE = frozenset({S.OPEN, S.REGISTERED, S.CHALLENGED, S.CHECKED})

ALLOWED: dict[Kind, dict[Command, frozenset[State]]] = {
    Kind.CONTRACT: {
        C.REGISTER: frozenset({S.OPEN, S.REGISTERED, S.PASSED}),
        C.APPROVE: frozenset({S.REGISTERED}),
    },
    Kind.LAYER: {
        C.REGISTER: BEFORE_FREEZE | {S.FROZEN, S.PASSED, S.FAILED, S.TAINTED, S.STALE},
        C.CHALLENGE: frozenset({S.REGISTERED, S.CHALLENGED, S.CHECKED}),
        C.CHECK: frozenset({S.CHALLENGED, S.CHECKED}),
        C.APPROVE: frozenset({S.CHECKED}),
        C.SUBMIT: frozenset({S.FROZEN, S.FAILED, S.TAINTED}),
        C.EXTEND: frozenset({S.PAUSED}),
    },
    Kind.GATE: {
        C.REGISTER: frozenset({S.OPEN, S.REGISTERED, S.STALE}),
        C.APPROVE: frozenset({S.REGISTERED}),
        C.SUBMIT: frozenset({S.FROZEN}),
    },
}


def transition(kind: Kind, state: State, command: Command, result: State | None = None) -> State:
    """State after ``command``. ``result`` is the outcome of a check or a submission, or the verdict an extension
    restores."""
    allowed = ALLOWED[kind].get(command, frozenset())
    if state not in allowed:
        expected = " or ".join(sorted(allowed)) if allowed else "never"
        raise WorkflowError(f"cannot {command} a {kind} stage that is {state} (allowed from: {expected})")
    if command is C.REGISTER:
        return S.REGISTERED
    if command is C.CHALLENGE:
        return S.CHALLENGED
    if command is C.CHECK:
        return S.CHECKED if _outcome(command, result, {S.PASSED, S.FAILED}) is S.PASSED else S.CHALLENGED
    if command is C.APPROVE:
        return S.PASSED if kind is Kind.CONTRACT else S.FROZEN
    if command is C.SUBMIT:
        return _outcome(command, result, VERDICTS)
    return _outcome(command, result, {S.FAILED, S.TAINTED})


def starts_new_version(state: State, command: Command) -> bool:
    return command is C.REGISTER and state not in BEFORE_FREEZE


def _outcome(command: Command, result: State | None, allowed: set[State] | frozenset[State]) -> State:
    if result is None or result not in allowed:
        raise WorkflowError(f"{command} needs a result in {sorted(allowed)}, got {result}")
    return result
