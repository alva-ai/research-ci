import pytest

from research_ci.workflow.machine import (
    ALLOWED,
    VERDICTS,
    Command,
    Kind,
    State,
    WorkflowError,
    starts_new_version,
    transition,
)

C = Command
S = State


def _walk(kind: Kind, steps: list[tuple[Command, State | None]]) -> State:
    state = S.OPEN
    for command, result in steps:
        state = transition(kind, state, command, result)
    return state


def test_a_layer_goes_from_open_to_passed_in_order() -> None:
    steps = [(C.REGISTER, None), (C.CHALLENGE, None), (C.CHECK, S.PASSED), (C.APPROVE, None), (C.SUBMIT, S.FAILED)]
    assert _walk(Kind.LAYER, steps) is S.FAILED
    assert transition(Kind.LAYER, S.FAILED, C.SUBMIT, S.PASSED) is S.PASSED


@pytest.mark.parametrize(
    ("state", "command"),
    [
        (S.OPEN, C.APPROVE),
        (S.OPEN, C.CHALLENGE),
        (S.OPEN, C.SUBMIT),
        (S.REGISTERED, C.CHECK),
        (S.REGISTERED, C.APPROVE),
        (S.CHALLENGED, C.APPROVE),
        (S.CHECKED, C.SUBMIT),
        (S.FROZEN, C.CHALLENGE),
        (S.PASSED, C.SUBMIT),
        (S.PAUSED, C.SUBMIT),
        (S.PAUSED, C.REGISTER),
        (S.STALE, C.SUBMIT),
    ],
)
def test_out_of_order_layer_commands_are_rejected(state: State, command: Command) -> None:
    with pytest.raises(WorkflowError, match=f"cannot {command}"):
        transition(Kind.LAYER, state, command, S.PASSED)


def test_a_failed_check_stays_challenged() -> None:
    assert transition(Kind.LAYER, S.CHALLENGED, C.CHECK, S.FAILED) is S.CHALLENGED
    assert transition(Kind.LAYER, S.CHECKED, C.CHECK, S.FAILED) is S.CHALLENGED


def test_new_nulls_undo_a_passed_check() -> None:
    assert transition(Kind.LAYER, S.CHECKED, C.CHALLENGE) is S.CHALLENGED


def test_changing_tests_after_the_freeze_starts_a_new_version() -> None:
    for state in (S.OPEN, S.REGISTERED, S.CHALLENGED, S.CHECKED):
        assert not starts_new_version(state, C.REGISTER)
    for state in (S.FROZEN, S.PASSED, S.FAILED, S.TAINTED, S.STALE):
        assert starts_new_version(state, C.REGISTER)
        assert transition(Kind.LAYER, state, C.REGISTER) is S.REGISTERED


def test_submit_needs_a_verdict() -> None:
    with pytest.raises(WorkflowError, match="needs a result"):
        transition(Kind.LAYER, S.FROZEN, C.SUBMIT)
    with pytest.raises(WorkflowError, match="needs a result"):
        transition(Kind.LAYER, S.FROZEN, C.SUBMIT, S.CHECKED)


def test_a_contract_is_derived_then_approved() -> None:
    assert _walk(Kind.CONTRACT, [(C.REGISTER, None), (C.APPROVE, None)]) is S.PASSED
    with pytest.raises(WorkflowError):
        transition(Kind.CONTRACT, S.REGISTERED, C.CHALLENGE)
    with pytest.raises(WorkflowError):
        transition(Kind.CONTRACT, S.OPEN, C.APPROVE)


def test_gates_run_once() -> None:
    assert _walk(Kind.GATE, [(C.REGISTER, None), (C.APPROVE, None), (C.SUBMIT, S.FAILED)]) is S.FAILED
    for state in (S.FAILED, S.TAINTED, S.PASSED):
        with pytest.raises(WorkflowError):
            transition(Kind.GATE, state, C.SUBMIT, S.PASSED)
        with pytest.raises(WorkflowError):
            transition(Kind.GATE, state, C.REGISTER)


def test_extending_a_paused_layer_restores_its_last_verdict() -> None:
    assert transition(Kind.LAYER, S.PAUSED, C.EXTEND, S.TAINTED) is S.TAINTED
    with pytest.raises(WorkflowError):
        transition(Kind.LAYER, S.PAUSED, C.EXTEND, S.PASSED)


def test_only_approve_and_submit_can_produce_a_pass() -> None:
    for kind, commands in ALLOWED.items():
        for command, sources in commands.items():
            for state in sources:
                for result in (None, *State):
                    try:
                        after = transition(kind, state, command, result)
                    except WorkflowError:
                        continue
                    if after is S.PASSED:
                        assert command is C.SUBMIT or (kind is Kind.CONTRACT and command is C.APPROVE)
                    if command is C.SUBMIT:
                        assert after in VERDICTS
                    assert after is not S.SKIPPED
