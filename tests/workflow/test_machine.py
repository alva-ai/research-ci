import pytest

from research_ci.workflow.machine import (
    ALLOWED,
    WORK,
    Command,
    Kind,
    State,
    WorkflowError,
    starts_new_version,
    transition,
)

C = Command
S = State


def test_the_agent_s_work_goes_up_for_review_when_it_is_ready_and_stays_open_when_not() -> None:
    for kind, command in ((Kind.CONTRACT, C.DERIVE), (Kind.LAYER, C.CHECK), (Kind.LAYER, C.SUBMIT)):
        assert transition(kind, S.OPEN, command, ready=True) is S.REVIEW
        assert transition(kind, S.OPEN, command, ready=False) is S.OPEN
        assert transition(kind, S.REVIEW, command, ready=False) is S.OPEN, "a later result replaces the one in review"
        assert transition(kind, S.REVIEW, C.APPROVE) is S.PASSED


@pytest.mark.parametrize(
    ("kind", "state", "command"),
    [
        (Kind.CONTRACT, S.OPEN, C.APPROVE),
        (Kind.CONTRACT, S.PASSED, C.APPROVE),
        (Kind.LAYER, S.OPEN, C.APPROVE),
        (Kind.LAYER, S.PASSED, C.APPROVE),
        (Kind.LAYER, S.STALE, C.APPROVE),
        (Kind.LAYER, S.PAUSED, C.SUBMIT),
        (Kind.LAYER, S.PAUSED, C.APPROVE),
        (Kind.LAYER, S.OPEN, C.EXTEND),
        (Kind.LAYER, S.OPEN, C.REGISTER),
        (Kind.CONTRACT, S.OPEN, C.SUBMIT),
    ],
)
def test_out_of_order_commands_are_rejected(kind: Kind, state: State, command: Command) -> None:
    with pytest.raises(WorkflowError, match=f"cannot {command}"):
        transition(kind, state, command, ready=True)


def test_work_on_a_passed_or_stale_stage_starts_a_new_version() -> None:
    for command in WORK:
        for state in (S.OPEN, S.REVIEW):
            assert not starts_new_version(state, command)
        for state in (S.PASSED, S.STALE):
            assert starts_new_version(state, command)
    assert not starts_new_version(S.REVIEW, C.APPROVE) and not starts_new_version(S.PAUSED, C.EXTEND)


def test_extending_a_paused_layer_opens_it_again() -> None:
    assert transition(Kind.LAYER, S.PAUSED, C.EXTEND) is S.OPEN


def test_a_checkpoint_registers_a_plan_a_person_approves_and_runs_once() -> None:
    assert transition(Kind.CHECKPOINT, S.OPEN, C.REGISTER, ready=True) is S.REVIEW
    assert transition(Kind.CHECKPOINT, S.STALE, C.REGISTER, ready=True, approved=True) is S.REVIEW
    assert transition(Kind.CHECKPOINT, S.REVIEW, C.APPROVE) is S.OPEN
    assert transition(Kind.CHECKPOINT, S.OPEN, C.RUN, ready=True, approved=True) is S.PASSED
    assert transition(Kind.CHECKPOINT, S.OPEN, C.RUN, ready=False, approved=True) is S.REVIEW
    with pytest.raises(WorkflowError, match="after a person approved its plan"):
        transition(Kind.CHECKPOINT, S.OPEN, C.RUN, ready=True)
    with pytest.raises(WorkflowError, match="its plan is approved"):
        transition(Kind.CHECKPOINT, S.OPEN, C.REGISTER, ready=True, approved=True)
    for command in (C.REGISTER, C.APPROVE, C.RUN):
        with pytest.raises(WorkflowError, match="it ran once"):
            transition(Kind.CHECKPOINT, S.REVIEW, command, ready=True, approved=True)


def test_only_a_person_s_approval_or_a_checkpoint_run_can_produce_a_pass() -> None:
    for kind, commands in ALLOWED.items():
        for command, sources in commands.items():
            for state in sources:
                for ready in (False, True):
                    for approved in (False, True):
                        try:
                            after = transition(kind, state, command, ready, approved)
                        except WorkflowError:
                            continue
                        if after is S.PASSED:
                            assert command is C.APPROVE or command is C.RUN
