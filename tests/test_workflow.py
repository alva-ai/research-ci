import pytest

from research_ci.workflow import (
    WorkflowError,
    approve,
    begin_confirmation,
    card_id,
    finish_confirmation,
    new_workflow,
    next_action,
    read_workflow,
    record_attempt,
    reject,
    reopen,
    require_stage_ready,
    submit_for_review,
    write_workflow,
)


def test_approval_unlocks_only_the_next_enabled_stage(tmp_path) -> None:
    state = new_workflow(("data", "feat", "sig", "algo"), "contract-card")
    write_workflow(tmp_path, state)

    with pytest.raises(WorkflowError, match="RCI-STAGE-NOT-READY"):
        require_stage_ready(read_workflow(tmp_path), "data")

    _, next_stage = approve(
        tmp_path,
        stage="contract",
        by="human",
        decision="continue",
        message="the research boundary is accepted",
    )

    assert next_stage == "data"
    state = read_workflow(tmp_path)
    assert state["stages"]["contract"] == "approved"
    assert state["stages"]["data"] == "ready"
    assert state["stages"]["feat"] == "locked"
    assert next_action(state)["stage"] == "data"
    assert next_action(state)["actor"] == "agent"


def test_only_a_valid_attempt_can_enter_review(tmp_path) -> None:
    write_workflow(
        tmp_path,
        new_workflow(("data", "feat", "sig", "algo"), "contract-card"),
    )
    approve(tmp_path, stage="contract", by="human", decision="continue")
    record_attempt(
        tmp_path,
        stage="data",
        attempt="0001",
        path="data/attempts/0001",
        valid=False,
        card="invalid-card",
        checks="invalid",
    )

    with pytest.raises(WorkflowError, match="RCI-ATTEMPT-INVALID"):
        submit_for_review(
            tmp_path,
            stage="data",
            card="invalid-card",
            recommendation="continue",
            attempt="0001",
        )

    record_attempt(
        tmp_path,
        stage="data",
        attempt="0002",
        path="data/attempts/0002",
        valid=True,
        card="draft-card",
        checks="valid",
    )
    submit_for_review(
        tmp_path,
        stage="data",
        card="data-card",
        recommendation="uncertain",
        attempt="0002",
    )

    state = read_workflow(tmp_path)
    assert state["stages"]["data"] == "review"
    assert state["recommendations"]["data"] == "uncertain"
    assert next_action(state)["actor"] == "reviewer"
    assert next_action(state)["allowed_actors"] == ["human", "agent"]
    assert card_id("data-card") != card_id("other-card")


def test_human_can_accept_a_valid_negative_result_and_stop(tmp_path) -> None:
    state = new_workflow(("data", "feat"), "contract-card")
    state["stages"].update({"contract": "approved", "data": "review"})
    state["cards"]["data"] = "data-card"
    state["recommendations"]["data"] = "stop"
    write_workflow(tmp_path, state)

    approved, next_stage = approve(
        tmp_path,
        stage="data",
        by="human",
        decision="stop",
        expected_card="data-card",
    )

    assert next_stage is None
    assert approved["stages"]["data"] == "stopped"
    assert approved["stages"]["feat"] == "locked"
    action = next_action(approved)
    assert action["action"] == "complete"
    assert action["outcome"] == "completed_no_go"


def test_rejection_requires_a_new_attempt(tmp_path) -> None:
    state = new_workflow(("data",), "contract-card")
    state["stages"].update({"contract": "approved", "data": "ready"})
    write_workflow(tmp_path, state)
    record_attempt(
        tmp_path,
        stage="data",
        attempt="0001",
        path="data/attempts/0001",
        valid=True,
        card="draft-card",
        checks="valid",
    )
    submit_for_review(
        tmp_path,
        stage="data",
        card="data-card",
        recommendation="continue",
        attempt="0001",
    )

    reject(
        tmp_path,
        stage="data",
        by="human",
        message="investigate the timestamp alignment",
        expected_card="data-card",
    )

    state = read_workflow(tmp_path)
    assert state["stages"]["data"] == "ready"
    assert state["attempts"]["data"][-1]["rejected"] is True
    assert next_action(state)["action"] == "run"
    with pytest.raises(WorkflowError, match="RCI-ATTEMPT-REJECTED"):
        submit_for_review(
            tmp_path,
            stage="data",
            card="data-card",
            recommendation="continue",
            attempt="0001",
        )


def test_only_human_reopen_invalidates_downstream(tmp_path) -> None:
    state = new_workflow(("data", "feat", "sig"), "contract-card")
    state["stages"].update(
        {
            "contract": "approved",
            "data": "approved",
            "feat": "approved",
            "sig": "ready",
        }
    )
    state["cards"].update({"data": "data-card", "feat": "feat-card"})
    write_workflow(tmp_path, state)

    with pytest.raises(WorkflowError, match="RCI-REOPEN-REQUIRES-HUMAN"):
        reopen(
            tmp_path,
            stage="data",
            by="agent",
            message="try another source",
        )

    reopened = reopen(
        tmp_path,
        stage="data",
        by="human",
        message="try another source",
    )

    assert reopened["stages"]["data"] == "ready"
    assert reopened["stages"]["feat"] == "locked"
    assert reopened["stages"]["sig"] == "locked"
    assert "data" not in reopened["cards"]
    assert "feat" not in reopened["cards"]


def test_reopen_requires_a_new_attempt(tmp_path) -> None:
    state = new_workflow(("data",), "contract-card")
    state["stages"].update({"contract": "approved", "data": "ready"})
    write_workflow(tmp_path, state)
    record_attempt(
        tmp_path,
        stage="data",
        attempt="0001",
        path="data/attempts/0001",
        valid=True,
        card="draft-card",
        checks="valid",
    )
    submit_for_review(
        tmp_path,
        stage="data",
        card="data-card",
        recommendation="continue",
        attempt="0001",
    )
    approve(
        tmp_path,
        stage="data",
        by="human",
        decision="continue",
        expected_card="data-card",
    )
    reopen(
        tmp_path,
        stage="data",
        by="human",
        message="rebuild with the revised source",
    )

    with pytest.raises(WorkflowError, match="RCI-ATTEMPT-SUBMITTED"):
        submit_for_review(
            tmp_path,
            stage="data",
            card="data-card",
            recommendation="continue",
            attempt="0001",
        )


def test_sig_confirmation_is_required_and_holdout_is_consumed_once(tmp_path) -> None:
    state = new_workflow(("sig",), "contract-card")
    state["stages"].update({"contract": "approved", "sig": "ready"})
    write_workflow(tmp_path, state)
    record_attempt(
        tmp_path,
        stage="sig",
        attempt="0001",
        path="sig/attempts/0001",
        valid=True,
        card="development-card",
        checks="valid",
        confirmation_required=True,
        evidence_scope="confirmation_pending",
    )

    with pytest.raises(WorkflowError, match="RCI-SIG-CONFIRMATION-REQUIRED"):
        submit_for_review(
            tmp_path,
            stage="sig",
            card="development-card",
            recommendation="continue",
            attempt="0001",
        )

    begin_confirmation(
        tmp_path,
        attempt="0001",
        holdout_id="future-001",
        holdout_hash="holdout-hash",
        sealed_hash="sealed-hash",
    )
    finish_confirmation(
        tmp_path,
        attempt="0001",
        evidence="sig/attempts/0001/confirmation/eval/report.jsonl",
        valid=True,
        card="confirmed-card",
    )
    submit_for_review(
        tmp_path,
        stage="sig",
        card="confirmed-card",
        recommendation="continue",
        attempt="0001",
    )
    state = read_workflow(tmp_path)
    assert state["attempts"]["sig"][0]["frozen"] is True
    assert state["holdouts"][0]["status"] == "revealed_valid"

    state["stages"]["sig"] = "ready"
    state["attempts"]["sig"][0]["submitted"] = True
    write_workflow(tmp_path, state)
    record_attempt(
        tmp_path,
        stage="sig",
        attempt="0002",
        path="sig/attempts/0002",
        valid=True,
        card="next-development-card",
        checks="valid",
        confirmation_required=True,
        evidence_scope="confirmation_pending",
    )
    with pytest.raises(WorkflowError, match="RCI-SIG-HOLDOUT-CONSUMED"):
        begin_confirmation(
            tmp_path,
            attempt="0002",
            holdout_id="renamed-future-block",
            holdout_hash="holdout-hash",
            sealed_hash="next-sealed-hash",
        )


def test_version_one_passed_state_is_read_as_approved(tmp_path) -> None:
    state = new_workflow(("data",), "contract-card")
    state["version"] = 1
    state["stages"]["contract"] = "passed"
    write_workflow(tmp_path, state)

    loaded = read_workflow(tmp_path)
    assert loaded["version"] == 3
    assert loaded["stages"]["contract"] == "approved"
