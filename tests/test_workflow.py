from research_ci.workflow import (
    approve,
    card_id,
    new_workflow,
    next_action,
    read_workflow,
    require_stage_ready,
    submit_for_review,
    write_workflow,
)


def test_approval_unlocks_only_the_next_enabled_stage(tmp_path) -> None:
    state = new_workflow(("data", "feat", "sig", "algo"), "contract-card")
    write_workflow(tmp_path, state)

    require_stage_ready(read_workflow(tmp_path), "contract")
    _, next_stage = approve(
        tmp_path,
        stage="contract",
        by="agent",
        message="contract structure is valid",
    )

    assert next_stage == "data"
    state = read_workflow(tmp_path)
    assert state["stages"]["contract"] == "passed"
    assert state["stages"]["data"] == "ready"
    assert state["stages"]["feat"] == "locked"
    assert next_action(state)["stage"] == "data"


def test_review_card_is_required_before_data_can_be_approved(tmp_path) -> None:
    write_workflow(
        tmp_path,
        new_workflow(("data", "feat", "sig", "algo"), "contract-card"),
    )
    approve(tmp_path, stage="contract", by="human")
    submit_for_review(
        tmp_path,
        stage="data",
        card="data-card",
        result_status="review",
    )

    state, next_stage = approve(
        tmp_path,
        stage="data",
        by="human",
        expected_card="data-card",
    )

    assert next_stage == "feat"
    assert state["stages"]["data"] == "passed"
    assert state["approvals"][-1]["card"] == "data-card"
    assert card_id("data-card") != card_id("other-card")
