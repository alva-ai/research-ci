"""Small review workflow for stage cards and approvals."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_ORDER = ("contract", "data", "feat", "sig", "algo")
WORKFLOW_FILENAME = "workflow.json"


class WorkflowError(ValueError):
    """Raised when a run cannot move to the requested workflow state."""


def card_id(card_body: str) -> str:
    """Return a short stable identifier for the rendered card body."""

    return hashlib.sha256(card_body.encode("utf-8")).hexdigest()[:12]


def render_card(body: str) -> tuple[str, str]:
    """Add a stable card identifier without hashing the identifier itself."""

    identifier = card_id(body)
    return f"{body}\n\n- card_id: `{identifier}`", identifier


def workflow_path(out: str | Path) -> Path:
    return Path(out) / WORKFLOW_FILENAME


def new_workflow(enabled_stages: tuple[str, ...], contract_card: str) -> dict[str, Any]:
    """Create a run state with Contract waiting for review."""

    enabled = set(enabled_stages)
    stages: dict[str, str] = {}
    for stage in STAGE_ORDER:
        if stage == "contract":
            stages[stage] = "review"
        elif stage in enabled:
            stages[stage] = "locked"
        else:
            stages[stage] = "disabled"
    return {
        "version": 1,
        "stages": stages,
        "cards": {"contract": contract_card},
        "results": {},
        "approvals": [],
    }


def read_workflow(out: str | Path) -> dict[str, Any]:
    path = workflow_path(out)
    if not path.exists():
        raise WorkflowError(f"workflow state does not exist: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read workflow state: {path}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("stages"), dict):
        raise WorkflowError(f"invalid workflow state: {path}")
    return state


def write_workflow(out: str | Path, state: dict[str, Any]) -> None:
    path = workflow_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def require_stage_ready(state: dict[str, Any], stage: str) -> None:
    """Require that the requested stage is unlocked by an earlier approval."""

    if stage not in STAGE_ORDER:
        raise WorkflowError(f"unknown workflow stage: {stage}")
    current = state["stages"].get(stage)
    if current not in {"ready", "review", "passed"}:
        raise WorkflowError(
            f"stage {stage!r} is {current!r}; approve the previous card first"
        )
    previous = _previous_enabled_stage(state, stage)
    if previous is not None and state["stages"].get(previous) != "passed":
        raise WorkflowError(f"stage {stage!r} is locked until {previous!r} is approved")


def submit_for_review(
    out: str | Path,
    *,
    stage: str,
    card: str,
    result_status: str,
) -> dict[str, Any]:
    """Set a newly produced card to review and invalidate later stages."""

    state = read_workflow(out)
    require_stage_ready(state, stage)
    state["stages"][stage] = "review"
    state.setdefault("cards", {})[stage] = card
    state.setdefault("results", {})[stage] = result_status
    for downstream in _downstream_enabled_stages(state, stage):
        state["stages"][downstream] = "locked"
        state.get("cards", {}).pop(downstream, None)
        state.get("results", {}).pop(downstream, None)
    write_workflow(out, state)
    return state


def approve(
    out: str | Path,
    *,
    stage: str,
    by: str,
    message: str = "",
    expected_card: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Approve a card and unlock the next enabled stage."""

    if by not in {"human", "agent"}:
        raise WorkflowError("approval source must be human or agent")
    state = read_workflow(out)
    if stage not in STAGE_ORDER:
        raise WorkflowError(f"unknown workflow stage: {stage}")
    if state["stages"].get(stage) != "review":
        raise WorkflowError(
            f"stage {stage!r} is {state['stages'].get(stage)!r}; "
            "only review cards can be approved"
        )
    active_card = state.get("cards", {}).get(stage)
    if not active_card:
        raise WorkflowError(f"stage {stage!r} has no active card")
    if expected_card is not None and expected_card != active_card:
        raise WorkflowError(
            f"card mismatch for {stage!r}: expected {active_card}, got {expected_card}"
        )

    state["stages"][stage] = "passed"
    state.setdefault("approvals", []).append(
        {
            "stage": stage,
            "card": active_card,
            "by": by,
            "message": message,
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    next_stage = _next_enabled_stage(state, stage)
    if next_stage is not None:
        state["stages"][next_stage] = "ready"
    write_workflow(out, state)
    return state, next_stage


def next_action(state: dict[str, Any]) -> dict[str, Any]:
    """Describe the next human/agent action for a run."""

    for stage in STAGE_ORDER:
        status = state["stages"].get(stage)
        if status == "review":
            return {
                "stage": stage,
                "action": "approve",
                "card": state.get("cards", {}).get(stage),
                "reason": "review is required before the next stage",
            }
        if status == "ready":
            return {
                "stage": stage,
                "action": "run",
                "card": state.get("cards", {}).get(stage),
                "reason": "the previous stage is approved",
            }
    return {
        "stage": None,
        "action": "complete",
        "card": None,
        "reason": "all enabled stages are approved",
    }


def _previous_enabled_stage(state: dict[str, Any], stage: str) -> str | None:
    index = STAGE_ORDER.index(stage)
    for candidate in reversed(STAGE_ORDER[:index]):
        if state["stages"].get(candidate) != "disabled":
            return candidate
    return None


def _next_enabled_stage(state: dict[str, Any], stage: str) -> str | None:
    index = STAGE_ORDER.index(stage)
    for candidate in STAGE_ORDER[index + 1 :]:
        if state["stages"].get(candidate) != "disabled":
            return candidate
    return None


def _downstream_enabled_stages(state: dict[str, Any], stage: str) -> tuple[str, ...]:
    index = STAGE_ORDER.index(stage)
    return tuple(
        candidate
        for candidate in STAGE_ORDER[index + 1 :]
        if state["stages"].get(candidate) != "disabled"
    )
