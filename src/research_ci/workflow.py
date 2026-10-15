"""Deterministic workflow transitions for stage attempts and review."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_ORDER = ("contract", "data", "feat", "sig", "algo")
WORKFLOW_FILENAME = "workflow.json"
WORKFLOW_STATES = {
    "locked",
    "ready",
    "review",
    "approved",
    "stopped",
    "disabled",
}
RECOMMENDATIONS = {"continue", "stop", "uncertain"}
REVIEW_DECISIONS = {"continue", "stop"}


class WorkflowError(ValueError):
    """Raised when a command is not a legal workflow transition."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RCI-WORKFLOW-REFUSED",
        action: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.action = action

    def __str__(self) -> str:
        lines = [f"REFUSED [{self.code}]", super().__str__()]
        if self.action and self.action.get("action") != "complete":
            lines.extend(("", "Next legal action:"))
            if self.action.get("actor"):
                lines.append(f"  actor: {self.action['actor']}")
            if self.action.get("command"):
                lines.append(f"  command: {self.action['command']}")
            if self.action.get("reason"):
                lines.append(f"  reason: {self.action['reason']}")
        return "\n".join(lines)


def card_id(card_body: str) -> str:
    """Return a short stable identifier for the rendered card body."""

    return hashlib.sha256(card_body.encode("utf-8")).hexdigest()[:12]


def render_card(body: str) -> tuple[str, str]:
    """Add a stable card identifier without hashing the identifier itself."""

    identifier = card_id(body)
    return f"{body}\n\n- card_id: `{identifier}`", identifier


def workflow_path(out: str | Path) -> Path:
    return Path(out) / WORKFLOW_FILENAME


def new_workflow(
    enabled_stages: tuple[str, ...],
    contract_card: str,
    *,
    commands: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Create a run with Contract waiting for human review."""

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
        "version": 2,
        "stages": stages,
        "cards": {"contract": contract_card},
        "recommendations": {"contract": "continue"},
        "attempts": {},
        "candidates": {},
        "approvals": [],
        "rejections": [],
        "reopenings": [],
        "commands": commands or {},
    }


def read_workflow(out: str | Path) -> dict[str, Any]:
    path = workflow_path(out)
    if not path.exists():
        raise WorkflowError(
            f"Workflow state does not exist: {path}",
            code="RCI-WORKFLOW-MISSING",
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(
            f"Cannot read workflow state: {path}",
            code="RCI-WORKFLOW-INVALID",
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("stages"), dict):
        raise WorkflowError(
            f"Invalid workflow state: {path}",
            code="RCI-WORKFLOW-INVALID",
        )
    state = _upgrade_workflow(raw)
    invalid = {
        stage: status
        for stage, status in state["stages"].items()
        if status not in WORKFLOW_STATES
    }
    if invalid:
        raise WorkflowError(
            f"Invalid workflow states: {invalid}",
            code="RCI-WORKFLOW-INVALID",
        )
    return state


def write_workflow(out: str | Path, state: dict[str, Any]) -> None:
    path = workflow_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _upgrade_workflow(state)
    path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def submit_contract_for_review(
    out: str | Path,
    *,
    card: str,
    enabled_stages: tuple[str, ...],
    commands: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Create or replace a Contract candidate without reopening an approval."""

    path = workflow_path(out)
    if not path.exists():
        state = new_workflow(enabled_stages, card, commands=commands)
        write_workflow(out, state)
        return state

    state = read_workflow(out)
    current = state["stages"].get("contract")
    if current != "ready":
        raise _refusal(
            state,
            f"Contract is {current!r}; reject or reopen it before deriving "
            "another Card.",
            code="RCI-STAGE-FROZEN",
        )
    enabled = set(enabled_stages)
    state["stages"]["contract"] = "review"
    for stage in STAGE_ORDER[1:]:
        state["stages"][stage] = "locked" if stage in enabled else "disabled"
    state["cards"] = {"contract": card}
    state["recommendations"] = {"contract": "continue"}
    state["candidates"] = {}
    state["commands"] = commands
    write_workflow(out, state)
    return state


def require_stage_ready(state: dict[str, Any], stage: str) -> None:
    """Require the exact state from which an Agent may run a stage."""

    _require_known_stage(stage)
    current = state["stages"].get(stage)
    if current != "ready":
        message = f"Cannot run stage {stage!r}: its workflow state is {current!r}."
        if current == "review":
            message += " Its submitted Card must be reviewed first."
        elif current in {"approved", "stopped"}:
            message += (
                " An accepted stage must be explicitly reopened before it changes."
            )
        elif current == "locked":
            previous = _previous_enabled_stage(state, stage)
            if previous is not None:
                message += (
                    f" Upstream stage {previous!r} is "
                    f"{state['stages'].get(previous)!r}."
                )
        raise _refusal(state, message, code="RCI-STAGE-NOT-READY")
    previous = _previous_enabled_stage(state, stage)
    if previous is not None and state["stages"].get(previous) != "approved":
        raise _refusal(
            state,
            f"Cannot run stage {stage!r} until {previous!r} is approved.",
            code="RCI-UPSTREAM-NOT-APPROVED",
        )


def record_attempt(
    out: str | Path,
    *,
    stage: str,
    attempt: str,
    path: str,
    valid: bool,
    card: str,
    checks: str,
) -> dict[str, Any]:
    """Record a completed Agent attempt while leaving the stage ready."""

    state = read_workflow(out)
    require_stage_ready(state, stage)
    records = state.setdefault("attempts", {}).setdefault(stage, [])
    if any(item.get("id") == attempt for item in records):
        raise WorkflowError(
            f"Attempt {attempt!r} already exists for stage {stage!r}.",
            code="RCI-ATTEMPT-EXISTS",
            action=next_action(state),
        )
    records.append(
        {
            "id": attempt,
            "path": path,
            "valid": valid,
            "checks": checks,
            "card": card,
            "rejected": False,
            "submitted": False,
            "at": _now(),
        }
    )
    write_workflow(out, state)
    return state


def latest_attempt(state: dict[str, Any], stage: str) -> dict[str, Any] | None:
    attempts = state.get("attempts", {}).get(stage, [])
    return attempts[-1] if attempts else None


def submit_for_review(
    out: str | Path,
    *,
    stage: str,
    card: str,
    recommendation: str,
    attempt: str,
) -> dict[str, Any]:
    """Submit one valid attempt and invalidate every downstream candidate."""

    if recommendation not in RECOMMENDATIONS:
        raise WorkflowError(
            f"Unknown recommendation {recommendation!r}; choose continue, stop, "
            "or uncertain.",
            code="RCI-RECOMMENDATION-INVALID",
        )
    state = read_workflow(out)
    require_stage_ready(state, stage)
    selected = _find_attempt(state, stage, attempt)
    if not selected.get("valid"):
        raise _refusal(
            state,
            f"Cannot submit invalid attempt {attempt!r}; fix the reported checks "
            "and run again.",
            code="RCI-ATTEMPT-INVALID",
        )
    if selected.get("rejected"):
        raise _refusal(
            state,
            f"Attempt {attempt!r} was rejected; create a new attempt before "
            "submitting again.",
            code="RCI-ATTEMPT-REJECTED",
        )
    if selected.get("submitted"):
        raise _refusal(
            state,
            f"Attempt {attempt!r} was already submitted; create a new attempt "
            "after reopening the stage.",
            code="RCI-ATTEMPT-SUBMITTED",
        )

    selected["submitted"] = True
    state["stages"][stage] = "review"
    state.setdefault("cards", {})[stage] = card
    state.setdefault("recommendations", {})[stage] = recommendation
    state.setdefault("candidates", {})[stage] = attempt
    _lock_downstream(state, stage)
    write_workflow(out, state)
    return state


def approve(
    out: str | Path,
    *,
    stage: str,
    by: str,
    decision: str = "continue",
    message: str = "",
    expected_card: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Accept a reviewed Card, then continue or stop the research."""

    if by not in {"human", "agent"}:
        raise WorkflowError(
            "Approval source must be human or agent.",
            code="RCI-APPROVAL-SOURCE-INVALID",
        )
    if decision not in REVIEW_DECISIONS:
        raise WorkflowError(
            f"Unknown review decision {decision!r}; choose continue or stop.",
            code="RCI-REVIEW-DECISION-INVALID",
        )
    state = read_workflow(out)
    _require_known_stage(stage)
    if state["stages"].get(stage) != "review":
        raise _refusal(
            state,
            f"Cannot approve stage {stage!r}: only a submitted review Card can "
            "be accepted.",
            code="RCI-NO-REVIEW-CARD",
        )
    active_card = state.get("cards", {}).get(stage)
    if not active_card:
        raise WorkflowError(
            f"Stage {stage!r} has no active Card.",
            code="RCI-CARD-MISSING",
            action=next_action(state),
        )
    if expected_card is not None and expected_card != active_card:
        raise WorkflowError(
            f"Card mismatch for {stage!r}: expected {active_card}, got "
            f"{expected_card}.",
            code="RCI-CARD-MISMATCH",
            action=next_action(state),
        )

    next_stage: str | None = None
    if decision == "continue":
        state["stages"][stage] = "approved"
        next_stage = _next_enabled_stage(state, stage)
        if next_stage is not None:
            state["stages"][next_stage] = "ready"
    else:
        state["stages"][stage] = "stopped"
        _lock_downstream(state, stage)
    state.setdefault("approvals", []).append(
        {
            "stage": stage,
            "card": active_card,
            "attempt": state.get("candidates", {}).get(stage),
            "recommendation": state.get("recommendations", {}).get(stage),
            "decision": decision,
            "by": by,
            "message": message,
            "at": _now(),
        }
    )
    write_workflow(out, state)
    return state, next_stage


def reject(
    out: str | Path,
    *,
    stage: str,
    by: str,
    message: str,
    expected_card: str | None = None,
) -> dict[str, Any]:
    """Return a submitted stage to the Agent's iteration loop."""

    if by not in {"human", "agent"}:
        raise WorkflowError(
            "Review source must be human or agent.",
            code="RCI-APPROVAL-SOURCE-INVALID",
        )
    if not message.strip():
        raise WorkflowError(
            "A rejection must explain what the Agent should revise.",
            code="RCI-REJECTION-MESSAGE-MISSING",
        )
    state = read_workflow(out)
    _require_known_stage(stage)
    if state["stages"].get(stage) != "review":
        raise _refusal(
            state,
            f"Cannot reject stage {stage!r}: it is not awaiting review.",
            code="RCI-NO-REVIEW-CARD",
        )
    active_card = state.get("cards", {}).get(stage)
    if expected_card is not None and expected_card != active_card:
        raise WorkflowError(
            f"Card mismatch for {stage!r}: expected {active_card}, got "
            f"{expected_card}.",
            code="RCI-CARD-MISMATCH",
            action=next_action(state),
        )
    attempt = state.get("candidates", {}).get(stage)
    if attempt is not None:
        _find_attempt(state, stage, attempt)["rejected"] = True
    state.setdefault("rejections", []).append(
        {
            "stage": stage,
            "card": active_card,
            "attempt": attempt,
            "by": by,
            "message": message,
            "at": _now(),
        }
    )
    state["stages"][stage] = "ready"
    state.get("cards", {}).pop(stage, None)
    state.get("recommendations", {}).pop(stage, None)
    state.get("candidates", {}).pop(stage, None)
    write_workflow(out, state)
    return state


def reopen(
    out: str | Path,
    *,
    stage: str,
    by: str,
    message: str,
) -> dict[str, Any]:
    """Explicitly reopen an accepted stage and lock every downstream stage."""

    if by != "human":
        raise WorkflowError(
            "Reopening an accepted stage requires a human decision.",
            code="RCI-REOPEN-REQUIRES-HUMAN",
        )
    if not message.strip():
        raise WorkflowError(
            "Reopening a stage requires a reason.",
            code="RCI-REOPEN-MESSAGE-MISSING",
        )
    state = read_workflow(out)
    _require_known_stage(stage)
    current = state["stages"].get(stage)
    if current not in {"approved", "stopped"}:
        raise _refusal(
            state,
            f"Cannot reopen stage {stage!r}: its state is {current!r}.",
            code="RCI-STAGE-NOT-ACCEPTED",
        )
    state["stages"][stage] = "ready"
    state.setdefault("reopenings", []).append(
        {"stage": stage, "by": by, "message": message, "at": _now()}
    )
    state.get("cards", {}).pop(stage, None)
    state.get("recommendations", {}).pop(stage, None)
    state.get("candidates", {}).pop(stage, None)
    _lock_downstream(state, stage)
    write_workflow(out, state)
    return state


def next_action(state: dict[str, Any]) -> dict[str, Any]:
    """Return the one current legal action for a human or Agent driver."""

    state = _upgrade_workflow(state)
    for stage in STAGE_ORDER:
        status = state["stages"].get(stage)
        if status == "stopped":
            return {
                "actor": None,
                "stage": stage,
                "state": status,
                "action": "complete",
                "command": "",
                "card": state.get("cards", {}).get(stage),
                "reason": f"research stopped after the accepted {stage} Card",
                "outcome": "completed_no_go",
            }
        if status == "review":
            card = state.get("cards", {}).get(stage)
            command = _command(state, stage, "approve")
            if not command:
                command = (
                    f"rci approve --stage {stage} --decision continue "
                    f"--card {card or '<card_id>'} --out <run> --by human"
                )
            return {
                "actor": "human",
                "stage": stage,
                "state": status,
                "action": "review",
                "command": command,
                "card": card,
                "reason": "review the submitted Card and continue, stop, or reject",
                "recommendation": state.get("recommendations", {}).get(stage),
            }
        if status == "ready":
            attempt = latest_attempt(state, stage)
            can_submit = bool(
                attempt
                and attempt.get("valid")
                and not attempt.get("rejected")
                and not attempt.get("submitted")
            )
            kind = "submit" if can_submit else "run"
            command = _command(state, stage, kind)
            if not command:
                command = (
                    f"rci {stage} submit --out <run> --recommendation "
                    "<continue|stop|uncertain>"
                    if can_submit
                    else f"rci {stage} run --out <run> ..."
                )
            reason = (
                f"attempt {attempt['id']} passed the mandatory checks; submit it "
                "or run another attempt"
                if can_submit
                else "run or revise this stage; only a valid submitted attempt "
                "can enter review"
            )
            return {
                "actor": "agent",
                "stage": stage,
                "state": status,
                "action": kind,
                "command": command,
                "card": None,
                "reason": reason,
                "attempt": attempt,
                "optional_help": (
                    [f"rci guide --stage {stage}"] if stage != "contract" else []
                ),
            }
    return {
        "actor": None,
        "stage": None,
        "state": None,
        "action": "complete",
        "command": "",
        "card": None,
        "reason": "all enabled stages are approved",
        "outcome": "completed",
    }


def _upgrade_workflow(raw: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(raw)
    if state.get("version") == 1:
        state["stages"] = {
            stage: "approved" if status == "passed" else status
            for stage, status in state.get("stages", {}).items()
        }
    state["version"] = 2
    state.setdefault("cards", {})
    state.setdefault("recommendations", {})
    state.setdefault("attempts", {})
    state.setdefault("candidates", {})
    state.setdefault("approvals", [])
    state.setdefault("rejections", [])
    state.setdefault("reopenings", [])
    state.setdefault("commands", {})
    return state


def _find_attempt(state: dict[str, Any], stage: str, attempt: str) -> dict[str, Any]:
    for item in state.get("attempts", {}).get(stage, []):
        if item.get("id") == attempt:
            return item
    raise WorkflowError(
        f"Unknown attempt {attempt!r} for stage {stage!r}.",
        code="RCI-ATTEMPT-MISSING",
        action=next_action(state),
    )


def _require_known_stage(stage: str) -> None:
    if stage not in STAGE_ORDER:
        raise WorkflowError(
            f"Unknown workflow stage: {stage}",
            code="RCI-STAGE-UNKNOWN",
        )


def _refusal(
    state: dict[str, Any],
    message: str,
    *,
    code: str,
) -> WorkflowError:
    return WorkflowError(message, code=code, action=next_action(state))


def _command(state: dict[str, Any], stage: str, kind: str) -> str:
    template = str(state.get("commands", {}).get(stage, {}).get(kind, ""))
    if not template:
        return ""
    attempt = latest_attempt(state, stage) or {}
    return template.format(
        card=state.get("cards", {}).get(stage, "<card_id>"),
        attempt=attempt.get("id", "<attempt>"),
    )


def _lock_downstream(state: dict[str, Any], stage: str) -> None:
    for downstream in _downstream_enabled_stages(state, stage):
        state["stages"][downstream] = "locked"
        state.get("cards", {}).pop(downstream, None)
        state.get("recommendations", {}).pop(downstream, None)
        state.get("candidates", {}).pop(downstream, None)


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
