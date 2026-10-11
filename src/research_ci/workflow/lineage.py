"""The one pipeline every idea goes through, in order, and what reopening a stage does to the stages after it."""

from collections.abc import Collection, Mapping

from research_ci.workflow.machine import Kind, State, WorkflowError

CONTRACT = "contract"
LAYERS = ("data", "feat", "signal", "algo")
CHECKPOINTS = ("seal", "forward", "capital", "monitor")
STAGES = (CONTRACT, *LAYERS, *CHECKPOINTS)


def kind_of(stage: str) -> Kind:
    if stage == CONTRACT:
        return Kind.CONTRACT
    return Kind.LAYER if stage in LAYERS else Kind.CHECKPOINT


def upstream(stage: str) -> tuple[str, ...]:
    return STAGES[: _index(stage)]


def downstream(stage: str) -> tuple[str, ...]:
    return STAGES[_index(stage) + 1 :]


def blockers(states: Mapping[str, State], stage: str) -> list[str]:
    return [s for s in upstream(stage) if states[s] is not State.PASSED]


def invalidated(states: Mapping[str, State], started: Collection[str], stage: str) -> list[str]:
    """Downstream stages that must go STALE when ``stage`` is reopened: every one that has started its current
    version, that is, that is not OPEN or has a record in it, and is not STALE already."""
    return [
        s for s in downstream(stage) if states[s] is not State.STALE and (states[s] is not State.OPEN or s in started)
    ]


def _index(stage: str) -> int:
    if stage not in STAGES:
        raise WorkflowError(f"{stage!r} is not a stage ({', '.join(STAGES)})")
    return STAGES.index(stage)
