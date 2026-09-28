from collections.abc import Mapping

from research_ci.workflow.machine import DONE, State, WorkflowError


def upstream(stages: tuple[str, ...], stage: str) -> tuple[str, ...]:
    return stages[: _index(stages, stage)]


def downstream(stages: tuple[str, ...], stage: str) -> tuple[str, ...]:
    return stages[_index(stages, stage) + 1 :]


def blockers(stages: tuple[str, ...], states: Mapping[str, State], stage: str) -> list[str]:
    return [s for s in upstream(stages, stage) if states[s] not in DONE]


def invalidated(stages: tuple[str, ...], states: Mapping[str, State], stage: str) -> list[str]:
    """Downstream stages that must go STALE when ``stage`` is reopened."""
    return [s for s in downstream(stages, stage) if states[s] not in {State.OPEN, State.SKIPPED, State.STALE}]


def _index(stages: tuple[str, ...], stage: str) -> int:
    if stage not in stages:
        raise WorkflowError(f"{stage!r} is not a stage of this idea ({', '.join(stages)})")
    return stages.index(stage)
