import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from research_ci.kernel.hashing import hash_obj
from research_ci.store.workspace import Workspace


@dataclass(frozen=True)
class Freeze:
    """Everything a stage verdict depends on, pinned by hash. Any change means a new version."""

    stage: str
    deps: dict[str, str]
    hash: str
    at_ns: int


def make_freeze(stage: str, deps: dict[str, str], at_ns: int | None = None) -> Freeze:
    return Freeze(
        stage=stage,
        deps=dict(sorted(deps.items())),
        hash=hash_obj({"stage": stage, "deps": deps}),
        at_ns=time.time_ns() if at_ns is None else at_ns,
    )


def changed(freeze: Freeze, deps: Mapping[str, str]) -> list[str]:
    return drift(freeze.deps, deps)


def drift(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))


def write_freeze(workspace: Workspace, freeze: Freeze) -> None:
    workspace.write_json(f"freeze/{freeze.stage}.json", asdict(freeze))


def read_freeze(workspace: Workspace, stage: str) -> Freeze | None:
    data = workspace.read_json(f"freeze/{stage}.json")
    return None if data is None else Freeze(**data)
