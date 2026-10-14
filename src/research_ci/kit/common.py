"""Small metadata types shared by the optional stage construction kits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

StageName = Literal["data", "feat", "sig", "algo"]
ToolKind = Literal["table", "expr", "split", "bundle"]


def _non_empty(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class ToolInfo:
    """Describe one optional helper without making it part of stage policy."""

    name: str
    stage: StageName
    kind: ToolKind
    purpose: str

    def __post_init__(self) -> None:
        _non_empty(self.name, "ToolInfo.name")
        _non_empty(self.purpose, "ToolInfo.purpose")
        if self.stage not in {"data", "feat", "sig", "algo"}:
            raise ValueError("ToolInfo.stage must be data, feat, sig, or algo")
        if self.kind not in {"table", "expr", "split", "bundle"}:
            raise ValueError("ToolInfo.kind must be table, expr, split, or bundle")

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "stage": self.stage,
            "kind": self.kind,
            "purpose": self.purpose,
        }


@dataclass(frozen=True, slots=True)
class BuildStep:
    """Record one named construction step in an external Lab recipe."""

    tool: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.tool, "BuildStep.tool")
        object.__setattr__(self, "inputs", tuple(dict.fromkeys(self.inputs)))
        object.__setattr__(self, "outputs", tuple(dict.fromkeys(self.outputs)))
        object.__setattr__(self, "parameters", dict(self.parameters))
        if any(not value.strip() for value in (*self.inputs, *self.outputs)):
            raise ValueError(
                "BuildStep inputs and outputs must not contain empty names"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class BuildInfo:
    """Explain how a Lab builder constructs one stage artifact."""

    stage: StageName
    summary: str
    steps: tuple[BuildStep, ...]

    def __post_init__(self) -> None:
        _non_empty(self.summary, "BuildInfo.summary")
        if self.stage not in {"data", "feat", "sig", "algo"}:
            raise ValueError("BuildInfo.stage must be data, feat, sig, or algo")
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise ValueError("BuildInfo.steps must contain at least one step")
        if any(not isinstance(step, BuildStep) for step in self.steps):
            raise TypeError("BuildInfo.steps must contain BuildStep values")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "summary": self.summary,
            "steps": [step.as_dict() for step in self.steps],
        }


__all__ = ["BuildInfo", "BuildStep", "StageName", "ToolInfo", "ToolKind"]
