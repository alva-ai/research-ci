"""Metadata used to discover and select the Engine's general eval methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StageName = Literal["data", "feat", "sig", "algo"]
InputKind = Literal[
    "main_table",
    "feature_panel",
    "feature_matrix",
    "evaluation_table",
    "trades",
]
MethodPurpose = Literal[
    "quality",
    "temporal",
    "structure",
    "predictive",
    "stability",
    "economic",
]
InputRole = Literal[
    "table",
    "keys",
    "features",
    "signals",
    "target",
    "trades",
]
MethodRequirement = Literal[
    "target",
    "horizon",
    "time_key",
    "signals",
    "positions",
    "trades",
    "pnl",
]
FitPolicy = Literal["none", "diagnostic_only", "train_only"]


@dataclass(frozen=True, slots=True)
class MethodInfo:
    """Describe a reusable library method without selecting its inputs."""

    name: str
    family: str
    stages: tuple[StageName, ...]
    input_kind: InputKind
    purpose: MethodPurpose
    input_roles: tuple[InputRole, ...]
    description: str
    requires: tuple[MethodRequirement, ...] = ()
    fit_policy: FitPolicy = "none"

    def __post_init__(self) -> None:
        if not self.name.startswith("library."):
            raise ValueError("library method names must start with 'library.'")
        if not self.family.strip() or not self.description.strip():
            raise ValueError(f"library method {self.name!r} needs metadata")
        if not self.stages:
            raise ValueError(f"library method {self.name!r} needs a stage")
        if not self.input_roles:
            raise ValueError(f"library method {self.name!r} needs input roles")

    def supports(self, stage: str, input_kind: str | None = None) -> bool:
        """Return whether this method can be considered for a stage/input."""

        return stage in self.stages and (
            input_kind is None or input_kind == self.input_kind
        )


@dataclass(frozen=True, slots=True)
class StagePolicy:
    """Hard semantic boundary for one research stage."""

    stage: StageName
    objective: str
    input_boundary: str
    output_boundary: str
    allowed_purposes: tuple[MethodPurpose, ...]
    forbidden_requirements: tuple[MethodRequirement, ...]
    next_stage: StageName | None


__all__ = [
    "FitPolicy",
    "InputKind",
    "InputRole",
    "MethodInfo",
    "MethodPurpose",
    "MethodRequirement",
    "StageName",
    "StagePolicy",
]
