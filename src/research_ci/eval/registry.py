"""Metadata used to discover and select the Engine's general eval methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StageName = Literal["data", "feat", "sig", "algo"]
InputKind = Literal[
    "main_table",
    "feature_panel",
    "evaluation_table",
    "trades",
]


@dataclass(frozen=True, slots=True)
class MethodInfo:
    """Describe a reusable library method without selecting its inputs."""

    name: str
    family: str
    stages: tuple[StageName, ...]
    input_kind: InputKind
    description: str

    def supports(self, stage: str, input_kind: str | None = None) -> bool:
        """Return whether this method can be considered for a stage/input."""

        return stage in self.stages and (
            input_kind is None or input_kind == self.input_kind
        )


__all__ = ["InputKind", "MethodInfo", "StageName"]
