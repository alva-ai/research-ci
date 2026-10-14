"""Compile stage-specific menus from the shared methodology registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .declarations import DataSet, SignalSet, StageDeclaration
from .defaults import default_method_names
from .library import METHODS
from .policies import stage_policy
from .registry import MethodInfo, StageName, StagePolicy

Availability = Literal["selectable", "unavailable"]


@dataclass(frozen=True, slots=True)
class MethodMenuEntry:
    """One method as seen through a stage policy and declaration."""

    method: MethodInfo
    availability: Availability
    recommended: bool
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.method.name,
            "family": self.method.family,
            "purpose": self.method.purpose,
            "stages": self.method.stages,
            "input_kind": self.method.input_kind,
            "input_roles": self.method.input_roles,
            "requires": self.method.requires,
            "fit_policy": self.method.fit_policy,
            "description": self.method.description,
            "availability": self.availability,
            "recommended": self.recommended,
            "reasons": self.reasons,
        }


@dataclass(frozen=True, slots=True)
class StageMenu:
    """Agent-facing stage brief plus its compiled method choices."""

    policy: StagePolicy
    entries: tuple[MethodMenuEntry, ...]

    @property
    def selectable(self) -> tuple[MethodMenuEntry, ...]:
        return tuple(
            entry for entry in self.entries if entry.availability == "selectable"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.policy.stage,
            "objective": self.policy.objective,
            "input_boundary": self.policy.input_boundary,
            "output_boundary": self.policy.output_boundary,
            "allowed_purposes": self.policy.allowed_purposes,
            "forbidden_requirements": self.policy.forbidden_requirements,
            "next_stage": self.policy.next_stage,
            "methods": [entry.as_dict() for entry in self.entries],
        }


def compile_stage_menu(
    stage: StageName | str,
    declaration: StageDeclaration | None = None,
    *,
    family: str | None = None,
    include_unavailable: bool = True,
) -> StageMenu:
    """Return methods permitted by both stage semantics and current inputs."""

    policy = stage_policy(stage)
    if declaration is not None and declaration.stage != stage:
        raise ValueError(
            f"{type(declaration).__name__} belongs to stage "
            f"{declaration.stage!r}, not {stage!r}"
        )
    recommendations = set(default_method_names(stage))
    entries: list[MethodMenuEntry] = []
    for method in METHODS:
        if family is not None and method.family != family:
            continue
        reasons = _unavailable_reasons(method, policy, declaration)
        entry = MethodMenuEntry(
            method=method,
            availability="unavailable" if reasons else "selectable",
            recommended=method.name in recommendations,
            reasons=reasons,
        )
        if include_unavailable or entry.availability == "selectable":
            entries.append(entry)
    return StageMenu(policy=policy, entries=tuple(entries))


def method_menu_entry(
    stage: StageName | str,
    method_name: str,
    declaration: StageDeclaration | None = None,
) -> MethodMenuEntry:
    """Return one compiled entry or raise for an unknown method name."""

    menu = compile_stage_menu(stage, declaration)
    for entry in menu.entries:
        if entry.method.name == method_name:
            return entry
    raise ValueError(f"unknown library method: {method_name}")


def require_selectable_method(
    stage: StageName | str,
    method_name: str,
    declaration: StageDeclaration | None = None,
) -> MethodInfo:
    """Reject a library method before data execution when it violates policy."""

    entry = method_menu_entry(stage, method_name, declaration)
    if entry.availability == "unavailable":
        reason = "; ".join(entry.reasons)
        raise ValueError(f"{method_name} is unavailable in stage {stage!r}: {reason}")
    return entry.method


def _unavailable_reasons(
    method: MethodInfo,
    policy: StagePolicy,
    declaration: StageDeclaration | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if policy.stage not in method.stages:
        supported = ", ".join(method.stages)
        reasons.append(f"method belongs to stage(s): {supported}")
    if method.purpose not in policy.allowed_purposes:
        reasons.append(
            f"purpose {method.purpose!r} is outside the {policy.stage} boundary"
        )
    forbidden = sorted(set(method.requires) & set(policy.forbidden_requirements))
    if forbidden:
        reasons.append("stage forbids requirement(s): " + ", ".join(forbidden))
    if declaration is not None:
        reasons.extend(_declaration_reasons(method, declaration))
    return tuple(dict.fromkeys(reasons))


def _declaration_reasons(
    method: MethodInfo,
    declaration: StageDeclaration,
) -> list[str]:
    reasons: list[str] = []
    requirements = set(method.requires)
    if requirements & {"target", "horizon"}:
        if not isinstance(declaration, SignalSet) or not declaration.targets:
            reasons.append("requires SignalSet.targets with an explicit horizon")
    if "time_key" in requirements:
        if not isinstance(declaration, DataSet) or declaration.time_key is None:
            reasons.append("requires DataSet.time_key")
    if "signals" in requirements:
        if not isinstance(declaration, SignalSet) or not declaration.signals:
            reasons.append("requires declared SignalSet.signals")
    return reasons


__all__ = [
    "Availability",
    "MethodMenuEntry",
    "StageMenu",
    "compile_stage_menu",
    "method_menu_entry",
    "require_selectable_method",
]
