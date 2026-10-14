"""Stage table declarations and agent-owned library selections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .protocol import FailurePolicy


def _names(values: tuple[str, ...] | list[str], label: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(values))
    if not result or any(not value.strip() for value in result):
        raise ValueError(f"{label} must contain at least one non-empty column name")
    return result


def _groups(
    values: Mapping[str, tuple[str, ...] | list[str]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for name, columns in values.items():
        if not name.strip():
            raise ValueError("column group names must not be empty")
        result[name] = _names(columns, f"column group {name!r}")
    return result


@dataclass(frozen=True, slots=True)
class DataSet:
    """Describe a Data-stage table without choosing its eval methods."""

    keys: tuple[str, ...]
    required: tuple[str, ...]
    numeric: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    time_key: str | None = None
    frequency: timedelta | None = None
    stage: str = field(default="data", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", _names(self.keys, "DataSet.keys"))
        object.__setattr__(self, "required", _names(self.required, "DataSet.required"))
        object.__setattr__(self, "numeric", _names(self.numeric, "DataSet.numeric"))
        object.__setattr__(self, "groups", _groups(self.groups))
        declared = set(self.required)
        referenced = set(self.keys) | set(self.numeric)
        referenced.update(
            column for columns in self.groups.values() for column in columns
        )
        missing = sorted(referenced - declared)
        if missing:
            raise ValueError(f"DataSet columns are absent from required: {missing}")
        if self.time_key is not None and self.time_key not in self.keys:
            raise ValueError("DataSet.time_key must be one of DataSet.keys")
        if self.frequency is not None and self.time_key is None:
            raise ValueError("DataSet.frequency requires DataSet.time_key")
        if self.frequency is not None and self.frequency <= timedelta(0):
            raise ValueError("DataSet.frequency must be positive")

    def columns(self, on: str | tuple[str, ...] | None) -> tuple[str, ...]:
        """Resolve a selection scope into concrete Data columns."""

        if on is None or on == "numeric":
            return self.numeric
        if on == "required":
            return self.required
        if on == "keys":
            return self.keys
        if isinstance(on, tuple):
            selected = _names(on, "selected columns")
        else:
            try:
                selected = self.groups[on]
            except KeyError as exc:
                raise ValueError(f"unknown DataSet column group: {on!r}") from exc
        unknown = sorted(set(selected) - set(self.required))
        if unknown:
            raise ValueError(f"selected DataSet columns are not declared: {unknown}")
        return selected


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Describe feature inputs and target sidecars without choosing methods."""

    features: tuple[str, ...]
    keys: tuple[str, ...]
    targets: Mapping[str, str] = field(default_factory=dict)
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    target_table: str = "target"
    cross_section_by: str | None = None
    stage: str = field(default="feat", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "features", _names(self.features, "FeatureSet.features")
        )
        object.__setattr__(self, "keys", _names(self.keys, "FeatureSet.keys"))
        object.__setattr__(self, "targets", dict(self.targets))
        object.__setattr__(self, "groups", _groups(self.groups))
        if not self.target_table.strip():
            raise ValueError("FeatureSet.target_table must not be empty")
        if any(
            not horizon.strip() or not column.strip()
            for horizon, column in self.targets.items()
        ):
            raise ValueError("FeatureSet.targets needs non-empty horizons and columns")
        unknown = sorted(
            {
                column
                for columns in self.groups.values()
                for column in columns
                if column not in self.features
            }
        )
        if unknown:
            raise ValueError(
                f"FeatureSet groups contain undeclared features: {unknown}"
            )
        if self.cross_section_by is not None and self.cross_section_by not in self.keys:
            raise ValueError(
                "FeatureSet.cross_section_by must be one of FeatureSet.keys"
            )

    @property
    def required(self) -> tuple[str, ...]:
        return (*self.keys, *self.features)

    def columns(self, on: str | tuple[str, ...] | None) -> tuple[str, ...]:
        """Resolve a selection scope into concrete feature columns."""

        if on is None or on == "features":
            return self.features
        if on == "keys":
            return self.keys
        if isinstance(on, tuple):
            selected = _names(on, "selected features")
        else:
            try:
                selected = self.groups[on]
            except KeyError as exc:
                raise ValueError(f"unknown FeatureSet group: {on!r}") from exc
        unknown = sorted(set(selected) - set(self.features))
        if unknown:
            raise ValueError(f"selected features are not declared: {unknown}")
        return selected


@dataclass(frozen=True, slots=True)
class SelectedMethod:
    """One general library method deliberately selected by an external agent."""

    name: str
    reason: str
    on: str | tuple[str, ...] | None = None
    targets: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    failure_policy: FailurePolicy = "record"
    alias: str = ""

    def __post_init__(self) -> None:
        if not self.name.startswith("library."):
            raise ValueError("selected method names must start with 'library.'")
        if not self.reason.strip():
            raise ValueError(f"selected method {self.name!r} needs a reason")
        object.__setattr__(self, "targets", tuple(dict.fromkeys(self.targets)))
        object.__setattr__(self, "parameters", dict(self.parameters))
        if self.failure_policy not in {"stop", "record"}:
            raise ValueError("selected method failure_policy must be stop or record")


@dataclass(frozen=True, slots=True)
class LibrarySelection:
    """The reusable methods an external agent selected for one case/stage."""

    methods: tuple[SelectedMethod, ...]

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("LibrarySelection must contain at least one method")
        object.__setattr__(self, "methods", tuple(self.methods))


StageDeclaration = DataSet | FeatureSet


__all__ = [
    "DataSet",
    "FeatureSet",
    "LibrarySelection",
    "SelectedMethod",
    "StageDeclaration",
]
