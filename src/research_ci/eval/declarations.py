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
    """Describe a target-free Feat table without choosing methods."""

    features: tuple[str, ...]
    keys: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    stage: str = field(default="feat", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "features", _names(self.features, "FeatureSet.features")
        )
        object.__setattr__(self, "keys", _names(self.keys, "FeatureSet.keys"))
        object.__setattr__(self, "groups", _groups(self.groups))
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
class SignalSet:
    """Describe the Sig feature universe, candidates, finals, and prediction targets.

    ``targets`` maps each research horizon to an observed label column.  The
    realized values live in the explicitly named label sidecar and never in
    the Feat or Sig main table.
    """

    features: tuple[str, ...]
    keys: tuple[str, ...]
    targets: Mapping[str, str] = field(default_factory=dict)
    candidates: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    label_table: str = "label"
    cross_section_by: str | None = None
    evaluation_by: str | None = None
    hypothesis: str = ""
    fit_policy: str = "none"
    stage: str = field(default="sig", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "features", _names(self.features, "SignalSet.features")
        )
        object.__setattr__(self, "keys", _names(self.keys, "SignalSet.keys"))
        object.__setattr__(self, "targets", dict(self.targets))
        object.__setattr__(self, "candidates", tuple(dict.fromkeys(self.candidates)))
        object.__setattr__(self, "signals", tuple(dict.fromkeys(self.signals)))
        object.__setattr__(self, "groups", _groups(self.groups))
        if any(not value.strip() for value in (*self.candidates, *self.signals)):
            raise ValueError("SignalSet candidate/signal names must not be empty")
        if not self.label_table.strip():
            raise ValueError("SignalSet.label_table must not be empty")
        if any(
            not horizon.strip() or not column.strip()
            for horizon, column in self.targets.items()
        ):
            raise ValueError("SignalSet.targets needs non-empty horizons and columns")
        if not self.targets and not self.hypothesis.strip():
            raise ValueError("targetless SignalSet requires an explicit hypothesis")
        if self.fit_policy not in {"none", "train_validation_test", "walk_forward"}:
            raise ValueError(
                "SignalSet.fit_policy must be none, train_validation_test, "
                "or walk_forward"
            )
        if self.cross_section_by is not None and self.cross_section_by not in self.keys:
            raise ValueError("SignalSet.cross_section_by must be one of SignalSet.keys")
        if self.evaluation_by is not None and not self.evaluation_by.strip():
            raise ValueError("SignalSet.evaluation_by must not be empty")
        available = set(self.features) | set(self.candidates) | set(self.signals)
        unknown = sorted(
            {
                column
                for columns in self.groups.values()
                for column in columns
                if column not in available
            }
        )
        if unknown:
            raise ValueError(f"SignalSet groups contain undeclared columns: {unknown}")
        overlaps = {
            "keys/features": set(self.keys) & set(self.features),
            "keys/candidates": set(self.keys) & set(self.candidates),
            "keys/signals": set(self.keys) & set(self.signals),
            "features/candidates": set(self.features) & set(self.candidates),
            "features/signals": set(self.features) & set(self.signals),
            "candidates/signals": set(self.candidates) & set(self.signals),
        }
        collisions = {
            name: sorted(columns) for name, columns in overlaps.items() if columns
        }
        if collisions:
            raise ValueError(f"SignalSet column roles overlap: {collisions}")

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (*self.keys, *self.features, *self.candidates, *self.signals)
            )
        )

    def columns(self, on: str | tuple[str, ...] | None) -> tuple[str, ...]:
        """Resolve a selection scope into declared feature or signal columns."""

        if on is None or on == "features":
            return self.features
        if on == "candidates":
            if not self.candidates:
                raise ValueError("SignalSet does not declare candidate columns")
            return self.candidates
        if on == "signals":
            if not self.signals:
                raise ValueError("SignalSet does not declare signal columns yet")
            return self.signals
        if on == "all":
            return tuple(
                dict.fromkeys((*self.features, *self.candidates, *self.signals))
            )
        if on == "keys":
            return self.keys
        if isinstance(on, tuple):
            selected = _names(on, "selected Sig columns")
        else:
            try:
                selected = self.groups[on]
            except KeyError as exc:
                raise ValueError(f"unknown SignalSet group: {on!r}") from exc
        available = set(self.features) | set(self.candidates) | set(self.signals)
        unknown = sorted(set(selected) - available)
        if unknown:
            raise ValueError(f"selected SignalSet columns are not declared: {unknown}")
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


StageDeclaration = DataSet | FeatureSet | SignalSet


__all__ = [
    "DataSet",
    "FeatureSet",
    "LibrarySelection",
    "SelectedMethod",
    "SignalSet",
    "StageDeclaration",
]
