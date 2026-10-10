import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


class ContractError(ValueError):
    pass


FLOOR_ALTERNATIVES = ("chance", "timing")
USAGE_TARGETS = ("claim", "costs", "labels")
LAYER_NAMES = ("data", "feat", "signal", "algo")


@dataclass(frozen=True)
class Use:
    """A plugin by entry-point name, with the options it is built with. The options are part of the contract."""

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractError("name the plugin")
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class Execution:
    """Decision times when no consumer declares them: every ``every``, starting ``offset`` after midnight UTC.

    A decision at ``t`` uses what is known at ``t`` and fills at the price known then; ``note`` says how, for people.
    """

    every: str
    offset: str = "0s"
    note: str = ""

    def __post_init__(self) -> None:
        every, offset = _span("every", self.every), _span("offset", self.offset)
        if every <= pd.Timedelta(0):
            raise ContractError("every must be positive")
        if not pd.Timedelta(0) <= offset < every:
            raise ContractError("offset must be at least 0 and shorter than every")


@dataclass(frozen=True)
class Universe:
    """The candidates at each decision time, from a universe plugin.

    ``clusters`` names the column whose members move together (nulls shuffle whole clusters, blocks count them once);
    ``groups`` are columns nulls may bucket by. ``membership="static"`` means the candidate list was fixed with
    hindsight; the engine then records that the history is explore-grade for the universe.
    """

    use: Use | str
    clusters: str
    groups: tuple[str, ...] = ()
    membership: Literal["pit", "static"] = "pit"

    def __post_init__(self) -> None:
        object.__setattr__(self, "use", as_use(self.use))
        object.__setattr__(self, "groups", tuple(self.groups))
        if not self.clusters.strip():
            raise ContractError("name the cluster column of the universe")


@dataclass(frozen=True)
class Label:
    """An outcome after each decision, computed by a label plugin for every horizon. The engine controls which data
    the plugin sees and who gets the values; the plugin defines what is measured."""

    name: str
    use: Use | str
    horizons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "use", as_use(self.use))
        object.__setattr__(self, "horizons", tuple(self.horizons))
        if not self.name.strip():
            raise ContractError("name the label")
        _horizons(self.horizons)


@dataclass(frozen=True)
class Effect:
    """The effect the claim is about, on one of the contract's labels. ``target`` is the plugin that turns each
    decision's label values and scores into the one number the claim is about; it must read the scores, such as
    ``Use("ref.mean", {"cluster": ..., "top": 0.2})``. Feat passes the effect when, with the submitted scores, it beats
    random scores significantly. A backtest is not an effect here: the consumer judges the algo layer.
    """

    on: str
    target: Use | str | None = None

    def __post_init__(self) -> None:
        if self.target is not None:
            object.__setattr__(self, "target", as_use(self.target))


@dataclass(frozen=True)
class Alternative:
    """Another reason the same result could appear. Null worlds and criteria refer to it by ``id``."""

    id: str
    text: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.text.strip():
            raise ContractError("an alternative needs an id and a sentence")


@dataclass(frozen=True)
class Usage:
    """A span of data already used to choose something; claims depending on ``affects`` are explore-grade there."""

    start: str
    end: str
    what: str
    affects: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "affects", tuple(self.affects))
        if _timestamp("start", self.start) >= _timestamp("end", self.end):
            raise ContractError(f"usage {self.what!r} must end after it starts")
        unknown = set(self.affects) - set(USAGE_TARGETS)
        if not self.affects or unknown:
            raise ContractError(f"affects must be some of {', '.join(USAGE_TARGETS)}")


@dataclass(frozen=True)
class Forward:
    """Where data after the freeze comes from: a data plugin that collects it live."""

    source: Use | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", as_use(self.source))


@dataclass(frozen=True)
class Contract:
    """The claim an idea makes and the rules its verdicts are held to. Written by the agent, frozen by a person.

    Everything here is a declaration; the engine derives the numbers. ``used`` must be stated even when empty.
    Decisions come from ``execution``, or from the timing of the consumer the algo layer will run. Explore data ends
    at ``explore_end`` (``"latest"`` is filled from the data at derive time); the sealed segment runs to ``seal_end``,
    or there is none and only ``forward`` data confirms. ``budget`` is the number of submissions each layer may use
    across all its versions, one number or one per layer.
    """

    claim: str
    decisive: tuple[Effect, ...]
    alternatives: tuple[Alternative, ...]
    used: tuple[Usage, ...] | None
    data: Use | str
    costs: Use | str
    explore_end: str = "latest"
    seal_end: str | None = None
    execution: Execution | None = None
    consumer: Use | str | None = None
    universe: Universe | None = None
    labels: tuple[Label, ...] = ()
    forward: Forward | None = None
    budget: int | Mapping[str, int] = 20

    def __post_init__(self) -> None:
        for name in ("decisive", "alternatives", "labels"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.used is not None:
            object.__setattr__(self, "used", tuple(self.used))
        object.__setattr__(self, "data", as_use(self.data))
        object.__setattr__(self, "costs", as_use(self.costs))
        if self.consumer is not None:
            object.__setattr__(self, "consumer", as_use(self.consumer))
        if isinstance(self.budget, Mapping):
            object.__setattr__(self, "budget", dict(self.budget))
        self._check_claim()
        self._check_tiers()
        self._check_trade()
        self._check_targets()
        self._check_budget()

    def _check_claim(self) -> None:
        if not self.claim.strip() or self.claim.startswith("TODO"):
            raise ContractError("state the claim in one sentence")
        if not self.decisive:
            raise ContractError("state the decisive effect: the label, and a target that reads the scores")
        if not self.alternatives:
            raise ContractError("list at least one alternative explanation specific to this idea")
        ids = [a.id for a in self.alternatives]
        if len(ids) != len(set(ids)):
            raise ContractError(f"alternative ids must be unique: {ids}")
        reserved = set(ids) & set(FLOOR_ALTERNATIVES)
        if reserved:
            raise ContractError(f"the engine adds {', '.join(sorted(reserved))} itself; use other ids")
        if self.used is None:
            raise ContractError("say which data was already used to choose anything here; used=() if none")

    def _check_tiers(self) -> None:
        latest = self.explore_end == "latest"
        explore_end = None if latest else _timestamp("explore_end", self.explore_end)
        if self.seal_end is not None:
            if latest or explore_end is None:
                raise ContractError("a sealed segment needs a fixed explore_end before it")
            seal_end = _timestamp("seal_end", self.seal_end)
            if seal_end <= explore_end:
                raise ContractError("seal_end must be after explore_end")
            for u in self.used or ():
                if _timestamp("start", u.start) < seal_end and _timestamp("end", u.end) > explore_end:
                    raise ContractError(f"the sealed segment overlaps data already used: {u.what}")
        elif self.forward is None:
            raise ContractError("with no sealed segment only forward data can confirm; name the forward source")

    def _check_trade(self) -> None:
        if (self.execution is None) == (self.consumer is None):
            raise ContractError("declare decision times with execution, or name the consumer that has them; not both")
        if self.labels and self.universe is None:
            raise ContractError("labels are per candidate; name the universe")

    def _check_targets(self) -> None:
        names = [label.name for label in self.labels]
        if len(names) != len(set(names)):
            raise ContractError(f"label names must be unique: {names}")
        for effect in self.decisive:
            if effect.on == "consumer":
                raise ContractError(
                    "a backtest is judged at the algo layer, not in the contract: make the decisive effect a label "
                    'effect, with a target that reads the scores, such as Use("ref.mean", {"cluster": ..., "top": 0.2})'
                )
            if effect.on not in names:
                known = ", ".join(names) or "none"
                raise ContractError(f"decisive effect on {effect.on!r}: not a label of this contract (labels: {known})")

    def _check_budget(self) -> None:
        budget = self.budget
        if isinstance(budget, Mapping):
            unknown = set(budget) - set(LAYER_NAMES)
            if unknown or not budget or min(budget.values()) < 1:
                raise ContractError(f"budget per layer must name layers ({', '.join(LAYER_NAMES)}) and be at least 1")
        elif budget < 1:
            raise ContractError("budget must allow at least one submission")

    def budget_for(self, layer: str, default: int = 20) -> int:
        if isinstance(self.budget, Mapping):
            return int(self.budget.get(layer, default))
        return int(self.budget)


def as_use(value: Use | str) -> Use:
    return value if isinstance(value, Use) else Use(value)


def _timestamp(name: str, value: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except ValueError as exc:
        raise ContractError(f"{name} is not a timestamp: {value!r}") from exc
    if ts.tzinfo is None:
        raise ContractError(f"{name} needs a time zone, for example {value}Z")
    return ts


def span(value: str) -> pd.Timedelta:
    """A duration from a contract, such as ``"8h"``, ``"7d"`` or ``"365 days"``."""
    return pd.Timedelta(re.sub(r"(?<=\d)\s*d\b", "D", value.strip()))


def _span(name: str, value: str) -> pd.Timedelta:
    try:
        return span(value)
    except ValueError as exc:
        raise ContractError(f"{name} is not a duration: {value!r}") from exc


def _horizons(horizons: tuple[str, ...]) -> None:
    if not horizons:
        raise ContractError("a label needs at least one horizon")
    spans = [_span("horizon", h) for h in horizons]
    if any(s <= pd.Timedelta(0) for s in spans) or spans != sorted(set(spans)):
        raise ContractError(f"horizons must be positive, distinct, and ascending: {horizons}")
