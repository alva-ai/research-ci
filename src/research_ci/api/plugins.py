"""Plugin protocols. Plugins live in a lab, import only ``research_ci.api``, and are found through the entry-point group
``research_ci.plugins``; each entry is a factory the engine calls with the options from the contract's ``Use``.

Plugins supply raw material only. The engine hands them the data, perturbs their inputs, and judges their outputs;
before it uses a plugin it runs protocol tests on it (point-in-time truncation, determinism, reconciliation, known
events). Everything a plugin reads must come from the fields it is given.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
import polars as pl

ENTRY_POINTS = "research_ci.plugins"
FIELD_COLUMNS = ("entity", "field", "event_time", "available_at", "value", "vintage", "source")
UNIVERSE_COLUMNS = ("t", "entity", "long", "short")


@runtime_checkable
class DataPlugin(Protocol):
    """Rows shaped like a FieldTable (``FIELD_COLUMNS``); ``available_at`` and ``vintage`` come from the plugin,
    never from the agent. The engine normalizes them, checks them, and keeps the snapshot."""

    def fetch(self) -> pl.DataFrame | pd.DataFrame: ...


@runtime_checkable
class NativeWriter(Protocol):
    """A data plugin whose data a consumer reads as files: writes fields back in the consumer's own layout, so the
    engine controls what the consumer sees (truncated, perturbed, or sliced)."""

    def write_native(self, fields: pl.DataFrame, root: Path) -> None: ...


@runtime_checkable
class UniversePlugin(Protocol):
    """The candidates at each decision time, computed only from ``fields``: ``UNIVERSE_COLUMNS`` plus cluster, group
    and score columns. ``t`` is the decision time; ``long`` and ``short`` name the FieldTable entities of the legs
    (empty for single-entity candidates)."""

    def candidates(self, fields: pl.DataFrame, times: Sequence[pd.Timestamp]) -> pl.DataFrame: ...


@dataclass(frozen=True)
class Timing:
    """When a consumer decides: every ``every`` from ``offset`` after midnight UTC. A decision at ``t`` may use what
    is known at ``t`` and fills at the price known at ``t``; ``note`` says how, for people."""

    every: str
    offset: str = "0s"
    note: str = ""


@dataclass(frozen=True)
class ConsumerResult:
    """What the real trading logic earned.

    ``net`` has one row per decision time: ``t``, ``net`` (what the period ending at ``t`` earned, as a fraction of
    the starting capital; the run's result is the sum) and optional component columns (``flow``, ``price``,
    ``fees``) that must add up to ``net``. ``marked`` names the components booked every period on the positions
    held; the others are booked only when trades happen. ``positions`` holds ``t``, ``entity``, ``weight``
    (fraction of capital per leg) after the decision at ``t``, plus the columns a label needs for every position
    held (such as ``long`` and ``short``), also when that pair is no longer a candidate; ``trades`` holds ``t``,
    ``entity``,
    ``from_weight``, ``to_weight``. ``turnover`` is the traded notional over capital. ``checks`` are the consumer's
    own consistency checks. ``policies`` states accounting rules people must see, such as a cap on gains.
    """

    net: pl.DataFrame
    positions: pl.DataFrame
    trades: pl.DataFrame
    turnover: float
    checks: Mapping[str, bool] = field(default_factory=dict)
    policies: Mapping[str, str] = field(default_factory=dict)
    marked: tuple[str, ...] = ("flow", "price")


@runtime_checkable
class ConsumerPlugin(Protocol):
    """Runs the strategy's own logic on ``fields``. With ``scores`` (``t``, ``entity``, ``score``) the strategy ranks
    and weights candidates by that score instead of its own, and a candidate without a score is not opened.

    It must be deterministic for the same inputs and read nothing but ``fields``. With ``fresh`` it computes the
    result again instead of reusing anything cached, so the engine can check that two runs agree.
    """

    def timing(self) -> Timing: ...

    def run(self, fields: pl.DataFrame, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult: ...


@dataclass(frozen=True)
class LabelCase:
    """A worked example the label's author computed by hand: small ``fields`` and ``candidates``, a horizon, and the
    rows the label must return (``t``, ``entity``, ``value`` and any component columns). The engine also mutates the
    example's inputs and requires the output to move, so an example that exercises nothing does not count."""

    fields: pl.DataFrame
    candidates: pl.DataFrame
    horizon: str
    expected: pl.DataFrame


@runtime_checkable
class LabelPlugin(Protocol):
    """What happened after each decision: one row per candidate and decision time (``t``, ``entity``, ``value``, and
    optional component columns such as ``flow`` and ``price`` that add up to ``value``), before costs.

    A row for ``t`` may use data known up to ``t + horizon`` and nothing later; the engine checks this by cutting the
    data. ``cases`` are the author's worked examples."""

    def values(self, fields: pl.DataFrame, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame: ...

    def cases(self) -> Sequence[LabelCase]: ...


@runtime_checkable
class CostModel(Protocol):
    """The cost of trading: ``round_trip`` per candidate row (entering and leaving, as a fraction of one leg's
    notional) and ``fees`` of a trade table (``t``, ``entity``, ``from_weight``, ``to_weight``) as a fraction of
    capital. The engine reconciles ``fees`` with what the consumer charged."""

    def round_trip(self, candidates: pl.DataFrame) -> pl.Series: ...

    def fees(self, trades: pl.DataFrame) -> float: ...


@runtime_checkable
class TargetPlugin(Protocol):
    """Turns one horizon's label rows (joined with the candidate columns) into the number the claim is about at each
    decision time (``t``, ``value``). The engine only ever reports how much that number varies, never its level."""

    def reduce(self, labels: pl.DataFrame, horizon: pd.Timedelta, costs: CostModel) -> pl.DataFrame: ...
