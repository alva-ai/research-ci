"""Plugin protocols. Plugins live in a lab, import ``research_ci.api`` and, for its methods, ``research_ci.eval``, and
are found through the entry-point group ``research_ci.plugins``; each entry is a factory the engine calls with the
options from the contract's ``Use``.

Plugins supply raw material only. The engine hands them the data, perturbs their inputs, and judges their outputs;
before it uses a plugin it runs protocol tests on it (point-in-time truncation, determinism, reconciliation, known
events). Everything a plugin reads must come from the sources, or from ``wide``, it is given.

The data an idea reads is its sources: one wide table per source, at the source's own cadence, with ``inst``,
``event_time``, ``available_at``, ``vintage`` and value columns whose names are unique across the sources. ``wide``
is the sources as of each decision time ``t``: ``t``, ``inst``, every value column, and ``age_<source>``, how long
before ``t`` the row used became known (``kernel/time/asof.py``).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd
import polars as pl

ENTRY_POINTS = "research_ci.plugins"
SOURCE_COLUMNS = ("inst", "event_time", "available_at", "vintage")
UNIVERSE_COLUMNS = ("t", "inst", "long", "short")
Sources = Mapping[str, pl.DataFrame]


@runtime_checkable
class DataPlugin(Protocol):
    """The sources, by name: each a table with ``SOURCE_COLUMNS`` and the source's value columns, numbers, at the
    source's own cadence; ``available_at`` and ``vintage`` come from the plugin, never from the agent. The engine
    normalizes them, checks them, and keeps the snapshot.

    ``fetch(as_of=T)`` returns what the plugin would have returned had it run at ``T``: only rows known by then, built
    only from what was known by then, including any repair the plugin applies. ``insts`` limits the fetch to those
    insts. Before a plugin's data is used, the engine replays it at cut points of its own choosing and compares
    each replay with the full fetch; a plugin that cannot fetch as of a time cannot be admitted.

    A class attribute ``levels`` (``research_ci.api.Levels``) names the value columns that are levels and how large and
    lasting a jump in them must be to count; without it, nothing is checked for level shifts.

    An optional class attribute ``sources`` maps each source's name to its value columns, for whoever writes a
    contract or features against it (``rci plugins`` shows it); it is documentation only, and nothing checks it.
    """

    def fetch(
        self, as_of: pd.Timestamp | None = None, insts: Sequence[str] | None = None
    ) -> Mapping[str, pl.DataFrame | pd.DataFrame]: ...


REQUEST_COLUMNS = ("source", "inst", "column", "start", "end")


@dataclass(frozen=True)
class Reference:
    """What an independent source holds for a batch of requests: ``rows``, by source, the rows it has of each
    request's inst in its ``[start, end]`` (``inst``, ``event_time`` and the value columns it holds), and ``answered``,
    the requests (``rid``) it could check at all. A request left out of ``answered`` (a column the source has no history
    of, a span too old) checks nothing."""

    rows: Mapping[str, pl.DataFrame]
    answered: pl.DataFrame


@runtime_checkable
class ReferenceSource(Protocol):
    """A data plugin that can ask an independent source, such as the venue's own public API, for the same columns.
    The engine samples the requests (``rid`` and ``REQUEST_COLUMNS``, one value column each) at random and adds every
    anomaly its checks found; the plugin never chooses them. ``reference_note`` says where the rows come from, for the
    person admitting the plugin: the engine cannot tell an independent source from one that reads the plugin's own
    files."""

    reference_note: str

    def reference(self, requests: pl.DataFrame) -> Reference: ...


@runtime_checkable
class UniversePlugin(Protocol):
    """The candidates at each decision time of ``wide``, computed only from ``wide`` and, where it needs native rows,
    from ``sources``: ``UNIVERSE_COLUMNS`` plus cluster, group and score columns. ``t`` is the decision time; ``long``
    and ``short`` name the insts of the legs (empty for single-inst candidates).

    A plugin whose legs are always one asset on two venues says so with a class attribute ``hedge``
    (``research_ci.api.Hedge``): the column of ``wide`` that prices both legs and how closely they must track. The
    engine then checks, at every decision time, that the legs trade at about the same level and move together, and drops
    the candidates where they do not."""

    def candidates(self, wide: pl.DataFrame, sources: Sources) -> pl.DataFrame: ...


@dataclass(frozen=True)
class Timing:
    """When a consumer decides: every ``every`` from ``offset`` after midnight UTC. A decision at ``t`` may use what
    is known at ``t`` and fills at the price known at ``t``; the positions it opens earn only what happens after
    ``t``, so a flow at ``t`` belongs to the positions held up to ``t``. ``note`` says how, for people."""

    every: str
    offset: str = "0s"
    note: str = ""


@dataclass(frozen=True)
class ConsumerResult:
    """What the real trading logic earned.

    ``net`` has one row per decision time: ``t``, ``net`` (what the period ending at ``t`` earned, as a fraction of
    the starting capital; the run's result is the sum) and optional component columns (``flow``, ``price``,
    ``fees``) that must add up to ``net``. ``marked`` names the components booked every period on the positions
    held, each the outcome's part ``y_<component>``; the others are booked only when trades happen. ``positions``
    holds ``t``, ``inst``, ``weight`` (fraction of capital per leg) after the decision at ``t``, plus the columns the
    outcome needs for every position held (such as ``long`` and ``short``), also when that pair is no longer a
    candidate; ``trades`` holds ``t``, ``inst``, ``from_weight``, ``to_weight``. ``turnover`` is the traded notional
    over capital. ``checks`` are the consumer's own consistency checks. ``policies`` states accounting rules people
    must see, such as a cap on gains.
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
    """Runs the strategy's own logic on ``sources``. With ``scores`` (``t``, ``inst``, ``score``) the strategy ranks
    and weights candidates by that score instead of its own, and a candidate without a score is not opened.

    It must be deterministic for the same inputs and read nothing but ``sources``. With ``fresh`` it computes the
    result again instead of reusing anything cached, so the engine can check that two runs agree.
    """

    def timing(self) -> Timing: ...

    def run(self, sources: Sources, scores: pl.DataFrame | None = None, fresh: bool = False) -> ConsumerResult: ...


@dataclass(frozen=True)
class OutcomeCase:
    """A worked example the outcome's author computed by hand: small ``sources`` and ``candidates``, a horizon, and
    the rows the outcome must return (``t``, ``inst``, ``y`` and any parts ``y_<part>``). The engine also mutates the
    example's inputs and requires the output to move, so an example that exercises nothing does not count."""

    sources: Sources
    candidates: pl.DataFrame
    horizon: str
    expected: pl.DataFrame


@runtime_checkable
class OutcomePlugin(Protocol):
    """What a unit position in each candidate earns over ``(t, t + h]``, gross, before costs: one row per candidate and
    decision time with ``t``, ``inst``, ``y``, optional parts ``y_<part>`` that add up to ``y`` (such as ``y_price`` and
    ``y_flow``), and ``y_end``, the last time whose data the row reads, for purging later.

    A row for ``t`` may use data known up to ``t + h`` and nothing later; the engine checks this by cutting the data.
    ``cases`` are the author's worked examples. An optional method ``values_each(sources, candidates, horizons)``
    yields the frames ``values`` gives at each horizon, in order, sharing the work that does not depend on the
    horizon; the engine calls it, when the class that defines ``values`` defines it too, to ask for every horizon at
    once."""

    def values(self, sources: Sources, candidates: pl.DataFrame, horizon: pd.Timedelta) -> pl.DataFrame: ...

    def cases(self) -> Sequence[OutcomeCase]: ...


@runtime_checkable
class CostModel(Protocol):
    """The cost of trading: ``round_trip`` per candidate row (entering and leaving, as a fraction of one leg's
    notional) and ``fees`` of a trade table (``t``, ``inst``, ``from_weight``, ``to_weight``) as a fraction of
    capital. The engine reconciles ``fees`` with what the consumer charged."""

    def round_trip(self, candidates: pl.DataFrame) -> pl.Series: ...

    def fees(self, trades: pl.DataFrame) -> float: ...


@runtime_checkable
class TargetPlugin(Protocol):
    """Turns one horizon's outcome rows (``y`` and its parts, joined with the candidate columns, and ``score`` once
    there are features) into the number the claim is about at each decision time (``t``, ``value``). Before any
    submission the engine reports only how much that number varies, never its level.

    Two optional class attributes tell the cards how to show the number. ``per`` is a short word for what one value
    is, such as ``"entry"``: values are then fractions shown in basis points per that unit. ``rolls = True`` says a
    value is a return over the horizon that can be repeated back to back, so a card may add what it comes to over a
    year, ``value × 365 days / h``. Without them, cards show the raw number.

    An optional method ``batched(horizon, costs)`` returns the same ``reduce`` computed on many shuffles of the scores
    at once (a form from ``research_ci.eval.statistics.shuffled``), or ``None``; the null worlds then run faster, with
    the same numbers."""

    def reduce(self, rows: pl.DataFrame, horizon: pd.Timedelta, costs: CostModel) -> pl.DataFrame: ...
