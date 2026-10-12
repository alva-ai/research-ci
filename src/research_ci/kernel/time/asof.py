"""What is known at a time: the rows of each source known by then, one source's events as known then, and ``wide``,
the research table every layer reads, every source as of each decision time."""

from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd
import polars as pl

from research_ci.kernel.tables import values
from research_ci.kernel.time.clock import TIME, lit_time, time_series, to_ns


def known(sources: Mapping[str, pl.DataFrame], t: int) -> dict[str, pl.DataFrame]:
    """Each source cut to the rows known by ``t`` (``available_at <= t``)."""
    return {name: table.filter(pl.col("available_at") <= lit_time(t)) for name, table in sources.items()}


def asof(source: pl.DataFrame, t: datetime | pd.Timestamp | int) -> pl.DataFrame:
    """One source's events as known at ``t``: the rows with ``available_at <= t``, each event at the latest vintage
    published by then; ``inst``, ``event_time`` and the value columns, by inst and event time."""
    cutoff = t if isinstance(t, int) else to_ns(t)
    seen = source.filter(pl.col("available_at") <= lit_time(cutoff))
    latest = seen.sort("vintage").unique(["inst", "event_time"], keep="last", maintain_order=True)
    return latest.select("inst", "event_time", *values(source)).sort("inst", "event_time")


def wide(sources: Mapping[str, pl.DataFrame], times: Sequence[int]) -> pl.DataFrame:
    """The sources as of each decision time ``t`` (ns): ``t``, ``inst``, every source's value columns, and
    ``age_<source>``. One row per time and inst that any source has a row of by then. For each source, the row is that
    of the latest event known by ``t``, at the latest vintage of it known by ``t``, and ``age_<source>`` is ``t`` less
    when that vintage became known; all null when the source has nothing of the inst yet. A late revision of an older
    event changes nothing: it is not the latest event."""
    grid = time_series(sorted(set(times)), "t").to_frame()
    since = pl.concat([table.select("inst", "available_at") for table in sources.values()])
    since = since.group_by("inst").agg(pl.col("available_at").min().alias("_since"))
    rows = grid.join(since, how="cross").filter(pl.col("t") >= pl.col("_since")).select("t", "inst").sort("t")
    columns, ages = [], []
    for name, table in sources.items():
        state = _current(table).rename({"available_at": "_at"})
        rows = rows.join_asof(state, left_on="t", right_on="_at", by="inst", check_sortedness=False)
        rows = rows.with_columns((pl.col("t") - pl.col("_at")).alias(f"age_{name}")).drop("_at")
        columns += values(table)
        ages.append(f"age_{name}")
    return rows.select("t", "inst", *columns, *ages).sort("t", "inst")


def wide_schema(sources: Mapping[str, pl.DataFrame]) -> dict[str, pl.DataType]:
    """The columns ``wide`` gives on these sources."""
    columns = {c: pl.Float64() for table in sources.values() for c in values(table)}
    ages = {f"age_{name}": pl.Duration("ns") for name in sources}
    return {"t": TIME, "inst": pl.String(), **columns, **ages}


def _current(table: pl.DataFrame) -> pl.DataFrame:
    """Each inst's latest event at each time a row of it becomes known, at the latest vintage of it known then: a row
    whose event is the latest of its inst seen so far is the new state, and the last such row at one time wins."""
    ordered = table.sort("inst", "available_at", "event_time", "vintage")
    newest = pl.col("event_time") == pl.col("event_time").cum_max().over("inst")
    state = ordered.filter(newest).unique(["inst", "available_at"], keep="last", maintain_order=True)
    return state.select("inst", "available_at", *values(table)).sort("available_at")
