"""What the stages and their lists share: when the contract decides, the target a decisive effect names, outcome,
candidate and per-time tables as the engine reads them, the treatments a step finds, and the JSON values the ledger
keeps."""

import math
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import DEFAULT_TARGET, Contract, ContractError, Effect, Use, as_use
from research_ci.api.plugins import OutcomePlugin, Sources, TargetPlugin, Timing
from research_ci.kernel.time.clock import TIME, from_ns

UNSCORED = 0.5
OUTCOME = (
    "an outcome returns, per candidate row, t, inst, y (what a unit position earns over (t, t + h]), its parts "
    "y_<part>, and y_end (the last time it reads)"
)


class DeriveError(ContractError):
    pass


@dataclass(frozen=True)
class Treatment:
    """What a treatment does to the sources (``on="sources"``) or to the candidates: ``apply`` gives them treated.
    ``removed`` gives the rows it drops or masks cells of (``source``, ``inst``, ``event_time``, ``available_at`` of a
    source row, or the candidate rows), so its influence can be weighed by leaving out the outcome rows that read them;
    a treatment without it changes identities instead, such as a split, and only ``apply`` says what it does. ``rows``
    counts the rows it touches."""

    name: str
    on: Literal["sources", "candidates"]
    reason: str
    rows: int
    apply: Callable[[Any], Any]
    removed: Callable[[Any], pl.DataFrame] | None = None


def cadence(c: Contract, plugin: Callable[[Use], Any]) -> tuple[Timing, str]:
    """When the contract decides, and where that comes from: the timing of the consumer it names, loaded by ``plugin``,
    or its execution."""
    if c.consumer is not None:
        return plugin(as_use(c.consumer)).timing(), f"consumer {as_use(c.consumer).name}"
    assert c.execution is not None
    e = c.execution
    return Timing(e.every, e.offset, e.note), "execution"


def parts(columns: Sequence[str]) -> list[str]:
    """The outcome's parts among ``columns``: the ``y_<part>`` columns but ``y_end``."""
    return [c for c in columns if c.startswith("y_") and c != "y_end"]


def target_use(effect: Effect, c: Contract) -> Use:
    """The target a decisive effect names; ``ref.mean`` over the universe's clusters when it names none."""
    assert c.universe is not None
    return as_use(effect.target) if effect.target else Use(DEFAULT_TARGET, {"cluster": c.universe.clusters})


def units(effect: Effect, target: TargetPlugin) -> dict[str, Any]:
    """The effect as cards show it, with the target's units (``per``, ``rolls``) when it declares them."""
    return {"name": effect.name, "per": getattr(target, "per", None), "rolls": bool(getattr(target, "rolls", False))}


def owner(plugin: Any, name: str) -> type | None:
    """The class that defines ``name``: an optional faster form counts only when it comes with the method it
    computes, so a subclass that overrides the method without it falls back to the method."""
    return next((k for k in type(plugin).__mro__ if name in vars(k)), None)


def outcome_values(
    plugin: OutcomePlugin, sources: Sources, candidates: pl.DataFrame, horizons: Sequence[pd.Timedelta]
) -> Iterator[pl.DataFrame]:
    """The outcome at each horizon in turn, from one ``values_each`` call when the plugin's class defines it with
    ``values``; each with ``t``, ``inst``, ``y``, its parts and ``y_end``."""
    each = getattr(plugin, "values_each", None) if owner(plugin, "values_each") is owner(plugin, "values") else None
    if callable(each):
        found = cast(Iterable[pl.DataFrame | pd.DataFrame], each(sources, candidates, list(horizons)))
        return (outcome_rows(frame) for frame in found)
    return (outcome_rows(plugin.values(sources, candidates, h)) for h in horizons)


def outcome_rows(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """An outcome's rows as the engine reads them, or a refusal that says what an outcome returns."""
    out = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    missing = [c for c in ("t", "inst", "y", "y_end") if c not in out.columns]
    if missing:
        raise DeriveError(f"the outcome returned no {', '.join(missing)}; {OUTCOME}")
    return out.select(
        pl.col("t").dt.convert_time_zone("UTC").cast(TIME),
        pl.col("inst").cast(pl.String),
        pl.col("y").cast(pl.Float64),
        *(pl.col(c).cast(pl.Float64) for c in parts(out.columns)),
        pl.col("y_end").dt.convert_time_zone("UTC").cast(TIME),
    )


def per_t(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    out = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    if not {"t", "value"} <= set(out.columns):
        raise DeriveError("a target must return columns t and value")
    return out.select(pl.col("t").cast(TIME), pl.col("value").cast(pl.Float64)).drop_nulls().sort("t")


def norm_candidates(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    frame = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
    missing = {"t", "inst"} - set(frame.columns)
    if missing:
        raise DeriveError(f"the universe must return columns t and inst; missing {', '.join(sorted(missing))}")
    return frame.with_columns(pl.col("t").dt.convert_time_zone("UTC").cast(TIME), pl.col("inst").cast(pl.String))


def quantile_times(times: pl.Series, k: int) -> list[int]:
    ns = sorted(int(x) for x in times.cast(pl.Int64).to_list())
    if not ns or k < 1:
        return []
    return sorted({ns[int(len(ns) * (i + 1) / (k + 1))] for i in range(k)})


def iso(ns: int) -> str:
    return from_ns(ns).strftime("%Y-%m-%dT%H:%MZ")


def median(series: pl.Series) -> float:
    return float(series.cast(pl.Float64).median() or 0.0)  # type: ignore[arg-type]


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def jsonable(obj: Any) -> Any:
    """Plain JSON values as the ledger keeps them: keys sorted, tuples become lists, and numbers that are not finite
    become null. A card rendered from this reads the same as one rendered from the ledger."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in sorted(((str(k), v) for k, v in obj.items()), key=lambda kv: kv[0])}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating | np.integer):
        return jsonable(obj.item())
    return obj
