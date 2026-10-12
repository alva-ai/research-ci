"""What the stages and their lists share: the target a decisive effect names, candidate and per-time tables as the
engine reads them, the treatments a step finds, and the JSON values the ledger keeps."""

import math
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import polars as pl

from research_ci.api.contract import Contract, ContractError, Effect, Use, as_use
from research_ci.api.plugins import LabelPlugin, TargetPlugin
from research_ci.kernel.time.clock import TIME, from_ns

DEFAULT_TARGET = "ref.mean"
UNSCORED = 0.5


class DeriveError(ContractError):
    pass


@dataclass(frozen=True)
class Treatment:
    """``removed`` gives the rows the treatment drops from a field table or a candidate table, so its influence can be
    weighed by leaving out the label rows that read them; a treatment without it changes rows instead, such as a
    split, and only ``apply`` says what it does."""

    name: str
    on: Literal["fields", "candidates"]
    reason: str
    rows: int
    apply: Callable[[pl.DataFrame], pl.DataFrame]
    removed: Callable[[pl.DataFrame], pl.DataFrame] | None = None


def target_use(effect: Effect, c: Contract) -> Use:
    """The target a decisive effect names; ``ref.mean`` over the universe's clusters when it names none."""
    assert c.universe is not None
    return as_use(effect.target) if effect.target else Use(DEFAULT_TARGET, {"cluster": c.universe.clusters})


def units(effect: Effect, target: TargetPlugin) -> dict[str, Any]:
    """The effect as cards show it, with the target's units (``per``, ``rolls``) when it declares them."""
    return {"on": effect.on, "per": getattr(target, "per", None), "rolls": bool(getattr(target, "rolls", False))}


def owner(plugin: Any, name: str) -> type | None:
    """The class that defines ``name``: an optional faster form counts only when it comes with the method it
    computes, so a subclass that overrides the method without it falls back to the method."""
    return next((k for k in type(plugin).__mro__ if name in vars(k)), None)


def label_values(
    plugin: LabelPlugin, fields: pl.DataFrame, candidates: pl.DataFrame, horizons: Sequence[pd.Timedelta]
) -> Iterator[pl.DataFrame]:
    """The label at each horizon in turn, from one ``values_each`` call when the plugin's class defines it with
    ``values``."""
    each = getattr(plugin, "values_each", None) if owner(plugin, "values_each") is owner(plugin, "values") else None
    if callable(each):
        found = cast(Iterable[pl.DataFrame | pd.DataFrame], each(fields, candidates, list(horizons)))
        return (_polars(frame) for frame in found)
    return (_polars(plugin.values(fields, candidates, h)) for h in horizons)


def _polars(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame


def per_t(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    out = pl.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    if not {"t", "value"} <= set(out.columns):
        raise DeriveError("a target must return columns t and value")
    return out.select(pl.col("t").cast(TIME), pl.col("value").cast(pl.Float64)).drop_nulls().sort("t")


def norm_candidates(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    frame = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
    missing = {"t", "entity"} - set(frame.columns)
    if missing:
        raise DeriveError(f"the universe must return columns t and entity; missing {', '.join(sorted(missing))}")
    return frame.with_columns(pl.col("t").dt.convert_time_zone("UTC").cast(TIME), pl.col("entity").cast(pl.String))


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
