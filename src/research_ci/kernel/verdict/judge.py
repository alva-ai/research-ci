"""Judging a criterion: its value per decision time, averaged by blocks of the horizon, against the same average in
each null world; and the decisive effect against the contract's smallest effect worth having."""

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import polars as pl

from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.power import block_means

Evaluate = Callable[[pl.DataFrame], pl.DataFrame]
Index = Callable[[int], np.ndarray]


class CriterionError(Exception):
    pass


def per_t(evaluate: Evaluate, table: pl.DataFrame) -> pl.DataFrame:
    out = evaluate(table)
    if not isinstance(out, pl.DataFrame) or not {"t", "value"} <= set(out.columns):
        raise CriterionError("a criterion must return a polars DataFrame with columns t and value")
    out = out.select(pl.col("t").cast(TIME), pl.col("value").cast(pl.Float64))
    return out.filter(pl.col("value").is_finite()).sort("t")


def blocks(values: pl.DataFrame, h_ns: int) -> tuple[float, float, int]:
    """Mean of the block means, its standard error, and the number of blocks."""
    means = block_means(values, h_ns)["value"].drop_nulls().to_numpy()
    n = len(means)
    if n == 0:
        return math.nan, math.nan, 0
    se = float(np.std(means, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return float(means.mean()), se, n


def with_scores(table: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    return table.with_columns(pl.Series("score", scores, dtype=pl.Float64))


def judge(
    evaluate: Evaluate | Mapping[str, Evaluate],
    tables: dict[str, tuple[pl.DataFrame, int]],
    index: Callable[[str], Index],
    reps: int,
    min_blocks: int,
    combine: str,
) -> dict[str, Any]:
    """One criterion against one null world over its horizons. ``tables`` maps a horizon to its evaluation table and
    length; ``index(h)(seed)`` gives the world's shuffle for that horizon; ``evaluate`` may differ by horizon.
    Horizons with fewer than ``min_blocks`` blocks are shown but not counted. With ``combine="max"`` the family's z
    compares the best standardized horizon with the best of each null world; with ``"each"`` it is the weakest
    horizon's z."""
    rows, nulls = [], []
    for h, (table, h_ns) in tables.items():
        f = evaluate[h] if isinstance(evaluate, Mapping) else evaluate
        real, se, n = blocks(per_t(f, table), h_ns)
        score = table["score"].to_numpy()
        shuffle = index(h)
        null = np.array([blocks(per_t(f, with_scores(table, score[shuffle(r)])), h_ns)[0] for r in range(reps)])
        mu, sd = float(np.nanmean(null)), float(np.nanstd(null, ddof=1))
        z = (real - mu) / sd if sd > 0 else math.nan
        counted = n >= min_blocks and math.isfinite(z)
        rows.append({"h": h, "real": real, "se": se, "blocks": n, "null_mean": mu, "null_sd": sd, "z": z})
        rows[-1]["counted"] = counted
        if counted:
            nulls.append((null - mu) / sd)
    counted = [r for r in rows if r["counted"]]
    if not counted:
        return {"z": math.nan, "best": None, "horizons": rows, "note": f"fewer than {min_blocks} blocks everywhere"}
    if combine == "each":
        worst = min(counted, key=lambda r: r["z"])
        return {"z": worst["z"], "best": worst["h"], "horizons": rows, "note": "weakest horizon"}
    best = max(counted, key=lambda r: r["z"])
    if len(counted) == 1:
        return {"z": best["z"], "best": best["h"], "horizons": rows, "note": "one horizon"}
    top = np.nanmax(np.vstack(nulls), axis=0)
    sd = float(np.nanstd(top, ddof=1))
    z = (best["z"] - float(np.nanmean(top))) / sd if sd > 0 else math.nan
    return {"z": z, "best": best["h"], "horizons": rows, "note": f"best of {len(counted)} horizons"}


def decide(effect: float, se: float, z: float, at_least: float, need: float, margin: float) -> str:
    """``failed`` when even ``effect + margin·se`` misses ``at_least``; ``passed`` when the effect beats its null by
    ``need`` and ``effect - margin·se`` still reaches ``at_least``; otherwise ``insufficient``."""
    if not (math.isfinite(effect) and math.isfinite(se)):
        return "insufficient"
    if effect + margin * se < at_least:
        return "failed"
    if math.isfinite(z) and z >= need and effect - margin * se >= at_least:
        return "passed"
    return "insufficient"
