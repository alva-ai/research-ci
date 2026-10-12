import math

import numpy as np
import polars as pl

from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.judge import judge
from research_ci.kernel.verdict.permute import Permutation

HOUR = 3_600 * 10**9


def _table(times: int = 60, n: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        (t * HOUR, f"e{i}", f"c{i // 2}", float(rng.standard_normal()), float(i))
        for t in range(times)
        for i in range(n)
    ]
    frame = pl.DataFrame(rows, schema=["t", "inst", "cluster", "score", "y"], orient="row")
    return frame.with_columns(pl.col("t").cast(TIME))


def _sum_of_score_times_y(table: pl.DataFrame) -> pl.DataFrame:
    return table.group_by("t").agg((pl.col("score") * pl.col("y")).sum().alias("value"))


def test_the_family_z_prices_in_picking_the_best_of_several_horizons() -> None:
    table = _table()
    tables = {"1h": (table, HOUR), "2h": (table, 2 * HOUR), "3h": (table, 3 * HOUR)}

    def index(h: str):
        return Permutation(table, None, "cluster", tables[h][1]).index

    result = judge(_sum_of_score_times_y, tables, index, 60, 5, "max")
    best = max(r["z"] for r in result["horizons"] if r["counted"])
    assert len(result["horizons"]) == 3 and result["note"] == "best of 3 horizons"
    assert math.isfinite(result["z"]) and result["z"] < best


def test_each_horizon_is_set_against_the_null_the_family_uses_and_the_best_gives_the_family_z() -> None:
    """Single-step max-T: a horizon's z against the best of each shuffle's standardized horizons; the best horizon's
    is the family's z, and no horizon's is above it."""
    table = _table()
    tables = {"1h": (table, HOUR), "2h": (table, 2 * HOUR), "3h": (table, 3 * HOUR)}

    def index(h: str):
        return Permutation(table, None, "cluster", tables[h][1]).index

    result = judge(_sum_of_score_times_y, tables, index, 60, 5, "max")
    rows = {r["h"]: r for r in result["horizons"]}
    assert math.isclose(rows[result["best"]]["adjusted"], result["z"], rel_tol=1e-12)
    assert all(r["adjusted"] <= result["z"] + 1e-12 for r in rows.values())
    each = judge(_sum_of_score_times_y, tables, index, 60, 5, "each")
    assert all(r["adjusted"] == r["z"] for r in each["horizons"])
