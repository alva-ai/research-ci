import numpy as np
import polars as pl

from research_ci.eval.calibration.stand_in import calibrate
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.statistics.spread import quantile_spread
from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.judge import decide, judge
from research_ci.kernel.verdict.permute import Permutation

HOUR = 3_600 * 10**9


def _table(signal: float, group_effect: float = 0.0, times: int = 120, n: int = 40, seed: int = 0) -> pl.DataFrame:
    """``n`` entities in clusters of two; ``y`` is ``signal`` times the score plus a group level plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(times):
        score = rng.standard_normal(n)
        noise = rng.standard_normal(n)
        for i in range(n):
            group = "g1" if i % 4 < 2 else "g2"
            y = signal * score[i] + (group_effect if group == "g1" else 0.0) + noise[i]
            rows.append((t * HOUR, f"e{i}", f"c{i // 2}", group, score[i], y))
    frame = pl.DataFrame(rows, schema=["t", "entity", "cluster", "group", "score", "y"], orient="row")
    return frame.with_columns(pl.col("t").cast(TIME))


def _run(table: pl.DataFrame, evaluate, bucket=None, reps: int = 60) -> dict:
    tables = {"1h": (table, HOUR)}

    def index(h: str):
        return Permutation(table, bucket, "cluster", HOUR).index

    return judge(evaluate, tables, index, reps, 5, "max")


def test_an_informative_score_beats_its_null_and_noise_does_not() -> None:
    assert _run(_table(0.3), rank_ic)["z"] > 4
    assert abs(_run(_table(0.0), rank_ic)["z"]) < 3
    assert _run(_table(0.3), quantile_spread)["z"] > 4


def test_the_decisive_effect_needs_both_information_and_size() -> None:
    assert decide(0.08, 0.01, 5.0, 0.05, 2.0, 2.0) == "passed"
    assert decide(0.01, 0.01, 5.0, 0.05, 2.0, 2.0) == "failed"
    assert decide(0.05, 0.02, 5.0, 0.05, 2.0, 2.0) == "insufficient"
    assert decide(0.08, 0.01, 1.0, 0.05, 2.0, 2.0) == "insufficient"


def test_a_null_world_passes_its_calibration_when_it_keeps_what_it_claims() -> None:
    table = _table(0.0, group_effect=1.0)
    for bucket in (None, "group"):
        shuffle = Permutation(table, bucket, "cluster", HOUR).index
        out = calibrate(rank_ic, table, HOUR, bucket, shuffle, reps=40, nulls=30, z_base=2.0)
        assert out["stand_in_dies"] >= 0.95 and out["planted_lives"] >= 0.8, out


def test_calibration_returns_a_world_that_does_not_keep_the_explanation_it_claims() -> None:
    """The group explains part of the label; a world that only shuffles within time lets the group stand-in live."""
    table = _table(0.0, group_effect=1.0)
    time_only = Permutation(table, None, "cluster", HOUR).index
    out = calibrate(rank_ic, table.with_columns(pl.col("group").alias("bucket")), HOUR, "group", time_only, 40, 30, 2.0)
    assert out["stand_in_dies"] < 0.95 and not out["ok"]
