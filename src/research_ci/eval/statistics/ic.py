"""Rank correlation of the score with the label: across the candidates at each decision time, or over each inst's
decision times."""

import polars as pl

from research_ci.api.evals import Case, Market, Power, Recovers, Size
from research_ci.eval.synthetic import HOUR_NS, hand


def rank_ic(table: pl.DataFrame, label: str = "y") -> pl.DataFrame:
    """Spearman correlation of score and label at each decision time. The correlation runs across the candidates at
    that time."""
    rows = table.filter(pl.col("score").is_not_null() & pl.col(label).is_not_null())
    return rows.group_by("t").agg(pl.corr("score", label, method="spearman").alias("value")).sort("t")


def ts_ic(table: pl.DataFrame, label: str = "y") -> pl.DataFrame:
    """Rank correlation of score and label over time, per inst. Each inst's score and label are ranked over its
    own decision times and standardized, and the value at a decision time is the mean over the insts of their
    product, so its mean over one inst's times is their Spearman correlation over time. Rows without a finite score
    or label are left out, and an inst whose score or label never changes counts nowhere."""
    rows = table.filter(pl.col("score").is_finite() & pl.col(label).is_finite())
    ranked = rows.with_columns(pl.col("score", label).rank().over("inst").cast(pl.Float64))
    standard = [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std(ddof=0)).over("inst") for c in ("score", label)]
    product = (standard[0] * standard[1]).fill_nan(None).alias("value")
    return ranked.select("t", product).group_by("t").agg(pl.col("value").mean()).sort("t")


def _one(scores: list[float | None], labels: list[float | None]) -> float:
    return rank_ic(hand({"score": scores, "y": labels}))["value"][0]


def _over_time(columns: dict[str, list]) -> float:
    """``ts_ic`` on a hand table whose rows are the hours 0, 1, 2, ... of each inst, averaged over the hours."""
    rows = len(columns["score"])
    insts = columns.get("inst", ["a"] * rows)
    frame = pl.DataFrame({**columns, "inst": insts}, strict=False).with_columns(
        t=(pl.int_range(pl.len()).over("inst") * HOUR_NS).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")
    )
    return float(ts_ic(frame)["value"].mean())  # type: ignore[arg-type]


SERIES = Market(times=300, insts=4, clusters=4, shape="series")

CASES = [
    Case("the same order", lambda: _one([1, 2, 3, 4], [0.1, 0.3, 0.5, 0.9]), 1.0),
    Case("the reverse order", lambda: _one([1, 2, 3, 4], [0.9, 0.5, 0.3, 0.1]), -1.0),
    Case("rows missing a score or a label are left out", lambda: _one([1, 2, None, 4], [0.1, 0.3, 0.2, None]), 1.0),
    Case(
        "over time: one swap in four, Spearman's 1 - 6·6/(4·15)",
        lambda: _over_time({"score": [1.0, 3.0, 2.0, 4.0], "y": [0.1, 0.2, 0.4, 0.3]}),
        0.4,
    ),
    Case(
        "each inst ranks its own times: one in order, one reversed",
        lambda: _over_time(
            {"inst": ["a"] * 3 + ["b"] * 3, "score": [1.0, 2.0, 3.0] * 2, "y": [1.0, 2.0, 3.0, 9.0, 8.0, 7.0]}
        ),
        0.0,
    ),
    Case(
        "over time, rows without a finite score are left out",
        lambda: _over_time({"score": [1.0, float("nan"), 2.0, 3.0], "y": [0.1, 0.9, 0.2, 0.3]}),
        1.0,
    ),
]

PROPERTIES = [
    Size(rank_ic),
    Power(rank_ic),
    Recovers(rank_ic, Market(ic=0.1, tail=None), expect=0.1, tol=0.01),
    Size(ts_ic, SERIES),
    Power(ts_ic, Market(times=300, insts=4, clusters=4, shape="series", ic=0.1)),
    Recovers(ts_ic, Market(times=300, insts=4, clusters=4, shape="series", ic=0.1, tail=None), 0.1, 0.015),
]
