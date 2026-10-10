"""Rank correlation of the score with the label across candidates, at each decision time."""

import polars as pl

from research_ci.api.methods import Case, Market, Power, Recovers, Size
from research_ci.eval.synthetic import hand


def rank_ic(table: pl.DataFrame, label: str = "y") -> pl.DataFrame:
    """Spearman correlation of score and label across the candidates at each decision time."""
    rows = table.filter(pl.col("score").is_not_null() & pl.col(label).is_not_null())
    return rows.group_by("t").agg(pl.corr("score", label, method="spearman").alias("value")).sort("t")


def _one(scores: list[float | None], labels: list[float | None]) -> float:
    return rank_ic(hand({"score": scores, "y": labels}))["value"][0]


CASES = [
    Case("the same order", lambda: _one([1, 2, 3, 4], [0.1, 0.3, 0.5, 0.9]), 1.0),
    Case("the reverse order", lambda: _one([1, 2, 3, 4], [0.9, 0.5, 0.3, 0.1]), -1.0),
    Case("rows missing a score or a label are left out", lambda: _one([1, 2, None, 4], [0.1, 0.3, 0.2, None]), 1.0),
]

PROPERTIES = [
    Size(rank_ic),
    Power(rank_ic),
    Recovers(rank_ic, Market(ic=0.1, tail=None), expect=0.1, tol=0.01),
]
