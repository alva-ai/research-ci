"""Rank correlation of the score with the label across candidates, at each decision time."""

import polars as pl


def rank_ic(table: pl.DataFrame, label: str = "y") -> pl.DataFrame:
    """Spearman correlation of score and label across the candidates at each decision time."""
    rows = table.filter(pl.col("score").is_not_null() & pl.col(label).is_not_null())
    return rows.group_by("t").agg(pl.corr("score", label, method="spearman").alias("value")).sort("t")
