"""Reference criteria: functions of the evaluation table that an idea may list in ``CRITERIA``. They have no
privileges; the engine judges them like an idea's own, with null worlds and calibration."""

import polars as pl


def rank_ic(table: pl.DataFrame, label: str = "y") -> pl.DataFrame:
    """Spearman correlation of score and label across the candidates at each decision time."""
    rows = table.filter(pl.col("score").is_not_null() & pl.col(label).is_not_null())
    return rows.group_by("t").agg(pl.corr("score", label, method="spearman").alias("value")).sort("t")


def quantile_spread(table: pl.DataFrame, share: float = 0.2, label: str = "y") -> pl.DataFrame:
    """Mean label of the top ``share`` of candidates by score minus that of the bottom ``share``, at each time."""
    rows = table.filter(pl.col("score").is_not_null() & pl.col(label).is_not_null())
    q = (pl.col("score").rank("ordinal") / pl.len()).over("t")
    rows = rows.with_columns(q=q)
    top = pl.col(label).filter(pl.col("q") > 1 - share).mean()
    bottom = pl.col(label).filter(pl.col("q") <= share).mean()
    return rows.group_by("t").agg((top - bottom).alias("value")).sort("t")


def without_top(table: pl.DataFrame, column: str, share: float, by: str = "y") -> pl.DataFrame:
    """The table without the ``share`` of ``column`` values (such as symbols) with the largest mean absolute ``by``."""
    sizes = table.group_by(column).agg(pl.col(by).abs().mean().alias("_m")).sort("_m", descending=True)
    drop = sizes.head(max(1, int(share * sizes.height)))[column]
    return table.filter(~pl.col(column).is_in(drop.implode()))
