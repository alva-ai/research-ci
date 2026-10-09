"""How far apart the labels of the highest and lowest scored candidates are, at each decision time."""

import polars as pl


def quantile_spread(table: pl.DataFrame, share: float = 0.2, label: str = "y") -> pl.DataFrame:
    """Mean label of the top ``share`` of candidates by score minus that of the bottom ``share``, at each time."""
    rows = table.filter(pl.col("score").is_not_null() & pl.col(label).is_not_null())
    q = (pl.col("score").rank("ordinal") / pl.len()).over("t")
    rows = rows.with_columns(q=q)
    top = pl.col(label).filter(pl.col("q") > 1 - share).mean()
    bottom = pl.col(label).filter(pl.col("q") <= share).mean()
    return rows.group_by("t").agg((top - bottom).alias("value")).sort("t")
