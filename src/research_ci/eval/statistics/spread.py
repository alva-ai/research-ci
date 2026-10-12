"""How far apart the labels of the highest and lowest scored candidates are, at each decision time."""

import polars as pl

from research_ci.api.methods import Case, Power, Size
from research_ci.eval.synthetic import hand


def quantile_spread(table: pl.DataFrame, share: float = 0.2, label: str = "y") -> pl.DataFrame:
    """Mean label of the top ``share`` less the bottom ``share``. The shares are of the candidates by score, at each
    time; a share is ``floor(share * n)`` candidates, at least one."""
    rows = table.filter(pl.col("score").is_not_null() & pl.col(label).is_not_null())
    n = pl.len().over("t")
    k = (n * share + 1e-9).floor().clip(lower_bound=1)
    rows = rows.with_columns(r=pl.col("score").rank("ordinal").over("t"), n=n, k=k)
    top = pl.col(label).filter(pl.col("r") > pl.col("n") - pl.col("k")).mean()
    bottom = pl.col(label).filter(pl.col("r") <= pl.col("k")).mean()
    return rows.group_by("t").agg((top - bottom).alias("value")).sort("t")


FIVE = {"score": [5.0, 1.0, 4.0, 2.0, 3.0], "y": [0.5, 0.1, 0.4, 0.2, 0.3]}

CASES = [
    Case("the top fifth minus the bottom fifth", lambda: quantile_spread(hand(FIVE))["value"][0], 0.4),
    Case("the top two fifths minus the bottom two", lambda: quantile_spread(hand(FIVE), share=0.4)["value"][0], 0.3),
    Case("at least one each way", lambda: quantile_spread(hand(FIVE), share=0.1)["value"][0], 0.4),
]

PROPERTIES = [Size(quantile_spread), Power(quantile_spread)]
