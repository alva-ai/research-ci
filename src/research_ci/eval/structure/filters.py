"""Tables cut down before a statistic reads them."""

import polars as pl


def without_top(table: pl.DataFrame, column: str, share: float, by: str = "y") -> pl.DataFrame:
    """The table without the ``share`` of ``column`` values (such as symbols) with the largest mean absolute ``by``."""
    sizes = table.group_by(column).agg(pl.col(by).abs().mean().alias("_m")).sort("_m", descending=True)
    drop = sizes.head(max(1, int(share * sizes.height)))[column]
    return table.filter(~pl.col(column).is_in(drop.implode()))
