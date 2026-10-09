"""Tables cut down before a statistic reads them."""

import polars as pl

from research_ci.api.methods import Case
from research_ci.eval.synthetic import hand


def without_top(table: pl.DataFrame, column: str, share: float, by: str = "y") -> pl.DataFrame:
    """The table without the ``share`` of ``column`` values (such as symbols) with the largest mean absolute ``by``."""
    sizes = table.group_by(column).agg(pl.col(by).abs().mean().alias("_m")).sort("_m", descending=True)
    drop = sizes.head(max(1, int(share * sizes.height)))[column]
    return table.filter(~pl.col(column).is_in(drop.implode()))


SYMBOLS = {"symbol": ["a", "a", "b", "c", "d"], "y": [0.9, -0.7, 0.1, -0.2, 0.05], "score": [1.0, 2, 3, 4, 5]}

CASES = [
    Case(
        "leaves out the symbol with the largest mean absolute label",
        lambda: sorted(without_top(hand(SYMBOLS), "symbol", 0.25)["symbol"].unique().to_list()),
        ["b", "c", "d"],
    ),
    Case(
        "leaves out at least one",
        lambda: sorted(without_top(hand(SYMBOLS), "symbol", 0.01)["symbol"].unique().to_list()),
        ["b", "c", "d"],
    ),
]
