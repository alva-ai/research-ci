"""Tables cut down before a statistic reads them."""

import polars as pl

from research_ci.api.evals import Case
from research_ci.eval.synthetic import hand


def without_top(table: pl.DataFrame, column: str, share: float, by: str = "y") -> pl.DataFrame:
    """The table without its ``share`` of most extreme ``column`` values. They are the ``share`` of ``column`` values
    (such as symbols) with the largest mean absolute ``by``."""
    return table.filter(kept_without_top(table, column, share, by))


def kept_without_top(table: pl.DataFrame, column: str, share: float, by: str = "y") -> pl.Series:
    """Which rows ``without_top`` keeps. Equal means fall in the order of the ``column`` values."""
    sizes = table.group_by(column).agg(pl.col(by).abs().mean().alias("_m"))
    sizes = sizes.sort(["_m", column], descending=[True, False])
    drop = sizes.head(max(1, int(share * sizes.height)))[column]
    return table.select(~pl.col(column).is_in(drop.implode()))[column]


SYMBOLS = {"symbol": ["a", "a", "b", "c", "d"], "y": [0.9, -0.7, 0.1, -0.2, 0.05], "score": [1.0, 2, 3, 4, 5]}

CASES = [
    Case(
        "leaves out the symbol with the largest mean absolute outcome",
        lambda: sorted(without_top(hand(SYMBOLS), "symbol", 0.25)["symbol"].unique().to_list()),
        ["b", "c", "d"],
    ),
    Case(
        "leaves out at least one",
        lambda: sorted(without_top(hand(SYMBOLS), "symbol", 0.01)["symbol"].unique().to_list()),
        ["b", "c", "d"],
    ),
    Case(
        "of equal means, leaves out the first value",
        lambda: kept_without_top(hand({"symbol": ["b", "a", "c"], "y": [0.5, -0.5, 0.1]}), "symbol", 0.3).to_list(),
        [True, False, True],
    ),
]
