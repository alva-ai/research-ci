"""The toy idea's lab evals: the flow part of the carry, and closes at zero or below."""

from datetime import UTC, datetime

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)


def funding_ic(table):
    """The rank correlation of the score with the flow part of the outcome, at each decision time."""
    rows = table.filter(pl.col("y_flow").is_not_null())
    return rows.group_by("t").agg(pl.corr("score", "y_flow", method="spearman").alias("value")).sort("t")


def closes_at_zero(sources):
    """Closes at zero or below, as cells of the candles."""
    zero = sources["candles"].filter(pl.col("close") <= 0)
    return zero.select(pl.lit("candles").alias("source"), "inst", "event_time", pl.lit("close").alias("column"))


FOUR = pl.DataFrame(
    {
        "t": [T] * 4,
        "inst": list("abcd"),
        "score": [1.0, 2.0, 3.0, 4.0],
        "y": [0.0] * 4,
        "y_flow": [0.1, 0.2, 0.4, 0.3],
    }
)
ROWS = {"candles": pl.DataFrame({"inst": ["a", "a"], "event_time": [T, T], "close": [1.0, 0.0]})}
CASES = [
    Case("one swap in four: 1 - 6·2/(4·15)", funding_ic, 0.8, given=FOUR),
    Case("a close at zero is flagged", closes_at_zero, 1, given=ROWS),
]
