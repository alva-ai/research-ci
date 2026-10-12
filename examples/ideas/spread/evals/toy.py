"""The toy idea's lab evals: the funding part of the carry, and closes at zero or below."""

from datetime import UTC, datetime

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)


def funding_ic(table):
    """The rank correlation of the score with the funding part of the carry, at each decision time."""
    rows = table.filter(pl.col("y_flow").is_not_null())
    return rows.group_by("t").agg(pl.corr("score", "y_flow", method="spearman").alias("value")).sort("t")


def closes_at_zero(fields):
    """Closes at zero or below."""
    return fields.filter((pl.col("field") == "close") & (pl.col("value") <= 0)).select("entity", "field", "event_time")


FOUR = pl.DataFrame(
    {
        "t": [T] * 4,
        "entity": list("abcd"),
        "score": [1.0, 2.0, 3.0, 4.0],
        "y": [0.0] * 4,
        "y_flow": [0.1, 0.2, 0.4, 0.3],
    }
)
ROWS = pl.DataFrame({"entity": ["a", "a"], "field": ["close"] * 2, "event_time": [T, T], "value": [1.0, 0.0]})
CASES = [
    Case("one swap in four: 1 - 6·2/(4·15)", funding_ic, 0.8, given=FOUR),
    Case("a close at zero is flagged", closes_at_zero, 1, given=ROWS),
]
