# rci:example -- the engine refuses to run this file while this line is in it. Copy it to
# research/<name>.py, delete this line in the copy, and replace every TODO with your own features.
"""TODO: one sentence on what the score measures and why it should rank the candidates."""

import polars as pl

COLUMN = "TODO"  # a column of the sources; `rci plugins` lists the columns a data plugin gives


def features(wide: pl.DataFrame, candidates: pl.DataFrame) -> pl.DataFrame:
    """One score per candidate row (t, inst, score), for every decision time at once.

    wide: t, inst, every source's value columns as known at t, and age_<source>, how long before t that row of the
    source became known; one row per decision time and inst, sorted by t and inst.
    candidates: t, inst, and the universe's columns (such as the legs long and short, the cluster and the groups).
    Both are polars DataFrames. A row of wide holds only what was known by its t, so a score at t may use the rows of
    wide up to t. Return a polars DataFrame or LazyFrame, or a pandas DataFrame.
    """
    # .lazy() lets polars plan the whole pipeline and run it once at .collect(); eager code works the same way.
    known = wide.lazy().select("t", "inst", pl.col(COLUMN).alias("value"))
    # TODO: transform each inst's series with expressions, for example
    #   known.with_columns(pl.col("value").rolling_mean(24).over("inst"))
    #   known.with_columns(pl.col("value").diff().over("inst").alias("change"))
    scored = candidates.lazy().join(known, on=["t", "inst"], how="left")
    # Candidates with two legs: look each leg up by its own column instead of inst, then take the difference:
    #   leg = known.rename({"inst": "long", "value": "long_value"})
    #   scored = candidates.lazy().join(leg, on=["t", "long"], how="left")
    # Ranks within each decision time: pl.col("score").rank().over("t").
    # pandas is fine for dates (pd.Timedelta, offsets, calendars), and any library runs. Whole-column polars
    # expressions like these are usually much faster than going row by row, which runs too, with a warning on the card;
    # the card also shows the time and peak memory the features took.
    return scored.select("t", "inst", pl.col("value").alias("score")).collect()
