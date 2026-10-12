# rci:example -- the engine refuses to run this file while this line is in it. Copy it to
# research/<name>.py, delete this line in the copy, and replace every TODO with your own features.
"""TODO: one sentence on what the score measures and why it should rank the candidates."""

import polars as pl

FIELD = "TODO"  # a field of the data; `rci plugins` lists the fields a data plugin declares


def features(fields: pl.DataFrame, candidates: pl.DataFrame) -> pl.DataFrame:
    """One score per candidate row (t, inst, score), for every decision time at once.

    fields: inst, field, event_time, available_at, value, vintage, source; all of the frozen data.
    candidates: t, inst, and the universe's columns (such as the legs long and short, the cluster and the groups).
    Both are polars DataFrames. A score at t may use only rows with available_at <= t; join_asof on available_at gives
    the latest of them. Return a polars DataFrame or LazyFrame, or a pandas DataFrame.
    """
    # .lazy() lets polars plan the whole pipeline and run it once at .collect(); eager code works the same way.
    known = (
        fields.lazy()
        .filter(pl.col("field") == FIELD)
        .select("inst", pl.col("available_at").alias("known_at"), "value")
        .sort("known_at")
    )
    # TODO: transform each inst's series with expressions, for example
    #   known.with_columns(pl.col("value").rolling_mean_by("known_at", window_size="24h").over("inst"))
    #   known.with_columns(pl.col("value").diff().over("inst").alias("change"))
    scored = (
        candidates.lazy()
        .sort("t")
        .join_asof(known, left_on="t", right_on="known_at", by="inst", check_sortedness=False)
    )
    # Candidates with two legs: look each leg up by its own column instead of inst, then take the difference:
    #   leg = known.rename({"inst": "long", "known_at": "long_at", "value": "long_value"})
    #   scored = candidates.lazy().sort("t").join_asof(leg, left_on="t", right_on="long_at", by="long",
    #                                                  check_sortedness=False)
    # Ranks within each decision time: pl.col("score").rank().over("t").
    # pandas is fine for dates (pd.Timedelta, offsets, calendars), and any library runs. Whole-column polars
    # expressions like these are usually much faster than going row by row, which runs too, with a warning on the card;
    # the card also shows the time and peak memory the features took.
    return scored.select("t", "inst", pl.col("value").alias("score")).collect()
