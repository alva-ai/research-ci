import polars as pl


def features(fields, candidates):
    return candidates.select("t", "entity", pl.col("spread_h").alias("score"))
