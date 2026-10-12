import polars as pl


def features(fields, candidates):
    return candidates.select("t", "inst", pl.col("spread_h").alias("score"))
