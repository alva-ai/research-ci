import polars as pl


def features(fields, candidates):
    return candidates.select("t", "inst", (pl.struct("t", "inst").hash(7) / 2.0**64).alias("score"))
