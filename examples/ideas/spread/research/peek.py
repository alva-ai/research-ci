import polars as pl


def features(wide, candidates):
    rate = wide.group_by("inst").agg(pl.col("funding_rate").mean())
    short = rate.rename({"inst": "short", "funding_rate": "s"})
    long = rate.rename({"inst": "long", "funding_rate": "l"})
    rows = candidates.join(short, on="short").join(long, on="long")
    return rows.select("t", "inst", (pl.col("s") - pl.col("l")).alias("score"))
