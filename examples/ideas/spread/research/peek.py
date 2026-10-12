import polars as pl


def features(fields, candidates):
    rate = fields.filter(pl.col("field") == "funding_rate").group_by("inst").agg(pl.col("value").mean())
    short = rate.rename({"inst": "short", "value": "s"})
    long = rate.rename({"inst": "long", "value": "l"})
    rows = candidates.join(short, on="short").join(long, on="long")
    return rows.select("t", "inst", (pl.col("s") - pl.col("l")).alias("score"))
