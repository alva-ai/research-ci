import polars as pl


def features(fields, candidates):
    rate = fields.filter(pl.col("field") == "funding_rate").group_by("entity").agg(pl.col("value").mean())
    short = rate.rename({"entity": "short", "value": "s"})
    long = rate.rename({"entity": "long", "value": "l"})
    rows = candidates.join(short, on="short").join(long, on="long")
    return rows.select("t", "entity", (pl.col("s") - pl.col("l")).alias("score"))
