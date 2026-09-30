import pandas as pd
import polars as pl

from research_ci.api.plugins import CostModel


class ClusterMean:
    """The mean label per decision time, each cluster weighted equally; with ``net``, after each candidate's round
    trip."""

    def __init__(self, cluster: str = "entity", net: bool = True) -> None:
        self.cluster, self.net = cluster, net

    def reduce(self, labels: pl.DataFrame, horizon: pd.Timedelta, costs: CostModel) -> pl.DataFrame:
        value = pl.col("value") - pl.Series(costs.round_trip(labels)) if self.net else pl.col("value")
        per = labels.with_columns(value.alias("_v")).group_by("t", self.cluster).agg(pl.col("_v").mean())
        return per.group_by("t").agg(pl.col("_v").mean().alias("value")).sort("t")
