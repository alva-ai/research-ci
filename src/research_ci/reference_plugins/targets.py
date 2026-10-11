import pandas as pd
import polars as pl

from research_ci.api.plugins import CostModel


class ClusterMean:
    """The mean label per decision time, each cluster weighted equally; with ``net``, after each candidate's round
    trip. With ``top``, when the rows carry a ``score``, only that share of the candidates with the highest scores at
    each time counts."""

    per = "entry"
    rolls = True

    def __init__(self, cluster: str = "entity", net: bool = True, top: float | None = None) -> None:
        if top is not None and not 0 < top <= 1:
            raise ValueError("top is a share in (0, 1]")
        self.cluster, self.net, self.top = cluster, net, top

    def reduce(self, labels: pl.DataFrame, horizon: pd.Timedelta, costs: CostModel) -> pl.DataFrame:
        if self.top is not None and "score" in labels.columns:
            labels = labels.filter(pl.col("score").is_not_null())
            rank = pl.col("score").rank("ordinal", descending=True).over("t") / pl.len().over("t")
            labels = labels.filter(rank <= self.top)
        value = pl.col("value") - pl.Series(costs.round_trip(labels)) if self.net else pl.col("value")
        per = labels.with_columns(value.alias("_v")).group_by("t", self.cluster).agg(pl.col("_v").mean())
        return per.group_by("t").agg(pl.col("_v").mean().alias("value")).sort("t")
