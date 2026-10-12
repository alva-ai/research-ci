import pandas as pd
import polars as pl

from research_ci.api.plugins import CostModel
from research_ci.eval.statistics.shuffled import TopClusterMean


class ClusterMean:
    """Which candidates do better (cross), or what follows an event (events): the mean outcome ``y`` per decision time
    over the candidates with a score, each cluster weighted equally; with ``net``, after each candidate's round trip.

    With ``top``, only that share of the candidates with the highest scores at each time counts, for a cross shape.
    Without it, every candidate with a score counts, so for events it is the mean outcome at the events, the rows the
    features scored; a row without a finite score never counts. Rows that carry no score at all, before any features
    exist, all count."""

    per = "entry"
    rolls = True

    def __init__(self, cluster: str = "inst", net: bool = True, top: float | None = None) -> None:
        if top is not None and not 0 < top <= 1:
            raise ValueError("top is a share in (0, 1]")
        self.cluster, self.net, self.top = cluster, net, top

    def reduce(self, rows: pl.DataFrame, horizon: pd.Timedelta, costs: CostModel) -> pl.DataFrame:
        if "score" in rows.columns:
            rows = rows.filter(pl.col("score").is_not_null() & pl.col("score").is_not_nan())
            if self.top is not None:
                rank = pl.col("score").rank("ordinal", descending=True).over("t") / pl.len().over("t")
                rows = rows.filter(rank <= self.top)
        value = pl.col("y") - pl.Series(costs.round_trip(rows)) if self.net else pl.col("y")
        per = rows.with_columns(value.alias("_v")).group_by("t", self.cluster).agg(pl.col("_v").mean())
        return per.group_by("t").agg(pl.col("_v").mean().alias("value")).sort("t")

    def batched(self, horizon: pd.Timedelta, costs: CostModel) -> TopClusterMean | None:
        """``reduce`` on many shuffles of the scores at once, when it reads them (with ``top``)."""
        return None if self.top is None else TopClusterMean(self.top, self.cluster, self.net, costs)


class SignedMean:
    """When to be long or short (series): the mean over the candidates at each decision time of sign(score) × ``y``,
    long where the score is positive, short where it is negative, flat at zero.

    A row without a finite score does not count; rows that carry no score at all, before any features exist, all count
    long."""

    per = "entry"
    rolls = True

    def reduce(self, rows: pl.DataFrame, horizon: pd.Timedelta, costs: CostModel) -> pl.DataFrame:
        side = pl.lit(1.0)
        if "score" in rows.columns:
            rows = rows.filter(pl.col("score").is_finite())
            side = pl.col("score").sign()
        return rows.group_by("t").agg((side * pl.col("y")).mean().alias("value")).sort("t")
