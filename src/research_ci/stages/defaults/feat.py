"""The diagnostics the engine shows on every feat submission, next to the idea's own evals from ``feat.py``.

Each is a step (``stages/steps.py``) that judges one eval on the label of the first decisive effect, under the step's
name (``{clusters}`` is the universe's cluster column), against the floor world and the idea's null worlds like any
eval; an idea eval of the same name is judged in its place. They never decide a verdict. So this list
can change with experience without touching any freeze: the engine hash leaves ``stages/defaults`` out.
"""

import polars as pl

from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.statistics.shuffled import QuantileSpread, RankCorrelation
from research_ci.eval.statistics.spread import quantile_spread
from research_ci.eval.structure.filters import kept_without_top, without_top
from research_ci.kernel.verdict.judge import Statistic
from research_ci.stages.floor.feat import Seen
from research_ci.stages.steps import Found, Step

SHARE = 0.2


def ic(seen: Seen) -> Found:
    """Rank IC: the Spearman correlation of score and label across the candidates at each decision time."""
    return seen.judge(Statistic(rank_ic, RankCorrelation()))


def ic_without_top(seen: Seen) -> Found:
    """Rank IC without the fifth of the cluster values (at least one) with the largest outcomes."""
    clusters = seen.clusters

    def evaluate(table: pl.DataFrame) -> pl.DataFrame:
        return rank_ic(without_top(table, clusters, SHARE))

    def kept(table: pl.DataFrame) -> pl.Series:
        return kept_without_top(table, clusters, SHARE)

    return seen.judge(Statistic(evaluate, RankCorrelation(keep=kept)))


def top_minus_bottom(seen: Seen) -> Found:
    """The mean label of the top fifth of the candidates by score less that of the bottom fifth."""
    return seen.judge(Statistic(lambda table: quantile_spread(table, SHARE), QuantileSpread(SHARE)))


CHECKS = [
    Step("rank ic", "shown", ic, "Rank correlation of score and label per time"),
    Step("rank ic without top {clusters}", "shown", ic_without_top, "Rank IC without the top fifth of clusters"),
    Step("top minus bottom fifth", "shown", top_minus_bottom, "Top fifth by score less the bottom fifth"),
]
