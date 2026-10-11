"""The diagnostics the engine shows on every feat submission, next to the idea's own criteria from ``feat.py``.

They are judged against the floor world and the idea's null worlds like any criterion, and they never decide a
verdict. So this list can change with experience without touching any freeze: the engine hash leaves
``stages/defaults`` out.
"""

import polars as pl

from research_ci.api.contract import Contract
from research_ci.api.study import Criterion
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.statistics.spread import quantile_spread
from research_ci.eval.structure.filters import without_top

SHARE = 0.2


def defaults(contract: Contract) -> list[Criterion]:
    """Rank IC, rank IC without the fifth of the cluster values (at least one) with the largest outcomes, and the top
    fifth by score less the bottom fifth, on the label of the first decisive effect."""
    assert contract.universe is not None
    label, clusters = contract.decisive[0].on, contract.universe.clusters

    def ic_without_top(table: pl.DataFrame) -> pl.DataFrame:
        return rank_ic(without_top(table, clusters, SHARE))

    return [
        Criterion("rank ic", rank_ic, label=label),
        Criterion(f"rank ic without top {clusters}", ic_without_top, label=label),
        Criterion("top minus bottom fifth", lambda table: quantile_spread(table, SHARE), label=label),
    ]
