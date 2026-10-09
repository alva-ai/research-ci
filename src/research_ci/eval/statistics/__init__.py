"""Statistics of the evaluation table: one value per decision time. Criteria pick them by name; they have no
privileges over an idea's own."""

from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.statistics.spread import quantile_spread

__all__ = ["quantile_spread", "rank_ic"]
