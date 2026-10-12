"""The built-in statistics on many shuffles at once.

A null world moves scores between the rows of one decision time and leaves everything else in place: each row's label,
its cluster, and which scores tie stay what they are, and only the row a score comes from changes. Each form here does
the work that does not change once per table, then takes a matrix of shuffles, one per row of ``index`` (row ``i``
takes the score of row ``index[i]``), and returns the statistic at each decision time for each shuffle, as the
statistic computes it on the shuffled table, to floating point; a missing score moves as NaN, as the shuffled table
holds it (``with_scores``), though feat judges only the candidates the features scored, so its tables hold none. NaN
marks a time without a value. The judge (``kernel/verdict/judge.py``) uses a form only
after it gives its statistic's numbers on the table as it is; tests compare every form with its statistic on shuffles.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from research_ci.api.methods import Case
from research_ci.api.plugins import CostModel
from research_ci.eval.synthetic import hand

Keep = Callable[[pl.DataFrame], pl.Series]


class _Layout:
    """The rows by decision time, kept in table order within a time (``p``: a row's place in that order)."""

    def __init__(self, table: pl.DataFrame) -> None:
        t = table["t"].dt.epoch("ns").to_numpy()
        self.n = t.size
        self.order = np.argsort(t, kind="stable")
        ts = t[self.order]
        first = np.concatenate([[True], ts[1:] != ts[:-1]]) if self.n else np.zeros(0, dtype=bool)
        self.starts = np.flatnonzero(first)
        self.times = ts[self.starts]
        self.seg = np.cumsum(first) - 1
        self.row_seg = np.empty(self.n, dtype=np.int64)
        self.row_seg[self.order] = self.seg

    def sums(self, x: np.ndarray) -> np.ndarray:
        """Per decision time sums of ``x`` (shuffles by places)."""
        if not self.n:
            return np.zeros((x.shape[0], 0))
        return np.add.reduceat(x, self.starts, axis=1)

    def sizes(self) -> np.ndarray:
        """How many rows each decision time has."""
        return np.diff(np.append(self.starts, self.n)).astype(np.float64)

    def donors(self, index: np.ndarray) -> np.ndarray:
        """The row whose score each place holds, per shuffle."""
        return index[:, self.order]

    def running(self, mask: np.ndarray) -> np.ndarray:
        """How many places before each one, at its decision time, are marked."""
        count = np.cumsum(mask, axis=1) - mask
        return count - count[:, self.starts][:, self.seg]


class _Ties:
    """The distinct ``values`` at each decision time, numbered across the table in time order and, within a time, in
    the order ``rank`` sorts them (NaN above every number); ``of[row]`` is the row's, -1 when null. ``whole`` and
    ``whole_below`` count every row of a time: how many hold each value, and how many hold a lower one."""

    def __init__(self, values: pl.Series, layout: _Layout, descending: bool = False) -> None:
        frame = pl.DataFrame({"g": layout.row_seg, "v": values})
        rank = frame.select(pl.col("v").rank("dense", descending=descending).over("g"))["v"]
        null = rank.is_null().to_numpy()
        local = rank.fill_null(0).cast(pl.Int64).to_numpy()
        distinct = np.zeros(layout.times.size, dtype=np.int64)
        np.maximum.at(distinct, layout.row_seg, local)
        first = np.cumsum(distinct) - distinct
        self.count = int(distinct.sum())
        self.base = np.repeat(first, distinct)
        self.time = np.repeat(np.arange(distinct.size), distinct)
        self.of = np.where(null, -1, first[layout.row_seg] + local - 1)
        self.whole = np.bincount(self.of[self.of >= 0], minlength=self.count)
        before = np.cumsum(self.whole) - self.whole
        self.whole_below = before - before[self.base] if self.count else before

    def counted(self, ties: np.ndarray, kept: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each shuffle, how many kept places hold each value, and how many hold a lower one at the same time."""
        shuffles = ties.shape[0]
        flat = (np.arange(shuffles)[:, None] * self.count + ties)[kept]
        counts = np.bincount(flat, minlength=shuffles * self.count).reshape(shuffles, self.count)
        before = np.cumsum(counts, axis=1) - counts
        return counts, before - before[:, self.base]

    def average(self, ties: np.ndarray, kept: np.ndarray) -> np.ndarray:
        """Each place's average rank among the kept places of its time, by the value it holds."""
        if not self.count:
            return np.zeros(ties.shape)
        counts, below = self.counted(ties, kept)
        rank = below + (counts + 1) / 2
        return np.take_along_axis(rank, np.maximum(ties, 0), axis=1)

    def at_most(self, ties: np.ndarray, kept: np.ndarray, layout: _Layout, k: np.ndarray) -> np.ndarray:
        """Whether each kept place's ordinal rank (ties in table order) is at most ``k`` of its time."""
        if not self.count:
            return np.zeros(ties.shape, dtype=bool)
        counts, below = self.counted(ties, kept)
        safe = np.maximum(ties, 0)
        lower = np.take_along_axis(below, safe, axis=1)
        same = np.take_along_axis(counts, safe, axis=1)
        cut = k[:, layout.seg]
        straddles = kept & (lower < cut) & (lower + same > cut)
        inside = kept & (lower + same <= cut)
        return inside | (straddles & (layout.running(straddles) < cut - lower))

    def average_whole(self) -> np.ndarray:
        """Each value's average rank among every row of its time."""
        return self.whole_below + (self.whole + 1) / 2

    def at_most_whole(self, ties: np.ndarray, layout: _Layout, k: np.ndarray) -> np.ndarray:
        """``at_most`` when every place is kept, so the counts are those of the table and ``k`` is one per time: each
        value is in or out as a whole, except the one that straddles ``k``, whose places count in table order."""
        if not self.count:
            return np.zeros(ties.shape, dtype=bool)
        cut = k[self.time]
        safe = np.maximum(ties, 0)
        inside = (self.whole_below + self.whole <= cut)[safe] & (ties >= 0)
        straddle = (self.whole_below < cut) & (self.whole_below + self.whole > cut)
        if not straddle.any():
            return inside
        straddles = straddle[safe] & (ties >= 0)
        return inside | (straddles & (layout.running(straddles) < (cut - self.whole_below)[safe]))


@dataclass(frozen=True)
class Prepared:
    """A form ready for one table: ``times`` (ns) and the statistic per time for a matrix of shuffles;
    ``with_scores`` gives the same form on the table with other scores, sharing everything that does not read them."""

    times: np.ndarray
    run: Callable[[np.ndarray], np.ndarray]
    rescored: Callable[[np.ndarray], "Prepared"]

    def __call__(self, index: np.ndarray) -> np.ndarray:
        return self.run(index)

    def with_scores(self, scores: np.ndarray) -> "Prepared":
        return self.rescored(scores)


@dataclass(frozen=True)
class RankCorrelation:
    """``rank_ic`` on many shuffles: the Spearman correlation of score and ``label`` across the rows at each time
    that ``keep`` keeps (a filter that reads no score, such as ``without_top``) and that hold both. When every row is
    kept, a shuffle only gathers each row's rank among its time's scores."""

    label: str = "y"
    keep: Keep | None = None

    def __call__(self, table: pl.DataFrame) -> Prepared:
        layout = _Layout(table)
        labels = _Ties(table[self.label], layout)
        label_of = labels.of[layout.order]
        own = label_of >= 0
        if self.keep is not None:
            own &= self.keep(table).fill_null(False).to_numpy()[layout.order]

        def scored(score: pl.Series) -> Prepared:
            scores = _Ties(score.fill_null(float("nan")), layout)
            run = _whole_ic(layout, scores, labels, label_of) if own.all() else _kept_ic(layout, scores, labels, own)
            return Prepared(layout.times, run, lambda s: scored(pl.Series(s, dtype=pl.Float64)))

        return scored(table["score"])


def _whole_ic(
    layout: _Layout, scores: _Ties, labels: _Ties, label_of: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    x_row = scores.average_whole()[scores.of]
    y = labels.average_whole()[label_of][None, :]
    n = layout.sizes()
    centre = n * ((n + 1) / 2) ** 2
    sxx = layout.sums(x_row[layout.order][None, :] ** 2)[0] - centre
    syy = layout.sums(y * y)[0] - centre

    def run(index: np.ndarray) -> np.ndarray:
        sxy = layout.sums(x_row[layout.donors(index)] * y) - centre
        with np.errstate(divide="ignore", invalid="ignore"):
            return sxy / np.sqrt(sxx * syy)

    return run


def _kept_ic(layout: _Layout, scores: _Ties, labels: _Ties, own: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    label_ties = np.broadcast_to(labels.of[layout.order], (1, layout.n))

    def run(index: np.ndarray) -> np.ndarray:
        held = scores.of[layout.donors(index)]
        kept = own & (held >= 0)
        x = np.where(kept, scores.average(held, kept), 0.0)
        y = np.where(kept, labels.average(np.broadcast_to(label_ties, held.shape), kept), 0.0)
        n = layout.sums(kept.astype(np.float64))
        centre = n * ((n + 1) / 2) ** 2
        sxy = layout.sums(x * y) - centre
        sxx = layout.sums(x * x) - centre
        syy = layout.sums(y * y) - centre
        with np.errstate(divide="ignore", invalid="ignore"):
            out = sxy / np.sqrt(sxx * syy)
        return np.where(n > 0, out, np.nan)

    return run


@dataclass(frozen=True)
class QuantileSpread:
    """``quantile_spread`` on many shuffles: the mean ``label`` of the top ``share`` of the rows by score less that of
    the bottom ``share``, at each time, over the rows that hold both."""

    share: float = 0.2
    label: str = "y"

    def __call__(self, table: pl.DataFrame) -> Prepared:
        layout = _Layout(table)
        y = table[self.label].to_numpy()[layout.order].astype(np.float64)
        own = ~table[self.label].is_null().to_numpy()[layout.order]
        y = np.where(own, y, 0.0)

        def scored(score: pl.Series) -> Prepared:
            scores = _Ties(score.fill_null(float("nan")), layout)
            whole = bool(own.all())
            n_whole = layout.sizes()
            k_whole = np.maximum(np.floor(n_whole * self.share + 1e-9), 1.0)

            def run(index: np.ndarray) -> np.ndarray:
                held = scores.of[layout.donors(index)]
                if whole:
                    bottom = scores.at_most_whole(held, layout, k_whole)
                    top = ~scores.at_most_whole(held, layout, n_whole - k_whole)
                else:
                    kept = own & (held >= 0)
                    n = layout.sums(kept.astype(np.float64))
                    k = np.maximum(np.floor(n * self.share + 1e-9), 1.0)
                    bottom = scores.at_most(held, kept, layout, k)
                    top = kept & ~scores.at_most(held, kept, layout, n - k)
                with np.errstate(divide="ignore", invalid="ignore"):
                    high = layout.sums(np.where(top, y, 0.0)) / layout.sums(top.astype(np.float64))
                    low = layout.sums(np.where(bottom, y, 0.0)) / layout.sums(bottom.astype(np.float64))
                return high - low

            return Prepared(layout.times, run, lambda s: scored(pl.Series(s, dtype=pl.Float64)))

        return scored(table["score"])


@dataclass(frozen=True)
class TopClusterMean:
    """The reference target ``ref.mean`` with ``top`` on many shuffles: at each time, the rows holding a score whose
    ordinal rank from the top (ties in table order) is within ``top`` of them, their ``value`` less the round trip of
    ``costs`` when ``net``, averaged per ``cluster``, then over clusters. ``costs`` prices each row on its own, as the
    cost model's ``round_trip`` does per candidate row."""

    top: float
    cluster: str
    net: bool
    costs: CostModel

    def __call__(self, table: pl.DataFrame) -> Prepared:
        layout = _Layout(table)
        value = pl.col("value") - pl.Series(self.costs.round_trip(table)) if self.net else pl.col("value")
        v = table.select(value.alias("v"))["v"]
        valid = ~v.is_null().to_numpy()[layout.order]
        v = np.where(valid, v.fill_null(0.0).to_numpy()[layout.order], 0.0)
        code = table.select(pl.col(self.cluster).rank("dense").over("t").fill_null(0).cast(pl.Int64))[self.cluster]
        group = _Groups(layout, code.to_numpy()[layout.order])
        n = layout.sizes()
        k = np.floor(n * self.top)
        k = k + ((k + 1) / n <= self.top)
        k = k - ((k > 0) & (k / n > self.top))

        def scored(score: pl.Series) -> Prepared:
            scores = _Ties(score.fill_null(float("nan")), layout, descending=True)

            def run(index: np.ndarray) -> np.ndarray:
                counted = scores.at_most_whole(scores.of[layout.donors(index)], layout, k) & valid
                means, has = group.means(np.where(counted, v, 0.0), counted)
                with np.errstate(divide="ignore", invalid="ignore"):
                    return group.per_time(np.where(has, means, 0.0)) / group.per_time(has.astype(np.float64))

            return Prepared(layout.times, run, lambda s: scored(pl.Series(s, dtype=pl.Float64)))

        return scored(table["score"])


class _Groups:
    """The clusters at each decision time, numbered across the table in time order."""

    def __init__(self, layout: _Layout, local: np.ndarray) -> None:
        distinct = np.zeros(layout.times.size, dtype=np.int64)
        np.maximum.at(distinct, layout.seg, local + 1)
        first = np.cumsum(distinct) - distinct
        self.of = first[layout.seg] + local
        self.count = int(distinct.sum())
        self.starts = first

    def means(self, x: np.ndarray, counted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each group's mean of ``x`` over its counted places per shuffle, and whether it has any."""
        shuffles = x.shape[0]
        flat = (np.arange(shuffles)[:, None] * self.count + self.of[None, :]).ravel()
        size = shuffles * self.count
        total = np.bincount(flat, weights=x.ravel(), minlength=size).reshape(shuffles, self.count)
        many = np.bincount(flat, weights=counted.ravel(), minlength=size).reshape(shuffles, self.count)
        has = many > 0
        return np.where(has, total / np.where(has, many, 1.0), 0.0), has

    def per_time(self, x: np.ndarray) -> np.ndarray:
        return np.add.reduceat(x, self.starts, axis=1) if self.count else np.zeros((x.shape[0], self.starts.size))


class _Free:
    """Trading that costs nothing, for the worked examples."""

    def round_trip(self, candidates: pl.DataFrame) -> pl.Series:
        return pl.Series("round_trip", [0.0] * candidates.height)

    def fees(self, trades: pl.DataFrame) -> float:
        return 0.0


_NONE = _Free()
FIVE = {"score": [5.0, 1.0, 4.0, 2.0, 3.0], "y": [0.5, 0.1, 0.4, 0.2, 0.3]}
REVERSED = [1, 0, 3, 2, 4]
CLUSTERED = {"score": [5.0, 4.0, 3.0, 2.0, 1.0], "value": [0.5, 0.3, 0.1, 0.2, 0.4], "c": ["a", "a", "b", "b", "c"]}


def _once(
    form: Callable[[pl.DataFrame], Prepared], columns: dict[str, list[Any]], index: list[int] | None = None
) -> float:
    table = hand(columns)
    shuffle = np.array([index if index is not None else list(range(table.height))])
    return float(form(table)(shuffle)[0, 0])


CASES = [
    Case("rank correlation with every score in place", lambda: _once(RankCorrelation(), FIVE), 1.0),
    Case("rank correlation with the scores swapped in reverse", lambda: _once(RankCorrelation(), FIVE, REVERSED), -1.0),
    Case(
        "tied scores take their average rank",
        lambda: _once(RankCorrelation(), {"score": [1.0, 1.0, 2.0, 2.0], "y": [0.1, 0.2, 0.3, 0.4]}),
        2 / math.sqrt(5),
    ),
    Case("the top fifth less the bottom fifth", lambda: _once(QuantileSpread(), FIVE), 0.4),
    Case("the same, the scores swapped in reverse", lambda: _once(QuantileSpread(), FIVE, REVERSED), -0.4),
    Case("the top two fifths, one cluster", lambda: _once(TopClusterMean(0.4, "c", False, _NONE), CLUSTERED), 0.4),
    Case(
        "the top two fifths after a swap, two clusters",
        lambda: _once(TopClusterMean(0.4, "c", False, _NONE), CLUSTERED, [0, 2, 1, 3, 4]),
        0.3,
    ),
    Case(
        "tied scores enter the top in table order",
        lambda: _once(
            TopClusterMean(0.5, "c", False, _NONE),
            {"score": [2.0, 2.0, 2.0, 1.0], "value": [0.1, 0.2, 0.9, 0.0], "c": ["a", "b", "c", "d"]},
        ),
        0.15,
    ),
]
