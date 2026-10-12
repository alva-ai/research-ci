"""Any eval on many shuffles at once, by stacking: one copy of the table per shuffle, each later in time, in one call.
It gives what the eval gives shuffle by shuffle, and the judge uses it only after checking that the eval reads each
copy on its own; an eval that reads across decision times or the calendar is run shuffle by shuffle."""

import numpy as np
import polars as pl
import pytest

from research_ci.api.evals import Market
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.synthetic import HOUR_NS, market
from research_ci.kernel.verdict.judge import Lane, Request, judge_all, per_t, stacked, with_scores
from research_ci.kernel.verdict.permute import Permutation
from research_ci.validation import calibration


def _spread(table: pl.DataFrame) -> pl.DataFrame:
    top = pl.col("y").filter(pl.col("score") >= pl.col("score").median()).mean()
    return table.group_by("t").agg(top.alias("value"))


def _inst_mean(table: pl.DataFrame) -> pl.DataFrame:
    """Each inst's mean score over every time: it reads across times."""
    m = table.group_by("inst").agg(pl.col("score").mean().alias("m"))
    return table.join(m, on="inst").group_by("t").agg(pl.corr("m", "y", method="spearman").alias("value"))


def _lagged(table: pl.DataFrame) -> pl.DataFrame:
    """The score of the time before: it reads across times."""
    s = table.sort("t").with_columns(p=pl.col("score").shift(1).over("inst"))
    return s.group_by("t").agg(pl.corr("p", "y").alias("value"))


def _monthly(table: pl.DataFrame) -> pl.DataFrame:
    """Weighted by the month: it reads the calendar."""
    return table.group_by("t").agg((pl.corr("score", "y") * pl.col("t").first().dt.month()).alias("value"))


def _by_shuffle(evaluate, table: pl.DataFrame, index: np.ndarray, times: np.ndarray) -> np.ndarray:
    out = np.full((index.shape[0], times.size), np.nan)
    score = table["score"].to_numpy()
    for i, rows in enumerate(index):
        v = per_t(evaluate, with_scores(table, score[rows]))
        out[i, np.searchsorted(times, v["t"].dt.epoch("ns").to_numpy())] = v["value"].to_numpy()
    return out


@pytest.mark.parametrize("evaluate", [rank_ic, _spread], ids=["rank ic", "spread"])
def test_stacked_shuffles_give_what_the_eval_gives_shuffle_by_shuffle(evaluate) -> None:
    table = market(Market(ic=0.05), 3)
    form = stacked(evaluate, table, per_t(evaluate, table))
    assert form is not None
    shuffle = Permutation(table, None, "cluster", HOUR_NS).index
    index = np.stack([shuffle(s) for s in range(12)])
    np.testing.assert_allclose(form(index), _by_shuffle(evaluate, table, index, form.times), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("evaluate", [_inst_mean, _lagged, _monthly], ids=["inst mean", "lagged", "monthly"])
def test_an_eval_that_reads_across_times_or_the_calendar_is_not_stacked(evaluate) -> None:
    table = market(Market(ic=0.05, times=24 * 120), 3)
    assert stacked(evaluate, table, per_t(evaluate, table)) is None
    lanes = {"1h": Lane(table, HOUR_NS, "lane", lambda: Permutation(table, None, "cluster", HOUR_NS).index)}
    request = Request(evaluate, lanes, 20, 5, "max")
    fast, slow = judge_all([request], 1)[0], judge_all([request], 1, batched=False)[0]
    assert isinstance(fast, dict) and isinstance(slow, dict) and fast["z"] == slow["z"]


def test_the_judge_and_the_calibration_stack_an_eval_without_a_batched_form() -> None:
    table = market(Market(ic=0.05), 5)
    lanes = {"1h": Lane(table, HOUR_NS, "lane", lambda: Permutation(table, None, "cluster", HOUR_NS).index)}
    request = Request(_spread, lanes, 40, 5, "max")
    fast, slow = judge_all([request], 1)[0], judge_all([request], 1, batched=False)[0]
    assert isinstance(fast, dict) and isinstance(slow, dict)
    assert fast["z"] == pytest.approx(slow["z"], rel=1e-9)
    assert calibration._batched(_spread, table) is not None
