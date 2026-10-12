# rci:example -- the engine refuses a lab eval from this file while this line is in it. Copy
# it to evals/<name>.py, delete this line in the copy, and replace every TODO with your own eval and its tests.
"""TODO: one sentence on what this idea needs tested that a generic eval cannot tell: the H0 the eval rejects."""

from datetime import UTC, datetime

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)


def ranks_label(table: pl.DataFrame) -> pl.DataFrame:
    """A feat eval. table: the evaluation table of one horizon (t, inst, score, y, the label's parts y_<part>, and
    the candidate columns). It returns one value per decision time (t, value); here the rank correlation of score and
    label. TODO: what yours measures, such as the score against one part of the label, y_<part>."""
    return table.group_by("t").agg(pl.corr("score", "y", method="spearman").alias("value")).sort("t")


def flags(fields: pl.DataFrame) -> pl.DataFrame:
    """A data eval. fields: the explore data (inst, field, event_time, available_at, value, vintage, source). It
    returns the rows it flags (inst, event_time, optionally field); none means it found nothing. TODO: what yours
    catches; here a value of FIELD at zero or below."""
    return fields.filter((pl.col("field") == FIELD) & (pl.col("value") <= 0)).select("inst", "field", "event_time")


FIELD = "TODO"
FOUR = pl.DataFrame({"t": [T] * 4, "inst": list("abcd"), "score": [1.0, 2.0, 3.0, 4.0], "y": [0.1, 0.2, 0.4, 0.3]})
ROWS = pl.DataFrame({"inst": ["a", "a"], "field": [FIELD] * 2, "event_time": [T, T], "value": [1.0, 0.0]})

# The tests the engine runs before the eval judges anything, worked by hand; a failed one refuses the layer.
CASES = [
    Case("TODO: one swap in four: 1 - 6·2/(4·15)", ranks_label, 0.8, given=FOUR),
    Case("TODO: the value at zero is flagged", flags, 1, given=ROWS),
]

# feat.py: from evals.<name> import ranks_label
#          EVALS = [Eval(ranks_label, h0="TODO: the H0 it rejects, such as: the scores do not rank the label")]
# data.py: from evals.<name> import flags
#          EVALS = [Eval(flags, h0="TODO: such as: every value of the field is above zero", on_fail="block")]
