"""The files ``rci init`` writes into a new idea: the contract skeleton, ``START.md`` for an agent that knows nothing
else, ``research/example.py`` and ``evals/example.py``, which the engine refuses to run until they are rewritten. They
shape only what is read, so the engine hash leaves this file out."""

from research_ci.runner.research import EXAMPLE_MARKER

CONTRACT_FILE = "contract.py"
START_FILE = "START.md"
EXAMPLE_FILE = "research/example.py"
EVALS_EXAMPLE_FILE = "evals/example.py"

SKELETON = """from research_ci.api import Alternative, Contract, Effect, Execution, Forward, Label, Universe, Usage, Use

CONTRACT = Contract(
    claim="TODO: state the claim in one sentence",
    # the target averages the label over the top share of the candidates by score at each decision time; keep that
    # to at least a few candidates (the ContractCard shows the median number of candidates)
    decisive=(Effect(on="return", target=Use("ref.mean", {"cluster": "entity", "top": 0.5})),),
    alternatives=(Alternative("TODO", "TODO: another reason the same result could appear"),),
    # if you looked at the data or at results for this idea before writing this (plugin data, notebooks, backtests),
    # declare it: used=(Usage("<start>", "<end>", "what was seen", ("claim",)),); () only if nothing was seen
    used=None,
    data="TODO-data-plugin",
    costs=Use("ref.flat_costs", {"commission_bps": 0.0, "slippage_bps": 0.0}),
    execution=Execution(every="24h"),
    # shape="cross" ranks the candidates at each time; "series" times one or a few assets (target ref.timing);
    # "events" asks what follows the rows the features score (target ref.mean without top)
    universe=Universe("TODO-universe-plugin", clusters="entity", shape="cross"),
    labels=(Label("return", Use("ref.return", {"price": "close"}), horizons=("24h",)),),
    forward=Forward("TODO-data-plugin"),
)
"""

START = """# Working on {name}

Research CI judges this idea with fixed code; you do the research with the `rci` command. Every command on the idea
takes `-i {idea}`. After every step, run `rci next -i {idea}` and do what it says: it names the command, the file to
write, and the helper commands.

1. Contract: fill in `contract.py`. `rci plugins` lists the installed data sources, universes, labels, targets, and
   cost models, with their options and fields. `Universe(..., shape=...)`: `"cross"` (default) ranks candidates,
   `ref.mean` with `top`; `"series"` times one or a few assets, `ref.timing`; `"events"` scores only events, `ref.mean`.
   If you looked at the data or at results for this idea (plugin data, notebooks, backtests), declare it in `used`;
   `used=()` only if nothing was seen. Run `rci contract derive`; a person approves the ContractCard first.
2. Data, your loop: list at least one lab eval in `EVALS` in `data.py` (Lab evals, below); run `rci data check`.
3. Feat, your loop: list at least one lab eval in `EVALS` in `feat.py`, and give each alternative of the contract a
   null world in `nulls/feat.py`: `NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>",
   block="horizon"), evals=("<eval>",))]`, or for a series or events `ShiftInTime(min_shift="72h")` (`from
   research_ci.api import Null, PermuteWithin, ShiftInTime`); a world judges the evals it names, or the decisive effect
   when it names none. Copy `research/example.py` to `research/<name>.py` and write your features there; try a file
   with `rci feat explore <file>` (free), then judge it with `rci feat submit <file>` (one trial of the budget).
4. A person reviews each result after your loop and approves it; then the next stage opens.

Lab evals: your tests of the claim; every layer with `EVALS` needs at least one, and each can veto. A lab eval is a
polars function in `evals/<name>.py` (copy `evals/example.py`) with its tests in the same module:
`CASES = [Case("<what it shows>", <eval>, <expected frame or number>, given=<input frame>)]`, required; `PROPERTIES`
and, for a data eval, `PITS`, optional. List it with the H0 it rejects, in one sentence: `from evals.<name> import
<eval>`, then `EVALS = [Eval(<eval>, h0="<the H0>")]`. In feat it, and each null world, must reach z ≥ 2 at every
horizon it covers, or the claim fails there; it never passes what the decisive test failed. In data, `on_fail` says
what is done with the rows it flags: `mask`, `drop_entities`, `ask`, or `block` (any row it flags, after the
treatments before it, blocks the data). The engine runs the tests before anything heavy and refuses a layer with a
lab eval that fails them; run `rci eval check evals/<name>.py` any time, for free. `EVALS` may also list the engine's
library (`rci methods`): shown, never vetoing, with no `h0`.

Features: `features(fields, candidates)` gets polars DataFrames and returns `t`, `entity`, `score` for every decision
time at once (a polars DataFrame or LazyFrame, or a pandas DataFrame). Research code is free: polars first, with
`fields.lazy() ... .collect()` for a longer pipeline; pandas is fine for dates; any library runs. Row-by-row code
(`iter_rows`, `.rows()`, `map_elements`, loops over data) runs too, with a warning on the card that names each line.
The card also shows the time and peak memory the features took.

Approvals: show the person the whole card, then relay their words with
`rci approve --stage <stage> --relayed "<their words>" --card <id>`, where the id is the one printed as `card <id>`
under the card they saw. Use `--auto` only when the person asked for fully automatic research.
"""

EXAMPLE = f'''# {EXAMPLE_MARKER} -- the engine refuses to run this file while this line is in it. Copy it to
# research/<name>.py, delete this line in the copy, and replace every TODO with your own features.
"""TODO: one sentence on what the score measures and why it should rank the candidates."""

import polars as pl

FIELD = "TODO"  # a field of the data; `rci plugins` lists the fields a data plugin declares


def features(fields: pl.DataFrame, candidates: pl.DataFrame) -> pl.DataFrame:
    """One score per candidate row (t, entity, score), for every decision time at once.

    fields: entity, field, event_time, available_at, value, vintage, source; all of the frozen data.
    candidates: t, entity, and the universe's columns (such as the legs long and short, the cluster and the groups).
    Both are polars DataFrames. A score at t may use only rows with available_at <= t; join_asof on available_at gives
    the latest of them. Return a polars DataFrame or LazyFrame, or a pandas DataFrame.
    """
    # .lazy() lets polars plan the whole pipeline and run it once at .collect(); eager code works the same way.
    known = (
        fields.lazy()
        .filter(pl.col("field") == FIELD)
        .select("entity", pl.col("available_at").alias("known_at"), "value")
        .sort("known_at")
    )
    # TODO: transform each entity's series with expressions, for example
    #   known.with_columns(pl.col("value").rolling_mean_by("known_at", window_size="24h").over("entity"))
    #   known.with_columns(pl.col("value").diff().over("entity").alias("change"))
    scored = (
        candidates.lazy()
        .sort("t")
        .join_asof(known, left_on="t", right_on="known_at", by="entity", check_sortedness=False)
    )
    # Candidates with two legs: look each leg up by its own column instead of entity, then take the difference:
    #   leg = known.rename({{"entity": "long", "known_at": "long_at", "value": "long_value"}})
    #   scored = candidates.lazy().sort("t").join_asof(leg, left_on="t", right_on="long_at", by="long",
    #                                                  check_sortedness=False)
    # Ranks within each decision time: pl.col("score").rank().over("t").
    # pandas is fine for dates (pd.Timedelta, offsets, calendars), and any library runs. Whole-column polars
    # expressions like these are usually much faster than going row by row, which runs too, with a warning on the card;
    # the card also shows the time and peak memory the features took.
    return scored.select("t", "entity", pl.col("value").alias("score")).collect()
'''

EVALS_EXAMPLE = f'''# {EXAMPLE_MARKER} -- the engine refuses a lab eval from this file while this line is in it. Copy
# it to evals/<name>.py, delete this line in the copy, and replace every TODO with your own eval and its tests.
"""TODO: one sentence on what this idea needs tested that a generic eval cannot tell: the H0 the eval rejects."""

from datetime import UTC, datetime

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)


def ranks_label(table: pl.DataFrame) -> pl.DataFrame:
    """A feat eval. table: the evaluation table of one horizon (t, entity, score, y, the label's parts y_<part>, and
    the candidate columns). It returns one value per decision time (t, value); here the rank correlation of score and
    label. TODO: what yours measures, such as the score against one part of the label, y_<part>."""
    return table.group_by("t").agg(pl.corr("score", "y", method="spearman").alias("value")).sort("t")


def flags(fields: pl.DataFrame) -> pl.DataFrame:
    """A data eval. fields: the explore data (entity, field, event_time, available_at, value, vintage, source). It
    returns the rows it flags (entity, event_time, optionally field); none means it found nothing. TODO: what yours
    catches; here a value of FIELD at zero or below."""
    return fields.filter((pl.col("field") == FIELD) & (pl.col("value") <= 0)).select("entity", "field", "event_time")


FIELD = "TODO"
FOUR = pl.DataFrame({{"t": [T] * 4, "entity": list("abcd"), "score": [1.0, 2.0, 3.0, 4.0], "y": [0.1, 0.2, 0.4, 0.3]}})
ROWS = pl.DataFrame({{"entity": ["a", "a"], "field": [FIELD] * 2, "event_time": [T, T], "value": [1.0, 0.0]}})

# The tests the engine runs before the eval judges anything, worked by hand; a failed one refuses the layer.
CASES = [
    Case("TODO: one swap in four: 1 - 6·2/(4·15)", ranks_label, 0.8, given=FOUR),
    Case("TODO: the value at zero is flagged", flags, 1, given=ROWS),
]

# feat.py: from evals.<name> import ranks_label
#          EVALS = [Eval(ranks_label, h0="TODO: the H0 it rejects, such as: the scores do not rank the label")]
# data.py: from evals.<name> import flags
#          EVALS = [Eval(flags, h0="TODO: such as: every value of the field is above zero", on_fail="block")]
'''
