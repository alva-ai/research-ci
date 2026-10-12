"""The files ``rci init`` writes into a new idea: the contract skeleton, ``START.md`` for an agent that knows nothing
else, and ``research/example.py``, which the engine refuses to run until it is rewritten. They shape only what is
read, so the engine hash leaves this file out."""

from research_ci.runner.research import EXAMPLE_MARKER

CONTRACT_FILE = "contract.py"
START_FILE = "START.md"
EXAMPLE_FILE = "research/example.py"

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
    universe=Universe("TODO-universe-plugin", clusters="entity"),
    labels=(Label("return", Use("ref.return", {"price": "close"}), horizons=("24h",)),),
    forward=Forward("TODO-data-plugin"),
)
"""

START = """# Working on {name}

Research CI judges this idea with fixed code; you do the research with the `rci` command. Every command on the idea
takes `-i {idea}`. After every step, run `rci next -i {idea}` and do what it says: it names the command, the file to
write, and the helper commands.

1. Contract: fill in `contract.py`. `rci plugins` lists the installed data sources, universes, labels, targets, and
   cost models, with their options and fields. If you looked at the data or at results for this idea (plugin data,
   notebooks, backtests), declare it in `used`; `used=()` only if nothing was seen. Run `rci contract derive`; a
   person approves the ContractCard before any other work starts.
2. Data, your loop: run `rci data check`. For a risk the default checks miss, add an eval to `EVALS` in `data.py`
   (see Evals below).
3. Feat, your loop: copy `research/example.py` to `research/<name>.py` and write your features there. Try a file with
   `rci feat explore <file>` (free), then judge it with `rci feat submit <file>` (one trial of the budget). Give each
   alternative of the contract a world in `nulls/feat.py`, or the ResultCard marks it uncovered: import `Null` and
   `PermuteWithin` from `research_ci.api`, then
   `NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", block="horizon"))]`. Optional
   diagnostics are evals in `EVALS` in `feat.py`; `rci methods` lists every field of a null world.
4. A person reviews each result after your loop and approves it; then the next stage opens.

Evals: `EVALS = [...]` in `data.py` or `feat.py` lists general evals of the engine's library (`rci methods` lists
them; they are trusted) and case evals of your own. A case eval is a polars function in a module of the idea, such as
`evals/<name>.py`, imported into `EVALS` (`from evals.<name> import <eval>`), with its tests in the same module:
`CASES = [Case("<what it shows>", <eval>, <expected frame or number>, given=<input frame>)]`, required, and
`PROPERTIES`, optional. The engine runs those tests before it runs the eval: one without tests is refused, and one
that fails a test does not run and is shown with the test, what it expected, and what it got. Run
`rci eval check evals/<name>.py` any time; it is free. Wrap an eval in `Eval(fn, ...)` when its layer needs more:
`on_fail` and `covers` in `data.py`, `label`, `horizons`, and `combine` in `feat.py`.

Features: `features(fields, candidates)` gets polars DataFrames and returns `t`, `entity`, `score` for every decision
time at once (a polars DataFrame or LazyFrame, or a pandas DataFrame). Research code is free: polars first, with
`fields.lazy() ... .collect()` for a longer pipeline; pandas is fine for dates; any library runs. Row-by-row code
(`iter_rows`, `.rows()`, `map_elements`, `.apply()`, `iterrows`, loops over data) runs too, with a warning on the card
that names each line; polars expressions (`over`, `rolling_*_by`, `join_asof`) are usually much faster. The card also
shows the time and peak memory the features took.

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
