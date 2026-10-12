# Working on spread

Research CI judges this idea with fixed code; you do the research with the `rci` command. Every command on the idea
takes `-i ideas/spread`. After every step, run `rci next -i ideas/spread` and do what it says: it names the command, the file to
write, and the helper commands.

1. Contract: fill in `contract.py`. `rci plugins` lists the installed data sources, universes, labels, targets, and
   cost models, with their options and fields. If you looked at the data or at results for this idea (plugin data,
   notebooks, backtests), declare it in `used`; `used=()` only if nothing was seen. Run `rci contract derive`; a
   person approves the ContractCard before any other work starts.
2. Data, your loop: list at least one lab eval in `EVALS` in `data.py` (Lab evals, below); run `rci data check`.
3. Feat, your loop: list at least one lab eval in `EVALS` in `feat.py`, and give each alternative of the contract a
   null world in `nulls/feat.py`: `NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>",
   block="horizon"), evals=("<eval>",))]` (`from research_ci.api import Null, PermuteWithin`); a world judges the evals
   it names, or the decisive effect when it names none. Copy `research/example.py` to `research/<name>.py` and write
   your features there; try a file with `rci feat explore <file>` (free), then judge it with `rci feat submit <file>`
   (one trial of the budget).
4. A person reviews each result after your loop and approves it; then the next stage opens.

Lab evals: your tests of the claim; every layer with `EVALS` needs at least one, and each can veto. A lab eval is a
polars function in `evals/<name>.py` (copy `evals/example.py`) with its tests in the same module:
`CASES = [Case("<what it shows>", <eval>, <expected frame or number>, given=<input frame>)]`, required; `PROPERTIES`
and, for a data eval, `PITS`, optional. List it with the H0 it rejects, in one sentence: `from evals.<name> import
<eval>`, then `EVALS = [Eval(<eval>, h0="<the H0>")]`. In feat it, and each null world, must reach z ≥ 2 at every
horizon it covers, or the claim fails there; it never passes what the decisive test failed. In data, `on_fail` says
what is done with the rows it flags: `mask`, `drop_insts`, `ask`, or `block` (any row it flags, after the
treatments before it, blocks the data). The
engine runs the tests before anything heavy and refuses a layer with a lab eval that fails them; run
`rci eval check evals/<name>.py` any time, for free. `EVALS` may also list the engine's library (`rci methods`):
shown, never vetoing, with no `h0`.

Features: `features(fields, candidates)` gets polars DataFrames and returns `t`, `inst`, `score` for every decision
time at once (a polars DataFrame or LazyFrame, or a pandas DataFrame). Research code is free: polars first, with
`fields.lazy() ... .collect()` for a longer pipeline; pandas is fine for dates; any library runs. Row-by-row code
(`iter_rows`, `.rows()`, `map_elements`, loops over data) runs too, with a warning on the card that names each line.
The card also shows the time and peak memory the features took.

Approvals: show the person the whole card, then relay their words with
`rci approve --stage <stage> --relayed "<their words>" --card <id>`, where the id is the one printed as `card <id>`
under the card they saw. Use `--auto` only when the person asked for fully automatic research.
