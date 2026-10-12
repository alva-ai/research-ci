# Research CI

A backtest framework for research: an alpha claim counts only after it passes CI, the same way code does.

> **Status:** early. The time kernel, the holdout, ledger, and workflow layer, the contract stage (plugin protocols, protocol tests, power, the ContractCard, approval provenance), the data layer (fidelity: source admission by replay and reconciliation; fitness: a mandatory default list of checks, and the idea's own evals; treatments weighed by their influence; the calibration of the decisive test; the DataCard), and the feat layer (the decisive effect judged against random scores, a lookahead check on cut data, evals judged against the null worlds of the contract's alternatives, the ResultCard), one format for the idea's evals (general evals of the engine's library, and case evals of the idea's own, which run only after their own tests pass, `rci eval check`), one lifecycle for every stage, and what lets an agent with no context follow it (`START.md`, `rci plugins`, `rci methods`, research code checked for row loops before it runs, errors and cards that say why something failed and how to fix it) are implemented. The signal and algo layers come next. See the [roadmap](docs/design.md#roadmap).

| Area | Works today | Planned |
| --- | --- | --- |
| Time | FieldTable, `asof`, decision grids; features, universes, and labels recomputed on data cut at a few decision times must not change | Decision times from events and latencies, walk-forward helpers |
| Holdout | Explore, seal, and forward tiers declared in the contract; every stage reads only the explore tier; labels reach evals only through the evaluation table | The seal and forward checkpoints, sealed data kept out of the agent's reach, labels through `fit` on training folds, vintage-aware labels |
| Contract | Every field validated; `derive` loads plugins, snapshots the data, runs protocol tests, checks that the decisive target reads the scores, computes power, and prints the ContractCard | Oracle bounds for maker strategies |
| Data | Fidelity, once per source: admission (replay as of past times, an independent source on engine-picked rows, canaries, field grades). Fitness, for every idea: a mandatory default list (freshness, level shifts, level sanity, identity breaks, coverage, the hedge guard, influence concentration) and the idea's own evals, a case eval only once its tests pass and the pits it claims to cover are caught; treatments applied by default and weighed by their influence in MDE units; the DataCard | Seal-tier checks, listings from the source to catch hindsight universes |
| Workflow | One pipeline for every idea and one lifecycle for every stage (OPEN, REVIEW, PASSED, PAUSED, STALE), budgets, lineage, `next_actions` that name the file to write and the helper commands, approval provenance; `rci init` writes `START.md` and an example features file | Caps by approval provenance, hooks after verdicts |
| Trust | Hash-chained ledger, head anchors, freeze records, data vault; plugin code and the data snapshot checked again at approval; the data freeze pins its check and the treated data; each feat submission records its features file, its evals and the modules of its case evals, null worlds, plugin code, and calibration, and approval checks them again | DEV marking for an editable engine |
| Feat | The minimal bar: the contract's decisive effect on a label must beat the engine's floor world (random scores) with a z that rises with the trial count; its calibration by stand-ins and planted signals at the data check, before any features exist; features on polars DataFrames, with row-by-row code shown as a warning and their time and peak memory shown, never judged; a lookahead check on cut data; only the candidates the features scored are judged, in every world alike; the engine's default evals and the idea's own, judged against the floor and the null worlds of the contract's alternatives and shown, never deciding; the ResultCard | Learning from labels with walk-forward, dedup by correlation |
| Judge | Protocol tests, power, minimum detectable effects; `PermuteWithin` shuffles, each drawn once per table and shared by every eval and world, with batched forms of the engine's statistics and, for any other eval, copies of the table stacked in one call once a check shows the eval reads each on its own; a decisive test that draws shuffles until its decision is settled; GridMax over horizons, z thresholds that rise with the trial count; the data checks, statistics, structure, and calibration in `eval` carry worked examples and properties on synthetic markets and sources; an idea's case evals are held to their own worked examples and properties before they run, the result kept by the hash of their files; the protocol tests are held to broken toy plugins; real incidents are kept as regression tests | Evidence levels and caps, stress worlds and limit criteria |
| Runner | Plugin loading by entry point, plugin code hashes, errors that name close plugins and a plugin's signature; the catalogs behind `rci plugins` and `rci methods`; loading an idea's files with its own modules importable, and validating its case evals; the static check that finds row-by-row research code for the cards' warning | Research code in a child process with Arrow exchange |
| Algo | Consumer protocol tests (point-in-time, causality, determinism, fees, labels, score pass-through) and the power of a backtest comparison, in `eval` and `kernel`; a consumer named in a contract declares the decision times | The algo layer: backtests through the consumer, from the baseline's own states or over the full path |
| Plugins | Protocols for data, reference sources, universes, labels, targets, cost models, consumers; reference labels, target, and cost model; real plugins live in a lab | Instruments, fill models, an LLM challenger |

## Documentation

| Read | For |
| --- | --- |
| This README | What Research CI is, the loops at a glance, a quickstart, the CLI, a glossary |
| [docs/design.md](docs/design.md) | The full design: principles, architecture, lifecycle, the contract, the data and feat layers, time and leaks, holdout and labels, judging, the challenger, trust, layers, plugins, key decisions and why, prior art, roadmap |
| [docs/walkthroughs.md](docs/walkthroughs.md) | One idea from the contract to a passed feat layer, with what every command printed |
| [docs/catalog.md](docs/catalog.md) | At a glance, generated from the code: every step of each stage's floor and default lists with its role, how an idea's case evals plug in, and every eval function with where it is used and what backs it |

## Why

Agents made research fast, but judgment did not keep up. The usual loop is: an agent prepares data, writes a strategy, sweeps parameters, shows the best PnL curve, and explains why it works. Every decision in that loop depends on context and gut feeling, and none of it is reproducible.

What a person needs instead is a green light they can trust without reading the research, the way a passing CI run lets you merge without rereading every line. Many agent sessions can then work on ideas in parallel, and the person only handles the few decisions that are theirs to make. A written summary is never the verdict: numbers are judged by code.

Research CI moves the judgment **before** the results and makes it executable:

- **Falsifiable**: before an idea starts, write down what result would prove it wrong.
- **Executable**: that standard is code. The engine reruns the submission and decides; it never trusts numbers an agent reports.
- **Reproducible**: the standard is frozen in the contract before any result is seen. Every shuffle and random draw uses a fixed seed, so the same inputs give the same verdict.

Research code is free: a features file is one function of the frozen data and the candidates, handed over as polars DataFrames, for every decision time at once, and the engine computes it again on the data cut at a few decision times to make sure nothing was read ahead. polars is the main path, with `.lazy()` … `.collect()` for a pipeline polars should plan as a whole, pandas is fine for dates, and loops and libraries such as sklearn run too; code that goes row by row gets a warning on the card that names each line, since polars expressions are usually much faster, and is never refused, and the card shows the time and peak memory the features took. Everything specific to a case is a plugin: data sources, candidates, labels, targets, cost models, backtesters. The engine is thin: it runs on polars, holds plugins and agent-written code to executable checks, keeps the methods it executes in `eval` (each with its own worked examples and properties), and owns exactly three things:

| Kernel duty | What it guarantees |
| --- | --- |
| **Time** | A score at a decision time may depend only on data known by then: the engine computes the features again on the data cut at a few decision times, and any change TAINTS the submission. Universes and labels pass the same kind of check. |
| **Holdout and labels** | Label definitions are plugins named in the contract; the engine decides which data a label sees, when it runs, and who gets its values. Every stage reads only the explore tier; keeping sealed data out of the agent's reach is planned. |
| **Verdict** | Scores are compared against null worlds, thresholds rise with the number of trials, and every verdict is logged in a hash-chained ledger. |

Engine code is held to whole columns: `eval` and `stages/*/defaults.py` run on every idea's data, so a loop over rows, entities, clusters, or decision times has no place there, and `tests/eval/test_columnar.py` fails on any `for` or `while` loop and any `map_groups`, `map_elements`, `apply`, `iter_rows`, or `iterrows` call that is not a reviewed exception. An exception is a structural loop, listed with its file, its function, and one line on why, such as one plugin fetch per cut point, one fold of a walk-forward at a time, PBO partitions taken in chunks, or the iterations of a fit.

## How it works

### Outer loop: one idea, end to end

Every idea goes through the same pipeline: the contract, the four layers below, and the checkpoints seal, forward, capital, and monitor. No stage is optional. The contract sets the floor; the loops of the layers set how far the idea goes. Each research layer is its own inner loop, and a layer can only use outputs that passed the layer before it. Seal, forward, and capital are one-shot checkpoints: their plan is registered in advance, they run once, and they send the idea back to an earlier layer if they fail.

![The outer loop: one idea from the contract through the four layers to the checkpoints](docs/assets/outer-loop.png)

| Layer | Question it answers | Output shape | Needs a backtest |
| --- | --- | --- | --- |
| `contract` | What is claimed, what result counts as wrong, and can the data decide it? | Contract, ContractCard | No; a consumer it names only declares the decision times |
| `data` | Can these fields support the contract (point-in-time, vintages, breaks)? | Treated FieldTable, DataCard | No |
| `feat` | Does the feature beat random, significantly? | Scores | No |
| `signal` | Does the combined signal predict? Raw signals and model outputs both live here. | Scores, Events | No |
| `algo` | Does acting on the signal make money after costs and constraints? | Positions, Quotes, Orders | Yes |

Labels are not a layer and neither is the backtest: labels are plugins held in the engine's custody, and the backtest is a consumer plugin, the evaluator of the `algo` layer. Nothing before `algo` runs it, so the contract, data, and feat are pure data analysis: a contract's decisive effect is on a label, with a target that reads the scores, and an effect on the consumer is refused.

### Inner loop: one lifecycle for every stage

Only the contract needs a person before work starts: it states the question and the one bar, the decisive test. Every layer after it is a loop the agent runs on its own; a person looks at the result after the loop, approves it, and the next stage starts. Every stage, of every kind, is in one of five states:

```text
         the agent: derive, check, or submit (repeatable)
OPEN  -------------------------------------------------->  REVIEW  -- a person: approve -->  PASSED
  ^      blocked, or a FAILED or TAINTED submission:          |
  |      stays OPEN                                           |
  +-----------------------------------------------------------+
         a newer result that is not ready takes REVIEW back to OPEN

PAUSED  -- a person: approve --extend N -->  OPEN     the last trial of a layer did not pass
PASSED  -- the agent: derive, check, or submit        reopens the stage as a new version; started stages after it go STALE
STALE   -- the agent: derive, check, or submit        runs the stage again as a new version, once the stages before it pass
```

- `rci contract derive` puts a DECIDABLE contract up for review; a BLOCKED one stays OPEN with the reason. A person approves the ContractCard, which freezes the claim, the decisive effect, and the budgets.
- `rci data check` asks two questions. Fidelity: is the data what the venue reported? The source's admission answers it once per source, and the first approval of a DataCard that uses the source admits it for every idea. Fitness: can faithful data be used for this research? A mandatory default list answers it for every idea, with the idea's own evals added: rows the venue really reported can still be a ticker reused for another asset, a price stuck or at zero, or an entity too short for the horizons. Every finding becomes a treatment weighed by its influence on the decisive number, never a silent change. Last, it calibrates the decisive test on the treated data, before any features exist: stand-ins must die and a planted signal must live in the floor world. A blocked check, a decisive test that cannot be calibrated included, stays OPEN; any other puts the DataCard up for review.
- `rci feat submit <file>` judges one features file, one trial. The data-analysis layers decide on one question: does the output beat random, significantly? The decisive test: the contract's decisive effect with the submitted scores must beat the floor world (the scores shuffled across clusters at each decision time) with `z ≥ 2·√(1+ln n)` on the n-th submission, a test the data check already calibrated. Nothing else decides, so potential alpha is not killed early; costs, absolute profit, and robustness are for later layers. The other evals, the engine's defaults and any the idea lists in its optional `feat.py`, are shown on every result and never decide; a case eval of the idea runs only when its own tests pass, and one that fails them is shown with the test it failed and no numbers. A PASSED submission goes up for review; FAILED and TAINTED ones leave the layer OPEN. A person can accept only a latest submission that PASSED: verdicts come from code.
- `rci feat explore` runs the checks a submission must pass without judging it. It never changes the state and is never counted as a trial.
- Each layer from feat on gets a submission budget from the contract, shared across its versions, so a new version does not reset it. The data layer has none: nothing in it is chosen by its result.
- A feat submission is refused, and costs no trial, when the file is still the example `rci init` wrote, it or the research code it imports does not parse, a layer file does not load or lacks its list, a case eval it lists has no tests, the features fail or return other columns, it repeats earlier scores, a null world cannot run, or a calibration it had to run again failed; each refusal says how to fix it. Row-by-row code is not refused: the card shows it as a warning. It is TAINTED when its scores read data from after their decision time, and that costs a trial. Every submission records what it used, and approving it is refused when any of it changed since.
- Checkpoints run once. A failed checkpoint sends the idea back to a layer.
- There is no separate state file. Every transition is a ledger record, the state is replayed from the ledger, and every command refuses to run on a ledger whose hash chain is broken.

### Evals: general and case

Everything the engine runs on a stage's result is an eval, in one format. The stages run their floor list and their default list; an idea adds its own in `EVALS = [...]` in `data.py` and `feat.py`: general evals of the engine's library (`research_ci.eval`, which `rci methods` lists), proven by the engine's CI and trusted, and case evals, polars functions of the idea's own with their tests in the same module. In every data and feat loop:

1. The agent researches, writes a case eval in the idea, such as `evals/<name>.py`, and its tests next to it: `CASES`, worked examples, required, and `PROPERTIES`, its behaviour on synthetic markets, optional.
2. The engine validates each case eval an idea uses by running only its tests, never the idea's other code and never pytest, and keeps the result in `.rci/evals/` by the hash of the module, the idea's files it imports, and the engine. A data eval that claims to catch a kind of problem (`covers`) must also flag the pits of that kind the engine plants. `rci eval check <file>` runs the tests any time, for free.
3. The engine runs the floor list, the default list, the library evals the idea selected, and the case evals that passed. A case eval without tests is refused; one that fails a test does not run and counts as no evidence: the DataCard is blocked with it, and the ResultCard shows it with no numbers.
4. A person reviews the card, where every eval that ran says whose it is (floor, default, library, or case), how it stands, and its result; every failure says why.

```python
# ideas/carry/evals/spreads.py
from datetime import datetime

import polars as pl

from research_ci.api import Case
from research_ci.eval.statistics.spread import quantile_spread

SIX = pl.DataFrame(
    {"t": [datetime(2026, 1, 1)] * 6, "score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "y": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}
)


def thirds(table: pl.DataFrame) -> pl.DataFrame:
    """The mean label of the top third by score less that of the bottom third, per decision time."""
    return quantile_spread(table, share=1 / 3)


CASES = [Case("the top two of six less the bottom two", thirds, 0.4, given=SIX)]
```

```python
# ideas/carry/feat.py
from functools import partial

from research_ci.api import Eval
from research_ci.eval.statistics.spread import quantile_spread
from evals.spreads import thirds

EVALS = [
    Eval(partial(quantile_spread, share=0.5), name="top minus bottom half"),  # library: trusted
    Eval(thirds, name="top minus bottom third", horizons=("24h",)),  # case: runs once its CASES pass
]
```

A `Case` takes frames in and gives frames or numbers out: `run(given)` must give `expect`, a frame matched on its own columns in any row order, or a number. `Eval(fn, ...)` adds what a layer reads: `on_fail` and `covers` in `data.py`, `label`, `horizons`, and `combine` in `feat.py`; a bare function needs nothing more. [docs/catalog.md](docs/catalog.md#case-evals) has every field.

### Who does what

```text
research agent           human                     scheduler
derive, check,           approve the contract,     forward, monitor
explore, submit          review results, extend
      |                     |                           |
      +---------------------+---------------------------+
                            |
                         rci CLI
                            |
                        workflow  ---------- spawns ------------>  challenger (planned)
                (lifecycle, next_actions)                          (separate LLM process,
                            |           <---- null worlds, tests    reads files only)
                            |
                 stages + eval + kernel
                (deterministic, no LLM)
```

The library is passive. It knows the state, decides which transitions are legal, and computes. Every driver reads the same `next_actions()` list: `rci next` prints it for one idea, and `rci inbox` collects everything that waits for a person across all ideas: results up for review and paused layers. `rci approve` asks a person at an interactive terminal by default. An agent can relay a person's approval with `--relayed "<the person's words>" --card <id>`, quoting the id printed under the card the person saw, or approve itself in fully automatic research with `--auto`; the ledger records which. A data card that asks a person to decide cannot be approved with `--auto`, and extending a budget always needs a terminal.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Install the engine into the project that will hold your ideas, pinned to a commit:

```sh
uv add "research-ci @ git+ssh://git@github.com/alva-ai/research-ci@<sha>"
```

The plugins a contract names must be installed in the same environment, registered under the entry-point group `research_ci.plugins`. Below, `ref.*` plugins ship with the engine and the `toy.*` plugins of the engine's tests stand in for a lab's; the output is what the commands printed ([walkthroughs.md](docs/walkthroughs.md) has the whole run).

Start an idea. `rci init` writes `START.md`, the page an agent with no other context reads first, a contract skeleton, and `research/example.py`; `rci next` says what to do after every step:

```text
$ rci init ideas/carry
created ideas/carry; read ideas/carry/START.md first: it says how the work goes
then fill in ideas/carry/contract.py and run `rci contract derive -i ideas/carry`
features start from ideas/carry/research/example.py; `rci next -i ideas/carry` says what to do after every step

$ rci next -i ideas/carry
agent      rci contract derive                  fill in contract.py (START.md says how; `rci plugins` lists the plugins, their options and fields), then derive the contract
```

`rci plugins` lists what a contract can name, read from the installed code (here the toy plugins, abridged):

```text
$ rci plugins
data: Contract(data=Use(name, {options}))
  toy.data(hours: int = 600, seed: int = 0)
    Replays as of any time, and answers reconciliation from the venue, here the same generator run apart; the venue keeps no history of funding intervals.
    fields: close, funding_rate, funding_interval_hours
    levels: close · factor 5 · window 24 · mask 24h
    reference source: answers the independent check at admission
...
```

Write the contract, and let a person approve it:

```python
# ideas/carry/contract.py
from research_ci.api import Alternative, Contract, Effect, Execution, Label, Universe, Use

CONTRACT = Contract(
    claim="Cross-venue funding spreads persist long enough to pay the round trip",
    decisive=(Effect(on="carry", target=Use("ref.mean", {"cluster": "symbol", "top": 0.34})),),
    alternatives=(Alternative("tail", "a few symbols with extreme funding carry the whole result"),),
    used=(),
    data="toy.data",
    costs=Use("ref.flat_costs", {"commission_bps": 1, "slippage_bps": 1, "legs": 2}),
    execution=Execution(every="1h", note="decide on the hour; fill both legs at the close known then"),
    universe=Universe("toy.pairs", clusters="symbol", groups=("venue_pair",)),
    labels=(Label("carry", Use("ref.carry", {"flow": "funding_rate"}), horizons=("8h", "24h")),),
    explore_end="2026-01-15T00:00:00Z",
    seal_end="2026-01-20T00:00:00Z",
    budget=3,
)
```

`rci contract derive` loads the plugins, stores the data in the vault, runs protocol tests on the universe, the label, and the target, computes power, and prints the ContractCard (the toy data ends in January 2026, so the card warns that it is old):

```text
$ rci contract derive -i ideas/carry
ContractCard · carry · v1
claim      Cross-venue funding spreads persist long enough to pay the round trip
decisive   carry: beats random significantly · target ref.mean(cluster=symbol, top=0.34)
decisions  every 1h from execution · 335 decision times
           decide on the hour; fill both legs at the close known then
data       toy.data afa06051cf36 · 3,780 rows · 9 entities · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 258.8 days old
universe   toy.pairs · median 9 candidates in 3 symbol clusters per decision · membership pit
costs      ref.flat_costs(commission_bps=1, legs=2, slippage_bps=1) · median round trip 8.0 bp
labels     carry: ref.carry · H 8h 24h
checks     [!]  data is current: latest data is 258.8 days old
           ok   universe point-in-time: 3 cut points give the same candidates
           ok   universe causal: values after 2026-01-08T04:00Z leave the candidates at it unchanged
           ok   label carry: worked examples: 2 ok
           ok   label carry: examples bite: shift, in-window changes and missing fields move it; later data does not
           ok   label carry: no data after t + h: 2 cut points, horizon 8h
           ok   label carry: no data after t + h: 2 cut points, horizon 24h
           ok   target reads the score: ref.mean moves with the scores
power      carry: the smallest gain over random the data can detect (MDE)
           H        decisions  blocks  MDE bp/entry  if rolled
           8h             321    41.9           5.9  ≈ 64.8%/yr
           24h            305    14.0          19.5  ≈ 71.1%/yr
alts       tail  (engine floor: chance, timing)
used       nothing declared
history    history can confirm
budget     3
verdict    DECIDABLE
           carry: noise measured at 8h, 24h; feat passes it when it beats random
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
           [!] carry: symbol Z carries 55% at 24h of the noise the MDE rests on; rci data check looks at whether its data is fit for research
----------------------------------------------------------------------------------------------------
card 3405b738
up for review; a person approves it with `rci approve --stage contract`

$ rci inbox ideas
ideas/carry              rci approve --stage contract         review the ContractCard and approve it; `rci contract show` prints it with the card id that a relayed approval quotes

$ rci approve --stage contract -i ideas/carry          # a person, at a terminal
approve contract v1 of carry
...                                                    (the card, as derive printed it)
card 3405b738
  frozen dependencies:
    contract 365c17d571b4
Approve contract v1? [y/N]: y
contract v1 PASSED: freeze 21eace54a830 · approved by a person at a terminal
to keep it in git: git add ideas/carry/.rci ideas/carry/contract.py && git commit -m 'rci: approve contract of carry'

$ rci status -i ideas/carry
carry
  contract  PASSED  v1           approved by a person at a terminal
  data      OPEN    v1
  feat      OPEN    v1   0/3
  signal    OPEN    v1   0/3
  algo      OPEN    v1   0/3
  seal      OPEN    v1
  forward   OPEN    v1
  capital   OPEN    v1
  monitor   OPEN    v1

$ rci next -i ideas/carry --json
[
  {
    "who": "agent",
    "stage": "data",
    "kind": "check",
    "command": "rci data check",
    "reason": "check the data against the contract; for a risk the default checks miss, list an eval in EVALS in data.py: one of the engine's library, or a case eval of your own in evals/<name>.py with its CASES, which `rci eval check <file>` runs for free (`rci methods` lists both)"
  }
]
```

Approval is refused if `contract.py`, a plugin's code, or the data snapshot changed since the derive. A contract that derive found BLOCKED (a protocol test failed, such as a decisive target that gives the same number whatever the scores, or the noise of a decisive effect cannot be measured) never goes up for review: it stays OPEN with the reason. From here the agent runs the loops on its own. `rci data check` runs the source's admission checks if this plugin, with these options and this code, was never admitted, checks the data against the contract with the default list in `stages/data/defaults.py` (a level that comes back at another level after a long gap is split into a new entity, prices at zero, stuck, or away for a moment are masked, entities shorter than the longest horizon are dropped, and the share of the noise each cluster carries is shown, with a warning above half) and the idea's optional evals in `data.py`, calibrates the decisive test, and prints the DataCard; `rci approve --stage data` freezes the treated data that feat reads and, the first time, admits the source. The agent then writes features files in `research/`, starting from `research/example.py`: `features(fields, candidates)` gets polars DataFrames and returns `t`, `entity`, `score` (a polars DataFrame or LazyFrame, or a pandas DataFrame), code that goes row by row runs with a warning on the card, and the card shows the time and peak memory the features took. `rci feat explore <file>` tries one for free; each `rci feat submit <file>` judges one and prints a ResultCard, the evals shown next to the decisive effect coming from the engine's default list and the optional `EVALS` in `feat.py` (see [Evals](#evals-general-and-case)), judged against the null worlds in `nulls/feat.py`, one per alternative of the contract; `rci approve --stage feat` accepts the latest PASSED one, whose card id `rci feat submit` and `rci feat show` print. When feat passes, `rci next` says the signal layer is not built yet and the idea is done for now. Engine state lives in `ideas/<idea>/.rci/` (commit it with the idea) and in `$RCI_HOME` (default `~/.rci`), which keeps each ledger's latest head outside the idea, the shared ledger of source admissions, and the data vault.

## CLI

| Command | Called by | Precondition | Available |
| --- | --- | --- | --- |
| `rci init <idea>` | agent | none; writes `START.md`, the contract skeleton, and `research/example.py` | yes |
| `rci plugins [--json]` | agent | none; every installed plugin with its kind, options, and declarations | yes |
| `rci methods [--json \| --markdown]` | agent | none; the general evals of the engine's library an idea's `EVALS` can list (feat evals, data evals) and the pits, with the worked examples that back each; how a case eval and its tests are written; the fields of a null world; `--markdown` prints [docs/catalog.md](docs/catalog.md) | yes |
| `rci eval check <file>` | agent, unlimited | none; runs the tests of every eval in a module of case evals and says how each stands, with the test that failed, what it expected and what it got; free, records nothing | yes |
| `rci contract derive` | agent, repeatable | none; prints the ContractCard; DECIDABLE goes up for review, BLOCKED stays OPEN | yes |
| `rci data check [--keep "<treatment>"]` | agent, repeatable | contract passed; runs the source's admission checks first if needed, calibrates the decisive test; prints the DataCard, with a warning for row-by-row code in `data.py` or `nulls/data.py`; `--keep` leaves a treatment a person rejected unapplied | yes |
| `rci feat submit <file>` | agent, 1 trial each | data passed, budget left, the file is not the example, scores not repeated, calibration holds; prints the ResultCard, with a warning for row-by-row code, and its card id when it PASSED | yes |
| `rci feat explore <file>` | agent, unlimited | data passed; runs only the checks a submission must pass | yes |
| `rci approve --stage <stage>` | **human**, interactive | the stage is up for review and nothing its result used changed since; for a layer with submissions, the latest PASSED | yes |
| `rci approve --stage <stage> --relayed "<words>" --card <id>` | agent, relaying a person's approval after showing the person the whole card | as above; the card has not changed since | yes |
| `rci approve --stage <stage> --auto` | agent, fully automatic research | as above; not for a DataCard that asks a person to decide | yes |
| `rci approve --stage <layer> --extend N` | **human**, interactive only | layer PAUSED | yes |
| `rci next`, `rci status`, `rci inbox`, `rci verify` | anyone | none | yes |
| `rci contract show`, `rci data show`, `rci feat show` | anyone | a derive, a check, or a submission exists; prints the card and the id of the one up for review | yes |
| `rci <layer> submit <file>`, `rci <layer> explore <file>` for signal and algo | agent | upstream passed | later |
| `rci <checkpoint> register` | agent | the layers passed | later |
| `rci seal run`, `rci capital run` | human | plan approved | later |
| `rci forward --all`, `rci monitor --all` | scheduler | plan approved | later |

`<layer>` is signal or algo, and `<checkpoint>` one of seal, forward, capital, and monitor. `status`, `next`, `inbox`, `plugins`, and `methods` take `--json` for agents. Commands on one idea take `-i <idea>` and default to the current directory; `init` takes the new idea's directory and `inbox` a root directory.

## Glossary

| Term | Meaning |
| --- | --- |
| Contract | The claim and its one bar, the decisive test, written in `contract.py` before any search: the decisive effect, the alternative explanations, the plugins, decision times, tier cutoffs, budget. It sets the floor, and it is the only thing a person approves before work starts. |
| Stage | The contract, a layer (`data`, `feat`, `signal`, `algo`), or a checkpoint (`seal`, `forward`, `capital`, `monitor`, whose plan is approved in advance and which runs once). Every idea goes through all of them, in order. |
| OPEN, REVIEW, PASSED | The agent's turn; a person's turn, with a result ready; approved and frozen, so the next stage may start. |
| PAUSED, STALE | A layer used its budget without a pass, and a person extends it or drops the idea; a stage before it was reopened, and the agent runs it again as a new version. |
| Decisive effect | The number the claim is about: a label's outcome turned into one value per decision time by a target that reads the scores. |
| Floor world | The scores shuffled across clusters at each decision time. At feat the decisive effect must beat it with `z ≥ z_need`, which rises with each submission: the minimal bar. |
| Submission | One judged features file: one trial, with the verdict PASSED, FAILED, or TAINTED. Verdicts come from code; a person can accept only a PASSED one. |
| Calibration | The test of the tests: at the data check, before any features exist, stand-ins must die and a planted signal must live in the floor world; in data, an eval must flag the pits it claims to cover. |
| Research code | Features files, `feat.py`, `nulls/feat.py`, `data.py`, `nulls/data.py`, case evals, and the modules of the idea and of installed labs they import. It is free (polars first, pandas for dates, any library); row-by-row code runs, with a warning on the card. |
| `START.md` | The page `rci init` writes for an agent with no other context: the loops, `rci next` after every step, where plugins and methods are listed, the rule for features, how approvals are relayed. |
| Eval | One check or statistic the engine runs, in one format everywhere: a step of a stage's floor or default list, a general eval of the engine's library (`research_ci.eval`, proven in the engine's CI, trusted), or a case eval of the idea's own. An idea lists its own in `EVALS` in `data.py` and `feat.py`, wrapped in `Eval(fn, ...)` where its layer needs more. Every card line says whose eval it is: floor, default, library, or case. |
| Case eval | A polars function of the idea's own, in a module such as `evals/<name>.py`, with its tests next to it: `CASES` (worked examples, required) and `PROPERTIES` (behaviour on synthetic markets, optional). The engine runs only those tests before it runs the eval, and keeps the result by the hash of its files: valid, it runs; invalid, it does not run, counts as no evidence, and the card says which test failed, what it expected and got, and how to fix it; without tests, it is refused. |
| Diagnostic | An eval shown on every feat result against the floor and the idea's null worlds, never deciding. |
| Source admission | Once per data source: replay as of past times and an independent source must agree with what the plugin gives. The first approval of a DataCard that uses the source admits it. |
| Treatment | A change the data check makes, such as a mask or a split, weighed by its influence on the decisive number in MDE units, the smallest gain over random the data can detect. |
| Card | What a person reads: the ContractCard, DataCard, and ResultCard (and the ExploreCard), rendered by code; every card a person may approve is printed with `card <id>`. |
| Freeze, ledger | The hashes of everything an approved result used; the hash-chained record of every state change, from which the state is replayed. |

## Architecture

```text
+------------------------------------------------------------------+
| cli        the rci command                                       |
+--------------------------------+---------------------------------+
| workflow                       | catalog                         |
| lifecycle, next_actions        | what rci plugins, methods list  |
+--------------------------------+---------------------------------+
| stages     per layer: run, floor and default lists; skeleton     |
+------------------------------------------------------------------+
| validation an eval's own tests, calibration, loading evals       |
+------------------------------------------------------------------+
| runner     plugins, research-code check, cost meter              |
+--------------------------------+---------------------------------+
| store                          | render                          |
| .rci/, ledger, freeze, vault   | cards from engine files         |
+--------------------------------+---------------------------------+
| eval       methods the engine executes, with worked examples     |
+------------------------------------------------------------------+
| kernel     time, holdout, verdict (pure functions, no I/O)       |
+------------------------------------------------------------------+
| api        contract, plugin protocols, evals, null worlds        |
+------------------------------------------------------------------+
     ^
     |  plugins (in a lab) import api, and eval for its methods
```

A layer may import only the layers below it; `workflow` and `catalog` do not import each other, nor do `store` and `render`. `validation` sits above `runner` because loading an idea's evals uses the plugin loader and the research-code check. import-linter enforces this in CI, and `reference_plugins` may import only `api` and `eval`, as a lab's plugins may: `eval` imports only `api` and `kernel`, so a plugin that uses a method gets nothing else. The engine repo holds no ideas and no real plugins, only a few generic reference plugins that get no privileges. A lab is any Python 3.11+ project with the engine and its plugins installed, keeping ideas under `ideas/<idea>/`.

| Package | Role | Status |
| --- | --- | --- |
| `api` | Building blocks for research code and plugins: `Contract` and its parts, plugin protocols, `Eval`, the tests a case eval declares (`Case`, `Size`, `Power`, `Recovers`), and null-world declarations. | Done |
| `kernel` | What a verdict rests on, pure functions, no I/O: `time` (asof), `holdout` (tiers), `hashing`, `verdict` (the floor shuffle and null worlds as the engine runs them (`worlds`), judging every eval on shared shuffles, batched or stacked, z, GridMax, the sequential rule, power, and how the decisive effects make a feat verdict (`rule`)). | Done |
| `eval` | The methods: plugin protocol checks, data checks for fidelity and fitness, the statistics and structure that evals use, and the synthetic markets and sources their properties run on. Each method's module carries worked examples and properties checked on synthetic markets and sources. Grows; imports only `api` and `kernel`, so criteria and plugins may import it. | Done for contract, data, and feat |
| `stages` | Per layer, a package `stages/<layer>/` of two lists of steps and the thin code that runs them: `floor.py`, the floor list (the steps every run takes, pinned by the engine hash), `defaults.py`, the default list (checks to adjust from experience, out of the engine hash), and `run.py`, which builds what the steps read and combines what they found; `skeleton.py` holds what every layer's lists are made of. Contract derivation, source admission and the data check with its treatments, and feat submissions today; signal (`stages/sig/`) and algo later. [docs/catalog.md](docs/catalog.md) lists every step with its role. | Contract, data, feat done |
| `validation` | An eval's own tests and their running: the runner of a module's worked examples and properties (`tests.py`), the calibration that holds the decisive test to its world before it judges (`calibration.py`), and loading an idea's files and validating its case evals (`load.py`). | Done |
| `workflow` | Lifecycle: the states every stage shares, the one pipeline with STALE invalidation, `next_actions`, approval provenance; one module of commands per stage; the files `rci init` writes. | Done |
| `catalog` | What `rci plugins` and `rci methods` list, read from the code: installed plugins, the library's methods and what backs them, and every layer's lists, printed as [docs/catalog.md](docs/catalog.md). Out of the engine hash. | Done |
| `runner` | Plugin loading by entry point and plugin code hashes, with errors that name close plugins and a plugin's signature; the static check that finds row-by-row research code for the cards' warning; the meter of what a call costs; later process isolation for research code. | Done; child process planned |
| `store` | All I/O: workspace `.rci/`, ledger and head anchors, freeze records, environment records, the content-hashed data vault. | Done |
| `render` | Deterministic cards from engine files: the Contract, Data, Result, and Explore cards; citation checks for summaries later. | Done for contract, data, and feat |
| `cli` | The `rci` command. | Partly done |
| `reference_plugins` | Labels `ref.return`, `ref.markout`, `ref.carry`; target `ref.mean`; cost model `ref.flat_costs`. | Done |

### Process model

Research code runs in the engine's process today: `feat` reads a features file for row-by-row code, for the card's warning, imports it, and calls it once with the frozen data and the candidates as polars DataFrames, then again on the data cut at a few decision times, timing each call and sampling the process's memory during the first. Running it in a child process is planned:

```text
engine (parent)                                     research code (child, planned)
      |                                                        |
      |-- verify the freeze, lineage, versions                 |
      |-- frozen data and candidates (Arrow) ----------------->|
      |<------------------------------- scores per (t, entity) |
      |-- the same, on data cut at a few decision times ------>|
      |<------------------------------------------------ scores |
      |                                                        |
      |-- labels (the child never sees them), evals,           |
      |   null worlds, calibration, verdict, ledger            |
```

Then a stray `np.random.seed`, an in-place edit, or a monkeypatch in research code cannot reach the engine, and the child runs with a timeout and a memory limit. This boundary is also where a trusted service would be split out later.

## Trust model

| Situation | Rule |
| --- | --- |
| Normal run in a lab | The engine should be installed at a pinned git SHA and read-only; a lab may install it in editable mode while both are being developed. A layer's freeze pins the contract, the engine source (without `render`, `cli`, `catalog`, and `workflow/templates.py`, which shape only what is shown or read, not what is judged, and the default lists `stages/*/defaults.py`: feat's diagnostics in `stages/feat/defaults.py`, which never decide, the data layer's checks in `stages/data/defaults.py`, whose result the data freeze pins by content, and derive's in `stages/contract/defaults.py`; the floor lists `stages/*/floor.py` are pinned), the interpreter, the engine's own dependencies, and the upstream freezes; the contract's pins its own file. Derive also records the code hash of every plugin and the hash of the data snapshot, and approving the contract checks them again; the data freeze also pins its check, with the decisive test's calibration, and the treated data; each feat submission records its features file, `feat.py` and the modules of the case evals it ran, null worlds, the code of every plugin it runs, and its calibration, and approving it pins them. Each submission also records every installed package and `uv.lock`. |
| Anything a result used changes before approval | Approval is refused and names what moved; the agent runs the stage again. Research code and the libraries it uses may change between submissions. |
| Approval relayed by an agent, or given by the agent | Recorded in the ledger with the channel and, for a relay, the person's words. The verdict machinery will cap what such stages can reach (planned). |
| Run from this repo in editable mode | To be marked DEV, so its verdicts do not count (planned). |
| Deliberate cheating | Out of scope for the MVP. Later: a separate OS user or a thin service `execute(bundle, vault, ledger) -> verdict`. |

Git is not a security mechanism, because agents can rewrite history. Trust comes from the engine's own freeze hashes and the hash-chained ledger.

The hash chain catches edited, deleted, and reordered records, but not records cut from the end, such as a failed submission deleted to win back a trial or an older ledger restored from git. For that the engine also keeps each idea's latest head in `$RCI_HOME/heads/` (default `~/.rci/heads/`), outside the idea and keyed by its first record, and refuses a ledger that no longer contains it. A ledger that grew elsewhere, for example pulled from another machine, is accepted as long as it still contains that head.

See [docs/design.md](docs/design.md) for the reasoning behind each rule and what the trust model does not cover.

## Development

```sh
uv sync
uv run rci --help
uv run pytest
uv run ruff check
uv run ruff format --check
uv run basedpyright
uv run lint-imports
npx markdownlint-cli2
```

```text
src/research_ci/
  api/  kernel/{time,holdout,verdict}/  eval/{protocol,data,statistics,structure}/  validation/
  stages/{contract,data,feat}/  workflow/  catalog/  runner/{plugins,research}  store/  render/  cli/  reference_plugins/
tests/<package>/  tests/incidents/ (real incidents)  tests/toy.py (toy plugins and broken variants)
docs/design.md  docs/walkthroughs.md  docs/catalog.md
```
