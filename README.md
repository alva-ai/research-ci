# Research CI

A backtest framework for research: an alpha claim counts only after it passes CI, the same way code does.

> **Status:** early. The time kernel (M1), the holdout, ledger, and workflow layer (M2), the contract stage (M3a: plugin protocols, protocol tests, power, the ContractCard, approval provenance), the data layer (M3b: source admission by replay and reconciliation, checks the contract and plugins declare, treatments weighed by their influence, the DataCard), and the feat layer (M3c: criteria judged against null worlds for every alternative, calibration, a lookahead check on cut data, the decisive effect, the MethodCard and ResultCard) are implemented. The signal and algo layers come next. See [Roadmap](#roadmap).

| Area | Works today | Planned |
| --- | --- | --- |
| Time | FieldTable, `asof`, stepping, lookahead checks, decision grids | Decision times from events and latencies, walk-forward helpers |
| Holdout | Explore, seal, and forward tiers; labels reach criteria only through the evaluation table | Labels through `fit` on training folds, vintage-aware labels |
| Contract | Every field validated; `derive` loads plugins, snapshots the data, runs protocol tests, computes power, pilots the consumer, and prints the ContractCard | Oracle bounds for maker strategies |
| Data | Source admission (replay as of past times, an independent source on engine-picked rows, canaries, field grades); checks the contract and plugins declare; treatments applied by default and weighed by their influence in MDE units; the DataCard | Influence on consumer effects, seal-tier checks, listings from the source to catch hindsight universes |
| Workflow | State machine, budgets, lineage, `next_actions`, templates, approval provenance | Caps by approval provenance, hooks after verdicts |
| Trust | Hash-chained ledger, head anchors, freeze records, data vault; plugin code, baseline files, and the data snapshot checked again at approval; the feat freeze pins its criteria, null worlds, and plugin code | Plugin hashes in the other layer freezes |
| Feat | Criteria on an evaluation table; the idea's null worlds and the engine's floor, one per alternative explanation; calibration by stand-ins and planted signals; a lookahead check on cut data; the decisive effect on a label or through the consumer; INSUFFICIENT when the data cannot decide; MethodCard and ResultCard | Learning from labels with walk-forward, other null generators, dedup by correlation |
| Judge | Protocol tests, power, minimum detectable effects; `PermuteWithin` shuffles, GridMax over horizons, z thresholds that rise with the trial count | Evidence levels and caps, stress worlds and limit criteria (M6) |
| Runner | Plugin loading by entry point, plugin code hashes | Research code in a child process with Arrow exchange (M4) |
| Plugins | Protocols for data, universes, labels, targets, cost models, consumers; reference labels, target, and cost model | First real data plugin and consumer (M5), fill models (M6), challenger |

## Documentation

| Read | For |
| --- | --- |
| This README | What Research CI is, the loops at a glance, a quickstart, the CLI |
| [docs/design.md](docs/design.md) | The full design: principles, architecture, lifecycle, the contract, time and leaks, holdout and labels, judging, the challenger, trust, layers, plugins, key decisions and why, prior art, roadmap |
| [docs/walkthroughs.md](docs/walkthroughs.md) | Three worked cases, step by step: a [futures inventory spread](docs/walkthroughs.md#futures-inventory-spread), a [cross-venue maker-taker](docs/walkthroughs.md#cross-venue-maker-taker) strategy, and [text event signals](docs/walkthroughs.md#text-event-signals) |

## Why

Agents made research fast, but judgment did not keep up. The usual loop is: an agent prepares data, writes a strategy, sweeps parameters, shows the best PnL curve, and explains why it works. Every decision in that loop depends on context and gut feeling, and none of it is reproducible.

What a person needs instead is a green light they can trust without reading the research, the way a passing CI run lets you merge without rereading every line. Many agent sessions can then work on ideas in parallel, and the person only handles the few decisions that are theirs to make. A written summary is never the verdict: numbers are judged by code, and summaries are checked against the verdict they cite.

Research CI moves the judgment **before** the results and makes it executable:

- **Falsifiable**: before an idea starts, write down what result would prove it wrong.
- **Executable**: that standard is code. The engine reruns the submission and decides; it never trusts numbers an agent reports.
- **Reproducible**: the standard is frozen before any result is seen. The same inputs always produce the same verdict, byte for byte.

Research code stays free (pandas or polars, rolling windows, sklearn, SHAP, anything); it declares which panel format it wants, and pandas is the default. Everything specific to a case is a plugin: data sources, candidates, labels, targets, cost models, backtesters. The engine is thin: it runs on polars, holds plugins and agent-written code to executable checks (and tests those checks), and owns exactly three things:

| Kernel duty | What it guarantees |
| --- | --- |
| **Time** | Research code only ever sees data with `available_at <= t` at each decision time. |
| **Holdout and labels** | Label definitions are plugins named in the contract; the engine decides which data a label sees, when it runs, and who gets its values. Sealed and forward data are out of the agent's reach. |
| **Verdict** | Scores are compared against null worlds, thresholds rise with the number of trials, and every verdict is logged in a hash-chained ledger. |

## How it works

### Outer loop: one idea, end to end

Each research layer is its own inner loop, and a layer can only use outputs that passed the layer before it. Seal, forward, and capital are one-shot checkpoints: registered in advance, run once, and sent back to an earlier layer if they fail.

![The outer loop: one idea from the contract through the four layers to the checkpoints](docs/assets/outer-loop.png)

| Layer | Question it answers | Output shape | Needs a backtest |
| --- | --- | --- | --- |
| `contract` | What is claimed, what result counts as wrong, and can the data decide it? | Contract, ContractCard | No; it runs the consumer only to measure its noise |
| `data` | Can these fields support the contract (point-in-time, vintages, breaks)? | FieldTable | No |
| `feat` | Does the feature add information? | Scores | No |
| `signal` | Does the combined signal predict? Raw signals and model outputs both live here. | Scores, Events | No |
| `algo` | Does acting on the signal make money after costs and constraints? | Positions, Quotes, Orders | Yes |

Labels are not a layer and neither is the backtest: labels are plugins held in the engine's custody, and the backtest is a consumer plugin, the evaluator of the `algo` layer.

### Inner loop: every layer runs the same state machine

```text
OPEN                                  STALE (an upstream stage was reopened)
 |  register (agent)                    |  register (agent): new version
 v                                      v
REGISTERED <--------------------------------------------+
 |  challenge (challenger adds null worlds, pitfalls)   |
 v                                                      |
CHALLENGED <------------+                               |
 |  check (calibration) |  calibration failed,          |  register again after
 v                      |  or new nulls were added      |  the freeze:
CHECKED ----------------+                               |  new version
 |  approve (who approved is recorded)                  |
 v                                                      |
FROZEN -------------------------------------------------+
 |  submit (1 trial)
 v
PASSED .............. the next stage can start
FAILED / TAINTED .... change the research code and submit again (1 more trial)
PAUSED .............. the budget is used up; a person extends it
```

- Exploration is allowed in any state. It never changes the state and is never counted as a trial.
- Each layer gets a submission budget from the contract, shared across its versions, so a new version does not reset it.
- A submission is TAINTED when anything frozen at approval has changed, for example the contract or the engine.
- Reopening a layer that has passed marks every started layer after it STALE, and nothing downstream can move until the reopened layer passes again.
- Checkpoints run once. A failed checkpoint sends the idea back to a layer.
- There is no separate state file. Every transition is a ledger record, the state is replayed from the ledger, and every command refuses to run on a ledger whose hash chain is broken.

### Who does what

```text
research agent           human                     scheduler
explore, submit          approve, capital card     forward, monitor
      |                     |                           |
      +---------------------+---------------------------+
                            |
                         rci CLI
                            |
                        workflow  ---------- spawns ------------>  challenger
              (state machine, next_actions)                        (separate LLM process,
                            |           <---- null worlds, tests    reads files only)
                            |
                     kernel + stages
                (deterministic, no LLM)
```

The library is passive. It knows the state, decides which transitions are legal, and computes. Every driver reads the same `next_actions()` list: `rci next` prints it for one idea, and `rci inbox` collects everything that waits for a person across all ideas. `rci approve` asks a person at an interactive terminal by default. An agent can relay a person's approval with `--relayed "<the person's words>"`, or approve itself in fully automatic research with `--auto`; the ledger records which, and extending a budget always needs a terminal.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Install the engine into the project that will hold your ideas, pinned to a commit:

```sh
uv add "research-ci @ git+ssh://git@github.com/alva-ai/research-ci@<sha>"
```

The plugins a contract names must be installed in the same environment, registered under the entry-point group `research_ci.plugins`. Below, `ref.*` plugins ship with the engine and `lab.*` stand for plugins in your lab. Numbers are illustrative.

Start an idea, write its contract, and let a person freeze it:

```text
$ rci init ideas/carry -t factor
created ideas/carry (factor); edit ideas/carry/contract.py, then run `rci contract derive -i ideas/carry`

$ rci next -i ideas/carry
agent      rci contract derive                  write contract.py, then derive the contract
```

```python
# ideas/carry/contract.py
from research_ci.api import Alternative, Contract, Effect, Execution, Label, Universe, Use

CONTRACT = Contract(
    template="factor",
    claim="Cross-venue funding spreads persist long enough to pay the round trip",
    decisive=(Effect(on="carry", at_least=0.001, unit="per entry after the round trip"),),
    alternatives=(Alternative("tail", "a few symbols with extreme funding carry the whole result"),),
    used=(),
    data=Use("lab.perp", {"root": "~/data/perp"}),
    costs=Use("ref.flat_costs", {"commission_bps": 1, "slippage_bps": 1, "legs": 2}),
    execution=Execution(every="1h", note="decide on the hour; fill both legs at the close known then"),
    universe=Universe("lab.venue_pairs", clusters="symbol", groups=("venue_pair",)),
    labels=(Label("carry", Use("ref.carry", {"flow": "funding_rate"}), horizons=("8h", "24h")),),
    explore_end="2026-01-15T00:00:00Z",
    seal_end="2026-01-20T00:00:00Z",
    budget=20,
)
```

`rci contract derive` loads the plugins, stores the data in the vault, runs protocol tests on every plugin, computes power, and prints the ContractCard:

```text
$ rci contract derive -i ideas/carry
ContractCard · carry · v1 · factor
claim      Cross-venue funding spreads persist long enough to pay the round trip
decisive   carry ≥ 10.0 bp per entry after the round trip · target ref.mean
decisions  every 1h from execution · 335 decision times
           decide on the hour; fill both legs at the close known then
data       lab.perp afa06051cf36 · 3,780 rows · 9 entities · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 0.4 days old
universe   lab.venue_pairs · median 9 candidates in 3 symbol clusters per decision · membership pit
costs      ref.flat_costs(commission_bps=1, slippage_bps=1, legs=2) · median round trip 8.0 bp
labels     carry: ref.carry · H 8h 24h
checks     ok   data is current: latest data is 0.4 days old
           ok   universe point-in-time: 3 cut points give the same candidates
           ok   universe causal: values after 2026-01-08T04:00Z leave the candidates at it unchanged
           ok   label carry: worked examples: 2 ok
           ok   label carry: examples bite: shift, in-window changes and missing fields move it; later data does not
           ok   label carry: no data after t + h: 2 cut points, horizon 8h
           ok   label carry: no data after t + h: 2 cut points, horizon 24h
power      carry:
           H        decisions  blocks   MDE bp  FAIL if <   new data
           8h             321    41.9      3.4        7.6        2 d
           24h            305    14.0     10.3        2.5       14 d
alts       tail  (engine floor: chance, timing)
used       nothing declared
history    history can confirm
budget     20    deadline 365 days
verdict    DECIDABLE
           carry: best horizon 8h, about 2 days of new data (history)
----------------------------------------------------------------------------------------------------
registered; a person freezes it with `rci approve --stage contract`

$ rci inbox ideas
ideas/carry              rci approve --stage contract         review the contract and freeze it

$ rci approve --stage contract -i ideas/carry          # a person, at a terminal
approve contract v1 of carry
ContractCard · carry · v1 · factor
...                                                    (the card, as derive printed it)
  frozen dependencies:
    contract             8fbeeb9e6ad8
Freeze contract v1? [y/N]: y
frozen contract v1: 0bb786ae7d27 · approved by a person at a terminal
to keep it in git: git add ideas/carry/.rci ideas/carry/contract.py && git commit -m 'rci: freeze contract of carry'

$ rci status -i ideas/carry
carry (factor)
  contract  PASSED      v1           approved by a person at a terminal
  data      OPEN        v1   0/20
  feat      OPEN        v1   0/20
  signal    OPEN        v1   0/20
  algo      OPEN        v1   0/20
  seal      OPEN        v1
  forward   OPEN        v1
  capital   OPEN        v1
  monitor   OPEN        v1

$ rci next -i ideas/carry --json
[
  {
    "who": "agent",
    "stage": "data",
    "kind": "check",
    "command": "rci data check",
    "reason": "check the data against the contract"
  }
]
```

Approval is refused if `contract.py`, a plugin's code, a baseline file, or the data snapshot changed since the derive, or if any protocol test failed. Next, `rci data check` admits the data source if this plugin, with these options and this code, was never admitted (a person approves that once, with `rci source approve`), checks the data against the contract, and prints the DataCard; `rci approve --stage data` freezes it and `rci data submit` records it as what feat reads. `rci feat register` fixes how features are judged (the criteria in `feat.py`, the challenger's null worlds in `nulls/feat.py`, and their calibration) and prints the MethodCard; after `rci approve --stage feat`, each `rci feat submit <file>` judges one features file and prints a ResultCard. Engine state lives in `ideas/<idea>/.rci/` (commit it with the idea) and in `$RCI_HOME` (default `~/.rci`), which keeps each ledger's latest head outside the idea and holds the data vault.

## CLI

| Command | Called by | Precondition | Available |
| --- | --- | --- | --- |
| `rci init <idea> --template <t>` | agent | none | yes |
| `rci contract derive` | agent | none; prints the ContractCard | yes |
| `rci approve --stage <stage>` | **human**, interactive | upstream passed; for the contract, nothing changed since derive and no protocol test failed; calibration passed for layers | yes |
| `rci approve --stage <stage> --relayed "<words>" --card <id>` | agent, relaying a person's approval after showing the person the whole card | as above; the card has not changed since | yes |
| `rci approve --stage <stage> --auto` | agent, fully automatic research | as above | yes |
| `rci approve --stage <layer> --extend N` | **human**, interactive only | layer PAUSED | yes |
| `rci next`, `rci status`, `rci inbox`, `rci verify` | anyone | none | yes |
| `rci data check [--keep "<treatment>"]` | agent | contract approved; admits the data source first if needed; prints the SourceCard and the DataCard; `--keep` leaves a treatment a person rejected unapplied | yes |
| `rci source approve` | **human**, interactive, or `--relayed "<words>" --card <id>` / `--auto` | the source passed admission; once per plugin, options, and code | yes |
| `rci data submit` | agent | data frozen | yes |
| `rci data show`, `rci source show` | anyone | none | yes |
| `rci feat register` | agent | data passed; registers, challenges, and checks the method; prints the MethodCard | yes |
| `rci feat explore <file>` | agent, unlimited | data passed; runs only the checks a submission must pass | yes |
| `rci feat submit <file>` | agent, 1 trial each | method frozen and unchanged, budget left, scores not repeated; prints the ResultCard | yes |
| `rci feat show` | anyone | method registered | yes |
| `rci <layer> explore/register/challenge/check/submit` for signal and algo | agent, challenger | upstream passed | later (challenger in M5) |
| `rci seal run`, `rci capital run` | human | plan declared and frozen in advance | later |
| `rci forward --all`, `rci monitor --all` | scheduler | none | later |

`<layer>` is signal or algo; data and feat have their own commands above. `status`, `next`, and `inbox` take `--json` for agents. Commands on one idea take `-i <idea>` and default to the current directory; `init` takes the new idea's directory and `inbox` a root directory.

## Templates

| Template | Research type | Layers |
| --- | --- | --- |
| `factor` | Cross-sectional factors | data, feat, signal (optional), algo (optional) |
| `rule` | Rule-based strategies | data, signal, algo |
| `event` | News and LLM events | data, feat, signal, algo; confirmed on forward data only |
| `taker_arb` | Taker arbitrage | data, signal (optional), algo |
| `maker_arb` | Maker arbitrage and market making | data, feat, signal, algo with a fill model |

All templates end with the same checkpoints: seal, forward, capital, monitor. The contract of a `taker_arb` idea may skip the signal layer with `skip=("signal",)`. A `factor` idea that improves an existing strategy may skip `signal` and `algo`; its contract must then name the baseline's config and code.

## Architecture

```text
+------------------------------------------------------------------+
| cli        the rci command                                       |
+------------------------------------------------------------------+
| llm        challenger, narration, summary audit (optional)       |
+------------------------------------------------------------------+
| workflow   state machine, lineage, templates, next_actions       |
+------------------------------------------------------------------+
| stages     per-stage policies: contract derivation today         |
+------------------------------------------------------------------+
| runner     plugin loading and code hashes; child process (M4)    |
+--------------------------------+---------------------------------+
| store                          | render                          |
| .rci/, ledger, freeze, vault   | cards from engine files         |
+--------------------------------+---------------------------------+
| kernel     time, holdout, checks, judge (pure functions, no I/O) |
+------------------------------------------------------------------+
| api        contract, plugin protocols, criteria, null worlds     |
+------------------------------------------------------------------+
     ^
     |  plugins (in a lab) import api only
```

A layer may import only the layers below it; `store` and `render` do not import each other. import-linter enforces this in CI, and `reference_plugins` may import only `api`. The engine repo holds no ideas and no real plugins, only a few generic reference plugins that get no privileges. A lab is any Python 3.11+ project with the engine and its plugins installed, keeping ideas under `ideas/<idea>/`.

| Package | Role | Status |
| --- | --- | --- |
| `api` | Building blocks for research code and plugins: `Contract` and its parts, plugin protocols, `Criterion` and null-world declarations. | Done |
| `kernel` | Mechanisms, pure functions, no I/O: `time` (asof, stepper), `holdout` (tiers), `hashing`, `checks`, `judge` (protocol tests, power, shuffles, calibration, verdicts). | Done for contract, data, and feat |
| `stages` | Policies, one file per stage: contract derivation, source admission, the data check, and the feat method and submissions today; signal and algo later. | Contract, data, feat done |
| `workflow` | Lifecycle: inner state machine, outer dependency graph with STALE invalidation, templates, `next_actions`, approval provenance. | Done |
| `runner` | Plugin loading by entry point and plugin code hashes; later process isolation for research code. | Loading done; child process M4 |
| `store` | All I/O: workspace `.rci/`, ledger and head anchors, freeze records, environment records, the content-hashed data vault. | Done |
| `render` | Deterministic cards from engine files: Contract, Source, Data, Method, and Result cards; citation checks for summaries later. | Done for contract, data, and feat |
| `llm` | Optional: challenger, card narration, summary audit. Never imported by the kernel or workflow. | Planned |
| `cli` | The `rci` command. | Partly done |
| `reference_plugins` | Labels `ref.return`, `ref.markout`, `ref.carry`; target `ref.mean`; cost model `ref.flat_costs`. | Done |

### Process model

Research code never runs in the kernel's process (planned for M4; until then contracts and plugins are loaded in-process).

```text
kernel (parent)                                       research code (child)
      |                                                        |
      |-- verify freeze hash, lineage, versions                |
      |-- start child with study code and plugins ------------>|
      |                                                        |
      |   for each decision time t:                            |
      |-- window with available_at <= t (Arrow) -------------->|
      |<------------------------------------------ output at t |
      |                                                        |
      |-- run label plugins (the child never sees the values)  |
      |-- evaluate(criterion, output, allowed labels) -------->|
      |<------------------------------------------------ score |
      |                                                        |
      |-- null worlds, calibration, verdict, ledger            |
```

A stray `np.random.seed`, an in-place edit, or a monkeypatch in research code cannot change the verdict. Seeds are derived from the freeze hash. This boundary is also where a trusted service would be split out later.

## Trust model

| Situation | Rule |
| --- | --- |
| Normal run in a lab | The engine is installed at a pinned git SHA and read-only. A freeze pins the contract, the engine source, the interpreter, the engine's own dependencies, and the upstream freezes. Derive also records the code hash of every plugin, the hash of every baseline file, and the hash of the data snapshot, and approving the contract checks them again; the feat freeze also pins its criteria, its null worlds, and the code of every plugin it runs. Each submission also records every installed package and `uv.lock`. |
| Anything frozen changes after approval | The submission is TAINTED. Research code and the libraries it uses may change between submissions. |
| Approval relayed by an agent, or given by the agent | Recorded in the ledger with the channel and, for a relay, the person's words. The verdict machinery will cap what such stages can reach (planned). |
| Run from this repo in editable mode | Marked DEV, capped at L1 (planned). |
| Deliberate cheating | Out of scope for the MVP. Later: a separate OS user or a thin service `execute(bundle, vault, ledger) -> verdict`. |

Git is not a security mechanism, because agents can rewrite history. Trust comes from the engine's own freeze hashes and the hash-chained ledger.

The hash chain catches edited, deleted, and reordered records, but not records cut from the end, such as a failed submission deleted to win back a trial or an older ledger restored from git. For that the engine also keeps each idea's latest head in `$RCI_HOME/heads/` (default `~/.rci/heads/`), outside the idea and keyed by its first record, and refuses a ledger that no longer contains it. A ledger that grew elsewhere, for example pulled from another machine, is accepted as long as it still contains that head.

See [docs/design.md](docs/design.md) for the reasoning behind each rule and what the trust model does not cover.

## Roadmap

| Step | Scope | Acceptance | Status |
| --- | --- | --- | --- |
| M0 | Scaffold, CI, import layers | CI green; layer violations are caught | Done |
| M1 | `tables` + `time` (asof, stepper) | Every planted lookahead is caught; plain rolling reports zero mismatches | Done |
| M2 | `holdout` + ledger + workflow skeleton + `rci next` | Out-of-order calls are rejected; a one-byte edit breaks ledger verification | Done |
| M3a | The contract: fields, plugin protocols and loading, data vault, decision grid, protocol tests, power, consumer pilot (`Branch`, `FullPath`), ContractCard, approval provenance, reference plugins | Every deliberately broken toy plugin is caught by its protocol test; approval refuses failed tests and anything changed since derive | Done |
| M3b | The data layer: source admission (replay, reconciliation, canaries, grades), invariants declared by plugins, idea tests and their calibration, treatments and their influence, DataCard, `rci data` and `rci source` commands | Every deliberately broken toy source is caught; a real source whose repairs used hindsight is refused until fixed | Done |
| M3c | The feat layer: criteria on an evaluation table, null worlds per alternative and the engine's floor, calibration, a lookahead check on cut data, decisive effects, INSUFFICIENT, MethodCard and ResultCard, `rci feat` commands | A spread passes, noise fails, a peek is tainted, and a null world that does not keep what it claims is returned, on the toy; the first real case runs end to end | Done |
| M4 | `runner`: child process + Arrow | Verdict cards stay byte-identical when research code reseeds or mutates data | Planned |
| M5 | First data plugin, consumer, minimal challenger | Plugins pass conformance | Planned |
| M6 | Algo: execution simulator, fill models, randomized decisions, constraint criteria | A maker-taker case end to end | Planned |
| Later | Checkpoint commands, verdict hooks, graveyard replay, a thin trusted service | Replaying past ideas kills some of them earlier than they were killed back then, kills no true positive, and every verdict reproduces | Planned |

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
  api/  kernel/{time,holdout,checks,judge}/  stages/  workflow/
  runner/  store/  render/  llm/  cli/  reference_plugins/
tests/unit/  tests/known_answers/  tests/toy.py (toy plugins and broken variants)
docs/design.md  docs/walkthroughs.md
```
