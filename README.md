# Alpha CI

A backtest framework for research: an alpha claim counts only after it passes CI, the same way code does.

> **Status:** early. The time kernel (M1), holdout tiers and labels, the ledger, freeze records, and the workflow with its first CLI commands (M2) are implemented. Nothing computes a verdict yet (M3). See [Roadmap](#roadmap).

## Why

Agents made research fast, but judgment did not keep up. The usual loop is: an agent prepares data, writes a strategy, sweeps parameters, shows the best PnL curve, and explains why it works. Every decision in that loop depends on context and gut feeling, and none of it is reproducible.

Alpha CI moves the judgment **before** the results and makes it executable:

- **Falsifiable**: before an idea starts, write down what result would prove it wrong.
- **Executable**: that standard is code. The engine reruns the submission and decides; it never trusts numbers an agent reports.
- **Reproducible**: the standard is frozen before any result is seen. The same inputs always produce the same verdict, byte for byte.

Research code stays free (pandas or polars, rolling windows, sklearn, SHAP, anything); it declares which panel format it wants, and pandas is the default. The kernel itself runs on polars and owns exactly three things:

| Kernel duty | What it guarantees |
| --- | --- |
| **Time** | Research code only ever sees data with `available_at <= t` at each decision time. |
| **Holdout and labels** | Labels are built by the engine from the contract; sealed and forward data are out of the agent's reach. |
| **Verdict** | Scores are compared against null worlds, thresholds rise with the number of trials, and every verdict is logged in a hash-chained ledger. |

## How it works

### Outer loop: one idea, end to end

Each research layer is its own inner loop, and a layer can only use outputs that passed the layer before it. Seal, forward, and capital are one-shot gates: registered in advance, run once, and sent back to an earlier layer if they fail.

```text
idea
 |
contract ....... reverse-derived thresholds, frozen before any search
 |
data ........... inner loop  <--+
 |                              |
feat ........... inner loop  <--|----------------------+      CI
 |                              |                      |
signal ......... inner loop     |  a later layer       |
 |                              |  finds a data        |
algo ........... inner loop  ---+  problem             |
 |               (the backtest evaluates this layer)   |
 |                                                     |  drift
seal ........... once, pass/fail only                  |  reopens
 |                                                     |  a layer    CD
forward ........ new data created after freeze         |
 |                                                     |
capital ........ human decision                        |
 |                                                     |
monitor ........ scheduled ----------------------------+
```

| Layer | Question it answers | Output shape | Needs a backtest |
| --- | --- | --- | --- |
| `contract` | What does the trade look like, and what result counts as wrong? | Spec | No |
| `data` | Can these fields support the contract (point-in-time, vintages, breaks)? | FieldTable | No |
| `feat` | Does the feature add information? | Scores | No |
| `signal` | Does the combined signal predict? Raw signals and model outputs both live here. | Scores, Events | No |
| `algo` | Does acting on the signal make money after costs and constraints? | Positions, Quotes, Orders | Yes |

### Inner loop: every layer runs the same state machine

```text
OPEN
 |  register evaluators
 v
REGISTERED <-----------------------------+
 |  challenger adds null worlds          |
 v                                       |
CHALLENGED                               |
 |  calibration passes                   |  change tests
 v                                       |  (new version, counted)
CHECKED                                  |
 |  human approves                       |
 v                                       |
FROZEN ----------------------------------+
 |  aci submit (1 trial)
 v
SUBMITTED ------------------> PASSED ----> next layer
 |      ^
 |      |  change code, resubmit (1 more trial)
 v      |
FAILED / TAINTED
```

Exploration is allowed in any state. It never changes the state and is never counted as a trial.

Each layer gets a submission budget from the contract, shared across its versions. When the budget runs out the layer is PAUSED until a person extends it. Reopening a layer that has passed marks every started layer after it STALE, and nothing downstream can move until the reopened layer passes again. Gates run once: a failed gate sends the idea back to a layer.

There is no separate state file. Every transition is a ledger record, the state is replayed from the ledger, and every command refuses to run on a ledger whose hash chain is broken.

### Who does what

```text
research agent           human                     scheduler
explore, submit          approve, capital card     forward, monitor
      |                     |                           |
      +---------------------+---------------------------+
                            |
                         aci CLI
                            |
                        workflow  ---------- spawns ------------>  challenger
              (state machine, next_actions)                        (separate LLM process,
                            |           <---- null worlds, tests    reads files only)
                            |
                     kernel + stages
                (deterministic, no LLM)
```

The library is passive. It knows the state, decides which transitions are legal, and computes. Every driver reads the same `next_actions()` list; `aci next` prints it.

## Architecture

```text
+------------------------------------------------------------------+
| cli        the aci command                                       |
+------------------------------------------------------------------+
| llm        challenger, narration, summary audit (optional)       |
+------------------------------------------------------------------+
| workflow   state machine, lineage, templates, next_actions       |
+----------------+----------------+----------------+---------------+
| stages         | store          | runner         | render        |
| per-layer      | .aci/, ledger, | research code  | cards from    |
| policies       | freeze, vault  | in a child     | verdict.json  |
+----------------+----------------+----------------+---------------+
| kernel     time, holdout, checks, judge (pure functions, no I/O) |
+------------------------------------------------------------------+
| api        building blocks and plugin protocols                  |
+------------------------------------------------------------------+
     ^
     |  plugins (in alpha-ci-lab) import api only
```

A layer may import only the layers below it. import-linter enforces this in CI, and `reference_plugins` may import only `api`.

| Package | Role |
| --- | --- |
| `api` | Building blocks for research code and plugins: `Contract`, `Angle`, null declarations, plugin protocols. |
| `kernel` | Mechanisms, pure functions, no I/O: `time` (asof, stepper, walk-forward), `holdout` (tiers, labels), `checks`, `judge` (derive, nulls, mutate, calibrate, stats, verdict). |
| `stages` | Policies, one file per layer: floor checks, floor null worlds, calibration recipe, caps. |
| `workflow` | Lifecycle: inner state machine, outer dependency graph with STALE invalidation, templates, `next_actions`. |
| `runner` | Process isolation for research code. |
| `store` | All I/O: workspace `.aci/`, ledger, freeze records, data vault. |
| `render` | Deterministic cards from `verdict.json`, citation checks for summaries. |
| `llm` | Optional: challenger, card narration, summary audit. Never imported by the kernel or workflow. |
| `cli` | The `aci` command. |

### Process model

Research code never runs in the kernel's process.

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
      |-- build labels (the child never sees them here)        |
      |-- evaluate(angle, output, allowed labels) ------------>|
      |<------------------------------------------------ score |
      |                                                        |
      |-- null worlds, calibration, verdict, ledger            |
```

A stray `np.random.seed`, an in-place edit, or a monkeypatch in research code cannot change the verdict. Seeds are derived from the freeze hash. This boundary is also where a trusted service would be split out later.

## CLI

| Command | Called by | Precondition |
| --- | --- | --- |
| `aci init <idea> --template <t>` | agent | none |
| `aci contract derive` | agent | none |
| `aci approve --stage <stage>` | **human**, interactive | calibration passed, no coverage gaps |
| `aci <layer> explore` | agent, unlimited | upstream passed |
| `aci <layer> register` | agent | upstream passed |
| `aci <layer> challenge` | challenger | evaluators registered |
| `aci <layer> check` | agent | null worlds present |
| `aci <layer> submit` | agent, 1 trial each | frozen, budget left; TAINTED if anything frozen changed |
| `aci approve --stage <layer> --extend N` | **human**, interactive | layer PAUSED |
| `aci seal run` | human | k candidates declared in advance |
| `aci forward --all`, `aci monitor --all` | scheduler | none |
| `aci next`, `aci status`, `aci inbox`, `aci verify` | anyone | none |

`<layer>` is one of data, feat, signal, algo. Implemented so far: `init`, `status`, `next`, `inbox`, `verify`, `contract derive`, and `approve`. The per-layer commands arrive with M3 and the gate commands after it; `aci next` already names them. `status`, `next`, and `inbox` take `--json` for agents.

```text
$ aci init ideas/tox -t factor
$ aci next -i ideas/tox
agent      aci contract derive                  write contract.py, then derive the contract
$ aci contract derive -i ideas/tox
$ aci approve --stage contract -i ideas/tox     # a person, at a terminal
approve contract v1 of tox
  claim: Taker toxicity on venue A predicts 5s markouts on venue B
  ...
Freeze contract v1? [y/N]: y
frozen contract v1: bc1490ba55b1
$ aci next -i ideas/tox
agent      aci data register                    declare how this stage will be judged
```

## Templates

| Template | Research type | Layers |
| --- | --- | --- |
| `factor` | Cross-sectional factors | data, feat, signal, algo |
| `rule` | Rule-based strategies | data, signal, algo |
| `event` | News and LLM events | data, feat, signal, algo; confirmed on forward data only |
| `taker_arb` | Taker arbitrage | data, signal (optional), algo |
| `maker_arb` | Maker arbitrage and market making | data, feat, signal, algo with a fill model |

All templates end with the same gates: seal, forward, capital, monitor.

## Trust model

| Situation | Rule |
| --- | --- |
| Normal run in a lab | The engine is installed at a pinned git SHA and read-only. A freeze pins the contract, the engine source, the interpreter, the engine's own dependencies, and the upstream freezes; from M3 also evaluators, nulls, plugins, and the data snapshot. Each submission also records every installed package and `uv.lock`. |
| Anything frozen changes after approval | The submission is TAINTED. Research code and the libraries it uses may change between submissions. |
| Run from this repo in editable mode | Marked DEV, capped at L1. |
| Deliberate cheating | Out of scope for the MVP. Later: a separate OS user or a thin service `execute(bundle, vault, ledger) -> verdict`. |

Git is not a security mechanism, because agents can rewrite history. Trust comes from the engine's own freeze hashes and the hash-chained ledger.

The hash chain catches edited, deleted, and reordered records, but not records cut from the end, such as a failed submission deleted to win back a trial or an older ledger restored from git. For that the engine also keeps each idea's latest head in `$ACI_HOME/heads/` (default `~/.aci/heads/`), outside the idea and keyed by its first record, and refuses a ledger that no longer contains it. A ledger that grew elsewhere, for example pulled from another machine, is accepted as long as it still contains that head.

See [docs/concepts.md](docs/concepts.md) for the vocabulary, how a verdict is computed, and an example verdict card.

## Roadmap

| Step | Scope | Acceptance | Status |
| --- | --- | --- | --- |
| M0 | Scaffold, CI, import layers | CI green; layer violations are caught | Done |
| M1 | `tables` + `time` (asof, stepper) | Every planted lookahead is caught; plain rolling reports zero mismatches | Done |
| M2 | `holdout` + ledger + workflow skeleton + `aci next` | Out-of-order calls are rejected; a one-byte edit breaks ledger verification | Done |
| M3 | `judge`: null worlds, z thresholds, evidence levels, stand-ins | Feat layer end to end on the first real case | Planned |
| M4 | `runner`: child process + Arrow | Verdict cards stay byte-identical when research code reseeds or mutates data | Planned |
| M5 | First data plugin, consumer, minimal challenger | Plugins pass conformance | Planned |
| M6 | Algo: execution simulator, fill models, randomized decisions, constraint angles | A maker-taker case end to end | Planned |

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run aci --help
uv run pytest
uv run ruff check
uv run ruff format --check
uv run basedpyright
uv run lint-imports
npx markdownlint-cli2
```

```text
src/alpha_ci/
  api/  kernel/{time,holdout,checks,judge}/  stages/  workflow/
  runner/  store/  render/  llm/  cli/  reference_plugins/
tests/unit/
docs/
```
