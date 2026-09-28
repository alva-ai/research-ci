# Research CI design

Research CI is a backtest framework for research: an alpha claim counts only after it passes CI, the same way code does. This document is the complete design. It covers the problem, the architecture, how an idea moves from a claim to a verdict, how each guarantee is enforced, and why each choice was made. The [README](../README.md) has a quickstart, and [walkthroughs.md](walkthroughs.md) follows three cases step by step.

Every major section starts with a status line. "Implemented" means the code in `src/research_ci/` does it and tests cover it. "Planned" means it is designed but not built yet, and its commands and APIs may change when it is built. Numbers in examples are illustrative unless the text says they were measured.

Contents:

1. [Purpose and principles](#purpose-and-principles)
2. [Architecture](#architecture)
3. [Lifecycle](#lifecycle)
4. [Time and lookahead](#time-and-lookahead)
5. [Holdout and labels](#holdout-and-labels)
6. [Judging](#judging)
7. [The challenger and summaries](#the-challenger-and-summaries)
8. [Trust and reproducibility](#trust-and-reproducibility)
9. [Layers and templates](#layers-and-templates)
10. [Plugins](#plugins)
11. [Key decisions](#key-decisions)
12. [Prior art](#prior-art)
13. [Roadmap and open decisions](#roadmap-and-open-decisions)

## Purpose and principles

> **Status:** partly implemented: time, holdout, the ledger, freezes, and the workflow in M1 and M2; judging planned for M3.

### The problem

Research agents now produce ideas, code, and backtests faster than a person can check them. The usual loop is: an agent prepares data, writes a strategy, sweeps parameters, shows the best PnL curve, and explains why it works. Every judgment in that loop depends on context and gut feeling, and none of it can be reproduced. A review of our own failed research cases showed the cost. One case took about three months to falsify. A holdout marked as frozen had been used two or three times. A trivial baseline beat the idea four times. At least five ideas died of costs or infeasibility, and five never had enough data to detect the effect they claimed. Most of these could have been found with a few hours of the right checks, before any strategy was written.

Software has an answer to the same problem. Many coding sessions run in parallel, and a person merges when CI is green without rereading every line, because the checks are executable and the same for everyone. Research needs that kind of green light:

- A person must be able to trust it without reading the details. A written summary is not a verdict, least of all one written by the agent that did the work.
- It must work across many ideas and many agent sessions at once. A person makes only the few decisions that are theirs: approving the rules before any result exists, extending a budget, committing capital.
- It must judge every stage, not only the final PnL, so a wrong idea dies at the first stage that can show it is wrong.
- It must accept any kind of idea, structured or not: cross-sectional factors, futures spreads, cross-venue arbitrage, market making, prediction markets, news and LLM-generated events.
- Its rules must be simple, executable, and reproducible.

### What the engine owns

The engine does not judge methods; it judges outputs, the way a backtester does not care whether orders came from a neural network or a chart pattern. Research code is free: pandas or polars, rolling windows, sklearn, SHAP, numba, compiled extensions. The engine owns exactly three things:

| Duty | What it guarantees | Status |
| --- | --- | --- |
| Time | At every decision time, research code sees only rows with `available_at <= t`. | Implemented (M1) |
| Holdout and labels | Labels are built by the engine from the contract. Sealed and forward data are out of the agent's reach. | Tiers and label builders in M2; delivery through evaluators in M3 and M4 |
| Verdict | Real scores are compared with null worlds, thresholds rise with the number of trials, and every submission is logged in a hash-chained ledger. | Ledger in M2; judge planned (M3) |

An agent may do anything else. It may not do three things: see sealed or forward labels and results; report its own numbers as a verdict; change labels, evaluators, or thresholds after seeing results without starting a new version, which the ledger counts.

### Principles

1. **Falsifiable, executable, reproducible.** Before an idea starts, it states what result would prove it wrong. That standard is code, and the engine reruns the submission to apply it; numbers the agent reports and explanations offered afterwards do not count. The standard is frozen before any result is seen, and the same inputs always give the same verdict.
2. **Derive backwards, execute forwards.** Thresholds are derived backwards from the final trade (decision time, action, holding period, costs) into a contract. Work then runs forwards through the layers, and each layer may use only what passed the layer before it.
3. **Judgment moves before results.** Claims, evaluators, and null worlds are written and frozen before the run. Changing them after seeing results is allowed, but it starts a new version and is recorded.
4. **Constraints are code, not prose.** Markdown explains; code enforces. In every automated research system that works in practice, the judge is code (see [Prior art](#prior-art)).
5. **The engine is thin and dumb on purpose.** The kernel is a few thousand lines of Python with no LLM in it. The clever parts (which claims to make, which alternative explanations to rule out) come from agents outside the engine and are checked by fixed machinery inside it. The engine never interprets a feature.
6. **Unseen data is the final judge.** Every angle computed on the explore tier reuses the same data, so the results are correlated and are not independent evidence. The seal and forward tiers are.
7. **Guard against false greens first.** An evaluator that is too strict only hurts its author. The system must stop results that should fail from passing. Calibration still tests both directions, because a null world that kills everything is as broken as one that kills nothing.
8. **The engine is passive.** It knows the state, decides which commands are legal, and computes. Something outside always drives it: a research agent, a person, a scheduler, later an orchestrator. All of them read the same `next_actions()`.
9. **The MVP defends against accidents, not attacks.** It stops honest mistakes and lazy shortcuts. A process or service boundary is added when deliberate cheating has to be stopped (see [Threat model](#threat-model)).

### Vocabulary

| Term | Meaning |
| --- | --- |
| Idea | One claim and everything done to test it: a directory with `contract.py`, research code, and the engine's `.rci/` state. |
| Contract | The trade hypothesis written before any search. Today: template, claim, tier cutoffs, budget, skipped stages. Planned: decision time, action, holding period, costs, mechanism, alternative explanations. |
| Spec | What the engine derives from the contract (planned): decision times, label definition, event clusters, effective sample size, minimum detectable effect (MDE), economic threshold, and how long to wait if the sample is too small. |
| Stage | One step of an idea: the `contract`, a layer, or a checkpoint. |
| Layer | `data`, `feat`, `signal`, or `algo`. Each runs the same inner loop. Labels are not a layer, and the backtest is not a layer; it is the evaluator of `algo`. |
| Checkpoint | `seal`, `forward`, `capital`, or `monitor`: registered in advance, run once. |
| Template | The list of layers an idea goes through, fixed at `rci init`. |
| Output shape | What a layer produces: a FieldTable for `data`, then Scores, Events, Positions, Quotes, or Orders. |
| Evaluator | A function written by the agent that turns an output and labels into one number. |
| Angle | One pre-registered way of testing the claim, backed by an evaluator. `stat` angles are compared with null worlds; `limit` angles are hard limits compared directly. |
| Null world | A world with the claimed information removed and the alternative explanation kept. The real score must beat the scores in these worlds. |
| Stress world | A world where an optimistic assumption is broken: thinner depth, failed hedges, a replayed incident. Used by `algo`. |
| Calibration | `calibrate`, the test of the tests: stand-ins must fail, planted signals must pass, mutants must be caught. |
| StandIn | A fake output that contains only the alternative explanation. If it passes, the null world is too weak. |
| Planted | An output with known real information planted in it. If it fails, the null world is too strict. |
| Mutant | A copy of the data or of an evaluator with a defect planted in it, such as a shifted `available_at` or a level break. The tests must catch it. |
| Pitfall | A domain-specific mutant, kept in the plugin of the data source it belongs to. |
| Challenger | A separate LLM process, fed only files, that writes null worlds and pitfalls against the registered tests. It never judges. |
| Tier | explore (numbers visible), seal (pass or fail only, one use per idea), forward (data created after the freeze). |
| Trial | One submission. Every submission counts, across all versions of a stage. Exploration never counts. |
| Version | A stage's tests as frozen at one approval. Changing the tests after the freeze starts a new version. |
| Freeze | Hashes of everything a stage's verdict depends on, taken when a person approves the stage. |
| Ledger | The append-only, hash-chained record of every state change of an idea. State and trial counts are replayed from it. |
| Head anchor | The latest ledger hash, kept outside the idea in `$RCI_HOME/heads/`, so records cut from the end are noticed. |
| Lineage | The upstream stages and outputs a stage depends on. Reopening an upstream stage makes started downstream stages STALE. |
| Evidence level | L0 to L4: how much independent evidence a verdict has, with caps. |
| Cards | ContractCard, StageCard, VerdictCard, CapitalCard: one-screen views rendered from engine files, never written by an LLM. |

## Architecture

> **Status:** partly implemented: `api` (the contract), `kernel` (tables, time, hashing, holdout), `store`, `workflow`, and part of `cli` in M1 and M2; `stages` and `kernel/judge` planned for M3, `runner` for M4, `llm` from M5, `render` later.

### Packages

The engine separates mechanism, policy, and lifecycle. The kernel holds mechanisms as pure functions, `stages` holds the rules of each layer, and `workflow` holds the lifecycle. A new rule for one layer changes a policy file, not the kernel.

| Package | Owns | Status |
| --- | --- | --- |
| `api` | The only package research code and plugins import: `Contract` today; `Angle`, null and stress declarations, study entry points, and plugin protocols later. | `Contract` in M2 |
| `kernel` | Pure functions without I/O: `tables` (FieldTable), `time` (`asof`, `step`, `check_leaks`), `hashing`, `holdout` (tiers, labels), `checks`, and `judge` (derive, nulls, mutate, calibrate, stats, verdict). | `tables` and `time` in M1; `hashing` and `holdout` in M2; `checks` and `judge` planned (M3) |
| `stages` | One policy file per stage: `data.py`, `feat.py`, `signal.py`, `algo.py`, and one per checkpoint, `seal.py`, `forward.py`, `capital.py`, `monitor.py` (`DataPolicy`, `FeatPolicy`, ..., `MonitorPolicy`): floor checks, floor null worlds, the calibration recipe, caps. | Planned (M3) |
| `workflow` | The state machine, lineage, templates, `next_actions`, and the commands that apply transitions. | Implemented (M2) |
| `runner` | Research code in a child process, with data exchanged as Arrow. | Planned (M4) |
| `store` | All I/O: the `.rci/` workspace with atomic writes and a lock, the ledger with head anchors, freeze records, environment records; later the data vault and plugin discovery. | Implemented (M2); vault planned |
| `render` | Cards rendered deterministically from verdict files, and citation checks for summaries. | Planned |
| `llm` | Optional: the challenger, card narration, summary audit. | Planned (M5 and later) |
| `cli` | The `rci` command, a thin layer over `workflow`. | Partly implemented (M2) |
| `reference_plugins` | Synthetic data and a minimal backtester, used only by known-answer and reconciliation tests. | Planned |

### Import layers

```text
cli                                  the rci command
 |
llm                                  optional LLM code: challenger, narration, audit
 |
workflow                             state machine, lineage, templates, next_actions
 |
stages    store    runner    render  independent: none of them imports another
 |
kernel                               pure mechanisms: no I/O, no LLM, no plugins
 |
api                                  building blocks and plugin protocols

reference_plugins, and every plugin in a lab: may import api only
```

A package may import only packages below it. import-linter enforces both rules in CI (`uv run lint-imports`), and the configuration was checked by planting violations on purpose. Three consequences follow. The kernel, the stage policies, and the workflow cannot import `llm`, so the code that computes verdicts cannot call an LLM. Plugins never see engine internals, and the kernel never imports plugins; they will be discovered through entry points. The challenger is started by a command, not called from the state machine. import-linter checks only imports between engine packages; that the kernel uses no model client and no network rests on its dependency list and on review.

### Library and lab

The engine repository is a library and a CLI. It holds no ideas, no strategies, and no data-source code. A lab is any Python 3.11+ environment with the engine and a plugin package installed. Usually it is a separate lab repository, mostly a plugin package plus the home for new ideas; any other Python 3.11+ project can host ideas by installing both.

```text
research-ci/                     the engine: library and CLI only
  src/research_ci/               packages above
  tests/unit/, tests/known_answers/, tests/conformance/ (planned)

<lab>/                           a lab repository, or any Python 3.11+ project
  pyproject.toml, uv.lock        research-ci pinned to a git SHA; a person commits that line
  src/<lab_package>/plugins/     real plugins: data sources, instruments, consumers, fill models, pitfalls
  src/<lab_package>/shared/      shared research tools
  ideas/<idea>/
    contract.py                  written by the agent, frozen by a person
    data.py feat.py signal.py algo.py   research code, one module per layer (rci init generates them from M3)
    nulls/                       written by the challenger
    .rci/                        written only by the engine

$RCI_HOME (default ~/.rci)       per user, outside every repository
  heads/                         ledger head anchors (M2)
  data/                          the data vault: fields/<source>/*.parquet, forward/, cache/ (planned)
```

- The engine changes slowly and every verdict pins it, so it is reviewed and versioned on its own. Ideas and plugins change daily. Keeping ideas out of the engine repository also keeps the engine generic.
- A lab is meant to install the engine at a pinned git SHA. Upgrading the engine changes the `engine` hash in every freeze, so a stage frozen under the old engine must be registered and approved again before it can submit without being TAINTED.
- A lab may install the engine in editable mode while both are being developed. Verdicts made that way will be marked DEV and capped at L1 (planned, M3), so a lab switches to a pinned SHA before real verdicts.
- Both repositories commit `uv.lock` and `.python-version`, because the lab's lock file is part of the recorded environment of each submission.
- Older research repositories are not moved. Lab plugins read their data or call their backtesters.
- Research notebooks are optional; submissions are code (modules or scripts), and the engine calls only agreed entry points.

The `.rci/` directory of an idea holds `ledger.jsonl`, `contract.json` (a copy of the derived contract, the minimal ContractCard), `freeze/<stage>.json` (copies of each freeze), and `lock`, which `.rci/.gitignore` excludes. Verdict files, cards, and audit reports will join it with M3 and later. Every JSON file is written to a temporary file and moved into place with `os.replace`, so a crash never leaves half a file.

### Data model

Six tables carry everything:

```text
FieldTable   entity, field, event_time, available_at, value, vintage, source      all data (M1)
Panel        rows (time, entity), one column per field, at one moment = asof(t)   (M1)
Output       Scores, Events, Positions, Quotes, or Orders                          (M3 onward)
Labels       (t, entity, horizon) -> outcome, built by the engine from the contract (builders in M2)
Verdict      state, per-angle scores and z, evidence level, caps, preconditions,
             and the hashes of everything it depends on                            (M3)
Ledger       append-only, every record carries the hash of the one before it       (M2)
```

The engine judges five output shapes. Methods are unlimited; shapes are not.

| Shape | What it is | How it is judged | Layer | Typical research |
| --- | --- | --- | --- | --- |
| Scores | A number per entity at each decision time | IC, ICIR, residual IC against null worlds | `feat`, `signal` | Factors, ML models, LLM scores |
| Events | Marks at certain moments | Outcome after the event against random times or unselected events from the same source | `signal` | Event studies, news |
| Positions | Holdings, or entries and exits | Backtest PnL against random entries and exits at the same rate | `algo` | Rules, rotations, portfolios |
| Quotes | Resting orders placed, moved, and cancelled | Fill model plugin in an optimistic and a pessimistic version; live trading decides | `algo` | Maker arbitrage, market making |
| Orders | Orders sent now that take liquidity | Simulated fills on recorded books, then small live trading | `algo` | Taker arbitrage |

A cross-sectional study is just one use of Scores: at each time, rank entities and check whether higher scores are followed by better outcomes.

### Data flow

```text
DataPlugin.fetch -> FieldTable (vault) -> visible(tier) -> asof(t) at each decision time -> research code
                                              |                                              |
                                              +--> labels (engine) -----+              output (shape)
                                                                        v                    |
                                                  evaluators, one per angle <----------------+
                                                             |
                                          real score vs null worlds (and stress worlds)
                                                             |
                                  z vs z_need (n from ledger), calibration, caps -> Verdict
                                                             |
                                                ledger record -> VerdictCard

implemented today: FieldTable, visible, asof, step, check_leaks, labels, ledger
```

### Process model

From M4, research code never runs in the kernel's process. Until then, `step` and `check_leaks` call research functions in the same process, and `rci contract derive` imports `contract.py` in-process.

```text
kernel (parent)                                       research code (child)
      |                                                        |
      |-- verify freeze hash, lineage, version                 |
      |-- load data for the tier                               |
      |-- start one child with study code and plugins -------->|
      |                                                        |
      |   for each decision time t:                            |
      |-- window with available_at <= t, as Arrow ------------>|
      |<------------------------------------------ output at t |
      |                                                        |
      |-- build labels in the parent                           |
      |-- evaluate(angle, output, labels the tier allows) ---->|
      |<------------------------------------------------ score |
      |                                                        |
      |-- null worlds, calibration, verdict, ledger            |
```

Seeds are derived from the freeze hash, and the child runs with a timeout and a memory limit. A stray `np.random.seed`, an in-place edit of shared data, or a monkeypatch in research code cannot reach the kernel. The acceptance test for M4 is that verdict cards stay byte-identical when research code reseeds or mutates its inputs. Arrow is the exchange format because both pandas and polars read it without conversion.

The process boundary is also where a trusted service would be split out later: `execute(bundle, vault, ledger) -> verdict`. The bundle is the frozen research code and declarations, the vault holds the holdout data, and the ledger is the append-only record. Today all three are local: the idea directory, the local data, and `ledger.jsonl`. The service would be a job executor, not a long-running platform. Status: later.

## Lifecycle

> **Status:** partly implemented: the state machine, versions, budgets, lineage, templates, `next_actions`, and the commands `init`, `status`, `next`, `inbox`, `verify`, `contract derive`, and `approve` in M2; per-layer CLI commands planned for M3; checkpoint commands later.

### Outer pipeline

```text
             reverse derivation: thresholds flow back from the final trade
   <-------------------------------------------------------------------------------
contract -> data -> feat -> signal -> algo -> seal -> forward -> capital -> monitor
   ------------------------------------------------------------------------------->
             forward execution: each stage uses only what passed before it

   |<-------- CI: layers, inner loops -------->|<--- CD: one-shot checkpoints --->|
                         ^                                                 |
                         +------- a drift rule reopens the matching layer --+
```

Every idea has one `contract`, the layers of its template, and the same four checkpoints. The template is chosen at `rci init` and fixed; the template named in `contract.py` must match it. The templates are listed in [Layers and templates](#templates). A later stage that finds a problem sends the idea back: a data problem found in `feat` reopens `data`, a failed checkpoint reopens a layer, and a drift rule in `monitor` reopens the layer it points at.

Each stage is one of three kinds (the contract, a layer, or a checkpoint), and each kind has its own state machine.

### Inner state machine

```text
contract

  OPEN --derive--> REGISTERED --approve (a person)--> PASSED
                    ^    |                              |
                    +----+ derive again                 | derive again: new version,
                    ^      (same version)               | every started stage -> STALE
                    +-----------------------------------+

layer: data, feat, signal, algo

  OPEN
   |  register
   v
  REGISTERED <------------------------------------------------+
   |  challenge                                               |
   v                                                          |
  CHALLENGED <------------------------+                       |
   |  check: calibration passed       |  check failed, or     |  register after the
   v                                  |  challenge again      |  freeze: new version
  CHECKED ----------------------------+  (new null worlds)    |  (from FROZEN, PASSED,
   |  approve: a person at a terminal                         |  FAILED, TAINTED, STALE)
   v                                                          |
  FROZEN -----------------------------------------------------+
   |  submit: 1 trial, the verdict is recorded at once
   v
  PASSED ............ the next stage may start
  FAILED, TAINTED ... change the research code and submit again (1 more trial)
  PAUSED ............ the budget ran out without a pass; a person extends it
  STALE ............. an upstream stage was reopened; register again (new version)

checkpoint: seal, forward, capital, monitor

  OPEN or STALE --register--> REGISTERED --approve (a person)--> FROZEN --run once--> PASSED
                                                                                      FAILED or TAINTED:
                                                                                      no rerun, reopen a layer
```

The full transition table, from `workflow/machine.py`:

| Kind | Command | Allowed from | Result |
| --- | --- | --- | --- |
| contract | register (`rci contract derive`) | OPEN, REGISTERED, PASSED | REGISTERED. From PASSED this is a new version, and every started downstream stage becomes STALE. |
| contract | approve | REGISTERED | PASSED. The contract's budget and `skip` list take effect. |
| layer | register | OPEN, REGISTERED, CHALLENGED, CHECKED | REGISTERED, same version |
| layer | register | FROZEN, PASSED, FAILED, TAINTED, STALE | REGISTERED, new version. From PASSED, started downstream stages become STALE. |
| layer | challenge | REGISTERED, CHALLENGED, CHECKED | CHALLENGED |
| layer | check | CHALLENGED, CHECKED | CHECKED if calibration passed, otherwise CHALLENGED |
| layer | approve | CHECKED | FROZEN |
| layer | submit | FROZEN, FAILED, TAINTED | PASSED, FAILED, or TAINTED. PAUSED instead of FAILED or TAINTED when this submission used the last trial. |
| layer | extend | PAUSED | The verdict of the last submission, FAILED or TAINTED |
| checkpoint | register | OPEN, REGISTERED, STALE | REGISTERED. From STALE this is a new version. |
| checkpoint | approve | REGISTERED | FROZEN |
| checkpoint | submit (the checkpoint run) | FROZEN | PASSED, FAILED, or TAINTED |

Everything else is refused with a message such as `cannot approve a layer stage that is OPEN (allowed from: CHECKED)`. Examples: submitting a stage that has PASSED, registering a PAUSED layer, running a checkpoint twice. Before the machine is even consulted, every command checks lineage: all upstream stages must be PASSED or SKIPPED, otherwise `feat is waiting for contract, data to pass`.

The tests hold three invariants over every allowed transition: only a submission (or the approval of a contract) can produce PASSED; a submission always ends in PASSED, FAILED, or TAINTED at the machine level; and no command produces SKIPPED directly. There is no SUBMITTED state: a submission runs synchronously and records its verdict in the same step.

### Versions

Every stage starts at v1. Registering a stage before its freeze (from OPEN, REGISTERED, CHALLENGED, or CHECKED) edits the current version. Registering it after the freeze (from FROZEN, PASSED, FAILED, TAINTED, or STALE) starts the next version, and the ledger records it. This is how "changing the tests after the freeze is a new version" is enforced.

Research code is different. After a FAILED or TAINTED verdict, the agent changes the research code and submits again against the same frozen tests; that is the normal loop and does not start a new version. Only the tests are frozen.

### Budgets and PAUSED

The contract's `budget` (default 20) is the number of submissions each layer may use across all its versions. Every submission counts, including TAINTED ones. A new version does not reset the count, so a failing idea cannot buy fresh trials by re-registering.

When a submission that does not pass uses the last trial, the layer becomes PAUSED instead of FAILED or TAINTED. A PAUSED layer accepts only `extend`; even `register` is refused. A person runs `rci approve --stage <layer> --extend N` at a terminal, and the layer returns to the verdict of its last submission, so the agent can submit again. Checkpoints have no budget, because they run once.

```text
$ rci status -i ideas/tox
tox (factor)
  contract  PASSED      v1
  data      PAUSED      v1   3/3
  feat      OPEN        v1   0/3
  signal    OPEN        v1   0/3
  algo      OPEN        v1   0/3
  seal      OPEN        v1
  forward   OPEN        v1
  capital   OPEN        v1
  monitor   OPEN        v1
$ rci next -i ideas/tox
human      rci approve --stage data --extend N  all 3 submissions used; extend the budget or drop the idea
```

The budget comes from the latest contract approval in the ledger, plus every extension of that layer. A person could also raise it by deriving and approving a new contract version, but that reopens the contract and makes every started stage STALE.

### Lineage and STALE

Every command requires all upstream stages to be PASSED or SKIPPED. Registering a stage that has PASSED reopens it; the same holds for deriving the contract again. Every downstream stage that has started (anything but OPEN, SKIPPED, or already STALE) becomes STALE, with a ledger record noting `data was reopened`. A STALE stage waits until the reopened stage passes again and then must be registered again, which starts a new version.

A second lock sits in the freeze. Each stage's freeze includes `upstream:<stage>`, the freeze hash of each upstream stage's latest approval. If an upstream stage is approved again, any downstream submission against the old freeze would be TAINTED even without the STALE mark.

From M3, each verdict also records the hashes of the upstream outputs it consumed, and a stage may only use upstream outputs that passed. For example, `signal` may only read feature versions that passed `feat`; it cannot compute a new, unjudged feature of its own.

### Optional stages

A template may mark stages as optional. Today only `taker_arb` does, for `signal`. The contract lists the stages it skips in `skip`, and `rci contract derive` refuses any stage the template does not mark as optional. The skip takes effect when the contract is approved: the stage becomes SKIPPED, counts as done for lineage, and is left out of downstream freezes. If a later contract version stops skipping it, the stage returns to OPEN.

### Checkpoints run once

A checkpoint is registered with its plan in advance: the k candidates for `seal`, the experiment design for `forward`, the capital plan for `capital`, the drift rules for `monitor`. A person approves the plan, and the checkpoint runs once. A checkpoint that ends FAILED or TAINTED cannot run again. `rci next` then asks a person to reopen a layer (`rci <layer> register`); that marks the checkpoint STALE, and it is registered again after the layers pass. A live experiment is not changed while it runs; a changed experiment is a new registration.

`seal` and `capital` are run by a person (`rci seal run`, `rci capital run`), and `forward` and `monitor` by a scheduler (`rci forward --all`, `rci monitor --all`). The state machine and `next_actions` already handle checkpoints; the commands are planned for later. In the M2 machine, `monitor` has the same one-shot shape as the other checkpoints; how a scheduled, ongoing monitor maps onto that shape is an [open decision](#open-decisions).

### Exploration

Exploration is allowed in any state. It never changes the state and is never counted as a trial. On the explore tier every number is shown, as often as the agent wants (planned command: `rci <layer> explore`, M3).

Exploration is free because it is how agents find ideas; counting it would only push work off the record. It is safe because it never produces a verdict, the tests are frozen before any submission, and the seal and forward tiers are out of reach. One weakness remains: an agent can look at explore results and then choose which angles to register. The seal and forward checkpoints are the backstop.

### Who drives the loop

The engine is passive. `next_actions(status)` returns what should happen next as `Action(who, stage, kind, command, reason)`, where `who` is one of `agent`, `human`, `challenger`, or `scheduler`. It looks at the first stage that is not PASSED or SKIPPED and returns one action for it, so an idea moves through its stages in order. `rci next` prints it; `rci next --json` gives it to agents.

| Stage | State | Who | Command | Reason printed |
| --- | --- | --- | --- | --- |
| contract | OPEN | agent | `rci contract derive` | write contract.py, then derive the contract |
| contract | REGISTERED | human | `rci approve --stage contract` | review the contract and freeze it |
| layer | OPEN | agent | `rci <layer> register` | declare how this stage will be judged |
| layer | REGISTERED | challenger | `rci <layer> challenge` | write the nulls and pitfalls |
| layer | CHALLENGED | agent | `rci <layer> check` | the note of a failed check, or the calibration rule |
| layer | CHECKED | human | `rci approve --stage <layer>` | review the tests and freeze them |
| layer | FROZEN | agent | `rci <layer> submit` | submit the research code, with trials left |
| layer | FAILED, TAINTED | agent | `rci <layer> submit` | the verdict and its note; change the research code |
| layer | PAUSED | human | `rci approve --stage <layer> --extend N` | all submissions used; extend the budget or drop the idea |
| layer | STALE | agent | `rci <layer> register` | which stage was reopened; register to start the next version |
| checkpoint | OPEN, STALE | agent | `rci <checkpoint> register` | declare the plan before the checkpoint runs |
| checkpoint | REGISTERED | human | `rci approve --stage <checkpoint>` | review the plan and freeze it |
| checkpoint | FROZEN | human or scheduler | `rci seal run`, `rci forward --all`, `rci capital run`, `rci monitor --all` | run the checkpoint once |
| checkpoint | FAILED, TAINTED | human | `rci <layer> register` | the checkpoint ran once; reopen a layer, which restarts the checkpoints |

The per-layer and checkpoint commands in this table do not exist yet (M3 and later), although `rci next` already prints them. Who does what:

```text
research agent           challenger              person                 scheduler
(any LLM session)        (separate process,      (at a terminal)        (cron, launchd)
derive, register,         files only)            approve, extend,       forward, monitor
explore, check, submit   challenge               seal run, capital run
      |                       |                        |                      |
      +-----------------------+------------+-----------+----------------------+
                                           |
                                    rci CLI (thin)
                                           |
                     workflow: state machine, lineage, next_actions()
                                           |
         kernel + stages (deterministic, no LLM)      store (.rci/, ledger, anchors)
```

### Running many ideas in parallel

Each idea is a directory with its own `.rci/`, ledger, and anchor. Commands on one idea are serialized by a file lock (`.rci/lock`); different ideas share no state. Any number of agent sessions can therefore work on different ideas at once, each reading `rci next --json` for its own idea. Within one idea the stages run in order, by design.

A person does not visit ideas one by one. `rci inbox [root]` scans `<root>/.rci`, `<root>/*/.rci`, and `<root>/*/*/.rci` for ledgers and lists only the actions that wait for a person, across all ideas. An idea whose ledger fails verification is listed too, as an error item.

A later orchestrator needs nothing new: it loops over ideas, reads `next_actions`, and dispatches agents, the challenger, and scheduled jobs.

### Commands that exist today

`rci init <idea> -t <template>` creates the idea, writes a `contract.py` skeleton, and opens the ledger with an `init` record. `rci contract derive` loads and validates `contract.py`, checks its template and `skip` list, and records every field in the ledger; the derivation of the Spec arrives in M3. `rci approve --stage <stage>` shows everything it will freeze, asks for confirmation, freezes, and prints a suggested git command; with `--extend N` it adds submissions to a PAUSED layer instead. `rci status`, `rci next`, and `rci inbox` show the state, the next action, and the person's queue, with `--json` for agents. `rci verify` checks the ledger and prints `ledger ok: N records`. `status`, `next`, `verify`, `contract derive`, and `approve` take `-i <idea>` and default to the current directory; `inbox` takes a root directory. The [README](../README.md#cli) has the full table.

`Contract(template, claim, explore_end, seal_end, budget=20, skip=())` refuses an empty claim or one that still starts with `TODO`, timestamps without a time zone, a `seal_end` that is not after `explore_end`, and a budget below 1.

## Time and lookahead

> **Status:** implemented (M1): FieldTable, `asof`, `step`, `check_leaks`. Decision times derived from the contract are planned (M3).

### FieldTable

All data enters as one long table. `field_table()` accepts pandas or polars and normalizes it:

| Column | Type after normalization | Required | Meaning |
| --- | --- | --- | --- |
| `entity` | String | yes | What the value is about: a contract, a stock, a venue pair |
| `field` | String | yes | The name of the quantity |
| `event_time` | Datetime, ns, UTC | yes | The moment the value describes |
| `available_at` | Datetime, ns, UTC | yes | The moment we could know it: the publication time, or the local receive time for market data |
| `value` | Float64 | yes | The value |
| `vintage` | Int64 | no, default 0 | The revision number |
| `source` | String | no, default empty | Where the row came from, such as a plugin and its version |

Normalization rules:

- Both time columns must be timezone-aware; they are converted to UTC and cast to nanoseconds. pandas 3 defaults to microseconds, so the cast is explicit.
- `entity`, `field`, `event_time`, `available_at`, and `vintage` must not be null.
- `(entity, field, event_time, vintage)` must be unique.
- A revision must not become available before the vintage it revises.
- Values are cast to Float64, so non-numeric data must be encoded as numbers.
- The result is sorted by `available_at` and flagged as sorted, which `asof` uses for speed.

`available_at >= event_time` is not enforced here; it is one of the data layer's floor checks (planned). The design also adds `gen_mode` and `model_cutoff` for LLM-generated data (see [Data generated by LLMs](#data-generated-by-llms)); they are not in the schema yet.

### Point-in-time panels

`asof(fields, t, lookback=None)` returns everything known at `t` as a wide polars frame: columns `time`, `entity`, then one column per field. Rows are event times. Only rows with `available_at <= t` are visible, and for each `(entity, field, event_time)` the latest vintage published by `t` wins. `lookback` keeps only event times after `t - lookback`. Every field of the table gets a column in a fixed, sorted order, even before its first row is visible, so research code always sees the same shape.

Two examples from the tests. An inventory figure for `T` published at `T + 30h` is invisible at `T + 29h` and visible at `T + 30h`, on the row for `T`. A figure first published at `T` and revised at `T + 10h` shows the first value until `T + 10h` and the revision from then on. Property tests check that `asof` matches a brute-force implementation and does not depend on row order.

### Stepping

`step(features, fields, decision_times, lookback=None, panel_format="pandas")` calls the research function once per decision time, on the panel known at that time, and keeps each entity's latest output row at or before `t`. The result has columns `t`, `entity`, and one column per output value. Research code can be written vectorized over the whole panel, as usual; causality comes from how it is called, not from how it is written. The values used for verdicts come from stepping on the real table.

The output contract: pandas features return a Series or DataFrame indexed by `(time, entity)`; polars features return a frame with `time` and `entity` columns. Values are cast to float and NaN becomes null. A column named `t` is reserved.

### The leak check

Stepping is causal by construction, so why check at all? Because research code is explored on full history. If its value at `t` depends on rows after `t`, the numbers the agent saw while exploring are not the numbers the same code produces live. The design treats such a submission as TAINTED instead of quietly judging different numbers. `check_leaks` finds these cases by comparing stepping with one run on the full history.

A naive comparison on the real table is wrong. Take a weekly inventory field published 30 hours after its event time, and a perfectly causal feature that forward-fills it. In a full run the value sits at its event time and fills forward from there. Stepping at a decision time 10 hours after the event time cannot see it yet, so the two runs differ and causal code looks like a leak. Revisions cause the same problem: the full run sees the final vintage, stepping sees the vintage known at `t`.

So the check runs on an availability-timed copy of the table:

1. Keep only the first vintage of each `(entity, field, event_time)`.
2. Move its `event_time` to its `available_at`. If several observations of the same entity and field become available at the same instant, keep the one with the latest event time.
3. On this copy every row sits at the moment it became known, so code that reads only past rows gives the same answer stepped or run once on everything.
4. Step, run once on everything, and compare at every decision time with `rtol=1e-9` and `atol=1e-12`; null equals null.

A mismatch can also mean that the declared lookback is too short, which is not cheating. When a lookback was declared, the engine reruns stepping without it at the decision times that mismatched. If the mismatches disappear, the report says `lookback` and the agent should declare a longer window. If any remain, it says `lookahead`. The report (`LeakReport`) gives the kind, the number of decision times checked, the number of mismatches, the first one (time, entity, column, stepped and full values), and counts per column.

The known-answer canaries run in both panel formats:

| Planted lookahead, must be caught | Causal feature, must be clean |
| --- | --- |
| `shift(-1)` | trailing rolling mean |
| z-score over the full sample | expanding mean |
| centered rolling mean | exponentially weighted mean |
| backward-filling a sparse field | percentage change |
| dividing by the future maximum | forward-filling a sparse field (published with a 30-hour lag) |

A trailing mean with a lookback shorter than its window is reported as `lookback`, and `shift(-1)` with a short lookback is still reported as `lookahead`.

### What the leak check reports

Only leaks that change a value at a decision time are reported. If an entity's all-time maximum occurs before the first decision time, then dividing by the future maximum gives the same value at every decision time as dividing by the past maximum. The check does not report it, and it does not affect the verdict either, because stepped values are what the verdict uses. The canary data drifts upward so that this kind of leak does show.

The check runs the code on the availability-timed copy, not on the real table. Its purpose is to find code that reads rows after `t`; it does not certify anything else about the real run.

### Panel formats

The kernel runs on polars. Research code chooses its panel with `panel_format`: `"pandas"` (the default) gives a wide DataFrame indexed by `(time, entity)`; `"polars"` gives a plain frame with `time` and `entity` columns. The known-answer tests run in both formats, and a control test requires the same feature written both ways to give the same values. Statistics that affect verdicts will be computed with numpy in a fixed order (M3), not with polars parallel aggregations, so that verdicts are reproducible bit for bit.

### Performance

`asof` uses a binary search on the sorted `available_at` column to slice the visible prefix, and falls back to a filter when the sorted flag is missing. The stepper computes the field list once per run. Measured during M1: on 1.2 million rows with a 168-hour lookback, one decision time costs 10.3 ms, against 35 ms for the same operation written in pandas. Incremental computation across decision times is left for later.

### Decision times

Today `step` takes decision times as given. From M3, the contract derivation computes them. The decision time is when the action could still happen, which is often earlier than the event. A maker quote that must be cancelled before a toxic taker trade arrives has its decision time at trade time minus the cancel latency (for example its p95), from a latency table the data layer provides. A trade on a weekly report is decided at the first tradable time after publication. A text signal is decided at its publication time, not at the time of the news behind it.

## Holdout and labels

> **Status:** partly implemented: tiers, `visible`, decision windows, and the `holding_return` and `markout` labels in M2; label delivery through evaluators and walk-forward in M3 and M4; vintage-aware labels and the LLM data rule planned.

### Tiers

`Tiers(explore_end, seal_end)` holds the cutoffs from the contract, in nanoseconds UTC.

| Tier | Rows visible (`visible`) | Decision times allowed (`decision_window`) | What the agent gets |
| --- | --- | --- | --- |
| explore | `available_at <= explore_end` | `t <= explore_end` | Every number, any number of runs |
| seal | `available_at <= seal_end`, including explore history | `explore_end < t <= seal_end` | Pass or fail only, for k candidates declared in advance; each sealed segment once per idea |
| forward | Everything; the freeze time is required | `t > frozen_at` | Data created after the freeze, collected on a schedule |

`require_in_window` rejects decision times outside the tier's window, and the windows never overlap. Tiers cut by `available_at`, not by `event_time`, because what matters is when a value became known: a value about an explore-period event that was published after `explore_end` is not visible in explore.

Planned behavior around the tiers: sealed data is not even downloaded while the idea is exploring; the ledger records when a sealed segment has been used, and a failed seal sends the idea back to a layer, where confirmation has to wait for new data; forward data is appended to the vault on a schedule by `rci forward --all`.

### Labels as known at the moment

Two label builders exist:

- `holding_return(fields, price_field, decision_times, horizons)`: `p(t + h) / p(t) - 1` for every entity, decision time, and horizon.
- `markout(fields, price_field, events, horizons)`: `side * (p(t + h) / p(t) - 1)` in basis points, for each event row `(t, entity, side)`.

Each price `p(x)` is the price as it was known at moment `x`:

- The time axis is `available_at`, not `event_time`.
- Only the first vintage of each price counts; later corrections were not known at the time.
- Only records that advance the latest `event_time` of the entity count, so a late-arriving older observation does not change the known price.
- `p(x)` is the latest such price known at `x`.

For example, a price for `T` known at `T` and the next price, for `T + 1h`, known only at `T + 3h`: the two-hour return from `T` is 0, because the new price was not known yet, and the three-hour return reflects it. Labels use first vintages only for now, because the price a trader saw is the first print; vintage-aware labels, for outcomes whose revisions matter, are planned.

### Missing horizons and purging

If no price for an entity became known at or after `t + h`, the data does not yet show what the price was at `t + h`, and the label is null. The row is kept, so the missing label is visible. An entity whose known prices do not cover `t` itself (none known yet, or none known at or after `t`) gets no row.

This rule makes a separate purge at the tier boundary unnecessary. Labels on `visible(fields, EXPLORE, tiers)` can never reach past `explore_end`: any label that would need a price published after the cutoff is null. The tests check exactly this.

Walk-forward is a different case and still needs purging (planned, M3). Code that learns from labels gets them only through `fit`. The engine passes `fit` only the labels of the training folds, and it drops training samples whose label window overlaps the test period.

### Who sees labels

Labels belong to the engine. Research code never builds the labels its verdict is judged on. The label definition is part of the contract, so the holding period cannot be changed after results; changing it is a new contract version, recorded in the same idea's ledger. From M3 and M4, labels reach research code only through evaluators and `fit`, and only the labels the current tier allows. Labels can be split by component, such as entry spread, exit cost, funding, and fees. Label types that need simulated fills call a plugin.

### Data generated by LLMs

LLM-generated data carries a special risk: a model that knows how events turned out can write "predictions" that look perfect. The data layer will apply a floor rule (planned):

- `available_at` is the actual publication time of the generated record.
- A record is usable for verdicts only if its event is after the model's knowledge cutoff and it was generated live, not in one retroactive pass over history.
- Everything else is explore-grade, which caps any verdict that uses it at L1.

Each record is judged against the context as it was at the time, for example the version of an author's thesis that a signal referred to, not the edited version that exists today. Data plugins must also return archived and closed objects, or failures disappear from the sample. The [text event walkthrough](walkthroughs.md#text-event-signals) shows these rules on a case.

## Judging

> **Status:** planned (M3), with the `algo` parts (fill models, `RandomizeTiming`, stress worlds, limit angles) in M6. Nothing computes a verdict today; the M2 workflow records verdicts passed in by tests.

### Evaluators and angles

The agent writes evaluators: functions that turn an output, labels, and a context into one number. Each is registered in advance as an angle and frozen at approval. A sketch of the planned API:

```python
from research_ci.api import Angle  # planned (M3); the final API may differ


@Angle("increment_over_b_move", kind="stat")
def resid_ic(scores, labels, ctx) -> float:
    return rank_ic(residualize(scores, ctx.b_move_200ms), labels.markout_60ms)
```

`Angle(kind="stat")` is compared with its null distribution. `Angle(kind="limit")` is a hard constraint, such as "the 99th percentile loss per cycle stays within the limit", compared directly with no null distribution.

The engine never reads what an evaluator means. It runs the same evaluator in the real world and in the null worlds and compares the numbers. Two properties follow. Any metric works, so the variety of ideas does not require the engine to understand them. And any search hidden inside an evaluator runs again in every null world, so its benefit is priced in automatically.

On top of the agent's angles, each stage policy adds floor checks and floor null worlds that the agent cannot remove. The agent's own tests follow four rules: written before running, allowed only to tighten the floor, executed by the engine, and able to fail, which calibration proves.

### Alternative explanations become null worlds

Every claim has alternative explanations: "it is only X". For each one, a null world keeps X and destroys only the information that is claimed. The real score must beat that world. If a null world is too weak, anything beats it and false greens appear, so each explanation needs the right generator:

| "It is only..." | Null generator | Keeps | Destroys |
| --- | --- | --- | --- |
| chance | `BlockPermute`: permute labels by time blocks or event clusters | autocorrelation | the pairing of output and outcome |
| a shape that happens to match | `ShiftTime` by a random offset that is not a whole year | the output's own structure | its alignment in time |
| seasonality | `ShiftTime` by whole years | the seasonal pattern | information specific to the year |
| timing or beta | `PermuteWithin(bucket_by=time)`: shuffle across entities at each time | each time's overall level | differences between entities |
| the group (sector, venue) | `PermuteWithin(bucket_by=group)` | the group effect | differences within the group |
| the obvious variable X | `PermuteWithin(bucket_by=X)`, moving whole clusters | the relation to X | information beyond X |
| clock skew | `ShiftTime` on one venue, within the measured skew | everything else | a lead created by misaligned clocks |
| any news at all | `RandomEvents`: unselected events from the same pool, same entity and density | the news flow | the selection |
| the mechanism is not needed | `ReplaceField`: scramble the mechanism inputs | all other inputs | the claimed cause |
| parameter search | `GridMax`: run the same grid in each null world and keep its maximum | the grid | the luck of picking the best cell |
| trading or cancelling less often | `RandomizeTiming(keep="rate")`, `algo` only | the decision rate | the decision timing |

`PermuteWithin` expresses most "it is only X" explanations. It always moves whole event clusters, so dependence inside a cluster survives. A bucket variable may be something that is not known at decision time, such as the size of the incoming trade: that cannot be a feature, because using it is lookahead, but it can bucket outcomes on the evaluation side.

Null worlds differ in cost. Worlds that only reorder labels or outputs (block permutation, shuffles within buckets) are cheap and run 200 times or more. Worlds that perturb inputs and recompute features (clock shifts, ablation, latency quantiles) are expensive and run 30 to 50 times in parallel. Sequential testing can stop early when a result is clearly failing.

Every alternative explanation listed on the ContractCard must map to at least one null world. Choosing the wrong generator produces false greens, for example ruling out seasonality with a random shift instead of a whole-year shift; the challenger and calibration exist to catch that.

### Stress worlds

In `algo`, most alternative explanations say that an optimistic assumption is false. `Stress` worlds break that assumption and rerun the backtest:

| "It only works because..." | Stress world |
| --- | --- |
| the simulation hands us all the depth | halve the depth on the hedge venue |
| hedge failures were ignored | inject rejects and timeouts at the measured rate |
| the size was picked to fit | a plateau over the size parameter |
| markets were calm | replay real incidents: a feed that lags for minutes, cancel latency jumping to 500 ms, 90% of the depth vanishing at once |
| venues never fail together | disconnect both venues at the same time |
| costs were underestimated | double the costs |
| fills were too optimistic | cut the pessimistic fill probability by another 30% |

Every real incident becomes a stress world that similar strategies must pass later. This is the same kind of accumulation as the pitfall library of the data layer.

### Calibration: the test of the tests

`calibrate` runs at `rci <layer> check`, before the freeze. A stage becomes CHECKED only when every test and null world passes it; failures are sent back and never count as trials.

- **StandIn**: an output that contains only the alternative explanation, for example `rank(X)` plus noise with the same correlation to X as the real feature. It must be killed in at least 95% of 50 runs. If it survives, the null world is too weak.
- **Planted**: the stand-in plus real information, `stand_in + lambda * residual(label on X)`, sized so its residual IC is about 1.5 times the MDE. It must pass in at least 80% of runs. If it is killed, the null world is too strict. It uses labels on purpose and is never a real feature.
- **Preservation**: a null world must keep what it claims to keep; the correlation of the permuted output with X must stay close to the original.
- **Mutant**: a copy of the data or of an evaluator with a planted defect. Each data test declares which risks it `covers`, must pass on clean synthetic data, and must fail on every mutant it covers. A mutant must be harmful: a naive feature on the mutated data must deviate clearly, for example show an inflated IC after a planted leak.
- **Coverage**: a risk flagged on a field, with no test that fails on the matching mutant, is a coverage gap, and approval is refused until it is closed.

In `algo`, calibration changes shape: a stand-in strategy that trades at random at the same rate must have a false pass rate of at most alpha against the random-timing null; an unprotected stand-in (no stops, no failure handling) must breach its limits in the stress worlds, which proves the stress is strong enough; and the execution simulator must pass known-answer tests (see [The algo layer](#the-algo-layer)). The rates, repetition counts, and sizes above are the planned recipe and may change in M3.

The [maker-taker walkthrough](walkthroughs.md#cross-venue-maker-taker) shows a calibration failure: a clock-skew null written too narrow let a stand-in survive and was sent back to the challenger before anything counted.

### Thresholds and trial counting

For a `stat` angle, the engine turns the null distribution into a z score and compares it with a threshold that rises with the number of trials:

```text
z      = (real - mean(null)) / sd(null)
z_need = z_base * sqrt(1 + ln n)
```

`n` is the number of submissions counted in the ledger for that stage, across all versions and including the current one. With `z_base = 2.0` as an example:

| n | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `z_need` | 2.00 | 2.60 | 2.90 | 3.23 | 3.63 | 4.00 | 4.43 |

Thresholds are z scores, not p-values, because permutation p-values run out of resolution: with 200 permutations the smallest p is about 0.005, and `z = 2.6` already corresponds to a one-sided p of about 0.005. After a few trials, a p threshold would ask for more precision than the permutations can give. The empirical rank of the real score in the null distribution stays as a secondary check. The threshold grows with the square root of `ln n`, the same rate at which the best of n independent z scores grows. The value of `z_base` is not fixed yet.

Every submission counts as a trial, including TAINTED ones and resubmissions of the same idea under a new look (see [Dedup](#dedup)). Exploration and calibration failures never count.

### Combining angles

An evidence level requires every angle registered for it to pass. The claim is the conjunction of its angles, which makes this an intersection-union test: if the claim is false, at least one angle's null hypothesis is true, and that angle alone passes with probability at most its level, so all of them together pass with at most that probability. The angles need no multiple-testing correction among themselves. The multiplicity that does need correction is the number of versions tried, and the trial count handles it. Failed angles stay on the card.

### Parameter grids and robustness

A parameter grid is declared in advance as a family. Each null world runs the whole grid and keeps its maximum (`GridMax`, in the style of White's Reality Check), so the threshold grows with the grid automatically, and the whole grid counts as one trial. The best cell must also sit on a plateau: enough of its neighbours must pass too. Exposed parameters, such as a latency quantile, are run at each registered value and judged by a rule registered in advance, for example "p95 must pass, p99 is informational".

### Dedup

A new output whose correlation with an existing one, such as an earlier version of the same idea, is above 0.9 counts as the same idea. It still costs a trial. This stops an agent from submitting the same signal under a new name to win a lower threshold.

### Evidence levels and caps

The draft levels:

| Level | Requires |
| --- | --- |
| L0 | Results on the explore tier only |
| L1 | Predictive power, increment over what is known, and stability under walk-forward |
| L2 | Plus mechanism ablation and a secondary prediction |
| L3 | Plus economic value after the consumer and costs |
| L4 | Plus at least one new-data angle: an independent source, the sealed tier, forward data, or live trading |

Caps:

- A DEV run (from an editable engine) is capped at L1.
- A verdict that uses explore-grade data is capped at L1.
- Quotes that pass only under the optimistic fill model are capped at L2.
- A strategy whose outcomes depend on its own actions is capped at L3 until live results exist.

Downstream stages consume by level; for example, `algo` may only use features at L2 or above, and capital requires L4. L4 needs new data because angles on the same data are not independent evidence. The final definitions follow the first real case in M3.

### Verdict states

A verdict is one of `PASS`, `FAIL`, `INSUFFICIENT` (the sample is too small to detect the claimed effect; the card says how much longer to collect at the current rate), `TAINTED` (something frozen changed, or a lookahead was found), `STALE` (an upstream stage was reopened), or `DEV` (made with an editable engine). In the M2 workflow a submission records PASSED, FAILED, or TAINTED, and STALE is a stage state. How INSUFFICIENT and DEV map onto stage states is decided in M3.

### Example verdict card

An illustrative card for a maker-taker toxicity feature. All numbers are made up.

```text
feat . toxicity score (B lead + book imbalance + staleness)     trial 3 of 20    z_need 2.90
decision time   trade time - 35 ms (p95 cancel latency)          leak check: 0 mismatches
sample          41,200 taker trades -> 3,100 independent clusters
calibration     stand-ins 5/5 killed, planted 5/5 kept
lineage         data v1 passed (freeze 3f2a91c0...)
--------------------------------------------------------------------------------------------
angle                     null world                          z     result
predictive                BlockPermute by cluster             4.1   pass
increment over B move     PermuteWithin(B move bucket)        3.2   pass
economic (drop top 20%)   PermuteWithin(B move bucket)        3.0   pass   -0.6 -> +0.4 bp per fill
horizon shape             PermuteWithin(hour)                 3.1   pass
mechanism ablation        ReplaceField(B inputs)              -     pass   IC falls to 0.01
latency robustness        p50 / p95 / p99                     -     p95 must pass: pass; p99 fails
--------------------------------------------------------------------------------------------
PASS . L3 . cap: L3 until live (action-dependent) . requirement for algo: cancel latency <= 40 ms
```

The card is rendered from the verdict file without an LLM. A narrated summary may be attached (see [Cards and summaries](#cards-and-summaries)).

### Labels for feat and signal, backtests for algo

`feat` and `signal` are judged on labels that do not depend on our own actions: returns, markouts, event outcomes. They need no backtest. For a strategy whose results depend on its own actions, such as a maker strategy, these layers use the terms of the profit formula that can be observed in history. For the maker-taker case: the adverse move on venue B after each taker trade on venue A (toxicity), and the taker volume that crosses a given price level (arrivals). The `signal` layer combines them into an expected net profit per opportunity, judged on historical taker trades that crossed a hypothetical quote, used as "if we had been filled" samples.

Everything that depends on our actions belongs to `algo`: whether to quote, where, when to cancel, how large. `algo` is judged with a backtest through a `ConsumerPlugin`, with a fill model, the `RandomizeTiming` null, stress worlds, and limit angles.

### Fill models

When a fill depends on our own resting order, history cannot say whether we would have been filled. A `FillModel` plugin answers in two versions:

- **Optimistic**: any taker trade that crosses our quote fills us completely.
- **Pessimistic**: we are filled only if our quote was strictly better than the best quote already resting, after subtracting the estimated queue ahead of us.

The hedge is simulated at fill time plus hedge latency by walking the recorded book on the hedge venue. Verdicts carry the precondition "under the pessimistic fill model", and a pass under the optimistic model alone is capped at L2. Live trading is the final judge. If the live fill rate falls between the two versions, the simulator is trusted and live data calibrates the fill model. If it falls below the pessimistic version, every upstream verdict that depends on the fill model is downgraded.

### Contract derivation

`rci contract derive` validates and records the contract today. From M3 it also derives the Spec, backwards from the final trade:

- decision times, from event times and action latencies;
- the label definition and horizons, for example hedge latency, 1 s, and 5 s;
- event clusters (for example trades less than 1 s apart) and the effective sample size, the number of independent clusters;
- the minimum detectable effect, from the spread of the null distribution under a cluster bootstrap;
- the economic threshold, for example the current net result per fill, which must turn positive;
- an oracle upper bound where one exists. For maker strategies it comes from public trade records (taker flow, adverse moves, competition, hedge cost, capacity) and is a mandatory kill check before any strategy code is written;
- when the sample is too small, how long to collect at the current event rate.

An illustration of why this comes first: ten years of weekly decisions give about 520 points, the standard error of an annualized Sharpe ratio is then about 0.32, and reaching `t > 2` needs a Sharpe ratio of at least 0.63. An idea that cannot plausibly reach that dies before any feature is written.

### Walk-forward and fit

There are two ways to run research code (planned, M3):

| Path | Written as | Run by the engine as |
| --- | --- | --- |
| Uses no labels | `features(panel)` | Stepping at each decision time |
| Learns from labels | `fit(X, y)` and `predict(X)` or `transform(X)` | Walk-forward: `fit` sees only training-fold labels, purged of overlap with the test period |

The rule is: anything that uses labels gets them through `fit`. Rolling statistics, z-scores, seasonal adjustment, and hedge ratios estimated on past prices need no labels. Selecting features by IC, IC weighting, target encoding, tree models, and symbolic search do; the whole search goes inside `fit` and is repeated in each fold. An agent that picks features by hand while looking at explore labels has done a manual `fit` the engine cannot see. Writing the selection as code, such as `SelectByIC(k=5)` inside a pipeline, gets it a nested walk-forward; otherwise the ledger, the seal, and forward data are the backstop. LLM scores need no labels, but the model's pretraining carries knowledge of the future, which is why the `event` template is confirmed on forward data only.

## The challenger and summaries

> **Status:** planned: a minimal challenger in M5; card narration and the summary audit later.

### Why a separate process

An agent's evaluation of its own work cannot be reproduced, and several sessions of the same model family make highly correlated mistakes. A reviewer that shares the research agent's context shares its blind spots. So the challenger is a separate LLM process that `rci <layer> challenge` starts with a prompt assembled only from files: the contract, the field metadata, the registered evaluators and tests. It never sees the research agent's conversation or reasoning, and a different model family is preferred. The CLI enforces the separation.

The challenger is an adversary, not a collaborator. It is not called a critic because that word suggests the cooperating half of an actor-critic pair.

### What the challenger writes

- Null worlds for `feat` and `signal`, choosing generators and parameters (`nulls/`).
- Pitfalls for `data`: domain mutants such as merged holiday weeks, night sessions assigned to the wrong day, jumps at futures rolls, publication days shifted by holidays.
- Stress worlds for `algo`.
- Extra tests, and the strongest counter-argument for the StageCard, for example "the takers on venue A are mostly arbitrageurs; the toxicity comes entirely from venue B having moved already".

Everything the challenger writes must pass calibration before the freeze, like the agent's tests. Duties are separated: whoever writes a test does not write the mutant that checks it. Tests declare what they `cover`; where a built-in mutation operator exists for that risk, it must be used, and the challenger writes the rest.

### Pitfall libraries

A pitfall that passes calibration (it is harmful, and the right test catches it) joins the pitfall library of the data source's plugin, `DataPlugin.mutations`. Every later contract that uses that source gets it automatically; a new pitfall is shown once on a card when it first joins. Real incidents join the stress worlds of similar strategies the same way. The library of plugins grows; the kernel does not.

### Cards and summaries

| Card | When | Shows |
| --- | --- | --- |
| ContractCard | Before any search | The claim, the derived Spec, alternative explanations, and the decisions a person must make |
| StageCard | Before a layer is frozen | Angles, null worlds, calibration results, coverage, the challenger's strongest counter-argument |
| VerdictCard | After each submission | The verdict, every angle with its z or limit, the level and caps, preconditions, lineage |
| CapitalCard | Before real money | Live results against the simulated range, the denominator (ideas and trials tried), capacity, hard requirements, known risks |

Cards are rendered from engine files without an LLM. Today the only one is `.rci/contract.json`, a plain copy of the derived contract.

Summaries are never verdicts. An optional `post_verdict` hook (later) may narrate a card, under two audits. First, every number in the summary must cite a field of the verdict file, and code checks each citation. Second, the qualitative statements that remain are reviewed by an independent reasoning LLM from a different model family, given only the output files. Neither audit can change the verdict. At a VerdictCard a person can kill the idea or ask for more work, but cannot turn a FAIL into a PASS; that would make verdicts irreproducible.

### The challenger never judges

Judging is done by deterministic kernel code. The challenger contributes tests and worlds and has no vote. Neither does any other LLM.

## Trust and reproducibility

> **Status:** partly implemented: the ledger, head anchors, state replay, freezes, environment records, and terminal-only approval in M2; DEV marking with M3; process isolation in M4; stronger hardening later.

### The ledger

`.rci/ledger.jsonl` holds one record per line in canonical JSON: sorted keys, no whitespace, UTF-8, NaN forbidden. Each record has `seq`, `at_ns`, `kind`, `stage`, `payload`, `prev`, and `hash`. `hash` is the SHA-256 of the canonical record without `hash`; `prev` is the previous record's hash, and the first record points at 64 zeros. Each append is flushed and synced to disk.

| Kind | Payload besides `from`, `to`, `version` |
| --- | --- |
| `init` | `template` only; stage `idea`; always the first record |
| `register` | For the contract: its file hash and every field. For other stages: the declarations (from M3). |
| `challenge` | The challenger's declarations (from M3) |
| `check` | `passed`, `note` |
| `approve` | `freeze` hash and `deps`; for the contract also `budget` and `skip` |
| `submit` | `verdict`, `note`, `environment` |
| `extend` | `trials`, `note` |
| `stale`, `skip` | `note`, written as follow-ups of a reopen or a contract approval |

An abridged `submit` record:

```json
{"at_ns":1790589079077636000,"hash":"a91664db...","kind":"submit","payload":{"environment":{"packages":"f56f5971...","uv.lock":"none"},"from":"FAILED","note":"","to":"PAUSED","verdict":"FAILED","version":1},"prev":"e655d565...","seq":10,"stage":"data"}
```

### What verification catches

Every command verifies the ledger before doing anything and refuses to run if verification fails. `rci verify` does the same and prints `ledger ok: N records`. For every line it checks valid JSON, canonical form, the sequence number, the link to the previous hash, and the record's own hash, and it reports the line number of the first failure. This catches a one-byte edit, a deleted line in the middle, two swapped lines, and a verdict edited by hand from FAILED to PASSED. `rci inbox` lists an idea that fails verification as an item for a person.

### Head anchors

A hash chain cannot show that records were cut from the end. Deleting the last failed submission to win back a trial leaves a valid chain; so does restoring an older ledger from git. So the engine keeps each ledger's latest head outside the idea, in `$RCI_HOME/heads/<hash of the first record>.json`, which holds the head's `seq`, its hash, and the ledger's path. Every append moves the anchor forward, and every verification checks that the record at the anchored position still exists with the same hash.

| Situation | Result |
| --- | --- |
| The last records deleted, for example a failed submission | Refused: records were removed or rewritten |
| An older ledger restored, for example from git | Refused |
| A consistent rewrite: a valid chain with different content at or before the anchored record | Refused |
| The ledger grew elsewhere, for example pulled from another machine | Accepted if it still contains the anchored record; the anchor moves forward |
| The idea directory renamed or moved | Accepted; the anchor is keyed by the first record, not the path |
| The idea directory copied | Both copies share one anchor, so once one moves on, the other is refused. A copy needs a fresh `.rci/` from its own `rci init`. |
| A new machine or an empty `$RCI_HOME` | Accepted; the first use is the starting point of trust |
| Anchor files edited or deleted, a ledger rewritten from its first record, or ledger and anchor rewritten together | Not caught; out of scope for the MVP |

### State is replayed from the ledger

There is no state file. Every transition is a ledger record whose payload carries `from`, `to`, and `version`, and `status()` replays the records in order: each one sets its stage's state and version, submissions add trials, extensions add budget, and contract approvals set the budget. Changing the state therefore means changing the ledger, and changing the ledger is caught. Replay costs one pass over a small file per command.

### Approval reads the ledger

`rci contract derive` writes the hash of `contract.py` and every contract field into the ledger. Approval uses those ledger fields; `.rci/contract.json` is a copy for people and for git, and editing it changes nothing (a test raises its budget to 1000 and the approval still uses 3). If `contract.py` changed after the derive, approval is refused: ``contract.py changed since it was derived; run `rci contract derive` again``. In the same way, a submission compares the `deps` recorded in the ledger's approval record with current values; `.rci/freeze/<stage>.json` is only a copy.

Approval requires a person: `rci approve` refuses to run unless stdin and stdout are both a terminal, so an agent's non-interactive shell cannot approve. Before asking, it prints everything it will freeze. Afterwards it prints a suggested command such as `git add ideas/tox/.rci ideas/tox/contract.py && git commit -m 'rci: freeze contract of tox'`. The engine never calls git; the person decides whether to commit.

### Freeze scope

A freeze is `Freeze(stage, deps, hash, at_ns)`, where the hash covers only the stage and its dependencies, so it does not depend on time or key order. What is frozen is the machinery that judges:

| Dependency | What it pins |
| --- | --- |
| `contract` | SHA-256 of the bytes of `contract.py`; even an edited comment taints |
| `engine` | A tree hash of the installed `research_ci` package source |
| `python` | Interpreter and version, for example `CPython 3.11.15` |
| `engine_packages` | The engine's version and the installed versions of its direct dependencies |
| `upstream:<stage>` | The freeze hash of each upstream stage's latest approval; skipped stages are left out |
| From M3 | Evaluators, null worlds, plugins, and the data snapshot |

The contract stage freezes only `contract`. At each submission the engine recomputes the dependencies and compares them with the ledger. Any change, added key, or missing key makes the submission TAINTED, with a note that lists what moved, for example `frozen dependencies changed: contract`.

The research environment is recorded, not frozen. Each submission records a hash of every installed package with its version and the hash of the nearest `uv.lock` in the idea directory or above it (`none` if there is none). The reason is the normal loop: after a FAIL the agent changes its code and resubmits, and adding a research library changes `uv.lock`. If the lock file were frozen, that normal resubmission would be TAINTED. What must not move is the machinery that judges; the rest is recorded so any verdict can be reproduced. The interpreter version is frozen because uv resolves dependencies per Python version (3.11 gets numpy 2.4.x, 3.12 gets 2.5.x), and floating-point results can differ across them.

### Table and tree hashes

Data snapshots (from M3) are hashed with the engine's own encoding per column type rather than polars' `hash_rows`, which is not guaranteed to be stable across versions. The hash ignores row and column order, keeps null, NaN, and 0.0 distinct, and treats -0.0 as 0.0. It supports nanosecond UTC datetimes, nanosecond durations, Int64, Float64, String, and Boolean, and refuses every other type rather than guessing. Directory hashes cover relative paths and contents and skip `.rci`, `__pycache__`, `.git`, `.venv`, and tool caches.

### DEV mode

A verdict made with an editable engine install will be marked DEV and capped at L1 (planned, M3; nothing detects it yet). Real verdicts need the engine pinned to a git SHA.

### Threat model

The MVP defends against accidents and lazy shortcuts: a forgotten publication lag, a retry loop until the result is green, a test edited after the results, a deleted failed submission, a stray random seed. It does not defend against deliberate attacks by a process with the same user's permissions, which could edit anchors, rewrite a ledger from its first record, or change the engine and approve a new freeze from a pseudo-terminal.

Git is not a security mechanism, because agents can rewrite history. The engine neither calls nor depends on git; trust comes only from its own freeze hashes, the hash chain, and the anchors. The decisions that matter stay with a person at a terminal, who runs `rci approve` in their own shell.

### Later hardening

When deliberate cheating has to be stopped, the options are, in order of cost: the M4 child process with seeds derived from the freeze hash, timeouts, and memory limits; a separate operating-system user that owns the vault, the ledgers, and the anchors; and the thin trusted service `execute(bundle, vault, ledger) -> verdict`, which would hold sealed data, confirmation verdicts, the ledger, and signatures.

## Layers and templates

> **Status:** partly implemented: templates and stage lists in M2; layer policies planned from M3 (`feat` first), data plugins in M5, `algo` in M6, checkpoint commands later.

Each layer runs the same inner loop. What changes is the question, the output shape, the labels, the kind of null world, and what calibration checks.

| | `data` | `feat` | `signal` | `algo` |
| --- | --- | --- | --- | --- |
| Question | Can these fields support the contract? | Does the feature add information? | Does the combined signal predict? | Does acting on it make money after costs and constraints? |
| Output | FieldTable | Scores | Scores, Events | Positions, Quotes, Orders |
| Labels | none | action-independent | action-independent | backtest through a `ConsumerPlugin` |
| Agent writes | data tests | evaluators | evaluators, baselines to beat | evaluators, limits, a configuration |
| Worlds | mutants | null worlds | null worlds | random-timing nulls, stress worlds |
| Calibration | mutants must be caught | stand-ins killed, planted kept | stand-ins killed, planted kept | random stand-in rejected, unprotected stand-in breaches limits, simulator known answers |

### The data layer

The data layer judges whether these fields can support this contract, not whether the data is "good". The same field can serve a weekly strategy and be useless for an intraday one. Its output is a FieldTable, plus the list of fields usable for verdicts, fields that are explore-grade only, masks (for example holiday weeks), splits at breaks, and a hash of the snapshot that `feat` must use.

Checks come in three tiers:

| Tier | Written by | Checks |
| --- | --- | --- |
| Generic floor | the engine | monotone timestamps without duplicates, time zones, gaps, `available_at >= event_time`, jumps and level breaks, units and magnitudes |
| Floor set by the contract | the engine, parameters from the contract | `available_at` before the decision time, coverage long enough for the required sample, freshness at the decision time, a universe that includes delisted and expired instruments, labels built on tradable prices |
| Source knowledge | the data plugin | publication schedules, vintage archives, methodology changes, reaction tests around releases |

The agent explores and then writes data tests. Each declares what it `covers` and what happens `on_fail`: split and recompute power, mask the affected weeks, use the actual `available_at`, drop the field, or raise a decision for a person. The generic mutation operators, all transformations of the one table format:

| Operator | Simulates |
| --- | --- |
| move `available_at` earlier | a wrong publication time |
| replace values with future values | a leak |
| multiply by (1 + k) after a date | a methodology break |
| delete or merge a stretch | a gap, a merged release |
| duplicate timestamps | duplicates |
| replace first vintages with revised ones | revised-only history |
| shift everything by some hours | a time zone error |
| insert spikes | outliers |
| multiply by 10 or 1000 | a unit error |
| remove delisted or expired entities | survivorship bias |

Execution strategies also need a latency table from the data layer: cancel latency, order-to-acknowledgement time, and hedge latency (p50, p95, p99) measured from our own order telemetry, the clock skew between us and each venue, and market data timed by local receive time rather than venue timestamps. Their tests check that the latency sample is large enough, stable across sessions, and that the skew was measured.

Typical pitfalls: history that exists only in revised form, level breaks when a survey changes its sample, merged holiday weeks, shifted publication days, night sessions that belong to the next trading day, continuous futures series with built-in lookahead, survivorship in instruments or archived objects, and LLM data generated after the fact.

The data layer is where many ideas should die early. If a core field exists only in revised form, history cannot produce a verdict, and only forward data can confirm the idea, at perhaps 52 samples a year for a weekly field. If a split at a break leaves too few samples, power is recomputed and the idea goes back to the contract as INSUFFICIENT. Passing unlocks `feat`, which may read only the passed fields; explore-grade fields cap everything downstream at L1. The [futures inventory spread walkthrough](walkthroughs.md#futures-inventory-spread) goes through the data layer in detail.

### The feat layer

The feat layer judges whether a feature adds information beyond what is already known. Its output is Scores. Features that use no labels are stepped; features that learn from labels go through `fit` and walk-forward (see [Walk-forward and fit](#walk-forward-and-fit)).

The agent picks angles from a menu and registers them in advance:

| Angle | Question |
| --- | --- |
| predictive | Is there a signal at all (IC, ICIR)? |
| increment | Is there something beyond the known variables (residual IC, nested out-of-sample comparison)? |
| economic | After a consumer and costs, is money left? |
| time stability | Is the sign consistent across years and regimes? |
| breadth | Does it hold on other entities and subsets? |
| mechanism ablation | Does it vanish when the claimed inputs are scrambled? |
| secondary prediction | Does the mechanism predict anything else, such as a stronger effect in peak season or a monotone response across buckets? |
| parameter robustness | Is the result a plateau or a spike? |
| multiple implementations | Do two or three definitions agree in sign? |
| independent source | Does it hold on different data? |
| seal and forward | Does it hold on data never seen? |

Typical null worlds are `BlockPermute`, `ShiftTime`, `PermuteWithin`, and `ReplaceField`. Calibration uses stand-ins, for example a pure seasonal feature plus noise that the whole-year shift must kill, and planted features that must survive.

Typical pitfalls: two-sided filters and seasonal decompositions (caught by stepping), normalization over the full sample, features built from the very event the strategy has to anticipate, and features picked by hand on explore labels. For a strategy that depends on its own actions, the labels are the observable terms of its profit formula, and decision times are event times minus the action's latency.

Passing stores the feature under its version hash. Downstream stages may reference only that version, at the level it reached.

### The signal layer

The signal layer judges whether the combined signal predicts: a raw signal, a rule, or a model's output (trees, neural networks, LLM scores). Its output is Scores or Events, judged on action-independent labels, with no backtest.

Its angles and null worlds are those of `feat`, applied to the combination: increment over its inputs and over simple baselines, stability, secondary predictions, and for Events the outcome after events against random times and against `RandomEvents` from the same pool. Models are fitted by walk-forward only. Small samples overfit easily; with about 520 weekly points, a gradient-boosted model finds patterns that are not there. A threshold picked from history, such as "cancel on the top 20% of toxicity", is a `fit` step too.

The engine checks lineage: `signal` may use only features that passed. In the maker-taker case, `signal` is "toxicity and arrivals give the expected net profit of each opportunity", judged on historical taker trades that crossed a hypothetical quote. Text and LLM signals use the `event` template and are confirmed only on forward data. Passing unlocks `algo`.

### The algo layer

The algo layer judges whether acting on the signal makes money after costs and constraints, safely, and at what size. Its output is Positions, Quotes, or Orders. The backtest is its evaluator: a `ConsumerPlugin`, which for millisecond strategies is an event-driven execution simulator and for daily strategies a bar-based backtester. It decides what the earlier layers do not: order size, how many quotes at once, what to do when a hedge fails, when to clean up residual delta, automatic stops (stale data, feed sequence gaps, latency above a limit, daily loss, repeated hedge failures), and margin and exposure per venue.

- **Angles**: `stat` angles for profit (for example daily net at the chosen size, against the null) and `limit` angles for risk (the 99th percentile loss per cycle, maximum daily drawdown, and how long a leg may stay open under injected hedge failures).
- **Null worlds**: `RandomizeTiming(keep="rate")`, a random strategy with the same decision rate, rules out "it only works because it trades or cancels less". Baseline strategies must be beaten too, such as a threshold-only rule that never cancels on toxicity. `GridMax` and plateaus cover sizes and limits.
- **Stress worlds**: as listed in [Stress worlds](#stress-worlds).
- **Calibration**: a random stand-in may pass the random-timing null no more often than the nominal rate alpha, a planted known-good signal on synthetic data must pass, and an unprotected stand-in must breach its limits under stress. The execution simulator passes known-answer tests before anything is judged on it: never quoting gives zero fills and zero PnL; quoting far away gives zero fills under the pessimistic model; doubling fees lowers PnL by exactly the number of fills times the fee difference; raising hedge latency from zero to p95 makes the adverse move grow monotonically; a larger size makes hedge slippage grow; zero depth on the hedge venue sends the simulation into the hedge-failure branch.
- **Caps**: Quotes that pass only under the optimistic fill model stop at L2; action-dependent strategies stop at L3 until live results.

Passing produces an executable configuration, such as an order size, a maximum number of open quotes, and a daily loss limit, which becomes the parameters of the forward experiment. For the maker-taker case, how the two hedge legs are eventually closed is an [open decision](#open-decisions); holding them adds funding, margin, and liquidation-distance angles.

### Checkpoints

The four checkpoints share the one-shot shape: register the plan, a person approves it, run once, and on failure go back to a layer.

| Checkpoint | Registered in advance | Run by | Returns | On failure |
| --- | --- | --- | --- | --- |
| `seal` | The k candidates | a person, `rci seal run` | Pass or fail only, no numbers; each sealed segment is used once per idea | Back to a layer; confirmation waits for new data |
| `forward` | The experiment: groups, duration from the MDE, stops from the algo limits | the scheduler, `rci forward --all` | Forward or live results; L4 evidence | Back to a layer, or downgrade upstream verdicts |
| `capital` | The capital plan, staged size increases | a person, `rci capital run` | The CapitalCard decision | The idea stops or waits |
| `monitor` | Drift rules and automatic actions | the scheduler, `rci monitor --all` | Pause, reduce size, or reopen a layer | Reopens the layer the rule points at |

For strategies whose fills depend on their own actions, `forward` is a pre-registered randomized experiment: each quoting cycle is assigned at random to groups (for example spread threshold by cancel rule), the duration comes from the MDE, stops come from the algo limits, and every live fill is reconciled with the simulator's prediction. Nothing changes while it runs. For the `event` template, `forward` is the only confirmation: it reruns on schedule and shows progress toward the required sample, counting only events created after the freeze, such as 40 of 196.

`monitor` watches the structure of results, not only the total, and each drift points at a layer:

| Drift | Meaning | Reopens |
| --- | --- | --- |
| hedge slippage grows | depth on the hedge venue got worse, or the size is too large | `algo` |
| adverse move during the hedge grows | toxicity changed, or cancelling got slower | `feat`, or the infrastructure |
| locked spread shrinks, fill rate drops | a competitor arrived | `signal` |
| live fills diverge from the simulator | the fill model is out of date | recalibrate the fill model |
| net results below the range for two weeks | the pre-registered pause rule | pause and ask a person |

### Templates

| Template | Research type | Layers | Notes |
| --- | --- | --- | --- |
| `factor` | Cross-sectional factors | `data`, `feat`, `signal`, `algo` | The most standard path |
| `rule` | Rule-based strategies | `data`, `signal`, `algo` | Features and signal are one step |
| `event` | News and LLM events | `data`, `feat`, `signal`, `algo` | Confirmed on forward data only |
| `taker_arb` | Taker arbitrage | `data`, `signal` (optional), `algo` | Orders; skip `signal` with `skip=("signal",)` |
| `maker_arb` | Maker arbitrage and market making | `data`, `feat`, `signal`, `algo` | Quotes; `algo` uses a fill model |

All templates end with `seal`, `forward`, `capital`, and `monitor`. The template is fixed at `rci init`.

## Plugins

> **Status:** planned: the first data plugin and consumer in M5, fill models in M6; conformance tests with them.

### Plugin interfaces

Everything domain-specific is a plugin. Plugins live in a lab, import only `research_ci.api`, and will be discovered through entry points.

```python
class DataPlugin:
    def fetch(self, request) -> FieldTable: ...  # rows carry available_at and vintage

    mutations: list[Mutation]  # the source's pitfall library


class InstrumentPlugin:
    def tradable_price(self, leg, t) -> float: ...  # contract months, sessions, rolls
    def cost(self, trades): ...  # fee tables, special fees, slippage
    def constraints(self, positions): ...  # price limits, delivery months, lot sizes


class ConsumerPlugin:
    def backtest(self, positions_or_quotes): ...  # the algo layer's evaluator


class FillModel:
    def fill(self, quote, market, mode): ...  # mode is "optimistic" or "pessimistic"
```

`available_at` and `vintage` come from the data plugin, never from the agent. A source that has not been reviewed yet is explore-grade. A backtester written in another language can be wrapped as a `ConsumerPlugin` that hands over the position table as a file; a reconciliation test then keeps the two in agreement.

### Conformance tests

Every plugin passes conformance tests before any verdict depends on it (planned, `tests/conformance/`):

- known answers: a planted effect is found, shuffled data is judged red;
- planted future rows in a data plugin's output are caught;
- costs reconcile with a hand-computed example;
- a consumer reproduces the reference backtester's PnL on the same position table within 1e-12;
- a fill model and its simulator pass the known-answer tests of the algo layer;
- a data plugin's mutants are harmful.

### Promotion from a lab into the engine

A plugin or tool moves from a lab into the engine only when all of these hold: at least two ideas use it; it has no hard-coded paths, parameters, or domain assumptions; it has known-answer tests and passes conformance; and a person has reviewed it.

## Key decisions

> **Status:** partly implemented: decisions about M1 and M2 are in the code; the others describe planned work.

### Polars kernel with pandas or polars panels

- **Decision**: the kernel (FieldTable, `asof`, stepping, comparisons) runs on polars. Research code chooses `panel_format="pandas"` (default) or `"polars"`.
- **Alternatives**: a pandas kernel; a polars-only research API.
- **Why**: `asof` is the hot path, and polars was about 3 times faster on it (10.3 ms against 35 ms per decision time). Forcing polars on research code would break the promise that research code is free, and most research code is pandas. Tests run in both formats, and statistics that decide verdicts will use numpy in a fixed order for exact reproducibility.

### Availability-timed leak check

- **Decision**: compare stepping with a full run on a copy where each first vintage sits at its `available_at`.
- **Alternatives**: compare with a full run on the real table; skip the comparison and rely on stepping alone.
- **Why**: on the real table, publication lags and revisions make causal code, such as a forward fill of a sparse field, look like a leak. Relying on stepping alone would hide the gap between explored numbers and live numbers. Rerunning without the lookback separates real lookahead from a window that is too short.

### No state file

- **Decision**: state is replayed from the ledger; every command verifies the chain first.
- **Alternatives**: a `state.json` beside the ledger.
- **Why**: two sources of truth drift, and an edited state file is undetectable. With replay, changing state means changing the ledger, which is caught.

### Freeze the judging machinery, record the research environment

- **Decision**: freeze the contract, the engine source, the interpreter, the engine's own dependencies, and upstream freezes (from M3 also evaluators, null worlds, plugins, the data snapshot). Record all installed packages and `uv.lock` with each submission.
- **Alternatives**: freeze `uv.lock` too, as the first design did; freeze nothing about the environment.
- **Why**: changing research code after a FAIL is the normal loop, and adding a library changes `uv.lock`; freezing it would taint honest resubmissions. Freezing nothing would make verdicts irreproducible.

### Ledger head anchors

- **Decision**: keep each ledger's latest head in `$RCI_HOME/heads/`, keyed by its first record.
- **Alternatives**: the hash chain alone; git history; signatures or a service.
- **Why**: the chain misses records cut from the end, which is exactly how a trial would be won back. Git can be rewritten by agents. Signatures and services are heavier than an MVP that defends against accidents needs. Keying by the first record lets ideas move and lets ledgers grow on other machines.

### No LLM in the kernel, and no separate small-model judge

- **Decision**: verdicts are computed only by deterministic kernel code. LLMs write tests (the challenger) and narration, never verdicts. Summaries are audited by a citation check and an independent reasoning LLM from a different model family.
- **Alternatives**: an LLM as judge; an earlier draft that had a separate small, cheap model review agents' summaries.
- **Why**: an LLM's judgment cannot be reproduced and is correlated with the agent it judges. Numbers are better checked by code, and the qualitative remainder needs a strong, independent reader, not a small one. The small-model judge was removed; such models may still appear inside research, for example as a classifier whose output is a feature.

### Local-only MVP, with the data boundary in the loader

- **Decision**: no service in the MVP. The explore tier's data boundary is enforced by the engine's loading functions (`visible`, `asof`), which never return rows after the cutoff.
- **Alternatives**: confirm only on forward data; never download sealed data during exploration; a trusted service from the start.
- **Why**: the loader boundary is simple and executable, and it keeps the historical sealed tier usable. Not downloading sealed data is still planned for the vault as an extra layer. A service is only needed against deliberate cheating.

### Library plus lab

- **Decision**: the engine repository holds only the library and CLI; ideas and real plugins live in a lab, which pins the engine by git SHA.
- **Alternatives**: one repository for engine and research; moving older research repositories into the lab.
- **Why**: every verdict pins the engine, so the engine must change slowly and under review, while research changes daily. The split keeps the engine generic and lets any Python 3.11+ environment host ideas.

### Python 3.11 floor, dependency floors at current releases

- **Decision**: Python 3.11 or later. Dependency floors at the latest releases: pandas 3.0.6, polars 1.44.2, pyarrow 25.0.1, typer 0.27.2; numpy 2.4.6 and scipy 1.17.1, the latest releases that still support 3.11.
- **Alternatives**: a 3.12 floor; older floors such as pandas 2.x.
- **Why**: environments pinned to 3.11 must be able to host ideas, so development defaults to 3.11. Current floors keep one set of behaviors to test (pandas 3 changed time resolution, for example). The cost: the engine cannot install next to pinned older versions such as pandas 2.x.

### Per-layer CLI deferred until the judge exists

- **Decision**: `register`, `challenge`, `check`, and `submit` are implemented and tested in the workflow, but their CLI commands arrive with M3. `rci next` already names them.
- **Alternatives**: expose them now.
- **Why**: without a judge, the workflow's `submit` takes the verdict as an argument. A CLI command like that would let an agent declare its own verdict, which is exactly what the engine exists to prevent.

### Thresholds as z scores

- **Decision**: compare z scores from the null distribution with `z_need = z_base * sqrt(1 + ln n)`.
- **Alternatives**: permutation p-values with a corrected alpha.
- **Why**: 200 permutations cannot resolve p below about 0.005, which the threshold reaches after two trials. z keeps resolution, and the empirical rank remains a secondary check.

## Prior art

> **Status:** background: other systems and what Research CI takes from them.

In the diagrams below, blue boxes are deterministic code, where results are judged; dashed orange boxes are agents or LLMs; green folded boxes are knowledge; rounded boxes are people; grey boxes are data and platforms. A red Gap badge marks a key gap for our purposes.

![Legend](assets/legend.png)

### RD-Agent(Q)

![RD-Agent(Q)](assets/prior-art-2-rd-agent.png)

An automated research loop for quantitative factors and models, built on Qlib ([paper](https://arxiv.org/abs/2505.15155)). It specifies the task as a structured tuple of background, data, output format, and execution environment, then loops: propose a hypothesis, write code, validate (deduplicate, then backtest), analyse against the best set so far. A bandit decides whether the next round changes factors or the model. Knowledge is structured: records of hypotheses and feedback, and a graph of tasks, code, and feedback used to retrieve earlier code.

- **Borrowed**: output formats and the execution environment written as code; deduplication by correlation (it uses IC correlation of 0.99 or more against its best set; Research CI uses 0.9 against earlier versions).
- **Different**: the paper does not say clearly whether in-loop feedback uses a validation or a test period, and its default configuration appears to backtest on the test period. It supports daily cross-sectional data only, does not count trials, stores trajectories rather than verified facts, and has no forward validation. Research CI keeps holdout data out of the agent's reach and counts every trial.

### AlphaLab

![AlphaLab](assets/prior-art-3-alphalab.png)

An autonomous research system that takes a dataset and a one-sentence goal and runs the whole study ([paper](https://arxiv.org/abs/2604.08590)). It generates its own adapter files and a domain knowledge document, explores the data, then has a builder write the evaluator, a critic without context review it for leaks and lookahead, and a tester write tests until they pass. Experiments are then proposed, implemented, and scheduled at scale, with a leaderboard. Knowledge passes through markdown files, including a playbook of methods and "do not try again" entries; the dispatcher is plain Python.

- **Borrowed**: a reviewer without the researcher's context (the challenger), a scheduler that is not an LLM, and pitfall lists that accumulate.
- **Different**: its evaluator is built by an LLM and constrained only by prompts, with no process isolation; results come from single runs; its domains have clean scalar metrics with no costs, capacity, or live trading. In Research CI the evaluator machinery is fixed, reviewed code, and the challenger's output must pass calibration.

### autoresearch

![autoresearch](assets/prior-art-1-autoresearch.png)

A minimal loop of a few hundred lines: a person edits an instruction file, the agent edits the training script, and a fixed preparation script owns the data and the metric. Each run trains for five minutes; a better validation score is kept, a worse one discarded.

- **Borrowed**: the three-part split of human instructions, agent-edited code, and a fixed evaluator.
- **Different**: it works because it has a cheap, objective scalar that is hard to game. Trading research has no such scalar, so Research CI builds the evaluator from pre-registered angles, null worlds, and calibration.

### AlphaEvolve

![AlphaEvolve](assets/prior-art-4-alphaevolve.png)

Evolves code and keeps a database of programs. The customer writes the evaluation function and runs it on their side; the system deliberately does not touch it.

- **Borrowed**: the recognition that the evaluator is the open slot every automated loop leaves to the user.
- **Different**: Research CI is that evaluator layer, for research claims.

### Qlib

![Qlib](assets/prior-art-5-qlib.png)

A quantitative research platform with no agent: data layer, factor expression engine, model zoo, backtesting, and experiment records that compute IC and portfolio analysis, driven by YAML workflows and Python.

- **Borrowed**: proof that a research engine can exist while research code stays free.
- **Different**: it is oriented to daily cross-sectional equities and does not manage point-in-time data, a trial ledger, or holdout permissions.

### Kaggle

Kaggle, WorldQuant BRAIN, and Lean side by side:

![Kaggle, WorldQuant BRAIN, and Lean](assets/prior-art-6-kaggle-brain-lean.png)

Competitions separate the host from the participants. Code competitions run submissions on hidden data, the private leaderboard is revealed once, and forecasting competitions score on data that did not exist at submission time.

- **Borrowed**: host and participant separation, a private holdout seen once (the seal tier), and scoring on the future (the forward tier).
- **Different**: every competition is designed by hand. Research CI derives the rules of each idea from its contract.

### WorldQuant BRAIN

A platform for alpha expressions: a DSL, a simulator, in-sample admission criteria, out-of-sample acceptance, and a correlation check against the existing alpha pool.

- **Borrowed**: admission before acceptance, true out-of-sample acceptance, and correlation checks against what already exists.
- **Different**: it supports one trading form. Research CI supports five output shapes and any method that produces them.

### Lean and Mathlib

A proof assistant with a small trusted kernel and a very large library. Tactics and library lemmas can be large and clever, but every proof is checked by the kernel, so an AI-written proof needs no human trust. This is why AI provers target Lean.

| Lean | Research CI |
| --- | --- |
| the kernel | the engine kernel |
| tactics | the research agent and the challenger |
| Mathlib | the plugin library and its pitfalls |
| `sorry` | explore-grade results |
| `#print axioms` | lineage |
| citing only proven lemmas | downstream stages consuming only passed outputs at a sufficient level |

- **Borrowed**: a small kernel, a large checked library, and the principle that only the kernel needs trust.
- **Different**: theorems are deterministic and permanent; statistical claims are probabilistic and decay. Hence the ledger, the trial count, and forward reruns.

### What they share and what is missing

![Side-by-side comparison](assets/prior-art-comparison.png)

In every system above that works in practice, the judge is code, not markdown; markdown carries knowledge and instructions and only shapes what agents propose. The loop and orchestration are open source already, or will be provided by vendors. What they lack is the judgment layer: holdout data outside the agent's permissions, trial counting per idea, point-in-time data, many research types, forward validation, and tests of the tests with null worlds. That layer is Research CI.

A reinforcement learning view makes the same point. The engine is an environment, a verdict is a reward, and the research agent is a policy; the whole design defends against reward hacking. The difference is that in research the final reward, future data, can be queried only once. The explore tier is the training environment, the seal and forward tiers are the test environment, and the ledger is the price of each query. Later, bandit methods could allocate trials across ideas, and pairs of verdicts and forward results would become trustworthy reward data.

## Roadmap and open decisions

> **Status:** M0 to M2 done; M3 is next.

### Roadmap

| Step | Scope | Acceptance | Status |
| --- | --- | --- | --- |
| M0 | Scaffold, CI, import layers | CI green; planted layer violations are caught | Done |
| M1 | `kernel/tables`, `kernel/time` (FieldTable, `asof`, `step`, `check_leaks`) | Every planted lookahead is caught; plain rolling features report zero mismatches | Done |
| M2 | `kernel/hashing`, `kernel/holdout` (tiers, labels), `store` (workspace, ledger with head anchors, freeze, environment), `workflow` (machine, templates, lineage, next, commands), `api/contract`, CLI `init`, `status`, `next`, `inbox`, `verify`, `contract derive`, `approve` (with `--extend`) | The state machine rejects out-of-order calls; a one-byte edit breaks ledger verification | Done |
| M3 | `kernel/judge` (null worlds, z thresholds, evidence levels, calibrate) and the per-layer CLI commands (`rci <layer> explore/register/challenge/check/submit`) | The `feat` layer runs end to end on the first real case | Planned |
| M4 | `runner`: research code in a child process, data exchanged as Arrow | Verdict cards stay byte-identical when research code reseeds or mutates data | Planned |
| M5 | First data plugin, consumer, minimal challenger | The plugins pass conformance | Planned |
| M6 | Algo: execution simulator, fill models, randomized decisions, constraint angles; a maker-taker case end to end | The cross-venue maker-taker case runs end to end | Planned |
| Later | Checkpoint commands (`rci seal run`, `rci forward --all`, `rci capital run`, `rci monitor --all`), `post_verdict` hook (card narration and citation check), graveyard replay, a thin trusted service `execute(bundle, vault, ledger) -> verdict` | Not set yet; graveyard replay is judged by the failure condition below | Planned |

### Open decisions

- **The first real case for M3.** Candidates: a cross-sectional residual factor whose data is already local; a futures spread time series; and the cross-venue maker-taker case, which is event-driven and depends on its own actions.
- **Final definitions of L0 to L4**, after M3 has run a real case. With them: the list of floor angles each stage enforces, the standard contents of the evaluator context, and the value of `z_base`.
- **How the maker-taker hedge legs are closed.** The strategy is opportunity-driven: it quotes on venue A and, after any fill, hedges on venue B at once, so it never carries one-sided inventory. Each cycle leaves a leg on each venue. Option A closes them with a reverse cycle when the opposite spread appears; option B holds them and closes periodically, which adds funding, margin, and liquidation-distance angles to `algo`; option C closes each cycle separately.
- **Graveyard cases**: which past failed research cases to replay, with what they died of and when.
- **How INSUFFICIENT and DEV map onto stage states**, and whether an INSUFFICIENT verdict costs a trial.
- **How an ongoing monitor fits the one-shot checkpoint**: in the M2 machine `monitor` runs once like the other checkpoints, while the design has it run on a schedule and reopen layers on drift.
- **Output shapes for options**: whether volatility surfaces or exposures to Greeks fit the five shapes.
- **For the text event case**: the knowledge cutoff of the LLM that produced the signals, and whether the unfiltered event pool that `RandomEvents` needs is available.

### When Research CI itself has failed

Research CI applies its own standard to itself. The project has failed if graveyard replay kills no case earlier than it was killed historically, or kills a true positive, or produces a verdict that cannot be reproduced.
