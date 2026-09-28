# Walkthroughs

> **Status:** partly implemented: the stage lifecycle, the ledger, freeze records, and the commands `rci init`, `rci contract derive`, `rci approve`, `rci next`, `rci status`, `rci inbox`, and `rci verify` in M2; judging, the per-layer commands, plugins, and the checkpoints are planned for M3 and later.

This document follows three ideas from the first sentence of a claim to the last checkpoint. Each case stresses a different part of Research CI.

| Case | Template | What it exercises |
| --- | --- | --- |
| [Futures inventory spread](#futures-inventory-spread) | `factor` | Data-heavy validation. Release times, revised-only values, level breaks, and holiday weeks decide most of the outcome before any return is computed. A data finding reopens a layer, and a later layer runs out of budget. |
| [Cross-venue maker-taker](#cross-venue-maker-taker) | `maker_arb` | A microstructure strategy through every layer and every checkpoint: decision times derived from latency, fill models, randomized-timing nulls, stress worlds, a small live experiment, a capital card, and drift rules that reopen layers. |
| [Text event signals](#text-event-signals) | `event` | Signals produced by an LLM from news: backfilled data that knows the outcome, publish time versus news time, survivorship, a power calculation that says "not yet", and confirmation on forward data only. |

Every number in these cases is illustrative: it shows the mechanics and is not a research result. Definitions of evidence levels, null worlds, and tiers are in [design.md](design.md).

## Reading a step

> **Status:** partly implemented: `rci next`, `rci status`, `rci inbox`, and the ledger records in M2; verdict fields in the ledger planned for M3.

Each step answers four questions: who acts, which command they run, what the engine checks or computes, and what is appended to the ledger.

| Actor | Who it is | What it may do |
| --- | --- | --- |
| agent | the research agent | write `contract.py`, the research code, and tests for its own claim; explore freely; submit |
| challenger | a separate LLM process fed only files, never the agent's conversation | write null worlds, stress worlds, and domain pitfalls against the registered tests; it never judges |
| human | a person at an interactive terminal | freeze a stage, extend a budget, run the seal and capital checkpoints |
| scheduler | a timed job | run the forward and monitor checkpoints |
| engine | deterministic kernel code with no LLM inside | check, compute, judge, and append to the ledger |

The engine is passive. Nothing moves unless a driver calls a command, and every driver reads the same list from `rci next`: who should act, the command, and the reason. `rci inbox` shows a person everything that waits for a human across ideas:

```text
$ rci inbox ideas
ideas/mill-margin        rci approve --stage signal --extend N all 8 submissions used; extend the budget or drop the idea
ideas/mm-toxicity        rci approve --stage algo             review the tests and freeze them
```

| Command | Who | Status |
| --- | --- | --- |
| `rci init <idea> --template <t>` | agent | implemented (M2) |
| `rci contract derive` | agent | implemented (M2): validates `contract.py` and records its fields; the derived spec (decision times, power, thresholds) is planned for M3 |
| `rci approve --stage <stage>`, `rci approve --stage <layer> --extend N` | human | implemented (M2): refuses without an interactive terminal, prints what it freezes, and suggests a `git commit` that the person may or may not run |
| `rci next`, `rci status`, `rci inbox`, `rci verify` | anyone | implemented (M2) |
| `rci <layer> explore` | agent | planned (M3) |
| `rci <layer> register`, `challenge`, `check`, `submit` | agent, challenger, agent, agent | planned (M3); the transitions behind them are implemented in `workflow` (M2), and `rci next` already prints them |
| `rci data run` | agent | planned (M5, with the first data plugin) |
| `rci <checkpoint> register`; `rci seal run`, `rci forward --all`, `rci capital run`, `rci monitor --all` | agent; human, scheduler, human, scheduler | planned (later) |

Every state change is one line of canonical JSON in `<idea>/.rci/ledger.jsonl`. There is no state file: `rci status` replays the ledger. Every command first verifies the hash chain and a head anchor kept outside the idea in `$RCI_HOME/heads/`, so deleting the last line (for example, a failed submission removed to win back a trial) is refused. A submission record today (one line in the file, wrapped here, hashes shortened):

```json
{"at_ns": 1790589015112083000, "hash": "e43e3bc9d2b3...", "kind": "submit",
 "payload": {"environment": {"packages": "f56f59714486...", "uv.lock": "none"},
             "from": "FROZEN", "note": "frozen dependencies changed: contract",
             "to": "TAINTED", "verdict": "TAINTED", "version": 1},
 "prev": "35830db044ea...", "seq": 12, "stage": "feat"}
```

Every payload carries `from`, `to`, and `version`. Beyond that, `init` records the template; `register` on the contract records the file hash and every field; `check` records `passed` and a note saying what failed; `approve` records the freeze hash and the frozen dependencies (plus the budget and `skip` for the contract); `submit` records the verdict, a note, and the research environment (hashes of all installed packages and of `uv.lock`); `extend`, `stale`, and `skip` record budget extensions and the state moves that follow a reopen or the contract approval. From M3 on, `submit` also records the angle scores, z values, evidence level, and caps.

Thresholds rise with n, the number of submissions the ledger counts for that stage: `z_need = z_base * sqrt(1 + ln n)`. With an illustrative `z_base = 2.0`:

| n | 1 | 2 | 3 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- |
| `z_need` | 2.0 | 2.6 | 2.9 | 3.5 | 3.6 | 3.6 |

## Futures inventory spread

> **Status:** partly implemented: the lookahead check in M1; the lifecycle and every `rci next` and `rci status` line shown in M2; data checks, mutants, null worlds, and verdicts planned for M3, data plugins for M5, and the backtest consumer for M6.

A weekly time-series idea on three futures legs. Most of the work is in the data layer, because the core field is a weekly industry survey with release-time, revision, and break problems. The case shows how much can be decided before anyone computes a return.

```text
contract    decision time, holding, instruments, power (about 520 weekly decisions)
data v1     floor checks, 7 tests, 7 mutants, a coverage gap closed before approval;
            human decision: the core field has revised values only -> explore-grade
feat v1     trial 1 TAINTED (two-sided seasonal filter); trial 2 PASS, capped at L1
data v2     reopened: a first-release archive is found  -->  feat goes STALE
feat v2     trial 3 PASS at L2, cap lifted
signal      8 submissions FAILED -> PAUSED -> a person extends by 2 -> trial 9 PASS
algo        a futures backtester is the evaluator
checkpoints seal -> forward -> capital -> monitor
```

### Futures: the claim and the contract

> **Status:** partly implemented: `Contract`, derive, and approve in M2; the derived spec planned for M3.

The idea: when rebar inventory falls faster than its seasonal norm, the futures mill margin (rebar − 1.6 × iron ore − 0.5 × coke) widens over the following week. The contract is derived backwards from the trade, before any search:

| Item | Value | What it constrains later |
| --- | --- | --- |
| Decision time | 21:00 night session on the day the weekly inventory survey is released | data released after 21:00 cannot be used that week |
| Holding | one week | the label is fixed now; another horizon is a new contract version |
| Instruments | specific contract months of rebar, iron ore, and coke | labels must use tradable prices, not a continuous main-contract series, whose roll adjustment looks ahead |
| Tiers | explore 2014-03 to 2024-08; sealed 2024-09 to 2026-08 | the agent never sees the sealed segment |
| Power | about 520 weekly decisions; over ten years the standard error of an annual Sharpe ratio is about 0.32, so t > 2 needs a Sharpe of at least 0.63 | if data problems shrink the sample, the bar rises |
| Controls | pure seasonality; pure margin mean reversion; random entry at the same frequency | become null worlds and baselines |
| Costs | three legs rebalanced weekly, including close-today fees | a high cost ceiling |

```python
from research_ci.api.contract import Contract

CONTRACT = Contract(
    template="factor",
    claim="faster-than-seasonal rebar inventory draws widen the mill margin over the next week",
    explore_end="2024-08-31T00:00:00Z",
    seal_end="2026-08-31T00:00:00Z",
    budget=8,
    skip=(),
)
```

The budget is small because weekly data has little power and every submission raises the threshold. Today `Contract` holds exactly these six fields and validates them: the claim must not be the TODO placeholder, both times need a time zone, `seal_end` must follow `explore_end`, the budget must allow one submission, and `skip` may name only optional stages of the template. The trade details in the table, and the spec derived from them, are planned for `derive` in M3.

| Step | Who | Command | Engine | Ledger |
| --- | --- | --- | --- | --- |
| C1 | agent | `rci init ideas/mill-margin --template factor` | writes a `contract.py` skeleton, opens the ledger | `init` |
| C2 | agent | `rci contract derive -i ideas/mill-margin` | loads and validates `contract.py`; checks the template and `skip` | `register` on `contract`, with the file hash and every field |
| C3 | human | `rci approve --stage contract -i ideas/mill-margin` | refuses if `contract.py` changed since derive; shows the fields as recorded in the ledger | `approve`, with the freeze hash, the budget, and `skip` |

```text
$ rci approve --stage contract -i ideas/mill-margin
approve contract v1 of mill-margin
  budget: 8
  claim: faster-than-seasonal rebar inventory draws widen the mill margin over the next week
  explore_end: 2024-08-31T00:00:00Z
  seal_end: 2026-08-31T00:00:00Z
  skip: []
  template: factor
  frozen dependencies:
    contract             0f3db034f091
Freeze contract v1? [y/N]: y
frozen contract v1: 319c24f1fad9
to keep it in git: git add ideas/mill-margin/.rci ideas/mill-margin/contract.py && git commit -m 'rci: freeze contract of mill-margin'
```

Approval reads the fields from the ledger, not from the copy in `.rci/contract.json`, so editing the copy changes nothing. Git is not part of the trust model: the engine never calls it; only its own hashes and the ledger count.

### Futures: data layer, steps D1 to D10

> **Status:** planned: floor checks, mutants, and calibration in M3; data plugins and `rci data run` in M5. The FieldTable is M1; the state transitions and `rci next` lines are M2.

The data layer does not ask whether the data is good. It asks whether these fields can support this contract: the same field can be fine for a weekly strategy and useless for an intraday one. So the engine runs one set of checks with parameters taken from the contract. Generic floor checks are blind to the strategy (monotone timestamps, duplicates, time zones, gaps, `available_at` not before `event_time`, level breaks, units). Contract floor checks take their parameters from the contract (`available_at` before the decision time, coverage for power, freshness, expired contracts kept, tradable label prices). Source knowledge lives in the data plugin, reviewed once and reused by every idea (release calendars, vintage archives, methodology changes). The agent cannot remove or loosen floor checks; its own tests are written before they run, only tighten, are run by the engine, and must be able to fail.

| Step | Who | Command | What happens | Ledger |
| --- | --- | --- | --- | --- |
| D1 | agent | writes `data.py` | lists the fields the claim needs | nothing |
| D2 | engine | `rci data run` (planned) | data plugins fetch the fields into a FieldTable | nothing: exploration does not change state |
| D3 | engine | same | floor checks run on every field | nothing |
| D4 | agent | `rci data run`, then `rci data register` (planned) | explores the explore tier, then registers its tests | `register` |
| D5 | challenger | `rci data challenge` (planned) | adds tests and domain pitfalls | `challenge` |
| D6 | engine | `rci data check` (planned) | calibration with mutants and a coverage check; fails twice, then passes | `check`, three times |
| D7 | human | reads the StageCard | one decision, before any return is seen | none of its own; the choice is part of what D8 freezes |
| D8 | human | `rci approve --stage data` | freezes the stage | `approve` |
| D9 | agent | `rci data submit` (planned) | the engine runs every check and applies the `on_fail` policies | `submit` |
| D10 | engine | part of the verdict | hands a frozen panel and its limits to the next layer | part of `submit` (planned) |

**D1 and D2, the request and the FieldTable.** The agent names fields from the catalog: weekly social inventory, mill inventory, and blast furnace utilization from an industry survey; monthly crude steel output from the statistics bureau; contract-month prices with the night session; the trading calendar. A source outside the catalog can be proposed as a new plugin and counts as explore-grade until it is reviewed. `available_at` and `vintage` come from the plugin; the agent never fills them in. The M1 `field_table()` already normalizes times to nanosecond UTC, defaults `vintage` to 0, and rejects duplicates and revisions published before the vintage they revise. Rows shown in exchange time:

```text
entity    field                      event_time  available_at          value   vintage  source
rebar     social_inventory           2021-03-18  2021-03-18 16:00 +08  1325.4  1        survey@v3
rebar     blast_furnace_utilization  2021-03-19  2021-03-19 15:00 +08  84.1    0        survey@v3
national  crude_steel_output         2021-02     2021-03-15 10:00 +08  ...     0        stats@v1
```

**D3, floor checks.** They run automatically, before the agent writes anything:

```text
[generic]
ok    timestamps monotone, no duplicates      all fields
ok    available_at >= event_time              all fields
warn  gaps                                    social_inventory: 1-2 weeks missing around each Spring Festival
ok    units and magnitude                     all fields
[contract]
ok    available_at <= decision time           inventory 16:00 < decision 21:00
warn  freshness at decision time              crude_steel_output lags about 45 days
ok    coverage from 2014-03                   all fields
warn  vintages                                social_inventory has revised values only, no first release
ok    label fields tradable                   contract months, not a continuous series
```

**D4, explore, then register tests.** On the explore tier the agent finds a step in inventory in mid-2019 (probably a larger survey sample), Spring Festival weeks published as one merged value, and a few releases moved from Thursday to Friday. Each finding becomes a test that declares what it `covers` and what happens `on_fail` (the decorator names are a sketch; the M3 interface may differ):

```python
@covers("level_break")
@on_fail("split_and_recompute_power")
def test_inventory_break(f):
    return structural_break(f.social_inventory, max_breaks=0)


@covers("release_time_wrong", "leak")
@on_fail("exception_card")
def test_release_reaction(f, px):
    return reaction_test(f.social_inventory, px.rebar, pre="24h", post="24h", rule="pre_corr < 0.5 * post_corr")
```

The other two are `test_spring_festival` (covers `holiday_merge`; masks the weeks on failure) and `test_publish_day` (covers `publish_day_shift`; uses the actual `available_at` on failure).

**D5, the challenger.** It sees the contract, the field metadata, and the agent's tests, but not the agent's reasoning. It adds a night-session alignment test (a Thursday night session belongs to Friday's trading day, and the order price must come after the release) and a consistency test between social and mill inventory breaks. The domain pitfalls it writes, such as merged holiday weeks and night-session assignment, go into the data plugin's pitfall library.

**D6, calibration.** The engine plants defects in a copy of the FieldTable. Every test must pass on clean data and fail on the defect it claims to cover.

| Mutant | Test that must catch it | Result |
| --- | --- | --- |
| M1 inventory `available_at` one day early | `test_publish_day` | caught |
| M2 inventory replaced by next week's price change (a planted leak) | `test_release_reaction` | caught |
| M3 a +15% step inserted in 2017 | `test_inventory_break` | caught |
| M4 Spring Festival weeks replaced by merged values | `test_spring_festival` | caught |
| M5 night session assigned to the same calendar day | night-session alignment | caught |
| M6 a break in mill inventory only | social-versus-mill consistency | not caught: back to the challenger |
| clean synthetic data | all tests | all pass |

A failed check happens before the freeze and does not count as a trial. `rci next` passes the note on:

```text
$ rci next -i ideas/mill-margin
agent      rci data check                       mutant M6 (a break in mill inventory only) was not caught
```

The challenger rewrites its test (`rci data challenge` is allowed again from CHALLENGED), and the next check catches M6 but fails on coverage: the plugin marks `social_inventory` as revised-only, and no test fails when first releases are replaced by revisions. The agent adds a `revision_drift` test that compares first releases with revisions in the years where an archive exists; a seventh mutant (M7, first release replaced by revision) shows that it can fail, and the third check passes. Approval is not offered while a coverage gap is open.

Because every source has the same table shape, most mutants come from about ten generic operators: move `available_at` earlier, replace a value with a future one, multiply by (1 + k) after a date, delete or merge a span, duplicate timestamps, replace first releases with revisions, shift by some hours, insert spikes, multiply by 10 or 1000, and drop expired instruments. The challenger writes the domain-specific ones. Two rules keep mutants honest: whoever writes a test does not write its mutant, and a mutant must actually harm (a naive feature computed on it must move clearly, for example an IC that jumps when a leak is planted).

**D7, the StageCard.** The person sees one screen and makes one decision, before any return exists:

```text
StageCard . mill-margin . data v1                                 illustrative
floor checks   9: 6 ok, 3 warn (gaps, freshness, vintages)
tests          7 (agent 5, challenger 2); mutants 7 of 7 caught; coverage complete
------------------------------------------------------------------------------
Decide: social_inventory has revised values only, and it is the core field.
  A  accept it as explore-grade: no historical verdict above L1, confirmation
     only from forward data, which gives about 52 decisions a year
  B  find an archive of first-release values before going on
```

The person chooses A for now, and the agent starts looking for an archive.

**D8, the freeze.** `rci approve --stage data` prints and freezes what M2 covers: the `contract.py` hash, the engine source tree (`engine`), the interpreter (`python`), the engine's own dependencies (`engine_packages`), and the upstream freeze hashes. From M3 it also freezes the registered tests, the null and pitfall list, plugin hashes, and the data snapshot. The research environment is recorded with each submission but not frozen, so adding a research library does not taint a resubmission.

**D9, the run.** `rci next` now says `rci data submit ... (8 of 8 submissions left)`. The engine runs the floor checks and the seven tests on the explore tier, and the `on_fail` policies take effect: the 18 Spring Festival weeks from 2015 to 2024 are masked, the history is split at the 2019 break, and the shifted release days use their actual `available_at`. Power is then recomputed on what is left; if the sample no longer supports the contract, the verdict is an insufficient-power card and the idea goes back to the contract. The sealed segment is not downloaded during exploration; the same tests run on it later, inside the seal checkpoint, which returns pass or fail only.

**D10, the handoff.** Feat may read only this frozen output:

```text
panel                 frozen hash
fields_for_verdict    blast_furnace_utilization, crude_steel_output (45-day lag), contract-month prices, calendar
fields_explore_only   social_inventory, mill_inventory (revised values only): any verdict using them is capped at L1
masks, splits         Spring Festival 2015-2024 (18 weeks); the 2019-06 break
```

### Futures: feat layer

> **Status:** partly implemented: `step` and `check_leaks`, which catch the lookahead below, in M1; angles, null worlds, calibration, and verdicts planned for M3.

The claim at this layer: at the decision time, the deviation of the inventory draw from its seasonal norm predicts the next week's margin change, beyond pure seasonality and beyond margin mean reversion.

| Alternative explanation | Null world |
| --- | --- |
| it is only seasonality | `ShiftTime` by whole years |
| it is only margin mean reversion | `PermuteWithin(bucket_by="margin_zscore")` |
| it is chance | `BlockPermute` by four-week blocks |
| the inventory mechanism is not needed | `ReplaceField("social_inventory")` (mechanism ablation) |

The other registered angles are a secondary prediction (the effect should be stronger in the peak construction season) and stability across years. Calibration tests each null world in both directions. A `StandIn` built from seasonality plus noise must be killed by the whole-year shift. A `Planted` feature, the stand-in plus a little of the true residual label (a residual IC of about 1.5 times the MDE), must survive; it uses labels on purpose, exists only to test the null world, and is never a candidate.

**Trial 1 is TAINTED.** The agent's first feature subtracts a centered seasonal fit, which uses weeks after the decision time. The stepper calls the feature once per decision time on the panel known at that time and compares the result with one full-history run; a mismatch that remains when the lookback is widened to all history is lookahead. This check runs today:

```text
check_leaks(features, fields, decision_times)
LeakReport(kind='lookahead', checked=520, mismatches=517,
           first=Mismatch(t=2014-10-09 13:00 UTC, entity='rebar', column='inv_dev', stepped=-2.59, full=-4.48))

$ rci next -i ideas/mill-margin
agent      rci feat submit                      TAINTED; change the research code (7 of 8 submissions left)
```

The trial counts. Had the agent declared a 13-week lookback for a 26-week trailing mean instead, `check_leaks` would report `kind='lookback'`: the mismatch disappears with the full history, so the window is too short, not cheating, and the agent is asked to declare a longer one.

**Trial 2 passes, capped.** The agent replaces the centered fit with a trailing seasonal norm (the same week in earlier years only):

```text
VerdictCard . mill-margin . feat v1 . inventory draw vs seasonal norm      trial 2 / 8   z_need 2.6
stepper       0 mismatches
sample        498 weekly decisions (18 masked), 2 segments split at 2019-06
calibration   stand-ins 3 of 3 killed, planted 3 of 3 kept
------------------------------------------------------------------------------------------
angle                          null world                            z      result
increment over seasonality     ShiftTime(whole years)                3.1    pass
increment over mean reversion  PermuteWithin(margin_zscore)          2.8    pass
stronger in peak season        PermuteWithin(month)                  2.7    pass
mechanism ablation             ReplaceField(social_inventory)        -      pass   IC 0.052 -> 0.012
stability                      by year                               -      8 of 10 years same sign
------------------------------------------------------------------------------------------
PASS . L2 by angles . capped at L1: social_inventory is explore-grade (revised values only)
```

This is what the choice at D7 meant: the idea may move on, but nothing it produces counts as evidence yet.

### Futures: reopening data

> **Status:** implemented (M2) for the state changes and outputs shown; the data work itself is planned (M3, M5).

The agent finds an archive of the survey's first-release values from 2016 on and reopens the data layer. `rci data register` on a PASSED stage starts version 2, and the engine marks every downstream stage that had started as STALE:

```text
$ rci status -i ideas/mill-margin
mill-margin (factor)
  contract  PASSED      v1
  data      REGISTERED  v2   1/8
  feat      STALE       v1   2/8     data was reopened
  signal    OPEN        v1   0/8
  algo      OPEN        v1   0/8
  seal      OPEN        v1
  forward   OPEN        v1
  capital   OPEN        v1
  monitor   OPEN        v1
```

Data v2 goes through challenge, check, approval, and submission again. `social_inventory` is now verdict-grade from 2016, `revision_drift` runs on the archive years, and power is recomputed: about 440 decisions over 8.7 years, a standard error of about 0.34, so the bar rises to a Sharpe of about 0.68. Once data passes, `rci next` points back at the stale layer:

```text
agent      rci feat register                    data was reopened; register again to start v2
```

Feat v2 registers the same tests, and its first submission is the layer's third: submissions count across versions, so reopening never refreshes a budget. At `z_need` 2.9 it passes at L2 with the cap lifted.

### Futures: signal layer and a spent budget

> **Status:** partly implemented: the budget, PAUSED, `--extend`, and the counts in M2; walk-forward, `GridMax`, and dedup planned for M3.

The signal layer combines the passed feature with blast furnace utilization and the margin level into one weekly score. Anything that learns from labels, such as a tree model or thresholds chosen by IC, must be written as `fit`. The engine then runs a purged walk-forward: `fit` receives only training-fold labels, without samples whose label window overlaps the test period. New angles are a mechanism ablation (without inventory, is the score just margin mean reversion?) and a secondary prediction (the hot-rolled coil margin should react the same way to its own inventory). The lookback and entry threshold are declared as one parameter family; `GridMax` runs the whole grid in every null world and keeps the maximum, so the grid is one submission and its search is priced in.

The agent tries gradient-boosted trees on about 440 samples, with different depths and feature sets. Out of fold, every one fails the increment over margin mean reversion: the trees learn the margin level itself. Submission 6 is a retuned submission 4 whose scores correlate 0.95 with it; above 0.9 it counts as the same idea and still costs a submission. The eighth failure pauses the layer, and `rci status` shows `signal PAUSED v1 8/8`:

```text
$ rci next -i ideas/mill-margin
human      rci approve --stage signal --extend N all 8 submissions used; extend the budget or drop the idea
```

The StageCard lists all eight submissions and every registered angle, failed ones included. Every failure is the same angle, which points to a simpler model, so the person extends by two instead of dropping the idea:

```text
$ rci approve --stage signal --extend 2 -i ideas/mill-margin
signal is PAUSED; 8 of 8 submissions used
Add 2 submissions to signal? [y/N]: y
signal is FAILED again

$ rci next -i ideas/mill-margin
agent      rci signal submit                    FAILED: budget extended by 2; change the research code (2 of 10 submissions left)
```

The extension adds submissions but does not reset n. Submission 9, a linear combination, faces `z_need` 3.6; its increment angle reaches z 3.8 and it passes at L2.

### Futures: algo and the checkpoints

> **Status:** planned: the backtest consumer in M6; checkpoint commands later.

The algo layer turns the score into positions, and the backtest is its evaluator: a futures backtester plugged in as a `ConsumerPlugin` that handles contract-month rolls, delivery months, margin, limit-up and limit-down days, close-today fees, and integer lots. Its angles are the net Sharpe after costs against `RandomizeTiming(keep="rate")` (random entries at the same frequency), a parameter plateau, and two limit angles, maximum drawdown and margin usage, compared directly. Doubled costs are reported and flagged on the capital card if they turn the net negative.

| Checkpoint | Plan registered before it runs | What it can show here |
| --- | --- | --- |
| seal | one candidate, the frozen algo configuration, run once on 2024-09 to 2026-08; pass or fail only; the data tests run on the segment too | the only historical out-of-sample check; each idea can use a sealed segment once. A research agent that is an LLM may know what happened in older years, so a recent sealed segment is the cleaner one. |
| forward | weekly, from the freeze on | about 52 decisions a year: too few for power, so mainly a check that live results match the backtest |
| capital | the CapitalCard | a person decides size, knowing the power bar was raised along the way |
| monitor | drift rules: a survey calendar or methodology change reopens data; live-versus-backtest slippage reopens algo | run by the scheduler |

### Futures: what can go wrong

| What goes wrong | Caught by | Result | Status |
| --- | --- | --- | --- |
| the survey is stamped a day earlier than it was published | mutant M1 and `test_publish_day` | the actual `available_at` is used | planned (M3) |
| a two-sided seasonal filter | stepper versus full run | TAINTED, and the submission counts | implemented (M1 check, M2 state) |
| a declared lookback shorter than the window used | stepper with the full history | `kind='lookback'`: declare a longer window, no taint | implemented (M1) |
| the core field has revised values only | vintage floor check, coverage check | explore-grade, verdicts capped at L1 | planned (M3) |
| a level break in 2019 | `test_inventory_break` | history split, power recomputed, possibly back to the contract | planned (M3) |
| a continuous main-contract series used for labels | contract floor check | the field is rejected for labels | planned (M3) |
| upstream data reopened | lineage | downstream stages STALE until they register again | implemented (M2) |
| `contract.py` edited after the freeze, for example to a two-week holding | freeze comparison at submit | TAINTED: `frozen dependencies changed: contract`; a new horizon needs `rci contract derive` again, which makes every later stage STALE | implemented (M2) |
| the budget runs out | submission count in the ledger | PAUSED until a person extends or drops the idea | implemented (M2) |
| the last failed submission is deleted to win a trial back | head anchor in `$RCI_HOME/heads/` | every command refuses: records were removed or rewritten | implemented (M2) |

### Futures: what this case taught the design

- The data layer judges fields against a contract. The same checks with different parameters serve every strategy, so the engine does not grow a check per idea.
- Many ideas should stop or pause in the data layer. A core field with only revised values cannot give a historical verdict, and weekly forward data brings about 52 samples a year.
- Tests must be able to fail. Mutants, separation of duties, and "the mutant must actually harm" make the agent's tests trustworthy without anyone reading them closely.
- A risk the plugin flags but no test covers is a coverage gap, and it blocks approval.
- The person decides once, on one screen, before any return exists.
- Domain pitfalls accumulate in plugins. The engine stays small; the plugin library grows.

## Cross-venue maker-taker

> **Status:** planned: this case runs end to end in M6 (execution simulator, fill models, randomized decisions, limit angles), and the checkpoints later. The lifecycle, freezes, taints, and every `rci next` and `rci status` line shown are implemented (M2).

A strategy whose outcome depends on its own orders, taken through every layer and checkpoint. It shows where action-free evidence ends and where fill models, stress worlds, and a live experiment have to take over.

```text
step 0    oracle bound from public trades, no strategy code
contract  decision time = venue A taker trade - p95 cancel latency
data      receive-time stamps, latency table, clock skew
feat      toxicity: adverse move on B after A taker trades            (action-free label)
signal    expected net per opportunity, from past A taker trades
          that crossed a hypothetical quote                           (action-free label)
algo      quote, cancel, size, risk: fill models, RandomizeTiming,
          stress worlds, limit angles                                 (depends on our orders)
seal      cheap historical check
forward   small randomized live experiment; calibrates the fill model
capital   CapitalCard
monitor   drift in the wear structure reopens feat, signal, or algo
```

### Maker-taker: the strategy and the open hedge decision

> **Status:** planned (M6).

Venue A has thin liquidity and venue B deep liquidity. The strategy is opportunity-driven, not continuous market making:

```text
spread between A and B wide enough?
   |
   v
post a maker quote on A, priced against the hedge price on B
   |
   +-- B moves and the spread is gone, or timeout ---> cancel
   |
   +-- filled by a taker on A ---> hedge at once with a taker order on B, whatever the market does
   v
book the cycle on its own:
   profit = spread locked at the fill
   wear   = hedge slippage + adverse move during the hedge latency + fees on both venues
```

Every fill is hedged immediately, so net inventory stays at zero: there is no one-sided inventory and no inventory skew in the quotes. What remains is leg risk during the hedge latency, residual delta from partial fills and minimum order sizes, and hedge failures. After each hedge, one leg is left open on each venue, and how those legs are closed is an open decision:

| Option | Meaning | Extra algo angles |
| --- | --- | --- |
| A | wait for the reverse spread and unwind with a reverse cycle | none beyond the base set |
| B | hold both legs and close them periodically | funding accrued over the holding time; margin usage, which limits concurrent quotes; distance to liquidation of each leg under a 10% move (a limit angle) |
| C | close each cycle's legs separately | the closing trades' cost enters each cycle's wear |

This walkthrough assumes A. The choice changes only which angles the algo layer registers.

### Maker-taker: the claim and the contract

> **Status:** partly implemented: `Contract` and the `markout` label (basis points, signed by side, from prices as known at each time) in M2; the oracle bound and the derived spec planned for M3.

```text
daily net = (A taker fills against us) x (delta - adverse move on B after the fill - hedge cost - fees) - fixed costs
```

Step 0 needs no strategy code. Public trade records give an oracle upper bound: taker flow on A, the adverse move after it, competition from other makers, hedge cost on B, and capacity. For maker ideas this bound is a mandatory kill check; an idea on a market with almost no taker flow dies on the first day instead of after months. What step 0 cannot see is how often we would be filled once our quotes are there, because that depends on our own orders. So the return formula is split by what can be observed without us:

| Term | Observable without our orders | Layer |
| --- | --- | --- |
| adverse move (toxicity) | the move on B in the taker's direction after each A taker trade, at the hedge latency, 1 s, and 5 s | feat |
| arrivals | A taker volume that crosses a given price in the next interval | feat |
| net per opportunity if filled | the locked spread minus a hedge simulated on B's recorded book, the adverse move, and fees, on past A taker trades that crossed a hypothetical quote | signal |
| whether we are filled, where to quote, when to cancel, how much | needs a fill model, and in the end live fills | algo |

To dodge a toxic taker, our cancel must land before the taker's order does, and cancels take time. So the decision time is the event time minus the cancel latency, and that is where the engine calls `asof()`:

```text
       information cut-off               taker trade on A
  -----------------|--------------------------|------------->
          t - cancel latency (p95)            t
```

```python
from research_ci.api.contract import Contract

CONTRACT = Contract(
    template="maker_arb",
    claim="maker quotes on venue A, hedged at once on venue B, earn a positive net per cycle after wear",
    explore_end="2026-06-30T00:00:00Z",
    seal_end="2026-08-31T00:00:00Z",
    budget=20,
)
```

The planned derive turns this into a spec: 41,200 A taker trades in the explore tier, merged into 3,100 independent clusters (gap 1 s); the label `markout_h = side x (B mid(t + h) / B mid(t) - 1)` in basis points for h at the hedge latency p95 (60 ms), 1 s, and 5 s; and an economic bar: with delta 4 bp, a mean adverse move of 3.1 bp, and 1.5 bp of hedge cost and fees, each fill loses 0.6 bp today, and that must turn positive.

### Maker-taker: data layer and the latency table

> **Status:** planned: checks in M3, plugins in M5; the FieldTable and `asof` are M1.

Besides the market fields (A trades with aggressor side, A best quotes, B's order book), the data layer of an execution strategy delivers what the decision time is derived from. Every `available_at` is the local receive time, never a venue timestamp. A latency table, measured from our own order telemetry, gives cancel latency (p50 22 ms, p95 35 ms, p99 80 ms), order-to-acknowledgement, and hedge latency (p95 60 ms). Clock skew between our host and each venue is measured, not assumed (p99 ±5 ms). The data tests check that the latency sample is large enough and stable across hours of the day, that the skew was measured, and that feed gaps are found and masked. The steps are the same as in the futures case, and data passes on its first submission.

### Maker-taker: feat layer, steps F0 to F10

> **Status:** planned (M3); the stepper that catches the lookahead in F7 is M1, and the states and `rci next` lines are M2.

| Step | Who | Command | What happens | Counts as a trial |
| --- | --- | --- | --- | --- |
| F0 | engine | none | data hands over fields and the latency table | no |
| F1 | engine | part of the spec (planned) | decision time = trade - 35 ms; economic bar: -0.6 bp per fill must turn positive | no |
| F2 | agent | `rci feat explore` (planned) | writes features, explores without limit | no |
| F3 | agent | `rci feat register` (planned) | angles, mechanism fields, exposed parameters | no |
| F4 | challenger | `rci feat challenge` (planned) | five null worlds; the clock-skew one is too narrow | no |
| F5 | engine | `rci feat check` (planned) | a stand-in survives; back to the challenger | no |
| F6 | human | `rci approve --stage feat` | freezes the tests | no |
| F7 | agent | `rci feat submit` (planned) | lookahead: TAINTED | yes |
| F8 | agent | `rci feat submit` | the economic angle fails: FAIL, evidence at L2 | yes |
| F9 | human | reads the VerdictCard | may kill or ask; cannot turn a FAIL into a PASS | no |
| F10 | agent | `rci feat submit` | PASS at L3; the feature is stored | yes |

**F2 and F3, features and angles.** The features are the recent move on B, B's book imbalance, and how stale A's quote is. The angles are registered before any submission (API sketch):

```python
@Angle("increment_over_b_move", kind="stat")
def increment(feat, label, ctx):
    return rank_ic(residualize(feat.score(), ctx.b_move_200ms), label.markout_60ms)


@Angle("economic", kind="stat")
def economic(feat, label, ctx):
    return net_edge_after_filter(feat.score(), label, ctx.costs, drop=0.2)


MECHANISM_FIELDS = ["b_mid", "b_bid_size", "b_ask_size"]
EXPOSED = {"latency_quantile": ["p50", "p95", "p99"]}
```

The other angles are predictive power, a horizon shape (the adverse move should grow up to the hedge latency and then flatten), and latency robustness, whose registered rule is that p95 must pass and p99 is for reference. An evaluator is any function that returns one number; the engine never reads its meaning, and any search inside it is repeated in every null world. The size of the incoming taker trade is not a feature: at the cancel decision the trade has not arrived, so using it is lookahead. It can still be a bucket on the evaluation side.

**F4, null worlds**, written by the challenger into `nulls/`:

| Alternative explanation | Null world |
| --- | --- |
| B already moved; the takers are arbitrageurs | `PermuteWithin(bucket_by="b_move_200ms")` |
| time of day | `PermuteWithin(bucket_by="hour")` |
| large trades are toxic (evaluation-side buckets only) | `PermuteWithin(bucket_by="trade_size")` |
| a false lead from clock skew | `ShiftTime(venue="a", max_ms=2)`: too narrow |
| chance | `BlockPermute(by="cluster")` |

Shuffles move whole event clusters, never single trades. Nulls that only reorder labels are cheap (200 or more runs); nulls that perturb inputs and recompute features, such as the clock shift, the ablation, and the latency quantiles, are expensive (30 to 50 runs in parallel).

**F5, calibration.** For each null world, the engine builds a stand-in that carries only the alternative explanation X, and a planted feature that also carries true information:

```python
stand_in = rank(X) + noise(match_corr_with=X, like=feat)
planted = stand_in + lam * residual(label, on=X)
must_kill = run_angle(stand_in, null, reps=50).fail_rate >= 0.95
must_keep = run_angle(planted, null, reps=50).pass_rate >= 0.80
preserves = abs(corr(null(feat), X) - corr(feat, X)) < tol
```

The other null worlds behave (stand-ins killed 49 or 50 of 50, planted features kept 43 to 46 of 50). The clock shift kills only 31 of 50 stand-ins: a B lead computed from a clock that is 3 ms off survives a ±2 ms shift, and the measured skew is ±5 ms. The check fails before the freeze and costs no trial:

```text
$ rci next -i ideas/mm-toxicity
agent      rci feat check                       stand-in 'B lead from a clock 3 ms off' survived ShiftTime(±2 ms): 31/50 killed
```

The challenger widens the shift to the measured ±5 ms, the stand-in is killed 50 of 50, and the check passes. At F6 the StageCard shows the angles, the null worlds, and the challenger's strongest objection (A's takers are mostly arbitrageurs, all the toxicity comes from B having moved, and the features add nothing), and the person approves.

**F7, trial 1 is TAINTED.** `b_ret_50ms` uses the window (trade - 50 ms, trade], whose last 35 ms fall after the decision time. The agent wrote it on full data, where the trade time looked natural. The stepper reports a mismatch at 41,200 of 41,200 decision times: the cut-off must be t minus the cancel latency, not the trade time. `rci next` reports `TAINTED; change the research code (19 of 20 submissions left)`.

**F8, trial 2 fails.** Every angle passes except the economic one: dropping the most toxic 20% moves the net per fill from -0.6 bp to -0.1 bp, not above zero. A registered angle failed, so the verdict is FAIL, with the evidence reaching L2. The mechanism ablation shows the IC falling to 0.01 when B's inputs are scrambled, and latency robustness passes by its rule, but at p99 (80 ms) the IC decays to zero. The feature predicts toxicity beyond "B already moved" only when cancels land within 40 ms, which becomes a hard infrastructure requirement for the algo layer and the capital card.

**F9 and F10.** The person may kill the idea or ask for more, but cannot turn the FAIL into a PASS: verdicts must be reproducible. The agent adds a sudden thinning of A's best level (other makers pulling out) and resubmits against the same frozen tests:

```text
VerdictCard . mm-toxicity . feat v1 . toxicity score                trial 3 / 20   z_need 2.9
decision time   trade time - 35 ms (p95 cancel latency)             stepper: 0 mismatches
sample          41,200 taker trades -> 3,100 independent clusters
calibration     stand-ins 5 of 5 killed, planted 5 of 5 kept
------------------------------------------------------------------------------------------
angle                    null world                        z      result
predictive               block permute by cluster          4.1    pass
increment over B move    permute within B-move buckets     3.2    pass
economic (drop top 20%)  permute within B-move buckets     3.0    pass   -0.6 -> +0.4 bp
horizon shape            permute within hour               3.1    pass
mechanism ablation       scramble B inputs                 -      pass   IC -> 0.01
latency robustness       p50 / p95 / p99                   -      pass   p99 fails (signal gone at 80 ms)
------------------------------------------------------------------------------------------
PASS . L3 . caps: none . requirement for algo: cancel latency <= 40 ms
```

The card is rendered from the verdict without any LLM. Two failures were stopped by two different mechanisms: the weak null world by calibration before the freeze, the lookahead by the stepper at submission.

### Maker-taker: signal layer

> **Status:** planned (M3; the lineage check on inputs arrives with the runner in M4).

The claim: at t minus the cancel latency, a score built from toxicity, arrivals, and the spread predicts the net a fill at the hypothetical quote would earn, and the best-scored opportunities earn a positive net after fees and the simulated hedge.

The label needs no fill model. Every past A taker trade that crossed the hypothetical quote price (the B hedge price plus or minus fees and the target) is an "if filled" sample. Its label is the locked spread, minus a unit-size hedge walked through B's recorded book at the trade time plus the hedge latency, minus the adverse move and fees. Whether our quote would actually have been filled depends on queue position and on our own orders, and is left to algo. Like feat, this layer uses only action-free labels.

| Step | Who | Command | What happens |
| --- | --- | --- | --- |
| S1 | agent | `rci signal register` (planned) | angles: predictive rank IC, increment over a spread-only score, net in the top predicted bucket, stability by day |
| S2 | challenger | `rci signal challenge` (planned) | `PermuteWithin(bucket_by="spread")`, `PermuteWithin(bucket_by="hour")`, `ShiftTime(venue="a", max_ms=5)`, `BlockPermute(by="cluster")` |
| S3 to S5 | engine, human, agent | `rci signal check`, `rci approve --stage signal`, `rci signal submit` | a stand-in (the spread plus noise) is killed within spread buckets and a planted score survives; the freeze includes the feat freeze hash (`upstream:feat`); PASS at L3 on the first submission |

The inputs may only be the stored `toxicity_score` and the fields data passed; the signal layer cannot quietly compute an unverified feature of its own. Thresholds chosen from history, such as "cut when toxicity is in the top 20%", learn from labels and must be written as `fit`, with the walk-forward run by the engine.

### Maker-taker: algo layer

> **Status:** planned (M6).

The signal layer says which opportunities are worth taking. The algo layer decides what to do about them, and answers whether that can run safely at a real size, and how large. It fixes whether to quote, at what price, and when to cancel; the size per quote (a larger size locks more per cycle but slips more on B and fills less on A); how many quotes and pairs run at once (simultaneous fills crowd the hedges and share margin); hedge-failure handling (retries, escalation, how long a leg may stay open); when to clean up residual delta; automatic stops (stale data, a feed sequence gap, cancel latency above 40 ms, the daily loss limit, repeated hedge failures); and per-venue margin and exposure limits.

The output shape is Quotes. The evaluator is a `ConsumerPlugin`, an event-driven execution simulator; a bar-close backtester cannot evaluate a millisecond strategy. Fills come from a `FillModel` plugin that always runs in two modes. Optimistic: any A taker trade that crosses our price fills us in full. Pessimistic: we are filled only if our price strictly improved the best price at that moment, and only after the volume estimated to be queued ahead of us. The hedge is simulated at fill time plus the hedge latency by walking B's recorded book. Verdicts are judged under the pessimistic model; if only the optimistic version passes, the evidence is capped at L2. Because the outcome depends on our own orders, the evidence is capped at L3 until live results exist.

The registered angles come in two kinds. Statistical angles (`kind="stat"`: net per cycle under pessimistic fills, beating a spread-only baseline, daily net at the chosen size) are compared with null worlds. Limit angles (`kind="limit"`: the 1st percentile of cycle PnL, the maximum daily drawdown, how long a leg stays open under injected hedge failures) are risk limits compared directly, with no null distribution. The exposed parameters are the spread threshold, the toxicity cut, the size, and the number of open quotes.

At this layer most alternative explanations say that some optimistic assumption does not hold, so most of what the challenger writes are stress worlds in which that assumption is broken:

| Alternative explanation | Null or stress world |
| --- | --- |
| cancelling more just means fewer fills and less toxicity | `RandomizeTiming(keep="rate")`: 200 random cancel policies with the same cancel rate and random timing; the real policy must beat their distribution |
| the parameters were tuned | `GridMax` over the exposed grid, plus a plateau: neighbouring settings must also pass |
| the fill model is too kind | `Stress`: pessimistic fill probability cut by another 30% |
| all the simulated depth was ours | `Stress`: B depth halved |
| hedge failures were ignored | `Stress`: rejects and timeouts injected at the measured rate (0.4%) |
| markets were always calm | `Stress`: replayed incidents: a feed stall of several minutes, A cancel latency spiking to 500 ms, B depth dropping 90% at once |
| the venues never fail together | `Stress`: both venues disconnect |
| costs were underestimated | `Stress`: costs doubled |

Every real incident becomes a stress world that later strategies of the same kind must pass, the same way data pitfalls accumulate in plugins. Calibration has three parts:

| Part | Test | Must happen |
| --- | --- | --- |
| random-policy null | `StandIn`: random cancels at the same rate | false pass rate at most alpha |
| random-policy null | `Planted`: synthetic data with a known toxicity signal, cancels driven by it | passes |
| simulator known answers | never quote; quote far away; fees doubled; hedge latency 0 then p95; size 0 then larger; B depth zero | zero fills and PnL; zero pessimistic fills; PnL falls by exactly fills times the fee difference; adverse move grows; slippage grows with size; the hedge-failure branch runs |
| stress strength | `StandIn` without any protection | loses through the limit in the stress worlds |
| stress strength | the protected version | passes |

The labels come from the simulator, so a simulator bug would corrupt every verdict; like a data plugin, it must pass known-answer tests first. The first check fails because the stress is too mild:

```text
$ rci next -i ideas/mm-toxicity
agent      rci algo check                       the unprotected stand-in stayed inside the loss limit in the halved-depth world
```

The challenger combines halved depth with injected hedge failures, the unprotected stand-in now loses through the limit, and the check passes. A person approves; from M3 on the freeze includes plugin hashes, the fill model among them.

**Trial 1 is TAINTED by a frozen dependency.** Between approval and submission, someone recalibrates the fill model plugin. The engine compares the dependencies recorded at approval with the current ones and names what moved (the key name is illustrative):

```text
$ rci next -i ideas/mm-toxicity
agent      rci algo submit                      TAINTED: frozen dependencies changed: plugins; change the research code (19 of 20 submissions left)
```

The fix is to restore the frozen plugin version and resubmit, or to register a new algo version that freezes the new plugin and goes through check and approval again. Either way the tainted submission stays counted.

**Trial 2 passes.** The grid covers size, concurrent quotes, and every stress world; each cell is an event-driven simulation of quote, pessimistic fill, hedge, failure handling, and booking.

```text
VerdictCard . mm-toxicity . algo v1 . spread quotes + toxicity cancels      trial 2 / 20   z_need 2.6
inputs        toxicity_score (feat v1, L3); net per opportunity (signal v1, L3)
calibration   simulator known answers 7 of 7; random-cancel stand-in false pass 4% (alpha 5%);
              unprotected stand-in loses through the limit in 5 of 5 stress worlds
--------------------------------------------------------------------------------------------
                                      pessimistic                optimistic
net per cycle                         +0.4 bp                    +0.9 bp
cycles where wear > profit            38%                        31%
beats spread-only baseline            z 3.4  pass                pass
beats random cancels (same rate)      z 3.0  pass                pass
parameter plateau                     7 of 9 neighbours pass
fill probability cut 30%              +0.1 bp  pass (narrow)
capacity (depth halved)               1x +$41/day  2x +$63  4x +$58  8x -$20  -> 2x
tail cycle loss, 1st percentile       -2.1 bp    limit -5 bp          pass
max daily drawdown                    -$35       limit -$150          pass
hedge failures at 0.4%                longest open leg 180 ms, limit 500 ms   pass
feed stall of several minutes         quoting stops within 120 ms, loss $3    pass
A cancel latency 500 ms               quoting stops                           pass
both venues disconnect                1 open leg, loss $18                    pass
costs doubled (reported)              -$12/day                                flag: cost-sensitive
--------------------------------------------------------------------------------------------
PASS . L3 . capped at L3 until live results . judged on the pessimistic fill model
configuration for the live experiment: size 2x, at most 3 open quotes, daily loss limit $150
```

The output of the algo layer is an executable configuration, and it becomes the parameters of the live experiment: size, stop, and protections.

### Maker-taker: seal checkpoint

> **Status:** planned (later); the checkpoint's state transitions and `rci next` lines are M2.

Checkpoints are one-shot: the plan is registered and frozen in advance, the checkpoint runs once, and a checkpoint that ends FAILED or TAINTED cannot run again; the idea goes back to an earlier layer instead. `rci next` walks the seal checkpoint like this:

```text
agent      rci seal register                    declare the plan before the checkpoint runs
human      rci approve --stage seal             review the plan and freeze it
human      rci seal run                         run the checkpoint once
```

The plan names one candidate, the frozen algo configuration, and the sealed segment from July to August. The run is cheap and confirms that the feat and signal conclusions hold on data nobody saw. It cannot resolve the largest uncertainty in this case, the fill model; only live fills can. The ledger records that this idea has used its sealed segment.

### Maker-taker: forward checkpoint, a small live experiment

> **Status:** planned (later).

Forward, for this case, is a small live experiment, pre-registered in full:

```text
arms        each quoting period assigned at random: spread threshold {3, 4, 5} bp x toxicity cancels {on, off}
duration    derived by the engine from the MDE: at least 400 fills per arm
metrics     realized fill rate; adverse move on our own fills; hedge slippage; cancel latency; profit and wear per cycle
stop        the daily loss limit from the algo card ($150)
reconcile   every live fill against the simulator: would the pessimistic model have filled it, with what wear?
changes     none during the run; a change ends the experiment, and a new one is registered and counted
```

The agent registers the plan (`rci forward register`), a person freezes it, and the scheduler runs `rci forward --all`. The checkpoint records its one verdict when the registered duration is reached or the stop fires. Reconciliation is a floor check: a live fill the simulator cannot explain is a finding, not noise.

| Live result | Conclusion |
| --- | --- |
| fill rate between the pessimistic and optimistic models | the simulator is credible; live data calibrates the fill model plugin |
| fill rate below the pessimistic model | the simulator is too kind; every upstream verdict that depends on the fill model is downgraded |
| cancel latency p95 above 40 ms | the feat conclusion does not apply; fix the infrastructure first |

The experiment brings L4 evidence and a calibrated fill model that later maker research can reuse.

### Maker-taker: capital checkpoint

> **Status:** planned (later).

A person runs `rci capital run`, reads the CapitalCard, and decides:

```text
CapitalCard . mm-toxicity                                             illustrative
evidence      L4: live experiment, 6 arms, 2,900 fills
live vs sim   fill rate 0.31 (pessimistic 0.24, optimistic 0.45): inside the range
              net per cycle +0.5 bp (pessimistic +0.4, optimistic +0.9)
denominator   1 idea; submissions: data 1, feat 3, signal 1, algo 2; sealed segment used once
capacity      size 2x, 3 open quotes: about +$60/day at pessimistic fills
requires      cancel latency p95 <= 40 ms (feat: the signal is gone at 80 ms)
risks         cost-sensitive: doubled costs turn the daily net negative
              hedge legs closed by option A, an open decision
plan          weeks 1-2 at 1x; weeks 3-4 at 2x if the net stays inside the live range
decision      approve / reject / ask for more
```

The card is rendered from verdict fields. A written summary may be attached later (planned: the `post_verdict` hook), but every number in it must cite a verdict field, checked by code, and the remaining sentences are audited by an independent reasoning LLM from a different model family. A summary never changes a verdict.

### Maker-taker: monitor checkpoint

> **Status:** planned (later); the reopen and STALE mechanics are M2.

Monitoring watches the wear structure, because each part of the wear points at a different layer. The drift rules are registered and frozen before the checkpoint runs:

| Drift | What it means | Layer reopened |
| --- | --- | --- |
| hedge slippage grows | B depth got worse, or the size is too large | algo: size |
| adverse move during the hedge latency grows | toxicity changed, or cancels got slower | feat, or the infrastructure |
| locked spread shrinks and fills fall | a competitor arrived | signal |
| live-versus-simulator gap grows | the fill model is out of date | recalibrate the fill model plugin |
| net below the live range two weeks in a row | the pre-registered automatic pause | pause, with an exception card |

The scheduler runs `rci monitor --all`. How an ongoing monitor fits the one-shot checkpoint shape is an open decision; in this walkthrough, when a rule fires, the checkpoint records FAILED and a person reopens the layer the rule names:

```text
$ rci next -i ideas/mm-toxicity
human      rci <layer> register                 monitor ran once and ended FAILED; reopen the layer to rework, which restarts the checkpoints
```

Reopening a passed layer marks every later stage STALE, the checkpoints included, so the whole CD half runs again. After `rci algo register`, `rci status` shows `algo REGISTERED v2 2/20`, and `seal`, `forward`, `capital`, and `monitor` all `STALE v1` with the note `algo was reopened`.

### Maker-taker: what can go wrong

| What goes wrong | Caught by | Result | Status |
| --- | --- | --- | --- |
| a null world narrower than the measured clock skew | calibration: the stand-in survives | back to the challenger, no trial | planned (M3); the state loop is M2 |
| a feature computed up to the trade time instead of t minus the cancel latency | stepper | TAINTED, trial counted | implemented (M1 check, M2 state) |
| the incoming trade's size used as a feature | stepper | TAINTED; allowed only as an evaluation bucket | implemented (M1 check) |
| only the optimistic fill model passes | the cap rule | evidence capped at L2 | planned (M6) |
| stress worlds too mild | calibration: the unprotected stand-in stays inside the limit | back to the challenger | planned (M6) |
| a plugin updated after the freeze | freeze comparison at submit | TAINTED, naming the changed dependency | mechanism implemented (M2); plugins join the freeze in M3 |
| live fills below the pessimistic model | forward reconciliation | every verdict that depends on the fill model is downgraded | planned (later) |
| parameters changed during the live experiment | the frozen plan | the experiment ends; a new one is registered and counted | planned (later) |
| drift after going live | monitor rules | FAILED, the named layer reopened, the checkpoints STALE | mechanism implemented (M2); rules planned (later) |

### Maker-taker: what this case taught the design

- A strategy that depends on its own orders can still be judged early. Split the return formula: judge the action-free terms in feat and signal with observable labels, and leave fills to algo, fill models, and live trading.
- Decision times are derived from the latency of the action (event time minus cancel latency), so the data layer must deliver a latency table from our own telemetry and stamp market fields with the local receive time.
- For maker ideas, the step 0 oracle bound from public trades is a kill check that needs no strategy code.
- Null worlds are tested with stand-ins: a null narrower than a measured quantity lets a stand-in through, and calibration catches it before the freeze. A stress world is strong enough only if the unprotected stand-in loses through the limit in it.
- Randomized-decision nulls, fill models, simulator known-answer tests, the optimistic-only cap, stress worlds, and limit angles all belong to the algo layer. Signal, like feat, uses only labels that do not depend on our actions.
- Live trading is an experiment: randomized, pre-registered, unchanged while it runs, and reconciled fill by fill against the simulator.
- Monitoring watches the wear structure, and each kind of drift names the layer to reopen.

## Text event signals

> **Status:** planned: judging in M3, the forward runs later. `RandomEvents`, the LLM point-in-time rule, and the generation columns (`gen_mode`, `model_cutoff`) are planned; the M1 FieldTable has seven columns (`entity`, `field`, `event_time`, `available_at`, `value`, `vintage`, `source`).

Authors publish investment theses: a direction on one or more instruments, with a written rationale. An LLM pipeline reads news from an append-only news-event ledger and emits signals that support or refute a thesis. A much cheaper classifier is an alternative way to produce the same stance.

```text
news-event ledger: append-only, monotone cursor, ingest time = available_at
      |
      v
LLM pipeline --- compares each event with each thesis ---> signal
                                                            stance: support | refute
                                                            thesis snapshot at signal time
                                                            published_at = decision time
      |
      +-- backfilled signals (generated later, in one pass) --> explore-grade only
      +-- live signals (generated as the news arrived) ------> verdict-grade
```

### Text events: the question

> **Status:** planned (M3).

Three questions could be asked of this data. (A) Do the theses themselves have alpha when followed? That becomes the consumer's baseline. (C) Do news events predict moves in related instruments? That is a generic event study and a possible foundation. The claim here is (B): after a refute signal, do thesis-direction returns really get worse? It tests what the pipeline promises, to notice in time that a thesis is going wrong. An offline evaluation with human labels can say whether a stance is right; it cannot say whether the market followed. It is also the hardest kind of research for Research CI, because the model's pretraining can itself be a view of the future.

### Text events: the claim and the contract

> **Status:** partly implemented: `Contract`, the `holding_return` label, and the forward decision window (decision times after the freeze) in M2; abnormal-return labels and the derived spec planned for M3.

| Item | Value |
| --- | --- |
| Decision time | the signal's publish time, executed at the next tradable time; a signal published after the close trades at the next open |
| Action | cut the thesis-direction position by x% on a refute signal (a second claim adds on a support signal) |
| Holding | 1, 5, and 20 trading days |
| Label | thesis-direction abnormal return, with sector or market beta removed, from the execution time |
| Consumer | a follow-the-thesis portfolio: long bullish theses, short bearish ones, equal weight, rebalanced daily |
| Mechanism | refuting information is absorbed gradually; holders who do not react pay |
| Alternative explanations | (1) any news moves prices; (2) the market or sector moved; (3) momentum after the fact: the signal follows news that already moved the price; (4) the LLM knows the outcome; (5) signals cluster on volatile instruments |

The decision time is the signal's publish time, never the news time. Signals are often triggered by news that has already moved the price; measured from the news time, the move is already inside the label, and the signal looks prescient. It is the same problem as "event time minus cancel latency" in the maker-taker case.

```python
from research_ci.api.contract import Contract

CONTRACT = Contract(
    template="event",
    claim="after a refute signal, the thesis-direction abnormal return over the next 5 trading days is negative",
    explore_end="2026-09-11T00:00:00Z",
    seal_end="2026-09-18T00:00:00Z",
    budget=20,
)
```

The `event` template has the same four layers as `factor`, but its claim is confirmed on forward data only. The current code still gives it a seal checkpoint that cannot be skipped: `skip` accepts only optional stages, and only `taker_arb` has one. How `event` ideas should treat the seal checkpoint is an open question. Here the sealed segment is the last live week before the freeze; it is too short to confirm anything, and whatever it returns, the claim waits for forward data.

### Text events: power first

> **Status:** planned (M3).

Derive answers the first question before any layer runs: can the data answer the claim at all, and if not, when will it?

```text
spec (planned, M3)
events          refute signals merged by (instrument, trading day)
verdict-grade   live signals only: 40 events in the explore tier
effect          detect a 1% 5-day abnormal return; its volatility is about 5%
needed          N = ((z_alpha + z_beta) x 5% / 1%)^2 = ((1.96 + 0.84) x 5)^2 = 196 events
confirmation    forward only: 196 events after the freeze
rate            about 15 refute events a week -> about 13 weeks after the freeze
verdict now     INSUFFICIENT
```

INSUFFICIENT with a date is a useful result: it says when an answer becomes possible, instead of forcing a conclusion from 40 events. The person approves the contract knowing that. The layers still run on the explore tier, to freeze the pipeline and the tests and to give directional clues.

### Text events: data layer

> **Status:** planned: checks in M3, plugins in M5.

This is where the case's main finding appears.

| Data | What `available_at` must be | Pitfall |
| --- | --- | --- |
| signal | the actual publish time | the event time may be empty, and must not be filled in with the publish time |
| thesis behind a signal | the thesis snapshot carried inside the signal | evaluating with today's thesis judges old signals against a thesis edited later |
| thesis cards | the generation time | historical cards were backfilled in one pass by an LLM from old posts, and that LLM may know how things ended |
| news events | the ingest time in the news-event ledger | none: an append-only ledger with a monotone cursor is ideal point-in-time data |
| prices | the close, at the close time | corporate actions, trading calendar, mapping thesis instruments to price series |
| generation metadata | per card and signal: which model produced it, and its knowledge cutoff | recorded in the planned `gen_mode` and `model_cutoff` columns |

A new floor check follows, the point-in-time rule for LLM-generated data:

```text
verdict-grade  =  available_at is the actual publish time
              and the event is after the generating model's knowledge cutoff
              and the row was generated live, not backfilled in one pass
anything else  =  explore-grade (verdicts capped at L1)
```

Two more pitfalls are specific to this case:

- **Survivorship.** The delivery logic that pushes alerts to authors refuses closed, paused, and archived theses, which is right for delivery. A data plugin that copies it and fetches only active theses drops failed theses systematically. The plugin must fetch archived theses too.
- **Direction field.** Only newer thesis cards carry an explicit direction. For older cards it is missing or must be re-inferred, and re-inferred values are explore-grade.

Calibration plants four mutants, and each must be caught: `available_at` replaced by the news time (publish-time check), archived theses deleted (survivorship coverage check), backfilled signals marked as live (generation-mode check), and the snapshot replaced by today's thesis (version-alignment check). The steps are the usual ones, and the data layer passes with backfilled signals and cards marked explore-grade.

### Text events: feat layer

> **Status:** planned (M3).

The claim at this layer: at the publish time, a refute stance predicts a negative thesis-direction abnormal return, and still does after removing (1) any news, (2) market and sector, and (3) the pre-publish move. The features are formulas that need no `fit`: stance, information type, number of sources, novelty of the news-event cluster, and the 5-day return before publication (as a control only).

The registered angles are predictive power; the increment over random news on the same instrument; a secondary prediction on the shape of the cumulative abnormal return (if the mechanism holds, the effect builds over the days after publication; if it is complete at publication, it was momentum); dose response (more sources and harder information types should give a stronger effect); breadth across authors and theses; and another implementation: the same news events labelled by a much cheaper classifier instead of the LLM that produced the signals. If the cheaper classifier does as well, the cost falls by orders of magnitude.

| Alternative explanation | Null world |
| --- | --- |
| (1) any news moves prices | `RandomEvents`: events from the same news-event ledger, same instrument and density, not selected by the pipeline |
| (2) market or sector | `PermuteWithin(bucket_by="trading_day")` |
| (3) momentum after the fact | `PermuteWithin(bucket_by="pre_publish_return_5d")` |
| (5) volatile instruments | `PermuteWithin(bucket_by="volatility")` |
| chance | `BlockPermute` by trading-day blocks |

Explanation (4), the LLM knowing the outcome, needs no null world: the data layer already keeps backfilled rows out of verdicts. A diagnostic angle compares backfilled and live signals directly; if the backfilled ones look much better, that is direct evidence of outcome leakage, and a finding in its own right. Calibration: a `StandIn` made of the sign of the pre-publish 5-day return (pure momentum) must be killed within pre-publish-return buckets; a `StandIn` that marks random news on the same instrument must be killed by `RandomEvents`; a `Planted` refute marker built from future abnormal returns must survive.

```text
VerdictCard . refute-signals . feat v1 . refute -> 5-day abnormal return      trial 1 / 20   z_need 2.0
decision time   signal publish time, executed at the next open
calibration     stand-ins 2 of 2 killed, planted 1 of 1 kept
-------------------------------------------------------------------------------------------
                               all explore rows (capped)     verdict-grade rows (live)
events                         310                           40 of 196 needed
predictive                     -2.1%   z 3.3                 -0.9%   z 1.4
increment over random news     -1.6%   z 2.6                 -0.6%   z 0.9
cumulative return shape        builds over days 1-3          right direction, too few events
diagnostic: backfilled vs live backfilled -2.3%, live -0.9%: suspected outcome leakage
cheaper classifier vs LLM      similar effect size (explore-grade)
-------------------------------------------------------------------------------------------
PASS . L1 (cap: explore-grade data) . verdict-grade: INSUFFICIENT, 40 / 196 . decided at forward
```

The contaminated rows are what make the left column look strong, and the cap is what keeps that from counting. The layer passes so that the pipeline can be frozen; the claim itself is left to forward. How an INSUFFICIENT verdict maps onto stage states, and whether it costs a submission, is still open (today `submit` records only PASSED, FAILED, or TAINTED); this walkthrough reports it on the card next to a capped PASS.

### Text events: signal and algo

> **Status:** planned (M3 for signal, M6 for algo).

The signal layer combines stance, number of sources, and information type into one event score (shape: Events). The algo layer overlays "cut on refute" on the follow-the-thesis portfolio and is judged by a daily bar-close backtester plugged in as a `ConsumerPlugin`, with costs and a per-instrument position cap. It must beat two baselines: the portfolio without the overlay, and `RandomizeTiming(keep="rate")`, which cuts at the same frequency on random days. Size is not a constraint. The baseline portfolio can use only live thesis cards, so its history is as short as the signals'.

### Text events: checkpoints, forward only

> **Status:** planned (later); the checkpoint states and `rci next` lines are M2.

The seal checkpoint runs once on the last live week and cannot confirm anything (see the contract above). The forward checkpoint is the only source of confirmation. Its plan registers the sample size from derive (196 refute events after the freeze), a time limit, and the frozen pipeline, including the model it uses. Once it is frozen, `rci status` shows every layer PASSED and `forward FROZEN`, and `rci next` hands the idea to the scheduler:

```text
$ rci next -i ideas/refute-signals
scheduler  rci forward --all                    run the checkpoint once
```

Each weekly run appends new data to the forward store and updates a progress line on the card, for example "61 of 196 verdict-grade events, week 4 of about 13". In this walkthrough the checkpoint stays FROZEN until the registered sample size is reached and then records its one verdict; how a checkpoint that must wait for its sample maps onto the one-shot checkpoint states is not settled yet. If the pipeline switches models during the run, the new signals come from a different source: the experiment ends, and a new one is registered.

For this case the CapitalCard is really a decision about a product claim, not about capital:

| Forward result | What may be claimed |
| --- | --- |
| refute signals work | the pipeline helps authors notice in time that a thesis is going wrong; alert delivery can put refute signals with strong evidence first |
| they do not work | the pipeline is still an information service, but it cannot claim early-warning value in market terms |
| the cheaper classifier matches the LLM | switch to the cheaper classifier |
| backfilled signals beat live ones | offline evaluations of the pipeline must use live data only, or only events after the model's knowledge cutoff |

### Text events: what can go wrong

All of these are planned: the checks and power in M3, the forward checkpoint later.

| What goes wrong | Caught by | Result |
| --- | --- | --- |
| backfilled signals treated as live | generation-mode check and its mutant | explore-grade, verdicts capped at L1 |
| the news time used as the decision time | publish-time check and its mutant | refused; the label starts at the publish time |
| only active theses fetched | survivorship coverage check | the plugin must fetch archived theses |
| today's thesis used instead of the snapshot | version-alignment check | refused |
| a conclusion drawn from 40 events | power in derive | INSUFFICIENT, with the number of weeks still needed |
| stopping forward early on a good-looking week | the frozen forward plan | one verdict, at the registered sample size |
| the unselected event pool is not available | `RandomEvents` cannot be built | calibration cannot pass; "any news would do" stays open |

### Text events: what this case taught the design

1. The point-in-time rule for LLM-generated data is a data floor check: `available_at` is the actual publish time, and anything backfilled in one pass, or about events before the model's knowledge cutoff, is explore-grade. The FieldTable records how each row was generated and the model's cutoff.
2. When the sample is too small, derive says how many weeks are still needed, and the forward checkpoint accumulates data and records the verdict when the sample is reached.
3. `RandomEvents` is a null world of its own: unselected events drawn from the same event pool stand in for the signal, to rule out "any news would do".
4. Signals are evaluated against the thesis snapshot taken when they were generated, never against a later version.
5. Data plugins must fetch archived and closed objects. Research data must not copy the safety filters of a delivery system.
6. The last card can be about a product claim, not only about capital.

Two items remain open before this case can run: the knowledge cutoff of the LLM that produced the signals, and whether the unselected event pool needed by `RandomEvents` can be fetched from the news-event ledger.
