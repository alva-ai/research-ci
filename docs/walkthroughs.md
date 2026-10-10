# Walkthrough

> **Status:** everything below runs today. The signal and algo layers and the checkpoints are planned.

One idea, from an empty directory to a passed feat layer. Every block is what the command printed, with only the features paths shortened. The idea runs on the toy plugins of the engine's tests: `toy.data` (hourly closes and funding rates for three symbols on three venues) and `toy.pairs` (every venue pair of a symbol, long the venue with the lower latest funding rate; `spread_h` is the latest spread per hour). A lab registers its own plugins the same way. The toy data ends in January 2026, so the cards warn that it is old.

| Actor | Who it is | What it does here |
| --- | --- | --- |
| agent | the research agent | writes `contract.py`, `feat.py`, the null worlds, and features files; runs the checks; submits |
| human | a person at a terminal, or an agent relaying the person's words with `--relayed "<words>" --card <id>` | admits the data source; freezes each stage |
| engine | the `rci` command; deterministic, no LLM | checks, computes, judges, and appends to the ledger |

Every driver reads `rci next`: who should act, the command, and why. A separate challenger that writes null worlds against the agent's tests, and a scheduler for the forward and monitor checkpoints, are planned.

## 1. The contract

```text
$ rci init ideas/carry --template factor
created ideas/carry (factor); edit ideas/carry/contract.py, then run `rci contract derive -i ideas/carry`

$ rci next -i ideas/carry
agent      rci contract derive                  write contract.py, then derive the contract
```

The claim, the smallest effect worth having, and the alternative explanations are written before any result is seen:

```python
# ideas/carry/contract.py
from research_ci.api import Alternative, Contract, Effect, Execution, Label, Universe, Use

CONTRACT = Contract(
    template="factor",
    claim="Cross-venue funding spreads persist long enough to pay the round trip",
    decisive=(
        Effect(
            on="carry",
            at_least=0.0003,
            unit="per entry after the round trip",
            target=Use("ref.mean", {"cluster": "symbol", "top": 0.34}),
        ),
    ),
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

The decisive effect names its target: at each decision time, the mean carry after the round trip of the third of the candidates with the highest scores, each symbol weighted once. The default target, `ref.mean` without `top`, averages every candidate whatever the scores say; feat would refuse to register a method whose decisive number cannot move with the scores.

`rci contract derive` loads the plugins, stores the data in the vault, runs the protocol tests on the universe and the label, and computes power from the dispersion of the label alone:

```text
$ rci contract derive -i ideas/carry
ContractCard · carry · v1 · factor
claim      Cross-venue funding spreads persist long enough to pay the round trip
decisive   carry ≥ 3.0 bp per entry after the round trip · target ref.mean
decisions  every 1h from execution · 335 decision times
           decide on the hour; fill both legs at the close known then
data       toy.data afa06051cf36 · 3,780 rows · 9 entities · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 257.2 days old
universe   toy.pairs · median 9 candidates in 3 symbol clusters per decision · membership pit
costs      ref.flat_costs(commission_bps=1, legs=2, slippage_bps=1) · median round trip 8.0 bp
labels     carry: ref.carry · H 8h 24h
checks     [!]  data is current: latest data is 257.2 days old
           ok   universe point-in-time: 3 cut points give the same candidates
           ok   universe causal: values after 2026-01-08T04:00Z leave the candidates at it unchanged
           ok   label carry: worked examples: 2 ok
           ok   label carry: examples bite: shift, in-window changes and missing fields move it; later data does not
           ok   label carry: no data after t + h: 2 cut points, horizon 8h
           ok   label carry: no data after t + h: 2 cut points, horizon 24h
power      carry:
           H        decisions  blocks   MDE bp  FAIL if <   new data
           8h             321    41.9      5.9       -1.2       52 d
           24h            305    14.0     19.5      -11.1      17 mo
alts       tail  (engine floor: chance, timing)
used       nothing declared
history    history can confirm
budget     3    deadline 365 days
verdict    DECIDABLE
           carry: best horizon 8h, about 52 days of new data (new data)
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
----------------------------------------------------------------------------------------------------
card dfddd3a0
registered; a person freezes it with `rci approve --stage contract`
```

- `checks`: the universe gives the same candidates on data cut at three times and does not change when later values change; the label matches its worked examples, moves when its inputs move, and reads nothing after `t + h`.
- `power`: on history the 8h label can detect 5.9 bp, more than the 3 bp worth having, so history alone may not decide; about 52 days of new data would, within the 365-day deadline, so the contract is DECIDABLE. An estimate below −1.2 bp counts as FAIL.
- `alts`: the engine adds `chance` and `timing` to the idea's own alternative; each must be answered by a null world at feat.

A person reads the card and freezes it. Approval reads the card from the ledger, so the id is the one derive printed:

```text
$ rci inbox ideas
ideas/carry              rci approve --stage contract         review the contract and freeze it

$ rci approve --stage contract -i ideas/carry
approve contract v1 of carry
...                                                    (the card, as printed before)
card dfddd3a0
  frozen dependencies:
    contract             d1ffaa8ba828
Freeze contract v1? [y/N]: y
frozen contract v1: f21f844c6120 · approved by a person at a terminal
to keep it in git: git add ideas/carry/.rci ideas/carry/contract.py && git commit -m 'rci: freeze contract of carry'

$ rci status -i ideas/carry
carry (factor)
  contract  PASSED      v1           approved by a person at a terminal
  data      OPEN        v1
  feat      OPEN        v1   0/3
  signal    OPEN        v1   0/3
  algo      OPEN        v1   0/3
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

## 2. The data

`rci data check` runs the source's admission checks the first time this plugin, with these options and this code, is used anywhere on the machine: the rows replayed as of past times must agree with the full fetch, an independent source must agree on rows the engine picks, and two canaries make sure both checks can fail. Then it checks the data against the contract:

```text
$ rci data check -i ideas/carry
SourceCard · toy.data · code c3abda4b41b7
fields     close funding_interval_hours funding_rate · 9 entities · 6,750 rows
baseline   ok   values are finite: none: values null, NaN or infinite
           ok   known no earlier than the event: none: rows known before their event time
           ok   no long gaps: none: gaps longer than 3× the usual spacing
           ok   no near-duplicate events: none: events within 60s of the last
           ok   no level shifts: none: lasting moves of 5× or more
replay     ok   replay as of past times: 5 cut points agree with the full fetch
reference  ok   independent source agrees: 24 of 24 checked rows agree with the source (12 it could not check)
                source: the toy venue
                close: match 12
                funding_interval_hours: unanswered 12
                funding_rate: match 12
canaries   ok   canary: replay catches a hindsight deletion: deleting a:Z close from 2026-01-10 19:00 was caught
           ok   canary: reconciliation catches altered values: 3 of 3 altered rows caught
grades     verified: close funding_rate · unverified: funding_interval_hours
verdict    ADMISSIBLE
           a person admits it once: rci source approve
----------------------------------------------------------------------------------------------------
card ed62e081
DataCard · carry · v1
data       toy.data afa06051cf36 · 3,780 rows · 9 entities · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 257.2 days old
source     toy.data WAITING for a person [!] · admission 867ec4547679
checks     [!]  data is current: latest data is 257.2 days old
           ok   hedged legs are one asset: every candidate row holds (close within 1.2x and returns correlated ≥ 0.3)
           ok   field grades: unverified fields are not read: funding_interval_hours
treatments none needed
power      H            MDE before   after   new data
           carry 8h            5.9     5.9       52 d
           carry 24h          19.5    19.5      17 mo
grades     verified: close funding_rate · unverified: funding_interval_hours
decide     nothing
verdict    BLOCKED
           source toy.data waits for a person: rci source approve
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
----------------------------------------------------------------------------------------------------
card fd05ebfe
blocked; see the verdict above
```

The data is blocked until a person admits the source, once for every idea that uses it:

```text
$ rci next -i ideas/carry
human      rci source approve                   source toy.data waits for a person: rci source approve

$ rci source approve -i ideas/carry
SourceCard · toy.data · code c3abda4b41b7
fields     close funding_interval_hours funding_rate · 9 entities · 6,750 rows
baseline   ok   values are finite: none: values null, NaN or infinite
           ok   known no earlier than the event: none: rows known before their event time
           ok   no long gaps: none: gaps longer than 3× the usual spacing
           ok   no near-duplicate events: none: events within 60s of the last
           ok   no level shifts: none: lasting moves of 5× or more
replay     ok   replay as of past times: 5 cut points agree with the full fetch
reference  ok   independent source agrees: 24 of 24 checked rows agree with the source (12 it could not check)
                source: the toy venue
                close: match 12
                funding_interval_hours: unanswered 12
                funding_rate: match 12
canaries   ok   canary: replay catches a hindsight deletion: deleting a:Z close from 2026-01-10 19:00 was caught
           ok   canary: reconciliation catches altered values: 3 of 3 altered rows caught
grades     verified: close funding_rate · unverified: funding_interval_hours
verdict    ADMISSIBLE
           a person admits it once: rci source approve
----------------------------------------------------------------------------------------------------
card ed62e081
Admit this source? [y/N]: y
admitted toy.data@b150400ded93ad0e: approved by a person at a terminal

$ rci data check -i ideas/carry
DataCard · carry · v1
data       toy.data afa06051cf36 · 3,780 rows · 9 entities · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 257.2 days old
source     toy.data admitted · admission 867ec4547679
checks     [!]  data is current: latest data is 257.2 days old
           ok   hedged legs are one asset: every candidate row holds (close within 1.2x and returns correlated ≥ 0.3)
           ok   field grades: unverified fields are not read: funding_interval_hours
treatments none needed
power      H            MDE before   after   new data
           carry 8h            5.9     5.9       52 d
           carry 24h          19.5    19.5      17 mo
grades     verified: close funding_rate · unverified: funding_interval_hours
decide     nothing
verdict    READY
           approving freezes the treated data; feat reads only that
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
----------------------------------------------------------------------------------------------------
card 60aa30b3
checked; a person freezes it with `rci approve --stage data`
```

The toy data needs no treatment. When a check flags rows, the card lists each treatment with its influence on the decisive number in MDE units; above half an MDE a person decides, and `rci data check --keep "<name>"` leaves a treatment unapplied. Freezing pins the check and the treated data; feat reads only that:

```text
$ rci approve --stage data -i ideas/carry
approve data v1 of carry
DataCard · carry · v1
data       toy.data afa06051cf36 · 3,780 rows · 9 entities · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 257.2 days old
source     toy.data admitted · admission 867ec4547679
checks     [!]  data is current: latest data is 257.2 days old
           ok   hedged legs are one asset: every candidate row holds (close within 1.2x and returns correlated ≥ 0.3)
           ok   field grades: unverified fields are not read: funding_interval_hours
treatments none needed
power      H            MDE before   after   new data
           carry 8h            5.9     5.9       52 d
           carry 24h          19.5    19.5      17 mo
grades     verified: close funding_rate · unverified: funding_interval_hours
decide     nothing
verdict    READY
           approving freezes the treated data; feat reads only that
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
----------------------------------------------------------------------------------------------------
card 60aa30b3
  frozen dependencies:
    contract             d1ffaa8ba828
    data_check           505c53874cc0
    engine               86d3b5f9433f
    engine_packages      5718313514dc
    exclusions           47c61605ee6a
    panel                b0a41d046f24
    python               CPython 3.11.15
Freeze data v1? [y/N]: y
frozen data v1: 611ddfff1d81 · approved by a person at a terminal
to keep it in git: git add ideas/carry/.rci ideas/carry/contract.py && git commit -m 'rci: freeze data of carry'

$ rci data submit -i ideas/carry
data PASSED

$ rci next -i ideas/carry
agent      rci feat register                    declare how this stage will be judged
```

The data layer has no budget and counts no trials: nothing in it is chosen by its result.

## 3. The feat method

Before any features exist, the agent writes how they will be judged. The criteria are functions of the evaluation table, one value per decision time; `research_ci.eval` has the common statistics:

```python
# ideas/carry/feat.py
from research_ci.api import Criterion
from research_ci.eval.statistics.ic import rank_ic
from research_ci.eval.structure.filters import without_top


def rest(table):
    return rank_ic(without_top(table, "symbol", 0.34))


CRITERIA = [
    Criterion("ic", rank_ic),
    Criterion("ic without the top symbol", rest),
]
```

Each alternative explanation in the contract needs a null world that answers it. Here the criterion does most of the work: `ic without the top symbol` leaves out the symbol with the largest outcomes, and the world it must beat shuffles the scores across whole symbols at each decision time while the labels stay:

```python
# ideas/carry/nulls/feat.py
from research_ci.api import Null, PermuteWithin

NULLS = [
    Null("tail", PermuteWithin("time", clusters="symbol", block="horizon"), criteria=("ic without the top symbol",)),
]
```

`rci feat register` adds the engine's floor world (chance and timing), checks that every alternative is kept by a world, and calibrates every criterion in every world before anything is judged:

```text
$ rci feat register -i ideas/carry
MethodCard · carry · feat v1
criteria   ic: carry · H 8h 24h · best of its horizons · required
criteria   ic without the top symbol: carry · H 8h 24h · best of its horizons · required
worlds     floor (engine): PermuteWithin(time, clusters=symbol, block=horizon) · keeps time · explains chance, timing · 200 reps · ic, ic without the top symbol
             scores shuffled across clusters at each decision time, the same way within a horizon
worlds     null 1 (idea): PermuteWithin(time, clusters=symbol, block=horizon) · keeps time · explains tail · 200 reps · ic without the top symbol
alts       tail: a few symbols with extreme funding carry the whole result
calibrate                          world     H        dies  lives  kept       MDE  dies needs
           ic                      floor     8h        96%   100%    --   0.09282  ≥ 92%
           ic without the top symb floor     8h        98%    96%    --    0.1061  ≥ 92%
           ic without the top symb null 1    8h        98%    96%    --    0.1061  ≥ 92%
           a stand-in carries only what the world keeps: it may live only as often as chance allows;
           a planted signal of 1.5 MDE must live ≥ 80%; kept: share of rows whose kept level survives the shuffle
decisive   carry ≥ 3.0 bp per entry after the round trip · target ref.mean with the scores
             H 8h 24h: z of the best against the floor world; passed when one horizon passes,
             failed when every horizon fails
tables     carry: 8h 321 times · 24h 305 times
           every decision time
judging    a criterion passes when it beats every world that judges it: z ≥ 2·√(1+ln n) on the n-th submission
           decisive: passed when z ≥ that and effect − 2·se ≥ at_least; failed when effect + 2·se < at_least;
           otherwise INSUFFICIENT · a leak in the scores TAINTS the submission
budget     3 submissions
checks     ok   null worlds: 1 declared, all runnable
           ok   alternatives covered: chance: floor · timing: floor · tail: null 1
           ok   calibration: 3 pairs hold
           ok   target reads the score: ref.mean moves with the scores
verdict    READY
           approving freezes the criteria and the worlds
----------------------------------------------------------------------------------------------------
card 2db23914
checked; a person freezes it with `rci approve --stage feat`
```

- `calibrate`: a stand-in that carries only what a world keeps must die at least 92% of the time (it may beat the world only by chance), and a planted signal of 1.5 MDE must live at least 80% of the time. A world that does not keep what it claims lets the stand-in live, and the method is returned.
- `decisive`: the contract's target with the submitted scores, judged against the floor world.
- `judging`: the bar rises with every submission: `z ≥ 2·√(1+ln n)`.

A failed calibration leaves the layer CHALLENGED, and `rci next` says to register again. Freezing pins the criteria, the null worlds, the method's ledger record, and the code of every plugin feat runs:

```text
$ rci next -i ideas/carry
human      rci approve --stage feat             review the tests and freeze them

$ rci approve --stage feat -i ideas/carry
approve feat v1 of carry
...                                                    (the card, as printed before)
card 2db23914
  frozen dependencies:
    contract             d1ffaa8ba828
    engine               86d3b5f9433f
    engine_packages      5718313514dc
    feat_file            73af41327aa7
    feat_method          dba6c90b538d
    nulls_file           f65a1432005f
    plugin:ref.carry     bcd1fd61eb3d
    plugin:ref.flat_costs bcd1fd61eb3d
    plugin:ref.mean      bcd1fd61eb3d
    plugin:toy.pairs     c3abda4b41b7
    python               CPython 3.11.15
    upstream:data        611ddfff1d81
Freeze feat v1? [y/N]: y
frozen feat v1: 757df88a8ae6 · approved by a person at a terminal
to keep it in git: git add ideas/carry/.rci ideas/carry/contract.py && git commit -m 'rci: freeze feat of carry'

$ rci next -i ideas/carry
agent      rci feat submit <file>               submit the research code (3 of 3 submissions left)
```

## 4. Submissions

A features file defines `features(fields, candidates)`: one score per candidate row, for every decision time at once, from the frozen data. Labels never reach it.

```python
# ideas/carry/research/spread.py
import polars as pl


def features(fields, candidates):
    return candidates.select("t", "entity", pl.col("spread_h").alias("score"))
```

`rci feat explore` runs the checks a submission must pass and records nothing. This file averages each venue's funding over the whole period, so its early scores change when later data is cut away:

```python
# ideas/carry/research/peek.py
import polars as pl


def features(fields, candidates):
    rate = fields.filter(pl.col("field") == "funding_rate").group_by("entity").agg(pl.col("value").mean())
    short = rate.rename({"entity": "short", "value": "s"})
    long = rate.rename({"entity": "long", "value": "l"})
    rows = candidates.join(short, on="short").join(long, on="long")
    return rows.select("t", "entity", (pl.col("s") - pl.col("l")).alias("score"))
```

```text
$ rci feat explore ideas/carry/research/peek.py -i ideas/carry
ExploreCard · carry · feat (not a submission; nothing is judged)
features   ideas/carry/research/peek.py · 2d50ba5b2d97
scores     2,961 rows · 100% of candidate rows · 5d7436c3d3f8
floors     ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           FAIL scores read nothing ahead: 2026-01-04T18:00Z: 747 values of score differ; 2026-01-08T04:00Z: 1485 values of score differ; 2026-01-11T14:00Z: 2223 values of score differ
----------------------------------------------------------------------------------------------------
```

Submitted, it would be TAINTED and cost a trial. Random scores are judged and fail:

```python
# ideas/carry/research/noise.py
import polars as pl


def features(fields, candidates):
    return candidates.select("t", "entity", (pl.struct("t", "entity").hash(7) / 2.0**64).alias("score"))
```

```text
$ rci feat submit ideas/carry/research/noise.py -i ideas/carry
ResultCard · carry · feat v1 · submission 1 of 3
features   ideas/carry/research/noise.py · 719e506e5111
scores     2,961 rows · 100% of candidate rows · e954ad32d08f
floors     ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
need       z ≥ 2.00 on submission 1
criteria   FAIL ic vs floor (chance, timing): z 1.519 at 24h · best of 2 horizons
                  8h     value   0.001575 ± 0.02114    null   0.001635 sd 0.02021      41 blocks  z -0.00298
                  24h    value    0.03418 ± 0.01805    null  -0.000238 sd 0.01888      13 blocks  z 1.823
criteria   FAIL ic without the top symbol vs floor (chance, timing): z 1.746 at 24h · best of 2 horizons
                  8h     value    0.01655 ± 0.02492    null   0.001566 sd 0.02665      41 blocks  z 0.5623
                  24h    value    0.04574 ± 0.02488    null -0.0004363 sd 0.02284      13 blocks  z 2.022
criteria   FAIL ic without the top symbol vs null 1 (tail): z 1.746 at 24h · best of 2 horizons
                  8h     value    0.01655 ± 0.02492    null   0.001566 sd 0.02665      41 blocks  z 0.5623
                  24h    value    0.04574 ± 0.02488    null -0.0004363 sd 0.02284      13 blocks  z 2.022
decisive   carry: 10.1 bp ± 4.0 at 24h · z 1.382 · needs ≥ 3.0 bp · INSUFFICIENT
verdict    FAILED
           ic does not beat floor (chance, timing): z 1.519
           ic without the top symbol does not beat floor (chance, timing): z 1.746
           ic without the top symbol does not beat null 1 (tail): z 1.746
----------------------------------------------------------------------------------------------------
feat FAILED

$ rci next -i ideas/carry
agent      rci feat submit <file>               FAILED; change the research code (2 of 3 submissions left)
```

The noise's decisive number is positive because the toy's candidates earn carry whatever the scores; against the floor world its z is 1.38, short of 2.00, so the effect is INSUFFICIENT, and the criteria fail the submission. Two submissions are refused without spending a trial: the same scores again, and any change to what the method was approved with:

```text
$ rci feat submit ideas/carry/research/noise.py -i ideas/carry
error: the same scores as submission 1; they would not be a new trial

$ rci feat submit ideas/carry/research/spread.py -i ideas/carry
error: changed since `rci feat register`: feat_file; register it again
```

The second error came after a comment was added to `feat.py`; with the file restored, the spread is the second submission, so it needs `z ≥ 2.60`:

```text
$ rci feat submit ideas/carry/research/spread.py -i ideas/carry
ResultCard · carry · feat v1 · submission 2 of 3
features   ideas/carry/research/spread.py · a5a0d9d57f06
scores     2,961 rows · 100% of candidate rows · 8af96e36d3cc
floors     ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
need       z ≥ 2.60 on submission 2
criteria   ok   ic vs floor (chance, timing): z 3.645 at 8h · best of 2 horizons
                  8h     value     0.1699 ± 0.03826    null  -0.003355 sd 0.04644      41 blocks  z 3.73
                  24h    value      0.269 ± 0.05959    null   -0.00837 sd 0.0793       13 blocks  z 3.498
criteria   ok   ic without the top symbol vs floor (chance, timing): z 3.853 at 8h · best of 2 horizons
                  8h     value     0.2087 ± 0.04598    null  -0.001457 sd 0.05367      41 blocks  z 3.916
                  24h    value     0.3152 ± 0.07336    null  -0.007472 sd 0.09522      13 blocks  z 3.389
criteria   ok   ic without the top symbol vs null 1 (tail): z 3.853 at 8h · best of 2 horizons
                  8h     value     0.2087 ± 0.04598    null  -0.001457 sd 0.05367      41 blocks  z 3.916
                  24h    value     0.3152 ± 0.07336    null  -0.007472 sd 0.09522      13 blocks  z 3.389
decisive   carry: 25.4 bp ± 6.4 at 24h · z 3.993 · needs ≥ 3.0 bp · PASSED
verdict    PASSED
           carry: passed (effect 0.002535 ± 0.0006382)
----------------------------------------------------------------------------------------------------
feat PASSED

$ rci status -i ideas/carry
carry (factor)
  contract  PASSED      v1           approved by a person at a terminal
  data      PASSED      v1
  feat      PASSED      v1   2/3
  signal    OPEN        v1   0/3
  algo      OPEN        v1   0/3
  seal      OPEN        v1
  forward   OPEN        v1
  capital   OPEN        v1
  monitor   OPEN        v1

$ rci next -i ideas/carry
agent      rci signal register                  declare how this stage will be judged
```

The signal layer's commands do not exist yet. `rci feat show` prints the MethodCard and every ResultCard again.

## 5. The ledger

```text
$ rci verify -i ideas/carry
ledger ok: 17 records
```

Each of those 17 records is one line of canonical JSON in `ideas/carry/.rci/ledger.jsonl`, chained by hash: `init`; the contract's `register` and `approve`; `register`, `challenge`, and `check` for each of the two data checks and for the feat method (feat's `register` records the method and what it pinned); `approve` for data and feat; data's `submit`; and one feat `submit` for each judged features file. The refused submissions left no record. Every command checks the chain and the head kept in `$RCI_HOME/heads/`, so a failed submission deleted from the end is noticed. The source's admission and approval are in `$RCI_HOME/sources/ledger.jsonl`, shared by every idea on the machine.

## Not shown here

- A decisive effect through a consumer (a backtester), with a pilot change and branches from the baseline: see [design.md](design.md#the-feat-layer).
- Data treatments and a card that asks a person to decide: see [The data check](design.md#the-data-check).
- INSUFFICIENT: the data cannot decide yet; it costs a trial, and a person decides what comes next.
- Reopening a layer: registering a passed layer again marks the started layers after it STALE.
