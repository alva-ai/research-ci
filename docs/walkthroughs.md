# Walkthrough

> **Status:** everything below runs today. The signal and algo layers and the checkpoints are planned.

One idea, from an empty directory to a passed feat layer. Every block is what the command printed, with only the features paths shortened. The idea runs on the toy plugins of the engine's tests: `toy.data` (hourly closes and funding rates for three symbols on three venues) and `toy.pairs` (every venue pair of a symbol, long the venue with the lower latest funding rate; `spread_h` is the latest spread per hour). A lab registers its own plugins the same way. The toy data ends in January 2026, so the cards warn that it is old.

The contract sets the floor; the loops of data and feat set how far it goes. Only the contract needs a person before work starts: it defines the question and the only gate. Every layer after it is a loop the agent runs on its own, and a person looks at the result after the loop and approves it, or not. The agent here starts with no context but the idea's name: everything it needs is on the paths it takes, in `START.md`, `rci next`, `rci plugins`, `rci methods`, and the messages of what the engine refuses or warns about.

| Actor | Who it is | What it does here |
| --- | --- | --- |
| agent | the research agent | reads `START.md`, writes `contract.py`, `feat.py`, the null worlds, and features files; derives the contract, checks the data, explores and submits features |
| human | a person at a terminal, or an agent relaying the person's words with `--relayed "<words>" --card <id>` | approves the contract before any work, then each layer's result after its loop |
| engine | the `rci` command; deterministic, no LLM | checks, computes, judges, refuses with how to fix it, and appends to the ledger |

Every stage is OPEN (the agent's turn), REVIEW (a person's turn: a result is ready), or PASSED; a layer whose budget is used up is PAUSED, and a stage after one that was reopened is STALE. Every driver reads `rci next`: who should act, the command, and why. A separate challenger that writes null worlds against the agent's tests, and a scheduler for the forward and monitor checkpoints, are planned.

## 1. Start, and the contract

`rci init` takes only the idea's directory: every idea goes through the same pipeline, the contract, the four layers, and the four checkpoints, which `rci status` lists below. It writes three files: `START.md`, the page an agent with no other context reads first; a `contract.py` skeleton; and `research/example.py`, which comes up in the feat loop.

```text
$ rci init ideas/carry
created ideas/carry; read ideas/carry/START.md first: it says how the work goes
then fill in ideas/carry/contract.py and run `rci contract derive -i ideas/carry`
features start from ideas/carry/research/example.py; `rci next -i ideas/carry` says what to do after every step
```

```text
$ cat ideas/carry/START.md
# Working on carry

Research CI judges this idea with fixed code; you do the research with the `rci` command. Every command on the idea
takes `-i ideas/carry`. After every step, run `rci next -i ideas/carry` and do what it says: it names the command, the file to
write, and the helper commands.

1. Contract: fill in `contract.py`. `rci plugins` lists the installed data sources, universes, labels, targets, and
   cost models, with their options and fields. If you looked at the data or at results for this idea (plugin data,
   notebooks, backtests), declare it in `used`; `used=()` only if nothing was seen. Run `rci contract derive`; a
   person approves the ContractCard before any other work starts.
2. Data, your loop: run `rci data check`. Add tests in `data.py` only for a risk the default checks miss;
   `rci methods` lists the functions they can call.
3. Feat, your loop: copy `research/example.py` to `research/<name>.py` and write your features there. Try a file with
   `rci feat explore <file>` (free), then judge it with `rci feat submit <file>` (one trial of the budget). Give each
   alternative of the contract a world in `nulls/feat.py`, or the ResultCard marks it uncovered: import `Null` and
   `PermuteWithin` from `research_ci.api`, then
   `NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", block="horizon"))]`. Optional
   diagnostics go in `feat.py`; `rci methods` lists what they can call and every field of a null world.
4. A person reviews each result after your loop and approves it; then the next stage opens.

Features: `features(fields, candidates)` gets polars DataFrames and returns `t`, `entity`, `score` for every decision
time at once (a polars DataFrame or LazyFrame, or a pandas DataFrame). Research code is free: polars first, with
`fields.lazy() ... .collect()` for a longer pipeline; pandas is fine for dates; any library runs. Row-by-row code
(`iter_rows`, `.rows()`, `map_elements`, `.apply()`, `iterrows`, loops over data) runs too, with a warning on the card
that names each line; polars expressions (`over`, `rolling_*_by`, `join_asof`) are usually much faster. The card also
shows the time and peak memory the features took.

Approvals: show the person the whole card, then relay their words with
`rci approve --stage <stage> --relayed "<their words>" --card <id>`, where the id is the one printed as `card <id>`
under the card they saw. Use `--auto` only when the person asked for fully automatic research.
```

```text
$ rci next -i ideas/carry
agent      rci contract derive                  fill in contract.py (START.md says how; `rci plugins` lists the plugins, their options and fields), then derive the contract
```

`rci plugins` lists what a contract can name, read from the installed code: each plugin's kind, where it goes in the contract, its options with their defaults, and what it declares. The agent does not need to read a plugin's source to know that `toy.data` gives `close` and `funding_rate`, or what `ref.mean` takes:

```text
$ rci plugins
data: Contract(data=Use(name, {options}))
  toy.data(hours: int = 600, seed: int = 0)
    Replays as of any time, and answers reconciliation from the venue, here the same generator run apart; the venue keeps no history of funding intervals.
    fields: close, funding_rate, funding_interval_hours
    levels: close · factor 5 · window 24 · mask 24h
    reference source: answers the independent check at admission
    from the registry
universe: Universe(name, clusters=<column>, groups=(<column>, ...))
  toy.pairs()
    Every venue pair of a symbol; long the venue with the lower latest funding rate per hour.
    from the registry
label: Label(<label name>, Use(name, {options}), horizons=(...))
  ref.carry(price: str = 'close', flow: str | None = None)
    A long/short pair held from `t` to `t + h`, before costs, per unit of one leg's notional.
    from an entry point of research-ci
  ref.markout(price: str = 'close')
    `side * (p(t + h) / p(t) - 1)` for event rows: candidates carry a `side` column (+1 bought, -1 sold).
    from an entry point of research-ci
  ref.return(price: str = 'close')
    `p(t + h) / p(t) - 1` of the candidate's own entity, each price as it was first known at that moment.
    from an entry point of research-ci
target: Effect(on=<label name>, target=Use(name, {options}))
  ref.mean(cluster: str = 'entity', net: bool = True, top: float | None = None)
    The mean label per decision time, each cluster weighted equally; with `net`, after each candidate's round trip.
    per: entry · rolls: yes
    from an entry point of research-ci
cost: Contract(costs=Use(name, {options}))
  ref.flat_costs(commission_bps: float, slippage_bps: float, legs: int = 1)
    Commission plus slippage per fill, in basis points of the traded notional, for candidates of `legs` legs traded together (weights are per leg).
    from an entry point of research-ci
consumer: Contract(consumer=Use(name, {options})), for its decision times
  toy.carry(k: int = 2, commission_bps: float = 1.0, slippage_bps: float = 1.0)
    Holds the top `k` pairs by score, equal weight per leg; flow and price accrue per hour after the decision.
    from the registry
```

The claim, the decisive effect, and the alternative explanations are written before any result is seen. The agent looked at no data or results for this idea, so `used` is empty:

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

The decisive effect names its target: at each decision time, the mean carry after the round trip of the third of the candidates with the highest scores, each symbol weighted once. Feat will pass it when, with the submitted scores, it beats random scores significantly; nothing else decides. The default target, `ref.mean` without `top`, averages every candidate whatever the scores say, so it could not judge any features; derive blocks a contract whose decisive number cannot move with the scores, and the contract stays OPEN with the reason. A plugin name that is not installed, or options a plugin rejects, stop derive with the close names or the plugin's signature.

`rci contract derive` loads the plugins, stores the data in the vault, runs the protocol tests on the universe and the label, and computes power from the dispersion of the target alone:

```text
$ rci contract derive -i ideas/carry
ContractCard · carry · v1
claim      Cross-venue funding spreads persist long enough to pay the round trip
decisive   carry: beats random significantly · target ref.mean(cluster=symbol, top=0.34)
decisions  every 1h from execution · 335 decision times
           decide on the hour; fill both legs at the close known then
data       toy.data afa06051cf36 · 3,780 rows · 9 entities · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 258.4 days old
universe   toy.pairs · median 9 candidates in 3 symbol clusters per decision · membership pit
costs      ref.flat_costs(commission_bps=1, legs=2, slippage_bps=1) · median round trip 8.0 bp
labels     carry: ref.carry · H 8h 24h
checks     [!]  data is current: latest data is 258.4 days old
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
card b489a2ab
up for review; a person approves it with `rci approve --stage contract`
```

- `decisive`: the target with its options, next to the median of 9 candidates on the `universe` line, so a person sees that the top third is 3 candidates.
- `checks`: the universe gives the same candidates on data cut at three times and does not change when later values change; the label matches its worked examples, moves when its inputs move, and reads nothing after `t + h`; the target gives other numbers for other scores. A label option that names no field of the data would add a line listing the fields there are.
- `power`: the smallest gain over random the data can detect, from the dispersion of the target alone: 5.9 bp per entry at 8h, about 65% a year if entered back to back, and 19.5 bp at 24h. `ref.mean` declares what one value is (`per = "entry"`) and that its values roll over a year (`rolls = True`); a target without them gets raw numbers. The noise can be measured, so the contract is DECIDABLE and goes up for review. The card shows only the scale of the noise, never where the target sits.
- `alts`: the engine adds `chance` and `timing` to the idea's own alternative; at feat the floor world answers them, and a null world of the idea may answer `tail`.
- `warnings`: Z, one of three symbols, carries 55% of the noise at 24h; derive only warns, and so does the data check.
- `card b489a2ab`: the id of this card. An agent that shows the card to a person and relays the answer quotes it with `--card`; `rci contract show` prints the card and the id again.

A person reads the card and approves it, which freezes it. Approval reads the card from the ledger, so the id is the one derive printed:

```text
$ rci inbox ideas
ideas/carry              rci approve --stage contract         review the ContractCard and approve it; `rci contract show` prints it with the card id that a relayed approval quotes

$ rci approve --stage contract -i ideas/carry
approve contract v1 of carry
...                                                    (the card, as printed before)
card b489a2ab
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
    "reason": "check the data against the contract; tests for risks the default checks miss go in data.py (`rci methods` lists what they can call)"
  }
]
```

## 2. The data

`rci data check` runs the source's admission checks the first time this plugin, with these options and this code, is used anywhere on the machine: the rows replayed as of past times must agree with the full fetch, an independent source must agree on rows the engine picks, and two canaries make sure both checks can fail. That is fidelity: the rows are what the venue reported. Then it checks fitness, whether faithful rows can be used for this research, with the mandatory default list in `stages/defaults/data.py` (freshness, level shifts the independent source did not confirm, level sanity, identity breaks, coverage, the hedge guard on legs of one asset, and influence concentration) and the idea's own tests in `data.py`, which this idea does not need. Last, it calibrates the gate on the treated data, before any features exist. No one has admitted `toy.data` yet, so the DataCard shows the admission under `source`:

```text
$ rci data check -i ideas/carry
DataCard · carry · v1
data       toy.data afa06051cf36 · 3,780 rows · 9 entities · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 258.4 days old
source     toy.data · admission 28a71abbc25d · not yet admitted: approving this card admits it for every idea [!]
           code c34ba7130b74 · 9 entities · 6,750 rows
           ok   values are finite: none: values null, NaN or infinite
           ok   known no earlier than the event: none: rows known before their event time
           ok   no long gaps: none: gaps longer than 3× the usual spacing
           ok   no near-duplicate events: none: events within 60s of the last
           ok   no level shifts: none: lasting moves of 5× or more
           ok   replay as of past times: 5 cut points agree with the full fetch
           ok   canary: replay catches a hindsight deletion: deleting a:Y close from 2026-01-15 18:00 was caught
           ok   independent source agrees: 24 of 24 checked rows agree with the source (12 it could not check)
                reference: the toy venue
           ok   canary: reconciliation catches altered values: 3 of 3 altered rows caught
checks     [!]  data is current: latest data is 258.4 days old
           ok   level sanity: no declared level at zero or below, stale, or spiking
           ok   identity breaks: no declared level comes back 5x away after a long gap
           ok   coverage: every entity covers 24h
           ok   hedged legs are one asset: every candidate row holds (close within 1.2x and returns correlated ≥ 0.3)
           [!]  influence concentration: more than 50% of the noise from one symbol: Z 55% at carry 24h; shown, not treated
           ok   field grades: unverified fields are not read: funding_interval_hours
treatments none needed
power      H            MDE before   after
           carry 8h            5.9     5.9
           carry 24h          19.5    19.5
calibrate  ok   carry at 8h vs floor: stand-ins die 97% (needs ≥ 92%) · a signal planted at 1.5 MDE (4.1 bp/entry) lives 98% (needs ≥ 80%)
clusters   share of the noise that goes away without each symbol, largest first
           carry 8h    X 0% · Y 0% · Z 0%
           carry 24h   Z 55% · X 0% · Y 0%
grades     verified: close funding_rate · unverified: funding_interval_hours
decide     nothing
verdict    READY
           approving freezes the treated data; feat reads only that
           approving also admits toy.data for every idea
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
----------------------------------------------------------------------------------------------------
card b503f2ce
up for review; a person approves it with `rci approve --stage data`
```

The toy data needs no treatment, and the card is READY. The `clusters` line shows how much of the label's noise, over all nine candidates, goes away without each symbol: Z carries 55% at 24h. Above half, the card warns and does nothing else: broken data is what the identity and sanity checks catch, an effect can sit in one symbol too, and with only three symbols one of them often carries this much. When a check flags rows, the card lists each treatment with its influence on the decisive number in MDE units; above half an MDE a person decides, and `rci data check --keep "<name>"` leaves a treatment unapplied. A blocked check, such as a source that fails admission, keeps the layer OPEN with the reason.

`calibrate` is the test of the gate itself, in the floor world (the scores shuffled across symbols at each decision time): a stand-in that carries only each decision time's level must die at least 92% of the time, since it may beat the floor only by chance, and a signal planted at 1.5 MDE must live at least 80% of the time. It needs the contract, the treated data, and the labels, but no scores, so it runs here; a target that cannot tell them apart, such as one that averages too few candidates or takes the lowest scores, blocks the data check, and the agent changes the target before writing any features. Feat submissions reuse it.

A person approves the card. That freezes the check, its calibration, and the treated data, which is all feat reads, and, since this is the first approval of data from `toy.data`, admits the source for every idea on the machine, in the shared sources ledger:

```text
$ rci next -i ideas/carry
human      rci approve --stage data             review the DataCard and approve it; `rci data show` prints it with the card id that a relayed approval quotes

$ rci approve --stage data -i ideas/carry
approve data v1 of carry
...                                                    (the card, as printed before)
card b503f2ce
  frozen dependencies:
    contract        365c17d571b4
    data_check      d8262089e677
    engine          bd316cde9ade
    engine_packages 5718313514dc
    exclusions      47c61605ee6a
    panel           b0a41d046f24
    python          CPython 3.11.15
Approve data v1? [y/N]: y
data v1 PASSED: freeze b2a60dbcc565 · approved by a person at a terminal
to keep it in git: git add ideas/carry/.rci ideas/carry/contract.py && git commit -m 'rci: approve data of carry'

$ rci next -i ideas/carry
agent      rci feat submit <file>               write research/<name>.py from research/example.py (polars expressions; row loops run with a warning), try it with `rci feat explore <file>`, then submit it; give each alternative of contract.py a world in nulls/feat.py, such as NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", block="horizon"))], or the ResultCard marks it uncovered; diagnostics in feat.py are optional; `rci methods` lists what diagnostics can call and the fields of a null world (3 of 3 submissions left)
```

The data layer has no budget and counts no trials: nothing in it is chosen by its result. `rci next` now names everything the feat loop needs: the file to start from, the free command to try a file, and the null worlds.

## 3. The feat loop

Nothing is registered before features exist. What decides is already in the contract: the decisive effect, with the submitted scores, must beat the engine's floor world by `z ≥ 2·√(1+ln n)` on the n-th submission. That is the minimal bar, and nothing else gates at feat, so potential alpha is not killed early; costs, absolute profit, and robustness are for later layers. Everything else is a diagnostic: judged and shown on every result, never deciding. The engine runs its default diagnostics on every submission (rank IC, rank IC without the top symbol, and the top fifth by score less the bottom fifth). `feat.py` is optional; `rci methods` lists the methods a criterion can call, with the worked examples that back each, and the data tests and pits further down:

```text
$ rci methods
criterion: feat.py: CRITERIA = [Criterion(name, evaluate)]; evaluate(table) returns t, value per decision time
  from research_ci.eval.statistics.ic import rank_ic
    rank_ic(table: pl.DataFrame, label: str = 'y') -> pl.DataFrame
    Spearman correlation of score and label across the candidates at each decision time.
    backed by 3 worked examples · 3 properties
  from research_ci.eval.statistics.spread import quantile_spread
    quantile_spread(table: pl.DataFrame, share: float = 0.2, label: str = 'y') -> pl.DataFrame
    Mean label of the top `share` of candidates by score minus that of the bottom `share`, at each time; a share is `floor(share * n)` candidates, at least one.
    backed by 3 worked examples · 2 properties
  from research_ci.eval.structure.filters import without_top
    without_top(table: pl.DataFrame, column: str, share: float, by: str = 'y') -> pl.DataFrame
    The table without the `share` of `column` values (such as symbols) with the largest mean absolute `by`.
    backed by 2 worked examples · 0 properties
...                                                    (data tests, pits, and null worlds follow)
```

Here `feat.py` adds one diagnostic of the idea's own, since a fifth of nine candidates is one:

```python
# ideas/carry/feat.py
from research_ci.api import Criterion
from research_ci.eval.statistics.spread import quantile_spread


def thirds(table):
    return quantile_spread(table, share=1 / 3)


CRITERIA = [Criterion("top minus bottom third", thirds)]
```

Each alternative explanation in the contract gets a null world that answers it, or the ResultCard marks it uncovered; the worlds judge diagnostics. The end of `rci methods` shows how one is written:

```text
$ rci methods
...                                                    (criteria, data tests, and pits)
null world: nulls/feat.py: NULLS = [Null(explains, world)]; give each alternative of the contract one, or the ResultCard marks it uncovered
  from research_ci.api import Null, PermuteWithin
    Null(explains: str, world: PermuteWithin, keeps: str | None = None, criteria: tuple[str, ...] = (), reps: int = 200, note: str = '')
      explains  the id of the contract's alternative this world answers
      world     a PermuteWithin: how the scores are shuffled while the labels stay
      keeps     the column the alternative lives in; None means the world's bucket_by
      criteria  the names of the criteria this world judges; () means every criterion of the layer
      reps      how many shuffles, at least 20
      note      free text kept with the world in the submission's record
    PermuteWithin(bucket_by: str, clusters: str | None = None, block: str | None = None)
      bucket_by  shuffle the scores across entities inside each bucket of this column; "time" is one bucket per decision time
      clusters   a column whose members move together and are shuffled as one
      block      repeat the same shuffle for every decision time inside a block this long; "horizon" uses each label horizon
  for example: NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", block="horizon"))]
```

Here the default `rank ic without top symbol` leaves out the symbol with the largest outcomes, and the world shuffles the scores across whole symbols at each decision time while the labels stay:

```python
# ideas/carry/nulls/feat.py
from research_ci.api import Null, PermuteWithin

NULLS = [
    Null("tail", PermuteWithin("time", clusters="symbol", block="horizon"), criteria=("rank ic without top symbol",)),
]
```

A features file defines `features(fields, candidates)`: one score per candidate row, for every decision time at once, from the frozen data. Both arrive as polars DataFrames, so eager polars code and pandas run as they are; `features` returns a polars DataFrame or LazyFrame, or a pandas DataFrame, and labels never reach it. `research/example.py`, from `rci init`, shows the shape and the usual patterns, polars first, with `.lazy()` and `.collect()` around a pipeline that polars plans as a whole, and carries a marker line:

```text
$ cat ideas/carry/research/example.py
# rci:example -- the engine refuses to run this file while this line is in it. Copy it to
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
    #   leg = known.rename({"entity": "long", "known_at": "long_at", "value": "long_value"})
    #   scored = candidates.lazy().sort("t").join_asof(leg, left_on="t", right_on="long_at", by="long",
    #                                                  check_sortedness=False)
    # Ranks within each decision time: pl.col("score").rank().over("t").
    # pandas is fine for dates (pd.Timedelta, offsets, calendars), and any library runs. Whole-column polars
    # expressions like these are usually much faster than going row by row, which runs too, with a warning on the card;
    # the card also shows the time and peak memory the features took.
    return scored.select("t", "entity", pl.col("value").alias("score")).collect()
```

The engine refuses to run the example as it is. Any other file runs: research code is free. The engine reads every features file, with `feat.py`, `nulls/feat.py`, and the research modules they import, for row-by-row code, and the card shows what it finds as one warning that names each line; it never refuses the file and never costs a trial. This file loops over rows:

```python
# ideas/carry/research/rows.py
import polars as pl


def features(fields, candidates):
    rows = candidates.sort("t", "entity")
    scores = [(t, e, s) for t, e, s in rows.select("t", "entity", "spread_h").iter_rows()]
    return pl.DataFrame(scores, schema=["t", "entity", "score"], orient="row")
```

```text
$ rci feat explore ideas/carry/research/example.py -i ideas/carry
error: ideas/carry/research/example.py is still the example; write your own: copy research/example.py to research/<name>.py, delete its `rci:example` line, and replace every TODO; nothing was recorded

$ rci feat explore ideas/carry/research/rows.py -i ideas/carry
ExploreCard · carry · feat (not a submission; nothing is judged)
features   ideas/carry/research/rows.py · a962f85d59f6
scores     2,961 rows · 100% of candidate rows · 8af96e36d3cc
floors     ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
cost       full call 2.4 ms, peak memory +0.1 MB (process 278 MB) · on cut data 0.9 ms, 1.8 ms, 2.9 ms · not judged
warnings   [!] row-wise code: research/rows.py:6 iter_rows — polars expressions are usually much faster
----------------------------------------------------------------------------------------------------
```

Both cards have a `cost` line: the wall time of the full call and of the calls on the cut data, and the peak memory of the full call, the process's resident size sampled while it runs, with how much it grew over the start of the call. It is there for the agent to see what its code costs, is kept in a submission's record, and is never judged. On the toy data every call is over in a few milliseconds; written on whole columns, the same score is one line, and on a real panel usually much faster:

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
cost       full call 1.6 ms, peak memory +0.2 MB (process 283 MB) · on cut data 1.2 ms, 1.7 ms, 1.1 ms · not judged
----------------------------------------------------------------------------------------------------
```

Submitted, it would be TAINTED and cost a trial. The agent submits random scores first:

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
cost       full call 0.3 ms, peak memory +0.0 MB (process 284 MB) · on cut data 0.2 ms, 0.2 ms, 0.1 ms · not judged
need       z ≥ 2.00 on submission 1
decisive   carry: gain over random 1.7 bp/entry ± 4.0 at 24h (≈ 6.4%/yr if rolled) · z 1.382 · FAILED
shown      diagnostics, not deciding: the engine's defaults and the idea's criteria
           [!]  rank ic (engine) vs floor (chance, timing): z 1.519 at 24h · best of 2 horizons
                  8h     value   0.001575 ± 0.02114    null   0.001635 sd 0.02021      41 blocks  z -0.00298
                  24h    value    0.03418 ± 0.01805    null  -0.000238 sd 0.01888      13 blocks  z 1.823
           [!]  rank ic without top symbol (engine) vs floor (chance, timing): z 1.746 at 24h · best of 2 horizons
                  8h     value    0.01655 ± 0.02492    null   0.001566 sd 0.02665      41 blocks  z 0.5623
                  24h    value    0.04574 ± 0.02488    null -0.0004363 sd 0.02284      13 blocks  z 2.022
           [!]  rank ic without top symbol (engine) vs null 1 (tail): z 1.746 at 24h · best of 2 horizons
                  8h     value    0.01655 ± 0.02492    null   0.001566 sd 0.02665      41 blocks  z 0.5623
                  24h    value    0.04574 ± 0.02488    null -0.0004363 sd 0.02284      13 blocks  z 2.022
           [!]  top minus bottom fifth (engine) vs floor (chance, timing): z 1.989 at 24h · best of 2 horizons
                  8h     value   0.000144 ± 0.0001743  null  8.296e-06 sd 0.0001819    41 blocks  z 0.7458
                  24h    value  0.0007446 ± 0.0002384  null -2.123e-05 sd 0.0003441    13 blocks  z 2.226
           [!]  top minus bottom third (idea) vs floor (chance, timing): z 1.421 at 24h · best of 2 horizons
                  8h     value  4.505e-05 ± 9.724e-05  null  3.853e-06 sd 9.43e-05     41 blocks  z 0.4369
                  24h    value   0.000283 ± 0.0001906  null -4.207e-06 sd 0.0001649    13 blocks  z 1.742
worlds     ok   alternatives covered: chance: floor · timing: floor · tail: null 1
verdict    FAILED
           carry: does not beat random, z 1.382 < 2.00
----------------------------------------------------------------------------------------------------
feat OPEN: not up for review; `rci next` says what to do and how many submissions are left

$ rci next -i ideas/carry
agent      rci feat submit <file>               submission 1 FAILED: carry: does not beat random, z 1.382 < 2.00; write a new features file in research/ and submit it (2 of 3 submissions left)
```

- `decisive`: the toy's candidates earn carry whatever the scores, so the target itself is well above zero; the card shows the gain over random, 1.7 bp per entry, and its z against the floor world, 1.38, short of 2.00, so the submission FAILED and the layer stays OPEN for the agent. It reused the calibration of the data check, which the DataCard shows.
- `shown`: diagnostics that do not beat a world are marked `[!]`; they are shown, not deciding. `worlds` says which world answers which alternative; one with none is a warning, not a block.

The same scores again are refused without spending a trial, since they would not be a new trial. The spread is the second submission, so it needs `z ≥ 2.60`:

```text
$ rci feat submit ideas/carry/research/noise.py -i ideas/carry
error: the same scores as submission 1; they would not be a new trial; no submission was spent

$ rci feat submit ideas/carry/research/spread.py -i ideas/carry
ResultCard · carry · feat v1 · submission 2 of 3
features   ideas/carry/research/spread.py · a5a0d9d57f06
scores     2,961 rows · 100% of candidate rows · 8af96e36d3cc
floors     ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
cost       full call 0.2 ms, peak memory +0.0 MB (process 290 MB) · on cut data 0.2 ms, 0.2 ms, 0.1 ms · not judged
need       z ≥ 2.60 on submission 2
decisive   carry: gain over random 17.6 bp/entry ± 6.4 at 24h (≈ 64.1%/yr if rolled) · z 3.993 · PASSED
shown      diagnostics, not deciding: the engine's defaults and the idea's criteria
           ok   rank ic (engine) vs floor (chance, timing): z 3.645 at 8h · best of 2 horizons
                  8h     value     0.1699 ± 0.03826    null  -0.003355 sd 0.04644      41 blocks  z 3.73
                  24h    value      0.269 ± 0.05959    null   -0.00837 sd 0.0793       13 blocks  z 3.498
           ok   rank ic without top symbol (engine) vs floor (chance, timing): z 3.853 at 8h · best of 2 horizons
                  8h     value     0.2087 ± 0.04598    null  -0.001457 sd 0.05367      41 blocks  z 3.916
                  24h    value     0.3152 ± 0.07336    null  -0.007472 sd 0.09522      13 blocks  z 3.389
           ok   rank ic without top symbol (engine) vs null 1 (tail): z 3.853 at 8h · best of 2 horizons
                  8h     value     0.2087 ± 0.04598    null  -0.001457 sd 0.05367      41 blocks  z 3.916
                  24h    value     0.3152 ± 0.07336    null  -0.007472 sd 0.09522      13 blocks  z 3.389
           [!]  top minus bottom fifth (engine) vs floor (chance, timing): z 2.278 at 8h · best of 2 horizons
                  8h     value   0.001056 ± 0.0003151  null  2.884e-06 sd 0.0004194    41 blocks  z 2.511
                  24h    value   0.002651 ± 0.0008472  null -1.641e-05 sd 0.001116     13 blocks  z 2.389
           ok   top minus bottom third (idea) vs floor (chance, timing): z 3.585 at 8h · best of 2 horizons
                  8h     value  0.0008304 ± 0.0002052  null -1.792e-05 sd 0.0002338    41 blocks  z 3.629
                  24h    value   0.002456 ± 0.0004698  null -9.899e-05 sd 0.0007608    13 blocks  z 3.358
worlds     ok   alternatives covered: chance: floor · timing: floor · tail: null 1
verdict    PASSED
           carry: beats random, z 3.993 ≥ 2.60
----------------------------------------------------------------------------------------------------
card cf7746a8
up for review; a person approves it with `rci approve --stage feat`
```

Its decisive effect beats random with z 3.99, so it PASSED, although the default top-minus-bottom-fifth diagnostic, one candidate each way, stayed below the bar. A PASSED submission puts the layer up for review, with the card id printed under its card; the agent could go on submitting, and a later FAILED one would take it back to OPEN, since a person can accept only a latest submission that PASSED:

```text
$ rci status -i ideas/carry
carry
  contract  PASSED  v1           approved by a person at a terminal
  data      PASSED  v1           approved by a person at a terminal
  feat      REVIEW  v1   2/3     submission 2 PASSED: carry: beats random, z 3.993 ≥ 2.60
  signal    OPEN    v1   0/3
  algo      OPEN    v1   0/3
  seal      OPEN    v1
  forward   OPEN    v1
  capital   OPEN    v1
  monitor   OPEN    v1

$ rci inbox ideas
ideas/carry              rci approve --stage feat             submission 2 PASSED: carry: beats random, z 3.993 ≥ 2.60; review the ResultCard and approve it; `rci feat show` prints it with the card id that a relayed approval quotes
```

Each submission records what it used: the features file, `feat.py` and `nulls/feat.py`, the code of every plugin feat runs, the engine, and the calibration. Approval checks them again. The first approval below came after a comment was added to `feat.py`; with the file restored, the person approves the ResultCard:

```text
$ rci approve --stage feat -i ideas/carry
error: changed since submission 2: feat_file; submit again

$ rci approve --stage feat -i ideas/carry
approve feat v1 of carry
...                                                    (the card, as printed before)
card cf7746a8
  frozen dependencies:
    calibration           d8262089e677
    contract              365c17d571b4
    engine                bd316cde9ade
    engine_packages       5718313514dc
    feat_file             3b90c8182116
    feat_submission       5da105ad28b2
    features              a5a0d9d57f06
    nulls_file            8d91a6f6b60e
    plugin:ref.carry      410af5211978
    plugin:ref.flat_costs 410af5211978
    plugin:ref.mean       410af5211978
    plugin:toy.pairs      c34ba7130b74
    python                CPython 3.11.15
    upstream:data         b2a60dbcc565
Approve feat v1? [y/N]: y
feat v1 PASSED: freeze b5b5be331212 · approved by a person at a terminal
to keep it in git: git add ideas/carry/.rci ideas/carry/contract.py && git commit -m 'rci: approve feat of carry'

$ rci status -i ideas/carry
carry
  contract  PASSED  v1           approved by a person at a terminal
  data      PASSED  v1           approved by a person at a terminal
  feat      PASSED  v1   2/3     approved by a person at a terminal
  signal    OPEN    v1   0/3
  algo      OPEN    v1   0/3
  seal      OPEN    v1
  forward   OPEN    v1
  capital   OPEN    v1
  monitor   OPEN    v1

$ rci next -i ideas/carry
agent      (nothing to run)                     the signal layer is not built yet, so this idea is done for now: every stage before signal passed; nothing to run until Research CI adds it
```

The signal layer is not built yet, so `rci next` says the idea is done for now and names no command. `rci feat show` prints every ResultCard again, with the card id under the one up for review.

## 4. The ledger

```text
$ rci verify -i ideas/carry
ledger ok: 8 records
```

Each of those 8 records is one line of canonical JSON in `ideas/carry/.rci/ledger.jsonl`, chained by hash: `init`; the contract's `derive` and `approve`; data's `check`, which holds the calibration, and `approve`; one `submit` for each judged features file; and feat's `approve`. The refused submission, the refused example, the explorations, and the refused approval left no record. Every command checks the chain and the head kept in `$RCI_HOME/heads/`, so a failed submission deleted from the end is noticed. The source's admission and its approval are in `$RCI_HOME/sources/ledger.jsonl`, shared by every idea on the machine.

## Not shown here

- A backtest. Nothing before the algo layer runs one: a contract may name a consumer (a backtester), but only for its decision times. Effects through the consumer, with a pilot change and branches from the baseline, belong to the planned algo layer: see [design.md](design.md#consumer-effects-branch-and-fullpath).
- Data treatments and their influence: see [The data check](design.md#the-data-check).
- A budget used up: the layer is PAUSED until a person extends it at a terminal: see [Budgets and PAUSED](design.md#budgets-and-paused).
- Reopening: deriving a passed contract again, checking passed data again, or submitting to a passed layer starts a new version and marks the started stages after it STALE; trial counts go on across versions.
