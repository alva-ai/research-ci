# Walkthrough

> **Status:** everything below runs today. The signal and algo layers and the checkpoints are planned.

One idea, from an empty directory to a passed feat layer. Every block is what the command printed, with only the features paths shortened. The idea runs on the toy plugins of the engine's tests: `toy.data` (hourly closes and funding rates for three symbols on three venues) and `toy.pairs` (every venue pair of a symbol, long the venue with the lower latest funding rate; `spread_h` is the latest spread per hour). A lab registers its own plugins the same way. The toy data ends in January 2026, so the cards warn that it is old.

The contract sets the floor; the loops of data and feat set how far it goes. Only the contract needs a person before work starts: it defines the question and the one bar, the decisive test. Every layer after it is a loop the agent runs on its own, and a person looks at the result after the loop and approves it, or not. The agent here starts with no context but the idea's name: everything it needs is on the paths it takes, in `START.md`, `rci next`, `rci plugins`, `rci methods`, and the messages of what the engine refuses or warns about.

| Actor | Who it is | What it does here |
| --- | --- | --- |
| agent | the research agent | reads `START.md`, writes `contract.py`, case evals with their tests, `feat.py`, the null worlds, and features files; derives the contract, checks the data, checks its evals, explores and submits features |
| human | a person at a terminal, or an agent relaying the person's words with `--relayed "<words>" --card <id>` | approves the contract before any work, then each layer's result after its loop |
| engine | the `rci` command; deterministic, no LLM | runs the tests of the agent's case evals before it runs them, checks, computes, judges, refuses with how to fix it, and appends to the ledger |

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
```

- `decisive`: the target with its options, next to the median of 9 candidates on the `universe` line, so a person sees that the top third is 3 candidates.
- `checks`: the universe gives the same candidates on data cut at three times and does not change when later values change; the label matches its worked examples, moves when its inputs move, and reads nothing after `t + h`; the target gives other numbers for other scores. A label option that names no field of the data would add a line listing the fields there are.
- `power`: the smallest gain over random the data can detect, from the dispersion of the target alone: 5.9 bp per entry at 8h, about 65% a year if entered back to back, and 19.5 bp at 24h. `ref.mean` declares what one value is (`per = "entry"`) and that its values roll over a year (`rolls = True`); a target without them gets raw numbers. The noise can be measured, so the contract is DECIDABLE and goes up for review. The card shows only the scale of the noise, never where the target sits.
- `alts`: the engine adds `chance` and `timing` to the idea's own alternative; at feat the floor world answers them, and a null world of the idea may answer `tail`.
- `warnings`: Z, one of three symbols, carries 55% of the noise at 24h; derive only warns, and so does the data check.
- `card 3405b738`: the id of this card. An agent that shows the card to a person and relays the answer quotes it with `--card`; `rci contract show` prints the card and the id again.

A person reads the card and approves it, which freezes it. Approval reads the card from the ledger, so the id is the one derive printed:

```text
$ rci inbox ideas
ideas/carry              rci approve --stage contract         review the ContractCard and approve it; `rci contract show` prints it with the card id that a relayed approval quotes

$ rci approve --stage contract -i ideas/carry
approve contract v1 of carry
...                                                    (the card, as printed before)
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

## 2. The data

`rci data check` runs the source's admission checks the first time this plugin, with these options and this code, is used anywhere on the machine: the rows replayed as of past times must agree with the full fetch, an independent source must agree on rows the engine picks, and two canaries make sure both checks can fail. That is fidelity: the rows are what the venue reported. Then it checks fitness, whether faithful rows can be used for this research, with the mandatory default list in `stages/defaults/data.py` (freshness, level shifts the independent source did not confirm, level sanity, identity breaks, coverage, the hedge guard on legs of one asset, and influence concentration) and the idea's own evals in `EVALS` in `data.py`, which this idea does not need. Every line under `evals` says whose eval it is: the stage's `floor` list, its `default` list, or the idea's own, a `library` eval of the engine or a `case` eval of the idea. Last, it calibrates the decisive test on the treated data, before any features exist. No one has admitted `toy.data` yet, so the DataCard shows the admission under `source`:

```text
$ rci data check -i ideas/carry
DataCard · carry · v1
data       toy.data afa06051cf36 · 3,780 rows · 9 entities · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 258.8 days old
source     toy.data · admission 9eb487bfd2ca · not yet admitted: approving this card admits it for every idea [!]
           code 29908a9eddf3 · 9 entities · 6,750 rows
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
evals      default  [!]  data is current: latest data is 258.8 days old
           default  ok   level sanity: no declared level at zero or below, stale, or spiking
           default  ok   identity breaks: no declared level comes back 5x away after a long gap
           default  ok   coverage: every entity covers 24h
           default  ok   hedged legs are one asset: every candidate row holds (close within 1.2x and returns correlated ≥ 0.3)
           default  [!]  influence concentration: more than 50% of the noise from one symbol: Z 55% at carry 24h; shown, not treated
           floor    ok   field grades: unverified fields are not read: funding_interval_hours
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
card 6855801e
up for review; a person approves it with `rci approve --stage data`
```

The toy data needs no treatment, and the card is READY. The `clusters` line shows how much of the label's noise, over all nine candidates, goes away without each symbol: Z carries 55% at 24h. Above half, the card warns and does nothing else: broken data is what the identity and sanity checks catch, an effect can sit in one symbol too, and with only three symbols one of them often carries this much. When a check flags rows, the card lists each treatment with its influence on the decisive number in MDE units; above half an MDE a person decides, and `rci data check --keep "<name>"` leaves a treatment unapplied. A blocked check, such as a source that fails admission, keeps the layer OPEN with the reason.

`calibrate` tests the decisive test itself, in the floor world (the scores shuffled across symbols at each decision time): a stand-in that carries only each decision time's level must die at least 92% of the time, since it may beat the floor only by chance, and a signal planted at 1.5 MDE must live at least 80% of the time. It needs the contract, the treated data, and the labels, but no scores, so it runs here; a target that cannot tell them apart, such as one that averages too few candidates or takes the lowest scores, blocks the data check, and the agent changes the target before writing any features. Feat submissions reuse it.

A person approves the card. That freezes the check, its calibration, and the treated data, which is all feat reads, and, since this is the first approval of data from `toy.data`, admits the source for every idea on the machine, in the shared sources ledger:

```text
$ rci next -i ideas/carry
human      rci approve --stage data             review the DataCard and approve it; `rci data show` prints it with the card id that a relayed approval quotes

$ rci approve --stage data -i ideas/carry
approve data v1 of carry
...                                                    (the card, as printed before)
card 6855801e
  frozen dependencies:
    contract        365c17d571b4
    data_check      95851be902d2
    engine          b54ecee968d1
    engine_packages 5718313514dc
    exclusions      47c61605ee6a
    panel           b0a41d046f24
    python          CPython 3.11.15
Approve data v1? [y/N]: y
data v1 PASSED: freeze ff3413989cb5 · approved by a person at a terminal
to keep it in git: git add ideas/carry/.rci ideas/carry/contract.py && git commit -m 'rci: approve data of carry'

$ rci next -i ideas/carry
agent      rci feat submit <file>               write research/<name>.py from research/example.py (polars expressions; row loops run with a warning), try it with `rci feat explore <file>`, then submit it; give each alternative of contract.py a world in nulls/feat.py, such as NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", block="horizon"))], or the ResultCard marks it uncovered; evals in feat.py (EVALS) are optional, the engine's library or case evals of your own in evals/<name>.py with their CASES (`rci eval check <file>`); `rci methods` lists the library, how a case eval is written, and the fields of a null world (3 of 3 submissions left)
```

The data layer has no budget and counts no trials: nothing in it is chosen by its result. `rci next` now names everything the feat loop needs: the file to start from, the free command to try a file, and the null worlds.

## 3. The feat loop

Nothing is registered before features exist. What decides is already in the contract: the decisive effect, with the submitted scores, must beat the engine's floor world by `z ≥ 2·√(1+ln n)` on the n-th submission. That is the decisive test, the minimal bar, and nothing else decides at feat, so potential alpha is not killed early; costs, absolute profit, and robustness are for later layers. Every other eval is a diagnostic: judged and shown on every result, never deciding. The engine runs its default evals on every submission (rank IC, rank IC without the top symbol, and the top fifth by score less the bottom fifth), and the idea adds its own in `EVALS` in `feat.py`, which is optional: general evals of the engine's library, which `rci methods` lists with the worked examples and properties that back each, and case evals of its own:

```text
$ rci methods
feat eval: feat.py defines EVALS = [fn, Eval(fn, label=..., horizons=(...), combine='max'), ...], where fn(table) returns t, value for each decision time from the evaluation table (t, entity, score, y, y_<component>, and the candidate columns); fn is an eval of the engine's library (`rci methods`; give it its options with functools.partial) or a case eval of the idea's own with CASES in its module (`rci eval check <file>`)
  from research_ci.eval.statistics.ic import rank_ic
    rank_ic(table: pl.DataFrame, label: str = 'y') -> pl.DataFrame
    Spearman correlation of score and label at each decision time.
    backed by 3 worked examples · 3 properties
  from research_ci.eval.statistics.spread import quantile_spread
    quantile_spread(table: pl.DataFrame, share: float = 0.2, label: str = 'y') -> pl.DataFrame
    Mean label of the top `share` less the bottom `share`.
    backed by 3 worked examples · 2 properties
  from research_ci.eval.structure.filters import kept_without_top
    kept_without_top(table: pl.DataFrame, column: str, share: float, by: str = 'y') -> pl.Series
    Which rows `without_top` keeps.
    backed by 3 worked examples · 0 properties
...                                                    (the rest of the library, data evals, and pits follow)
```

A fifth of nine candidates is one, so the agent wants the top third less the bottom third, and the top half less the bottom half. `Eval(partial(quantile_spread, share=1 / 3))` would list the library's eval with another share, trusted as it is; here the agent writes the two as case evals of its own instead, polars functions in a module of the idea with their tests next to them, the way the library's modules carry theirs. `rci methods` says how:

```text
$ rci methods
...                                                    (the library, data evals, and pits)
case eval: a function of the idea's own, in a module such as `evals/<name>.py`, with its tests next to it: CASES (required) and PROPERTIES (optional); data.py or feat.py imports it into EVALS (`from evals.<name> import <eval>`). Before anything runs, the engine runs its tests, the cases and properties that run it, and only a case eval that passes them all runs; one without tests is refused, an invalid one is shown with the test it failed and no result. `rci eval check <file>` runs them any time, for free; a validation is kept until the module's files change
  from research_ci.api import Eval, Case, Size, Power, Recovers, Market, Source
    Eval(of, name='', on_fail=None, covers=(), label=None, horizons=None, combine=None, note='')
      an entry of EVALS with what its layer reads: on_fail, covers (data.py); label, horizons, combine (feat.py)
    Case(name, run, expect, tol=1e-09, given=None)
      `run()` must give `expect`; with `given`, `run(given)` must, so a case can name the eval and the frame it reads: `Case("the top two of six", thirds, 0.4, given=SIX)`.
    Size(of, market=Market(), markets=100)
      On markets with no information, the statistic must beat the floor world (z ≥ 2) no more often than chance allows, over `markets` seeds.
    Power(of, market=Market(), markets=40, rate=0.8)
      On markets with `market.ic` of information, the statistic must beat the floor world at least `rate` of the time.
    Recovers(of, market, expect, tol, markets=20)
      Averaged over `markets` seeds and decision times, the statistic must come within `tol` of `expect`.
    Market(times=120, entities=100, clusters=20, groups=4, cluster_share=0.3, group_share=0.0, tail=4.0, ic=0.0)
      A synthetic evaluation table: `entities` candidates in `clusters` (and the clusters in `groups`) at `times` hourly decision times.
    Source(entities=12, hours=336, settle=8, lag=0, revised=0.0)
      A synthetic field table: `entities` entities with an hourly `price` (a random walk) and a `flow` settled every `settle` hours and known `lag` hours after it; a share `revised` of the flows gets a second vintage one settlement later.
  for example: CASES = [Case("the top two of six less the bottom two", thirds, 0.4, given=SIX)], where SIX is a polars DataFrame of the evaluation table (t, entity, score, y) and thirds(SIX) returns t, value
```

```python
# ideas/carry/evals/spreads.py
"""Spreads of the label between the candidates with the highest and the lowest scores, at each decision time."""

from datetime import datetime

import polars as pl

from research_ci.api import Case
from research_ci.eval.statistics.spread import quantile_spread

SIX = pl.DataFrame(
    {"t": [datetime(2026, 1, 1)] * 6, "score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "y": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}
)


def thirds(table: pl.DataFrame) -> pl.DataFrame:
    """The mean label of the top third by score less that of the bottom third."""
    return quantile_spread(table, share=1 / 3)


def halves(table: pl.DataFrame) -> pl.DataFrame:
    """The mean label of the top half by score less that of the bottom half."""
    return quantile_spread(table, share=0.5)


CASES = [
    Case("the top two of six less the bottom two", thirds, 0.4, given=SIX),
    Case("reversed scores turn it around", thirds, -0.4, given=SIX.with_columns(-pl.col("score"))),
    Case("the top three of six less the bottom three", halves, 0.4, given=SIX),
]
```

`feat.py` imports them into `EVALS`, wrapped in `Eval` to name them, and to judge the halves at 24h only:

```python
# ideas/carry/feat.py
from research_ci.api import Eval
from evals.spreads import halves, thirds

EVALS = [
    Eval(thirds, name="top minus bottom third"),
    Eval(halves, name="top minus bottom half", horizons=("24h",)),
]
```

The engine runs a case eval only after its tests pass: the cases and properties of its module that run it, never the idea's other code and never pytest. `rci eval check` runs them any time, for free, and records nothing. The third case was worked wrong by hand: the top three of six have a mean label of 0.5 and the bottom three of 0.2, so the spread is 0.3, not 0.4:

```text
$ rci eval check ideas/carry/evals/spreads.py
ideas/carry/evals/spreads.py
  halves  invalid 0/1
    FAIL case 'the top three of six less the bottom three': expected 0.4, got 0.3
    fix the eval or its CASES in evals/spreads.py
  thirds  valid 2/2
    ok   case 'the top two of six less the bottom two': as worked by hand
    ok   case 'reversed scores turn it around': as worked by hand
pits that a data eval claims in covers are planted by `rci data check`, on the idea's data
```

Each alternative explanation in the contract gets a null world that answers it, or the ResultCard marks it uncovered; the worlds judge the evals. The end of `rci methods` shows how one is written:

```text
$ rci methods
...                                                    (the library, data evals, pits, and case evals)
null world: nulls/feat.py: NULLS = [Null(explains, world)]; give each alternative of the contract one, or the ResultCard marks it uncovered
  from research_ci.api import Null, PermuteWithin
    Null(explains: str, world: PermuteWithin, keeps: str | None = None, evals: tuple[str, ...] = (), reps: int = 50, note: str = '')
      explains  the id of the contract's alternative this world answers
      world     a PermuteWithin: how the scores are shuffled while the labels stay
      keeps     the column the alternative lives in; None means the world's bucket_by
      evals     the names of the evals this world judges; () means every eval of the layer, defaults included
      reps      how many shuffles, at least 20; 50 by default
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
    Null("tail", PermuteWithin("time", clusters="symbol", block="horizon"), evals=("rank ic without top symbol",)),
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
floors     floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
cost       full call 2.4 ms, peak memory +0.0 MB (process 277 MB) · on cut data 0.8 ms, 2.2 ms, 3.2 ms · not judged
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
floors     floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    FAIL scores read nothing ahead: 2026-01-04T18:00Z: 747 values of score differ; 2026-01-08T04:00Z: 1485 values of score differ; 2026-01-11T14:00Z: 2223 values of score differ
cost       full call 1.5 ms, peak memory +0.2 MB (process 282 MB) · on cut data 1.3 ms, 1.4 ms, 1.4 ms · not judged
----------------------------------------------------------------------------------------------------
```

Submitted, it would be TAINTED and cost a trial. The agent submits random scores first, with the halves still invalid:

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
floors     floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
cost       full call 0.3 ms, peak memory +0.1 MB (process 283 MB) · on cut data 0.1 ms, 0.1 ms, 0.2 ms · not judged
need       z ≥ 2.00 on submission 1
decisive   floor    carry: gain over random 1.8 bp/entry ± 4.0 at 24h (≈ 6.4%/yr if rolled) · z 1.238 < 2.00 · 50 shuffles · FAILED
evals      shown, not deciding: the engine's default evals and the idea's own (library or case)
           default  [!]  rank ic vs floor (chance, timing): z 1.59 < 2.00 at 24h · best of 2 horizons · 50 shuffles
                          8h     value   0.001575 ± 0.02114    null -0.0006697 sd 0.02031      41 blocks  z 0.1106
                          24h    value    0.03418 ± 0.01805    null  -0.002541 sd 0.02013      13 blocks  z 1.824
           default  [!]  rank ic without top symbol vs floor (chance, timing): z 1.768 < 2.00 at 24h · best of 2 horizons · 50 shuffles
                          8h     value    0.01655 ± 0.02492    null   0.001519 sd 0.02481      41 blocks  z 0.6059
                          24h    value    0.04574 ± 0.02488    null  -0.001533 sd 0.02482      13 blocks  z 1.904
           default  [!]  rank ic without top symbol vs null 1 (tail): z 1.768 < 2.00 at 24h · best of 2 horizons · 50 shuffles
                          8h     value    0.01655 ± 0.02492    null   0.001519 sd 0.02481      41 blocks  z 0.6059
                          24h    value    0.04574 ± 0.02488    null  -0.001533 sd 0.02482      13 blocks  z 1.904
           default  ok   top minus bottom fifth vs floor (chance, timing): z 2.148 at 24h · best of 2 horizons · 50 shuffles
                          8h     value   0.000144 ± 0.0001743  null -1.341e-07 sd 0.0001971    41 blocks  z 0.7312
                          24h    value  0.0007446 ± 0.0002384  null  -2.82e-05 sd 0.0003219    13 blocks  z 2.401
           case     [!]  top minus bottom third (valid 2/2) vs floor (chance, timing): z 1.406 < 2.00 at 24h · best of 2 horizons · 50 shuffles
                          8h     value  4.505e-05 ± 9.724e-05  null -3.123e-06 sd 0.0001044    41 blocks  z 0.4614
                          24h    value   0.000283 ± 0.0001906  null -3.052e-05 sd 0.0001836    13 blocks  z 1.708
           case     FAIL top minus bottom half: invalid 0/1: case 'the top three of six less the bottom three': expected 0.4, got 0.3; not judged, so no numbers; fix the eval or its CASES in evals/spreads.py; `rci eval check evals/spreads.py` runs them for free
worlds     floor    ok   alternatives covered: chance: floor · timing: floor · tail: null 1
verdict    FAILED
           carry: does not beat random, z 1.238 < 2.00 (decisive test, floor): gain over random 1.76 bp per entry ± 3.97 at 24h
----------------------------------------------------------------------------------------------------
feat OPEN: not up for review; `rci next` says what to do and how many submissions are left

$ rci next -i ideas/carry
agent      rci feat submit <file>               submission 1 FAILED: carry: does not beat random, z 1.238 < 2.00 (decisive test, floor): gain over random 1.76 bp per entry ± 3.97 at 24h; write a new features file in research/ and submit it (2 of 3 submissions left)
```

- `floors` and `decisive`: the floor list's evals. The toy's candidates earn carry whatever the scores, so the target itself is well above zero; the card shows the gain over random, 1.8 bp per entry, and its z against the floor world, 1.24, short of 2.00, so the submission FAILED and the layer stays OPEN for the agent; the verdict says which test failed, its z against the bar, and the effect. The decisive test draws its shuffles 50 at a time, up to 200, and stops once more could not change its decision: 1.24 is far enough below 2.00 after the first 50. It reused the calibration of the data check, which the DataCard shows.
- `evals`: the default evals and the idea's own, each with whose it is; one that does not beat a world is marked `[!]` with its z against the bar, and each takes 50 shuffles. `null 1` shuffles the way the floor world does, so its row is judged once and repeats the floor's. The case eval `top minus bottom third` passed its two tests and is judged; `top minus bottom half` failed one, so it is shown with the test, what it expected and what it got, and how to fix it, and with no numbers: it counts as no evidence. `worlds` says which world answers which alternative; one with none is a warning, not a block.

The agent fixes the case, from 0.4 to 0.3, and checks the module again:

```text
$ rci eval check ideas/carry/evals/spreads.py
ideas/carry/evals/spreads.py
  halves  valid 1/1
    ok   case 'the top three of six less the bottom three': as worked by hand
  thirds  valid 2/2
    ok   case 'the top two of six less the bottom two': as worked by hand
    ok   case 'reversed scores turn it around': as worked by hand
pits that a data eval claims in covers are planted by `rci data check`, on the idea's data
```

A validation is kept in `.rci/evals/` under the hash of the module, the idea's files it imports, and the engine, so the next submission runs the tests again only for the module that changed. The same scores again are refused without spending a trial, since they would not be a new trial. The spread is the second submission, so it needs `z ≥ 2.60`:

```text
$ rci feat submit ideas/carry/research/noise.py -i ideas/carry
error: the same scores as submission 1; they would not be a new trial; no submission was spent

$ rci feat submit ideas/carry/research/spread.py -i ideas/carry
ResultCard · carry · feat v1 · submission 2 of 3
features   ideas/carry/research/spread.py · a5a0d9d57f06
scores     2,961 rows · 100% of candidate rows · 8af96e36d3cc
floors     floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
cost       full call 0.2 ms, peak memory +0.0 MB (process 338 MB) · on cut data 0.1 ms, 0.1 ms, 0.2 ms · not judged
need       z ≥ 2.60 on submission 2
decisive   floor    carry: gain over random 18.0 bp/entry ± 6.4 at 24h (≈ 65.5%/yr if rolled) · z 4.164 ≥ 2.60 · 100 shuffles · PASSED
evals      shown, not deciding: the engine's default evals and the idea's own (library or case)
           default  ok   rank ic vs floor (chance, timing): z 3.139 at 24h · best of 2 horizons · 50 shuffles
                          8h     value     0.1699 ± 0.03826    null    0.00302 sd 0.05813      41 blocks  z 2.87
                          24h    value      0.269 ± 0.05959    null  -0.003007 sd 0.0835       13 blocks  z 3.258
           default  ok   rank ic without top symbol vs floor (chance, timing): z 3.533 at 24h · best of 2 horizons · 50 shuffles
                          8h     value     0.2087 ± 0.04598    null   0.002167 sd 0.06351      41 blocks  z 3.252
                          24h    value     0.3152 ± 0.07336    null -0.0007996 sd 0.08935      13 blocks  z 3.537
           default  ok   rank ic without top symbol vs null 1 (tail): z 3.533 at 24h · best of 2 horizons · 50 shuffles
                          8h     value     0.2087 ± 0.04598    null   0.002167 sd 0.06351      41 blocks  z 3.252
                          24h    value     0.3152 ± 0.07336    null -0.0007996 sd 0.08935      13 blocks  z 3.537
           default  [!]  top minus bottom fifth vs floor (chance, timing): z 1.736 < 2.60 at 24h · best of 2 horizons · 50 shuffles
                          8h     value   0.001056 ± 0.0003151  null  8.123e-05 sd 0.0005137    41 blocks  z 1.897
                          24h    value   0.002651 ± 0.0008472  null  9.465e-05 sd 0.001279     13 blocks  z 1.999
           case     ok   top minus bottom third (valid 2/2) vs floor (chance, timing): z 3.132 at 24h · best of 2 horizons · 50 shuffles
                          8h     value  0.0008304 ± 0.0002052  null   9.05e-06 sd 0.0002922    41 blocks  z 2.811
                          24h    value   0.002456 ± 0.0004698  null -4.131e-05 sd 0.0007705    13 blocks  z 3.241
           case     ok   top minus bottom half (valid 1/1) vs floor (chance, timing): z 3.133 at 24h · one horizon · 50 shuffles
                          24h    value   0.001922 ± 0.0003696  null -3.254e-05 sd 0.0006237    13 blocks  z 3.133
worlds     floor    ok   alternatives covered: chance: floor · timing: floor · tail: null 1
verdict    PASSED
           carry: beats random, z 4.164 ≥ 2.60
----------------------------------------------------------------------------------------------------
card 8e071f68
up for review; a person approves it with `rci approve --stage feat`
```

Its decisive effect beats random with z 4.16; nearer the bar this time, the decisive test drew 100 shuffles before its decision was settled. It PASSED although the default top-minus-bottom-fifth eval, one candidate each way, stayed below the bar; both case evals are valid now and beat the floor world. A PASSED submission puts the layer up for review, with the card id printed under its card; the agent could go on submitting, and a later FAILED one would take it back to OPEN, since a person can accept only a latest submission that PASSED:

```text
$ rci status -i ideas/carry
carry
  contract  PASSED  v1           approved by a person at a terminal
  data      PASSED  v1           approved by a person at a terminal
  feat      REVIEW  v1   2/3     submission 2 PASSED: carry: beats random, z 4.164 ≥ 2.60
  signal    OPEN    v1   0/3
  algo      OPEN    v1   0/3
  seal      OPEN    v1
  forward   OPEN    v1
  capital   OPEN    v1
  monitor   OPEN    v1

$ rci inbox ideas
ideas/carry              rci approve --stage feat             submission 2 PASSED: carry: beats random, z 4.164 ≥ 2.60; review the ResultCard and approve it; `rci feat show` prints it with the card id that a relayed approval quotes
```

Each submission records what it used: the features file, `feat.py` and `nulls/feat.py`, the module of every case eval it ran (`eval:evals/spreads.py`), the code of every plugin feat runs, the engine, and the calibration. Approval checks them again. The first approval below came after a comment was added to `feat.py`; with the file restored, the person approves the ResultCard:

```text
$ rci approve --stage feat -i ideas/carry
error: changed since submission 2: feat_file; submit again

$ rci approve --stage feat -i ideas/carry
approve feat v1 of carry
...                                                    (the card, as printed before)
card 8e071f68
  frozen dependencies:
    calibration           95851be902d2
    contract              365c17d571b4
    engine                b54ecee968d1
    engine_packages       5718313514dc
    eval:evals/spreads.py c785ab89d27a
    feat_file             d99fb5ed33e7
    feat_submission       eca6e3812197
    features              a5a0d9d57f06
    nulls_file            a2067050ac0e
    plugin:ref.carry      599e2f89646d
    plugin:ref.flat_costs 599e2f89646d
    plugin:ref.mean       599e2f89646d
    plugin:toy.pairs      29908a9eddf3
    python                CPython 3.11.15
    upstream:data         ff3413989cb5
Approve feat v1? [y/N]: y
feat v1 PASSED: freeze 1dc2ca788082 · approved by a person at a terminal
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

Each of those 8 records is one line of canonical JSON in `ideas/carry/.rci/ledger.jsonl`, chained by hash: `init`; the contract's `derive` and `approve`; data's `check`, which holds the calibration, and `approve`; one `submit` for each judged features file, with the validation of each case eval it ran; and feat's `approve`. The refused submission, the refused example, the explorations, the eval checks, and the refused approval left no record. Every command checks the chain and the head kept in `$RCI_HOME/heads/`, so a failed submission deleted from the end is noticed. The source's admission and its approval are in `$RCI_HOME/sources/ledger.jsonl`, shared by every idea on the machine.

## Not shown here

- A backtest. Nothing before the algo layer runs one: a contract may name a consumer (a backtester), but only for its decision times. Effects through the consumer, with a pilot change and branches from the baseline, belong to the planned algo layer: see [design.md](design.md#consumer-effects-branch-and-fullpath).
- Data treatments and their influence, and data evals with the pits they cover: see [The data check](design.md#the-data-check).
- A budget used up: the layer is PAUSED until a person extends it at a terminal: see [Budgets and PAUSED](design.md#budgets-and-paused).
- Reopening: deriving a passed contract again, checking passed data again, or submitting to a passed layer starts a new version and marks the started stages after it STALE; trial counts go on across versions.
