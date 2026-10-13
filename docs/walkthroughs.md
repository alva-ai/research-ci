# Walkthrough

> **Status:** everything below runs today. The signal and algo layers and the checkpoints are planned.

One idea, from an empty directory to a passed feat layer. Every block is what the command printed, with only the features paths shortened. The idea runs on the toy plugins of the engine's tests: `toy.data`, two sources for three symbols on three venues (`candles`, an hourly `close`, and `funding`, a `funding_rate` and its `funding_interval_hours` every 8 hours), and `toy.pairs` (every venue pair of a symbol, read from `wide`, long the venue with the lower latest funding rate; `spread_h` is the latest spread per hour). The printed blocks were recorded before sources, `wide`, and the outcome replaced the field table and labels: today's commands print sources, columns, and the outcome where these print fields and labels, while the files the agent writes are shown as they are written now. A lab registers its own plugins the same way. The toy data ends in January 2026, so the cards warn that it is old.

The contract sets the floor; the loops of data and feat set how far it goes. Only the contract needs a person before work starts: it defines the question and the one bar, the decisive test. Every layer after it is a loop the agent runs on its own, and a person looks at the result after the loop and approves it, or not. The agent here starts with no context but the idea's name: everything it needs is on the paths it takes, in `START.md`, `rci next`, `rci plugins`, `rci methods`, and the messages of what the engine refuses or warns about.

Every layer is the same four slots. Slot 1 is the engine's floor: the premises and the decisive test, the only thing that can pass a claim. Slot 2 is the engine's lists: the default checks and diagnostics, and the library evals an idea lists, which test generic H0s and are shown, or treat the data. Slot 3 is the lab's: the evals the agent writes for this case, each naming the H0 it rejects, and the null worlds; they can veto, never pass. Slot 4 is the lab evals' own tests, which the engine runs before a lab eval judges anything. Every layer with evals needs at least one lab eval.

| Actor | Who it is | What it does here |
| --- | --- | --- |
| agent | the research agent | reads `START.md`, writes `contract.py`, lab evals with their h0 and tests, `data.py`, `feat.py`, the null worlds, and features files; derives the contract, checks the data, checks its evals, explores and submits features |
| human | a person at a terminal, or an agent relaying the person's words with `--relayed "<words>" --card <id>` | approves the contract before any work, then each layer's result after its loop |
| engine | the `rci` command; deterministic, no LLM | runs the tests of the agent's lab evals before anything heavy, checks, computes, judges, refuses with how to fix it, and appends to the ledger |

Every stage is OPEN (the agent's turn), REVIEW (a person's turn: a result is ready), or PASSED; a layer whose budget is used up is PAUSED, and a stage after one that was reopened is STALE. Every driver reads `rci next`: who should act, the command, and why. A separate challenger that writes null worlds against the agent's tests, and a scheduler for the forward and monitor checkpoints, are planned.

## 1. Start, and the contract

`rci init` takes only the idea's directory: every idea goes through the same pipeline, the contract, the four layers, and the four checkpoints, which `rci status` lists below. It writes four files: `START.md`, the page an agent with no other context reads first; a `contract.py` skeleton; `evals/example.py`, the skeleton of a lab eval, which comes up in the data loop; and `research/example.py`, which comes up in the feat loop.

```text
$ rci init ideas/carry
created ideas/carry; read ideas/carry/START.md first: it says how the work goes
then fill in ideas/carry/contract.py and run `rci contract derive -i ideas/carry`
lab evals start from ideas/carry/evals/example.py, features from ideas/carry/research/example.py
`rci next -i ideas/carry` says what to do after every step
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
```

```text
$ rci next -i ideas/carry
agent      rci contract derive                  fill in contract.py (START.md says how; `rci plugins` lists the plugins, their options and fields), then derive the contract
```

`rci plugins` lists what a contract can name, read from the installed code: each plugin's kind, where it goes in the contract, its options with their defaults, and what it declares. The agent does not need to read a plugin's source to know that `toy.data` gives a source `candles` with `close` and a source `funding` with `funding_rate`, or what `ref.mean` takes:

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
    `p(t + h) / p(t) - 1` of the candidate's own inst, each price as it was first known at that moment.
    from an entry point of research-ci
target: Effect(on=<label name>, target=Use(name, {options}))
  ref.mean(cluster: str = 'inst', net: bool = True, top: float | None = None)
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
from research_ci.api import Alternative, Contract, Effect, Execution, Universe, Use

CONTRACT = Contract(
    claim="Cross-venue funding spreads persist long enough to pay the round trip",
    decisive=(Effect(target=Use("ref.mean", {"cluster": "symbol", "top": 0.34})),),
    alternatives=(Alternative("tail", "a few symbols with extreme funding carry the whole result"),),
    used=(),
    data="toy.data",
    costs=Use("ref.flat_costs", {"commission_bps": 1, "slippage_bps": 1, "legs": 2}),
    execution=Execution(every="1h", note="decide on the hour; fill both legs at the close known then"),
    universe=Universe("toy.pairs", clusters="symbol", groups=("venue_pair",)),
    outcome=Use("ref.carry", {"flow": "funding_rate"}),
    horizons=("8h", "24h"),
    explore_end="2026-01-15T00:00:00Z",
    seal_end="2026-01-20T00:00:00Z",
    budget=3,
)
```

The outcome is what a unit position in each pair earns over each horizon, gross: `ref.carry` gives `y`, its price part `y_price` and its funding part `y_flow`, and `y_end`, the last time it reads; without `horizons` the contract would take the decision period times 1, 4 and 16. The decisive effect names its target, and goes by its name: at each decision time, the mean carry after the round trip of the third of the candidates with the highest scores, each symbol weighted once. Feat will pass it when, with the submitted scores, it beats random scores significantly, and no lab eval or null world rejects it. The default target, `ref.mean` without `top`, averages every candidate whatever the scores say, so it could not judge any features; derive blocks a contract whose decisive number cannot move with the scores, and the contract stays OPEN with the reason. A plugin name that is not installed, or options a plugin rejects, stop derive with the close names or the plugin's signature.

`rci contract derive` loads the plugins, stores the sources in the vault, runs the protocol tests on the universe and the outcome, and computes power from the dispersion of the target alone:

```text
$ rci contract derive -i ideas/carry
ContractCard · carry · v1
claim      Cross-venue funding spreads persist long enough to pay the round trip
decisive   carry: beats random significantly · target ref.mean(cluster=symbol, top=0.34)
decisions  every 1h from execution · 335 decision times
           decide on the hour; fill both legs at the close known then
data       toy.data afa06051cf36 · 3,780 rows · 9 insts · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 259.3 days old
universe   toy.pairs · median 9 candidates in 3 symbol clusters per decision · membership pit
costs      ref.flat_costs(commission_bps=1, legs=2, slippage_bps=1) · median round trip 8.0 bp
labels     carry: ref.carry · H 8h 24h
checks     [!]  data is current: latest data is 259.3 days old
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
           each needs a null world in nulls/feat.py, or feat refuses a submission before it is counted; a world can veto
used       nothing declared
history    history can confirm
budget     3
verdict    DECIDABLE
           carry: noise measured at 8h, 24h; feat passes it where it beats random and no lab eval or null world rejects it
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
           [!] carry: symbol Z carries 55% at 24h of the noise the MDE rests on; rci data check looks at whether its data is fit for research
----------------------------------------------------------------------------------------------------
card 7c62db1e
up for review; a person approves it with `rci approve --stage contract`
```

- `decisive`: the target with its options, next to the median of 9 candidates on the `universe` line, so a person sees that the top third is 3 candidates.
- `checks`: the universe gives the same candidates on the sources cut at three times, with `wide` built again on them, and does not change when later values change; the outcome matches its worked examples, moves when its inputs move, and reads nothing after `t + h`; the target gives other numbers for other scores. An outcome option that names no column of the sources would add a line listing the columns there are.
- `power`: the smallest gain over random the data can detect, from the dispersion of the target alone: 5.9 bp per entry at 8h, about 65% a year if entered back to back, and 19.5 bp at 24h. `ref.mean` declares what one value is (`per = "entry"`) and that its values roll over a year (`rolls = True`); a target without them gets raw numbers. The noise can be measured, so the contract is DECIDABLE and goes up for review. The card shows only the scale of the noise, never where the target sits.
- `alts`: the engine adds `chance` and `timing` to the idea's own alternative; at feat the floor world answers them, and the idea's `tail` needs a null world of its own: feat refuses a submission until it has one, and that world can veto.
- `warnings`: Z, one of three symbols, carries 55% of the noise at 24h; derive only warns, and so does the data check.
- `card 7c62db1e`: the id of this card. An agent that shows the card to a person and relays the answer quotes it with `--card`; `rci contract show` prints the card and the id again.

A person reads the card and approves it, which freezes it. Approval reads the card from the ledger, so the id is the one derive printed:

```text
$ rci inbox ideas
ideas/carry              rci approve --stage contract         review the ContractCard and approve it; `rci contract show` prints it with the card id that a relayed approval quotes

$ rci approve --stage contract -i ideas/carry
approve contract v1 of carry
...                                                    (the card, as printed before)
card 7c62db1e
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
    "reason": "list at least one lab eval in EVALS in data.py, then check the data against the contract: a lab eval is a function of your own in evals/<name>.py (copy evals/example.py) with its CASES, listed with the H0 it rejects, Eval(fn, h0='<the H0>', on_fail=...), where on_fail='block' makes any row it flags after the treatments before it block the data; `rci eval check <file>` runs its tests for free, and the engine's library (`rci methods`) may be listed too"
  }
]
```

## 2. The data

The agent tries the data check before it has written anything for it. The data layer has evals, so it needs at least one lab eval, a test of this case's data written for this idea, and the check is refused before any data is fetched, saying what to write and where:

```text
$ rci data check -i ideas/carry
error: slot 3, lab evals: there is no data.py, and the data layer needs at least one, a test of the claim written for this idea, which can veto it: write a function of your own in evals/<name>.py with its CASES next to it (`rci eval check evals/<name>.py` runs them for free) and list it with the H0 it rejects; data.py defines EVALS = [Eval(fn, h0='<the H0 it rejects>', on_fail=..., covers=[(category, field)]), ...], with at least one lab eval: fn(fields) returns the rows it flags (inst, event_time, optionally field), on_fail is mask (the default), drop_insts, ask, or block (a lab eval only: any row it flags in the data after the treatments before it blocks the data), and covers names the pits it claims to catch; a lab eval is a function of the idea's own in evals/<name>.py with CASES in its module (`rci eval check <file>`), and h0 names in one sentence the H0 it rejects; an eval of the engine's library (`rci methods`; its options with functools.partial) sets no h0; nothing was recorded
```

`rci init` wrote the skeleton of a lab eval, with a feat eval and a data eval and the tests next to them. The engine refuses an eval from it while its marker line is there, so it is never used as it is:

```text
$ cat ideas/carry/evals/example.py
# rci:example -- the engine refuses a lab eval from this file while this line is in it. Copy
# it to evals/<name>.py, delete this line in the copy, and replace every TODO with your own eval and its tests.
"""TODO: one sentence on what this idea needs tested that a generic eval cannot tell: the H0 the eval rejects."""

from datetime import UTC, datetime

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)


def ranks_label(table: pl.DataFrame) -> pl.DataFrame:
    """A feat eval. table: the evaluation table of one horizon (t, inst, score, y, the label's parts y_<part>, and
    the candidate columns). It returns one value per decision time (t, value); here the rank correlation of score and
    label. TODO: what yours measures, such as the score against one part of the label, y_<part>."""
    return table.group_by("t").agg(pl.corr("score", "y", method="spearman").alias("value")).sort("t")


def flags(fields: pl.DataFrame) -> pl.DataFrame:
    """A data eval. fields: the explore data (inst, field, event_time, available_at, value, vintage, source). It
    returns the rows it flags (inst, event_time, optionally field); none means it found nothing. TODO: what yours
    catches; here a value of FIELD at zero or below."""
    return fields.filter((pl.col("field") == FIELD) & (pl.col("value") <= 0)).select("inst", "field", "event_time")


FIELD = "TODO"
FOUR = pl.DataFrame({"t": [T] * 4, "inst": list("abcd"), "score": [1.0, 2.0, 3.0, 4.0], "y": [0.1, 0.2, 0.4, 0.3]})
ROWS = pl.DataFrame({"inst": ["a", "a"], "field": [FIELD] * 2, "event_time": [T, T], "value": [1.0, 0.0]})

# The tests the engine runs before the eval judges anything, worked by hand; a failed one refuses the layer.
CASES = [
    Case("TODO: one swap in four: 1 - 6·2/(4·15)", ranks_label, 0.8, given=FOUR),
    Case("TODO: the value at zero is flagged", flags, 1, given=ROWS),
]

# feat.py: from evals.<name> import ranks_label
#          EVALS = [Eval(ranks_label, h0="TODO: the H0 it rejects, such as: the scores do not rank the label")]
# data.py: from evals.<name> import flags
#          EVALS = [Eval(flags, h0="TODO: such as: every value of the field is above zero", on_fail="block")]
```

The toy's pairs earn their carry from both legs' prices, and a leg whose close is at zero has no price to carry. The agent writes one lab eval for that, in a module of its own with its case worked by hand, and lists it in `data.py` with the H0 it rejects and `on_fail="block"`: a single close at zero blocks the data, rather than being masked. A data eval reads what its parameters name: `sources`, each source at its own cadence, as here, or `wide`, the sources as of each decision time, or both; it returns cells of a source (`source`, `inst`, `event_time`, and `column` when it names one) or rows of `wide` (`t`, `inst`):

```python
# ideas/carry/evals/closes.py
"""Closes at zero or below: a leg with no price cannot be carried, whatever its funding."""

from datetime import UTC, datetime

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=UTC)
TWO = {"candles": pl.DataFrame({"inst": ["a:X", "a:X"], "event_time": [T, T], "close": [101.0, 0.0]})}


def closes_at_zero(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The closes at zero or below, as the cells to flag."""
    zero = sources["candles"].filter(pl.col("close") <= 0)
    return zero.select(pl.lit("candles").alias("source"), "inst", "event_time", pl.lit("close").alias("column"))


CASES = [Case("of a close above zero and one at zero, the one at zero is flagged", closes_at_zero, 1, given=TWO)]
```

```python
# ideas/carry/data.py
from research_ci.api import Eval
from evals.closes import closes_at_zero

EVALS = [Eval(closes_at_zero, h0="every close the venue reported is above zero", on_fail="block")]
```

`rci eval check` runs the module's tests any time, for free, records nothing, and says how `data.py` lists the eval and what the engine would refuse there, such as a lab eval without `h0`:

```text
$ rci eval check ideas/carry/evals/closes.py
ideas/carry/evals/closes.py
  closes_at_zero  valid 1/1
    ok   case 'of a close above zero and one at zero, the one at zero is flagged': as worked by hand
  data.py lists 'closes_at_zero' · H0: every close the venue reported is above zero
pits that a data eval claims in covers, built in or in PITS, are planted by `rci data check`, on the idea's data
```

`rci data check` runs the lab eval's tests first, before anything heavy (slot 4). Then the source's admission, the first time this plugin, with these options and this code, is used anywhere on the machine: the rows replayed as of past times must agree with the full fetch, an independent source must agree on cells the engine picks, each about one column of one source, and two canaries make sure both checks can fail. That is fidelity, source by source and column by column at each source's own cadence: the rows are what the venue reported. Then it checks fitness, whether faithful rows can be used for this research, with the mandatory default list in `stages/data/defaults.py` (freshness, level shifts the independent source did not confirm, level sanity, identity breaks, coverage, the hedge guard on legs of one asset, and influence concentration) and the lab evals of `data.py`. Last, it calibrates the decisive test on the treated data, before any features exist. The DataCard is laid out by slot; no one has admitted `toy.data` yet, so slot 1 shows the admission under `source`:

```text
$ rci data check -i ideas/carry
DataCard · carry · v1
data       toy.data afa06051cf36 · 3,780 rows · 9 insts · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 259.3 days old
1 floor · the source's admission, and what the engine weighs on every idea's data
source     toy.data · admission 032a0e19d535 · not yet admitted: approving this card admits it for every idea [!]
           code b3b6e8b74f9c · 9 insts · 6,750 rows
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
checks     ok   field grades: unverified fields are not read: funding_interval_hours
grades     verified: close funding_rate · unverified: funding_interval_hours
treatments none needed
power      H            MDE before   after
           carry 8h            5.9     5.9
           carry 24h          19.5    19.5
calibrate  ok   carry at 8h vs floor: stand-ins die 97% (needs ≥ 92%) · a signal planted at 1.5 MDE (4.1 bp/entry) lives 98% (needs ≥ 80%)
2 engine lists · the default checks and the library evals data.py lists; what they flag is treated by default
checks     default  [!]  data is current: latest data is 259.3 days old
           default  ok   level sanity: no declared level at zero or below, stale, or spiking
           default  ok   identity breaks: no declared level comes back 5x away after a long gap
           default  ok   coverage: every inst covers 24h
           default  ok   hedged legs are one asset: every candidate row holds (close within 1.2x and returns correlated ≥ 0.3)
           default  [!]  influence concentration: more than 50% of the noise from one symbol: Z 55% at carry 24h; shown, not treated
clusters   share of the noise that goes away without each symbol, largest first
           carry 8h    X 0% · Y 0% · Z 0%
           carry 24h   Z 55% · X 0% · Y 0%
3 lab evals · tests of this idea's data: what each flags is treated as its on_fail says; with on_fail=block any row it flags in the data after the treatments before it vetoes the data
lab evals  ok   closes_at_zero (evals/closes.py) · H0: every close the venue reported is above zero
                on_fail block: valid 1/1 · flagged 0 rows of the data as it came
4 tests of lab evals · their CASES and PROPERTIES before the data was fetched, then the pits each covers
tests      ok   closes_at_zero (evals/closes.py): valid 1/1
decide     nothing
verdict    READY
           approving freezes the treated data; feat reads only that
           approving also admits toy.data for every idea
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
----------------------------------------------------------------------------------------------------
card 4de2b4ef
up for review; a person approves it with `rci approve --stage data`
```

- `1 floor`: the source's admission, the column grades (each column as `<source>.<column>`), the treatments with their influence on the decisive number (none here), the power before and after, and the calibration. `calibrate` tests the decisive test itself, in the floor world (the scores shuffled across symbols at each decision time): a stand-in that carries only each decision time's level must die at least 92% of the time, since it may beat the floor only by chance, and a signal planted at 1.5 MDE must live at least 80% of the time. It needs the contract, the treated sources, and the outcomes, but no scores, so it runs here; a target that cannot tell them apart blocks the data check, and the agent changes the target before writing any features. Feat submissions reuse it.
- `2 engine lists`: the default checks, each marked `default`; a library eval that `data.py` listed would follow, marked `library`. The `clusters` line shows how much of the outcome's noise, over all nine candidates, goes away without each symbol: Z carries 55% at 24h. Above half, the card warns and does nothing else: broken data is what the identity and sanity checks catch, an effect can sit in one symbol too, and with only three symbols one of them often carries this much.
- `3 lab evals`: each lab eval with its H0, what it flagged, and what its `on_fail` did with it. With `on_fail="block"` the eval judges the data after the treatments found before it, here none, so it vetoes only what they leave; a flagged row would make the card BLOCKED, naming the eval, its H0, how many rows, and a few of them.
- `4 tests of lab evals`: how each lab eval stood before the data was fetched, and the pits it claims to cover, planted in a column of clean copies of the sources (none here).

The toy data needs no treatment, and the card is READY. When a check flags something, the card lists each treatment, cells of a source masked to null, an inst dropped or split in every source, or candidate rows dropped, with its influence on the decisive number in MDE units; above half an MDE a person decides, and `rci data check --keep "<name>"` leaves a treatment unapplied. A blocked check keeps the layer OPEN with the reason.

A person approves the card. That freezes the check, its calibration, the treated sources, and `wide` built on them on the contract's decision times, which is all feat reads, and, since this is the first approval of data from `toy.data`, admits the source for every idea on the machine, in the shared sources ledger:

```text
$ rci next -i ideas/carry
human      rci approve --stage data             review the DataCard and approve it; `rci data show` prints it with the card id that a relayed approval quotes

$ rci approve --stage data -i ideas/carry
approve data v1 of carry
...                                                    (the card, as printed before)
card 4de2b4ef
  frozen dependencies:
    contract        365c17d571b4
    data_check      9f76784ba3de
    engine          74abe3689deb
    engine_packages 5718313514dc
    exclusions      47c61605ee6a
    panel           b0a41d046f24
    python          CPython 3.11.15
Approve data v1? [y/N]: y
data v1 PASSED: freeze ef6c2a3142f2 · approved by a person at a terminal
to keep it in git: git add ideas/carry/.rci ideas/carry/contract.py && git commit -m 'rci: approve data of carry'

$ rci next -i ideas/carry
agent      rci feat submit <file>               list at least one lab eval in EVALS in feat.py, a function of your own in evals/<name>.py with its CASES, listed as Eval(fn, h0='<the H0 it rejects>'); give each alternative of contract.py a null world in nulls/feat.py, such as NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", block="horizon"), evals=("<eval>",))]; each lab eval and world must reach z ≥ 2 at every horizon it covers, or the claim fails there; then write research/<name>.py from research/example.py (polars expressions; row loops run with a warning), try it with `rci feat explore <file>`, and submit it; `rci methods` lists the library, how a lab eval is written, and the fields of a null world (3 of 3 submissions left)
```

The data layer has no budget and counts no trials: nothing in it is chosen by its result. `rci next` now names everything the feat loop needs: the lab evals, the null worlds, the file to start the features from, and the free command to try a file.

## 3. The feat loop

Nothing is registered before features exist. What can pass the claim is already in the contract: the decisive effect, with the submitted scores, must beat the engine's floor world by `z ≥ 2·√(1+ln n)` on the n-th submission, the decisive test of slot 1, and it passes at each horizon where its z against the family's null reaches that bar. The engine runs its default diagnostics on every submission (rank IC, rank IC without the top symbol, and the top fifth by score less the bottom fifth), slot 2, shown and never deciding, as are the library evals an idea lists, which `rci methods` lists with the worked examples and properties that back each:

```text
$ rci methods
feat eval: feat.py defines EVALS = [Eval(fn, h0='<the H0 it rejects>', label=..., horizons=(...), combine='max'), ...], with at least one lab eval: fn(table) returns t, value for each decision time from the evaluation table (t, inst, score, y, y_<component>, and the candidate columns); a lab eval is a function of the idea's own in evals/<name>.py with CASES in its module (`rci eval check <file>`), h0 names in one sentence the H0 it rejects, and it must reach z ≥ 2 at every horizon it covers; an eval of the engine's library (`rci methods`; its options with functools.partial) sets no h0 and is shown
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

Slot 3 is the agent's: lab evals in `feat.py`, at least one, each a test of the claim with the H0 it rejects, and a null world for every alternative of the contract. In the floor world, and in every null world that names it, a lab eval must reach `z ≥ 2` at every horizon it covers, or the claim fails there; it can never pass what the decisive test failed, whose bar rises with the submissions, so it needs no pre-registration. `rci methods` says how one is written:

```text
$ rci methods
...                                                    (the library, data evals, and pits)
lab eval: a function of the idea's own, in a module such as `evals/<name>.py`, with its tests next to it: CASES (required), PROPERTIES (optional) and, for a data eval, PITS (optional); data.py or feat.py imports it into EVALS (`from evals.<name> import <eval>`) as `Eval(<eval>, h0='<the H0 it rejects>')`. It is a test of the claim: why a case is wrong is specific to the case, so every layer with EVALS needs at least one, its h0 is required (and refused on a library eval, which tests a generic H0), and it can veto: in feat, it and each null world must reach z ≥ 2.00 at every horizon it covers, or the claim fails there; in data, with on_fail='block', any row it flags in the data after the treatments before it blocks the data. It never passes what the decisive test failed, whose bar rises with the submissions, so it needs no pre-registration. Before anything heavy runs, the engine runs its tests, the cases and properties that run it, and refuses a layer with a lab eval that fails them; `rci eval check <file>` runs them any time, for free, and a validation is kept until the module's files change
  from research_ci.api import Eval, Case, Size, Power, Recovers, Market, Source, Pit
    Eval(of, name='', h0='', on_fail=None, covers=(), label=None, horizons=None, combine=None, note='')
      an entry of EVALS with what its layer reads: h0, the H0 a lab eval rejects (required for one, refused on a library eval); on_fail, covers (data.py); label, horizons, combine (feat.py)
    Case(name, run, expect, tol=1e-09, given=None)
      `run()` must give `expect`; with `given`, `run(given)` must, so a case can name the eval and the frame it reads: `Case("the top two of six", thirds, 0.4, given=SIX)`.
    Size(of, market=Market(), markets=100)
      On markets with no information, the statistic must beat the floor world (z ≥ 2) no more often than chance allows, over `markets` seeds.
    Power(of, market=Market(), markets=40, rate=0.8)
      On markets with `market.ic` of information, the statistic must beat the floor world at least `rate` of the time.
    Recovers(of, market, expect, tol, markets=20)
      Averaged over `markets` seeds and decision times, the statistic must come within `tol` of `expect`.
    Market(times=120, insts=100, clusters=20, groups=4, cluster_share=0.3, group_share=0.0, tail=4.0, ic=0.0)
      A synthetic evaluation table: `insts` candidates in `clusters` (and the clusters in `groups`) at `times` hourly decision times.
    Source(insts=12, hours=336, settle=8, lag=0, revised=0.0)
      A synthetic field table: `insts` insts with an hourly `price` (a random walk) and a `flow` settled every `settle` hours and known `lag` hours after it; a share `revised` of the flows gets a second vintage one settlement later.
    Pit(category, plant, note='')
      A planted problem of `category` in `field`: `plant(fields, field, rng)` returns a copy with it planted in one inst where the field varies.
  for example: CASES = [Case("the top two of six less the bottom two", thirds, 0.4, given=SIX)], where SIX is a polars DataFrame of the evaluation table (t, inst, score, y) and thirds(SIX) returns t, value
```

The claim is about funding, so the agent asks whether the pairs it ranks highest earn more of the funding part of the carry, `y_flow`, than those it ranks lowest: the top third less the bottom third, and the top half less the bottom half. A fifth of nine candidates is one, so the engine's default spread says little here:

```python
# ideas/carry/evals/spreads.py
"""Does the ranking pick the pairs whose funding pays? The funding part of the carry, y_flow, of the candidates with
the highest scores less that of the candidates with the lowest, at each decision time."""

from datetime import datetime

import polars as pl

from research_ci.api import Case
from research_ci.eval.statistics.spread import quantile_spread

SIX = pl.DataFrame(
    {"t": [datetime(2026, 1, 1)] * 6, "score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "y_flow": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}
)


def thirds(table: pl.DataFrame) -> pl.DataFrame:
    """The mean funding part of the top third by score less that of the bottom third."""
    return quantile_spread(table.with_columns(y=pl.col("y_flow")), share=1 / 3)


def halves(table: pl.DataFrame) -> pl.DataFrame:
    """The mean funding part of the top half by score less that of the bottom half."""
    return quantile_spread(table.with_columns(y=pl.col("y_flow")), share=0.5)


CASES = [
    Case("the top two of six less the bottom two", thirds, 0.4, given=SIX),
    Case("reversed scores turn it around", thirds, -0.4, given=SIX.with_columns(-pl.col("score"))),
    Case("the top three of six less the bottom three", halves, 0.4, given=SIX),
]
```

`feat.py` lists them with the H0 each rejects, and judges the halves at 24h only:

```python
# ideas/carry/feat.py
from research_ci.api import Eval
from evals.spreads import halves, thirds

EVALS = [
    Eval(
        thirds,
        name="funding, top minus bottom third",
        h0="the top third by score earns no more funding than the bottom",
    ),
    Eval(
        halves,
        name="funding, top minus bottom half",
        h0="the top half earns no more funding than the bottom half",
        horizons=("24h",),
    ),
]
```

The third case was worked wrong by hand: the top three of six have a mean funding part of 0.5 and the bottom three of 0.2, so the spread is 0.3, not 0.4:

```text
$ rci eval check ideas/carry/evals/spreads.py
ideas/carry/evals/spreads.py
  halves  invalid 0/1
    FAIL case 'the top three of six less the bottom three': expected 0.4, got 0.3
    fix the eval or its CASES in evals/spreads.py
  thirds  valid 2/2
    ok   case 'the top two of six less the bottom two': as worked by hand
    ok   case 'reversed scores turn it around': as worked by hand
  feat.py lists 'funding, top minus bottom third' · H0: the top third by score earns no more funding than the bottom
  feat.py lists 'funding, top minus bottom half' · H0: the top half earns no more funding than the bottom half
pits that a data eval claims in covers, built in or in PITS, are planted by `rci data check`, on the idea's data
```

Every alternative explanation in the contract needs a null world, or feat refuses the submission before it is counted. The end of `rci methods` shows how one is written:

```text
$ rci methods
...                                                    (the library, data evals, pits, and lab evals)
null world: nulls/feat.py: NULLS = [Null(explains, world, evals=(...))]; every alternative of the contract needs one, or feat refuses the submission before it is counted; a world judges the evals it names, or the decisive effect, and must reach z ≥ 2.00 at every horizon it judges
  from research_ci.api import Null, PermuteWithin
    Null(explains: str, world: PermuteWithin, keeps: str | None = None, evals: tuple[str, ...] = (), reps: int = 200, note: str = '')
      explains  the id of the contract's alternative this world answers
      world     a PermuteWithin: how the scores are shuffled while the labels stay
      keeps     the column the alternative lives in; None means the world's bucket_by
      evals     the names of the evals this world judges, any eval of the layer; () means the decisive effect
      reps      the most shuffles it draws, a batch at a time until its votes settle; at least 20, 200 by default
      note      free text kept with the world in the submission's record
    PermuteWithin(bucket_by: str, clusters: str | None = None, block: str | None = None)
      bucket_by  shuffle the scores across insts inside each bucket of this column; "time" is one bucket per decision time
      clusters   a column whose members move together and are shuffled as one
      block      repeat the same shuffle for every decision time inside a block this long; "horizon" uses each label horizon
  for example: NULLS = [Null("<id>", PermuteWithin("time", clusters="<cluster column>", block="horizon"), evals=("<eval>",))]
```

Here the world for `tail` shuffles the scores across whole symbols at each decision time while the outcomes stay, and judges the default `rank ic without top symbol`, which leaves out the symbol with the largest outcomes; it must reach `z ≥ 2` there too:

```python
# ideas/carry/nulls/feat.py
from research_ci.api import Null, PermuteWithin

NULLS = [
    Null("tail", PermuteWithin("time", clusters="symbol", block="horizon"), evals=("rank ic without top symbol",)),
]
```

A features file defines `features(wide, candidates)`: one score per candidate row, for every decision time at once, from `wide`, the frozen sources as of each decision time, one row per decision time and inst with every source's columns and `age_<source>`, how long before `t` that row became known. Both arrive as polars DataFrames, so eager polars code and pandas run as they are; `features` returns a polars DataFrame or LazyFrame, or a pandas DataFrame, and outcomes never reach it. A row of `wide` holds only what was known by its `t`, so the engine's check that nothing is read ahead builds `wide` again on the sources cut at a few decision times and runs the features on it. `research/example.py`, from `rci init`, shows the shape and the usual patterns, polars first, with `.lazy()` and `.collect()` around a pipeline that polars plans as a whole, and carries a marker line:

```text
$ cat ideas/carry/research/example.py
# rci:example -- the engine refuses to run this file while this line is in it. Copy it to
# research/<name>.py, delete this line in the copy, and replace every TODO with your own features.
"""TODO: one sentence on what the score measures and why it should rank the candidates."""

import polars as pl

FIELD = "TODO"  # a field of the data; `rci plugins` lists the fields a data plugin declares


def features(fields: pl.DataFrame, candidates: pl.DataFrame) -> pl.DataFrame:
    """One score per candidate row (t, inst, score), for every decision time at once.

    fields: inst, field, event_time, available_at, value, vintage, source; all of the frozen data.
    candidates: t, inst, and the universe's columns (such as the legs long and short, the cluster and the groups).
    Both are polars DataFrames. A score at t may use only rows with available_at <= t; join_asof on available_at gives
    the latest of them. Return a polars DataFrame or LazyFrame, or a pandas DataFrame.
    """
    # .lazy() lets polars plan the whole pipeline and run it once at .collect(); eager code works the same way.
    known = (
        fields.lazy()
        .filter(pl.col("field") == FIELD)
        .select("inst", pl.col("available_at").alias("known_at"), "value")
        .sort("known_at")
    )
    # TODO: transform each inst's series with expressions, for example
    #   known.with_columns(pl.col("value").rolling_mean_by("known_at", window_size="24h").over("inst"))
    #   known.with_columns(pl.col("value").diff().over("inst").alias("change"))
    scored = (
        candidates.lazy()
        .sort("t")
        .join_asof(known, left_on="t", right_on="known_at", by="inst", check_sortedness=False)
    )
    # Candidates with two legs: look each leg up by its own column instead of inst, then take the difference:
    #   leg = known.rename({"inst": "long", "known_at": "long_at", "value": "long_value"})
    #   scored = candidates.lazy().sort("t").join_asof(leg, left_on="t", right_on="long_at", by="long",
    #                                                  check_sortedness=False)
    # Ranks within each decision time: pl.col("score").rank().over("t").
    # pandas is fine for dates (pd.Timedelta, offsets, calendars), and any library runs. Whole-column polars
    # expressions like these are usually much faster than going row by row, which runs too, with a warning on the card;
    # the card also shows the time and peak memory the features took.
    return scored.select("t", "inst", pl.col("value").alias("score")).collect()
```

The engine refuses to run the example as it is. Any other file runs: research code is free. The engine reads every features file, with `feat.py`, `nulls/feat.py`, and the research modules they import, for row-by-row code, and the card shows what it finds as one warning that names each line; it never refuses the file and never costs a trial. This file loops over rows:

```python
# ideas/carry/research/rows.py
import polars as pl


def features(wide, candidates):
    rows = candidates.sort("t", "inst")
    scores = [(t, e, s) for t, e, s in rows.select("t", "inst", "spread_h").iter_rows()]
    return pl.DataFrame(scores, schema=["t", "inst", "score"], orient="row")
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
cost       full call 2.9 ms, peak memory +0.0 MB (process 281 MB) · on cut data 3.5 ms, 3.8 ms, 3.7 ms · not judged
warnings   [!] row-wise code: research/rows.py:6 iter_rows — polars expressions are usually much faster
----------------------------------------------------------------------------------------------------
```

Both cards have a `cost` line: the wall time of the full call and of the calls on the cut data, and the peak memory of the full call, the process's resident size sampled while it runs, with how much it grew over the start of the call. It is there for the agent to see what its code costs, is kept in a submission's record, and is never judged. On the toy data every call is over in a few milliseconds; written on whole columns, the same score is one line, and on a real panel usually much faster:

```python
# ideas/carry/research/spread.py
import polars as pl


def features(wide, candidates):
    return candidates.select("t", "inst", pl.col("spread_h").alias("score"))
```

`rci feat explore` runs the checks a submission must pass and records nothing. This file averages each venue's funding over the whole period, so its early scores change when later data is cut away:

```python
# ideas/carry/research/peek.py
import polars as pl


def features(wide, candidates):
    rate = wide.group_by("inst").agg(pl.col("funding_rate").mean())
    short = rate.rename({"inst": "short", "funding_rate": "s"})
    long = rate.rename({"inst": "long", "funding_rate": "l"})
    rows = candidates.join(short, on="short").join(long, on="long")
    return rows.select("t", "inst", (pl.col("s") - pl.col("l")).alias("score"))
```

```text
$ rci feat explore ideas/carry/research/peek.py -i ideas/carry
ExploreCard · carry · feat (not a submission; nothing is judged)
features   ideas/carry/research/peek.py · 2d50ba5b2d97
scores     2,961 rows · 100% of candidate rows · 5d7436c3d3f8
floors     floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    FAIL scores read nothing ahead: 2026-01-04T18:00Z: 747 values of score differ; 2026-01-08T04:00Z: 1485 values of score differ; 2026-01-11T14:00Z: 2223 values of score differ
cost       full call 2.6 ms, peak memory +0.2 MB (process 285 MB) · on cut data 1.8 ms, 2.0 ms, 2.3 ms · not judged
----------------------------------------------------------------------------------------------------
```

Submitted, it would be TAINTED and cost a trial. The agent submits random scores first, with the halves still wrong. A lab eval whose tests fail cannot judge, and listing it cannot set its veto aside: the submission is refused before it is counted, with the test, what it expected and got, and how to fix it:

```python
# ideas/carry/research/noise.py
import polars as pl


def features(wide, candidates):
    return candidates.select("t", "inst", (pl.struct("t", "inst").hash(7) / 2.0**64).alias("score"))
```

```text
$ rci feat submit ideas/carry/research/noise.py -i ideas/carry
error: slot 4, tests of lab evals: the lab eval 'funding, top minus bottom half' (evals/spreads.py) failed its tests, so it cannot judge: invalid 0/1: case 'the top three of six less the bottom three': expected 0.4, got 0.3; fix the eval or its CASES in evals/spreads.py; `rci eval check evals/spreads.py` runs them for free; no submission was spent
```

The agent fixes the case, from 0.4 to 0.3, and checks the module again:

```text
$ rci eval check ideas/carry/evals/spreads.py
ideas/carry/evals/spreads.py
  halves  valid 1/1
    ok   case 'the top three of six less the bottom three': as worked by hand
  thirds  valid 2/2
    ok   case 'the top two of six less the bottom two': as worked by hand
    ok   case 'reversed scores turn it around': as worked by hand
  feat.py lists 'funding, top minus bottom third' · H0: the top third by score earns no more funding than the bottom
  feat.py lists 'funding, top minus bottom half' · H0: the top half earns no more funding than the bottom half
pits that a data eval claims in covers, built in or in PITS, are planted by `rci data check`, on the idea's data
```

```text
$ rci feat submit ideas/carry/research/noise.py -i ideas/carry
ResultCard · carry · feat v1 · submission 1 of 3
features   ideas/carry/research/noise.py · 719e506e5111
scores     2,961 rows · 100% of candidate rows · e954ad32d08f
cost       full call 0.4 ms, peak memory +0.0 MB (process 286 MB) · on cut data 0.4 ms, 0.2 ms, 0.1 ms · not judged
1 floor · the premises and the decisive test, fixed before any result; only the decisive test can pass a claim
premises   floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
need       z ≥ 2.00 on submission 1
decisive   carry: gain over random 1.8 bp/entry ± 4.0 at 24h (≈ 6.4%/yr if rolled) · z 1.238 < 2.00 · 50 shuffles · FAILED
             8h     gain       0.4 ± 1.3       MDE      1.6    41 blocks  z 0.7849   vs the family's null 0.3139 ✗
             24h    gain       1.8 ± 4.0       MDE      3.1    13 blocks  z 1.617    vs the family's null 1.238 ✗
2 engine lists · the default diagnostics and the library evals feat.py lists, vs the floor world; shown only
shown      [!]  rank ic (default) vs floor (chance, timing): z 1.59 < 2.00 at 24h · best of 2 horizons · 50 shuffles
                8h     value   0.001575 ± 0.02114    null -0.0006697 sd 0.02031      41 blocks  z 0.1106
                24h    value    0.03418 ± 0.01805    null  -0.002541 sd 0.02013      13 blocks  z 1.824
           [!]  rank ic without top symbol (default) vs floor (chance, timing): z 1.768 < 2.00 at 24h · best of 2 horizons · 50 shuffles
                8h     value    0.01655 ± 0.02492    null   0.001519 sd 0.02481      41 blocks  z 0.6059
                24h    value    0.04574 ± 0.02488    null  -0.001533 sd 0.02482      13 blocks  z 1.904
           ok   top minus bottom fifth (default) vs floor (chance, timing): z 2.148 at 24h · best of 2 horizons · 50 shuffles
                8h     value   0.000144 ± 0.0001743  null -1.341e-07 sd 0.0001971    41 blocks  z 0.7312
                24h    value  0.0007446 ± 0.0002384  null  -2.82e-05 sd 0.0003219    13 blocks  z 2.401
3 lab evals · tests of the claim: each lab eval, and each null world, must reach z ≥ 2.00 at every horizon it covers, or the claim fails there
lab evals  ✗ funding, top minus bottom third (evals/spreads.py) · H0: the top third by score earns no more funding than the bottom
               vs floor (chance, timing): z 1.45 < 2.00 · one vote for all its horizons · best of 2 horizons · 150 shuffles
                  8h     value   2.52e-05 ± 1.388e-05  null  5.321e-07 sd 1.363e-05    41 blocks  MDE 3.872e-05  z 1.81 ✗
                  24h    value  5.868e-05 ± 4.507e-05  null -2.112e-08 sd 3.962e-05    13 blocks  MDE 0.0001126  z 1.482 ✗
           ✗ funding, top minus bottom half (evals/spreads.py) · H0: the top half earns no more funding than the bottom half
               vs floor (chance, timing): z 1.974 < 2.00 · one vote for all its horizons · one horizon · 200 shuffles
                  24h    value  8.251e-05 ± 4.604e-05  null  3.051e-06 sd 4.024e-05    13 blocks  MDE 0.0001144  z 1.974 ✗
worlds     ✗ null 1 (tail: a few symbols with extreme funding carry the whole result) judges rank ic without top symbol (default)
               vs null 1 (tail): z 1.746 < 2.00 · one vote for all its horizons · best of 2 horizons · 200 shuffles
                  8h     value    0.01655 ± 0.02492    null   0.001566 sd 0.02665      41 blocks  MDE 0.07573    z 0.5623 ✗
                  24h    value    0.04574 ± 0.02488    null -0.0004363 sd 0.02284      13 blocks  MDE 0.06491    z 2.022 ✗
           ok   alternatives covered: chance: floor · timing: floor · tail: null 1
4 tests of lab evals · run before the submission was counted
tests      ok   funding, top minus bottom third (evals/spreads.py): valid 2/2
           ok   funding, top minus bottom half (evals/spreads.py): valid 1/1
verdict    FAILED
           carry: does not beat random, z 1.238 < 2.00 (decisive test, slot 1): gain over random 1.76 bp per entry ± 3.97 at 24h
----------------------------------------------------------------------------------------------------
feat OPEN: not up for review; `rci next` says what to do and how many submissions are left

$ rci next -i ideas/carry
agent      rci feat submit <file>               submission 1 FAILED: carry: does not beat random, z 1.238 < 2.00 (decisive test, slot 1): gain over random 1.76 bp per entry ± 3.97 at 24h; write a new features file in research/ and submit it (2 of 3 submissions left)
```

The ResultCard is laid out by slot, then the verdict:

- `1 floor`: the premises, that the scores rank the candidates and read nothing ahead, and the decisive test. The toy's candidates earn carry whatever the scores, so the target itself is well above zero; the card shows the gain over random, 1.8 bp per entry at 24h, and its z against the floor world, 1.24, short of 2.00, with each horizon on its own line: its gain, the MDE at the bar, its own z, and its z against the family's null, which says whether the decisive test passes there. It passes at no horizon, so the submission FAILED, attributed to slot 1, and the layer stays OPEN for the agent. The decisive test draws its shuffles 50 at a time, up to 200, and stops once more could not change the decision at any horizon: 50 were enough. It reused the calibration of the data check, which the DataCard shows.
- `2 engine lists`: the default diagnostics against the floor world, on 50 shuffles each; one below the bar is marked `[!]` with its z against it, and decides nothing.
- `3 lab evals`: each lab eval with its H0, judged against the floor world and against every world that names it, each horizon with its value, its null, its MDE, the smallest effect the eval detects there four times in five against a bar of 2 (`(2 + 0.84) × the null's spread`, from the shuffles already drawn), its z, and its vote. A lab eval with `combine="max"`, the default, votes once for all its horizons with its family's z. These lab evals and the world are below 2, and drew up to 200 shuffles, a batch at a time, until their votes were settled; they would have rejected the claim, but slot 1 already failed it, and a lab eval can never pass what the decisive test failed. `worlds` shows the null world, which judges the default `rank ic without top symbol` here, and which world answers which alternative.
- `4 tests of lab evals`: how each lab eval stood before the submission was counted.

A validation is kept in `.rci/evals/` under the hash of the module, the idea's files it imports, and the engine, so the next submission runs the tests again only for the module that changed. The same scores again are refused without spending a trial, since they would not be a new trial. The spread is the second submission, so it needs `z ≥ 2.60`:

```text
$ rci feat submit ideas/carry/research/noise.py -i ideas/carry
error: the same scores as submission 1; they would not be a new trial; no submission was spent

$ rci feat submit ideas/carry/research/spread.py -i ideas/carry
ResultCard · carry · feat v1 · submission 2 of 3
features   ideas/carry/research/spread.py · a5a0d9d57f06
scores     2,961 rows · 100% of candidate rows · 8af96e36d3cc
cost       full call 0.4 ms, peak memory +0.0 MB (process 341 MB) · on cut data 0.2 ms, 0.2 ms, 0.2 ms · not judged
1 floor · the premises and the decisive test, fixed before any result; only the decisive test can pass a claim
premises   floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
need       z ≥ 2.60 on submission 2
decisive   carry: gain over random 18.0 bp/entry ± 6.4 at 24h (≈ 65.5%/yr if rolled) · z 4.164 ≥ 2.60 · 100 shuffles · PASSED at 8h, 24h
             8h     gain       5.3 ± 2.0       MDE      5.0    41 blocks  z 3.622    vs the family's null 3.763 ✓
             24h    gain      18.0 ± 6.4       MDE     15.6    13 blocks  z 3.954    vs the family's null 4.164 ✓
2 engine lists · the default diagnostics and the library evals feat.py lists, vs the floor world; shown only
shown      ok   rank ic (default) vs floor (chance, timing): z 3.139 at 24h · best of 2 horizons · 50 shuffles
                8h     value     0.1699 ± 0.03826    null    0.00302 sd 0.05813      41 blocks  z 2.87
                24h    value      0.269 ± 0.05959    null  -0.003007 sd 0.0835       13 blocks  z 3.258
           ok   rank ic without top symbol (default) vs floor (chance, timing): z 3.533 at 24h · best of 2 horizons · 50 shuffles
                8h     value     0.2087 ± 0.04598    null   0.002167 sd 0.06351      41 blocks  z 3.252
                24h    value     0.3152 ± 0.07336    null -0.0007996 sd 0.08935      13 blocks  z 3.537
           [!]  top minus bottom fifth (default) vs floor (chance, timing): z 1.736 < 2.60 at 24h · best of 2 horizons · 50 shuffles
                8h     value   0.001056 ± 0.0003151  null  8.123e-05 sd 0.0005137    41 blocks  z 1.897
                24h    value   0.002651 ± 0.0008472  null  9.465e-05 sd 0.001279     13 blocks  z 1.999
3 lab evals · tests of the claim: each lab eval, and each null world, must reach z ≥ 2.00 at every horizon it covers, or the claim fails there
lab evals  ✓ funding, top minus bottom third (evals/spreads.py) · H0: the top third by score earns no more funding than the bottom
               vs floor (chance, timing): z 15.95 ≥ 2.00 · one vote for all its horizons · best of 2 horizons · 50 shuffles
                  8h     value  0.0005378 ± 9.187e-06  null  6.595e-06 sd 3.542e-05    41 blocks  MDE 0.0001007  z 14.99 ✓
                  24h    value   0.001628 ± 2.133e-05  null -2.551e-05 sd 0.0001888    13 blocks  MDE 0.0005364  z 8.759 ✓
           ✓ funding, top minus bottom half (evals/spreads.py) · H0: the top half earns no more funding than the bottom half
               vs floor (chance, timing): z 9.286 ≥ 2.00 · one vote for all its horizons · one horizon · 50 shuffles
                  24h    value    0.00131 ± 1.564e-05  null -3.676e-05 sd 0.000145     13 blocks  MDE 0.000412   z 9.286 ✓
worlds     ✓ null 1 (tail: a few symbols with extreme funding carry the whole result) judges rank ic without top symbol (default)
               vs null 1 (tail): z 3.533 ≥ 2.00 · one vote for all its horizons · best of 2 horizons · 50 shuffles
                  8h     value     0.2087 ± 0.04598    null   0.002167 sd 0.06351      41 blocks  MDE 0.1805     z 3.252 ✓
                  24h    value     0.3152 ± 0.07336    null -0.0007996 sd 0.08935      13 blocks  MDE 0.2539     z 3.537 ✓
           ok   alternatives covered: chance: floor · timing: floor · tail: null 1
4 tests of lab evals · run before the submission was counted
tests      ok   funding, top minus bottom third (evals/spreads.py): valid 2/2
           ok   funding, top minus bottom half (evals/spreads.py): valid 1/1
verdict    PASSED at carry 8h, 24h
           carry: passes at 8h, 24h: the decisive test beats random there, z 4.164 ≥ 2.60 (slot 1), and every lab eval and null world that covers them reaches z ≥ 2.00 (slot 3)
----------------------------------------------------------------------------------------------------
card 66771cff
up for review; a person approves it with `rci approve --stage feat`
```

Its decisive effect beats random with z 4.16, and passes at both horizons; nearer the bar this time, the decisive test drew 100 shuffles before every horizon's decision was settled. Both lab evals reach 2 far above it, on 50 shuffles, and so does the world, so no horizon is rejected and the verdict names the passing horizons, `PASSED at carry 8h, 24h`, although the default top-minus-bottom-fifth diagnostic, one candidate each way, stayed below its bar: slot 2 decides nothing. A PASSED submission puts the layer up for review, with the card id printed under its card; the agent could go on submitting, and a later FAILED one would take it back to OPEN, since a person can accept only a latest submission that PASSED:

```text
$ rci status -i ideas/carry
carry
  contract  PASSED  v1           approved by a person at a terminal
  data      PASSED  v1           approved by a person at a terminal
  feat      REVIEW  v1   2/3     submission 2 PASSED: carry: passes at 8h, 24h: the decisive test beats random there, z 4.164 ≥ 2.60 (slot 1), and every lab eval and null world that covers them reaches z ≥ 2.00 (slot 3)
  signal    OPEN    v1   0/3
  algo      OPEN    v1   0/3
  seal      OPEN    v1
  forward   OPEN    v1
  capital   OPEN    v1
  monitor   OPEN    v1

$ rci inbox ideas
ideas/carry              rci approve --stage feat             submission 2 PASSED: carry: passes at 8h, 24h: the decisive test beats random there, z 4.164 ≥ 2.60 (slot 1), and every lab eval and null world that covers them reaches z ≥ 2.00 (slot 3); review the ResultCard and approve it; `rci feat show` prints it with the card id that a relayed approval quotes
```

Each submission records what it used: the features file, `feat.py` and `nulls/feat.py`, the module of every lab eval it ran (`eval:evals/spreads.py`), the code of every plugin feat runs, the engine, and the calibration. Approval checks them again. The first approval below came after a comment was added to `feat.py`; with the file restored, the person approves the ResultCard:

```text
$ rci approve --stage feat -i ideas/carry
error: changed since submission 2: feat_file; submit again

$ rci approve --stage feat -i ideas/carry
approve feat v1 of carry
...                                                    (the card, as printed before)
card 66771cff
  frozen dependencies:
    calibration           9f76784ba3de
    contract              365c17d571b4
    engine                74abe3689deb
    engine_packages       5718313514dc
    eval:evals/spreads.py 1c2c2b52d75c
    feat_file             e23f0f705f24
    feat_submission       16ff586774ce
    features              a5a0d9d57f06
    nulls_file            a2067050ac0e
    plugin:ref.carry      599e2f89646d
    plugin:ref.flat_costs 599e2f89646d
    plugin:ref.mean       599e2f89646d
    plugin:toy.pairs      b3b6e8b74f9c
    python                CPython 3.11.15
    upstream:data         ef6c2a3142f2
Approve feat v1? [y/N]: y
feat v1 PASSED: freeze 1b67acc467c1 · approved by a person at a terminal
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

Each of those 8 records is one line of canonical JSON in `ideas/carry/.rci/ledger.jsonl`, chained by hash: `init`; the contract's `derive` and `approve`; data's `check`, which holds the calibration, and `approve`; one `submit` for each judged features file, with the validation of each lab eval it ran; and feat's `approve`. The refused data check, the refused submissions, the refused example, the explorations, the eval checks, and the refused approval left no record. Every command checks the chain and the head kept in `$RCI_HOME/heads/`, so a failed submission deleted from the end is noticed. The source's admission and its approval are in `$RCI_HOME/sources/ledger.jsonl`, shared by every idea on the machine.

## Not shown here

- A veto: a submission whose decisive effect beats random but that a lab eval or a null world rejects at every horizon FAILED in slot 3, with each veto's H0, its z against 2, and whether its evidence points the other way or it could have seen an effect that size: see [The ResultCard](design.md#the-resultcard). A data lab eval with `on_fail="block"` that still flags rows after the treatments before it blocks the data: see [The DataCard](design.md#the-datacard).
- A backtest. Nothing before the algo layer runs one: a contract may name a consumer (a backtester), but only for its decision times. Effects through the consumer, with a pilot change and branches from the baseline, belong to the planned algo layer: see [design.md](design.md#consumer-effects-branch-and-fullpath).
- Data treatments and their influence, and data evals with the pits they cover: see [The data check](design.md#the-data-check).
- A budget used up: the layer is PAUSED until a person extends it at a terminal: see [Budgets and PAUSED](design.md#budgets-and-paused).
- Reopening: deriving a passed contract again, checking passed data again, or submitting to a passed layer starts a new version and marks the started stages after it STALE; trial counts go on across versions.
