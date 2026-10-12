# Research CI catalog

Generated from the code by `rci methods --markdown`; a test fails when it is out of date: `uv run rci methods --markdown > docs/catalog.md`.

## Stages

### contract: `rci contract derive`

Lists: `stages/contract/floor.py`, `stages/contract/defaults.py`.

| Step | Checks or computes | Role | List | Eval method |
| --- | --- | --- | --- | --- |
| data is current | Warns when the newest row is too old | warns | default | — |
| universe point-in-time | Same candidates when the data is cut | blocks | floor | `protocol.universe_pit` |
| universe causal | Later values leave earlier candidates unchanged | blocks | floor | `protocol.universe_causal` |
| universe membership | A static candidate list makes history explore-grade | warns | floor | — |
| label: worked examples | Worked examples hold, and move when inputs move | blocks | floor | `protocol.label_cases` |
| label: fields | Label options name fields; fails when no rows result | blocks | floor | — |
| label: no data after t + h | Labels unchanged when the data stops early | blocks | floor | `protocol.label_truncation` |
| power | MDE per horizon; blocks when no noise is measurable | blocks | floor | `data.concentration.shares` |
| target reads the score | The decisive target moves with the scores | blocks | floor | `protocol.target_reads_score` |

### data: `rci data check`

Lists: `stages/data/floor.py`, `stages/data/defaults.py`, `stages/data/source.py`.

| Step | Checks or computes | Role | List | Eval method |
| --- | --- | --- | --- | --- |
| **admission, once per source** | | | | |
| baseline | Gaps, near-duplicates, and level shifts in the rows | warns | floor | `data.baseline.baseline` |
| replay as of past times | Fetches as of past times match the full fetch | blocks | floor | `data.replay.{cut_points, replay}` |
| canary: replay | The replay catches a deletion made with hindsight | blocks | floor | `data.baseline.events` · `data.replay.replay` |
| independent source agrees *(reference)* | Another source agrees; grades every field | blocks | floor | `data.reconcile.{summarize, requests, compare}` · `data.baseline.baseline` |
| canary: reconciliation *(reference)* | The comparison catches altered values | blocks | floor | `data.reconcile.{compare, requests}` · `data.baseline.baseline` |
| row-by-row code *(data.py, nulls/data.py)* | Row-by-row code in the idea's data files | warns | floor | — |
| data is current | Warns when the newest row is too old | warns | default | — |
| level shifts *(levels)* | Masks level jumps the source did not confirm | treats | default | `data.baseline.level_shift_anomalies` |
| level sanity *(levels)* | Masks levels at zero, stuck, or spiking | treats | default | `data.sanity.{nonpositive, stale, spikes}` |
| identity breaks *(levels)* | Splits an entity that returns at another level | treats | default | `data.identity.{breaks, renames}` |
| coverage | Drops entities shorter than the longest horizon | treats | default | `data.coverage.{short_lived, thin}` |
| hedge guard *(hedge)* | Drops candidates whose legs are not one asset | treats | default | `data.invariants.{hedged_legs, hedge_check}` |
| influence concentration | Shows the clusters most of the noise rests on | warns | default | `data.concentration.shares` |
| eval validation *(data.py)* | Case evals pass their tests; covered pits are caught | blocks | floor | `data.pits.{unit, level_break, gap, duplicate, shift, leak, spike, swap}` |
| the idea's evals *(data.py)* | Masks, drops, or asks about what a valid eval flags | treats | floor | — |
| field grades | The decisive number reads only verified fields | warns | floor | — |
| treatment influence | Each treatment's move of the decisive number, in MDE | warns | floor | — |
| power | MDE per horizon, before and after the treatments | shown | floor | — |
| decisive test calibration | Stand-ins die and planted signals live | blocks | floor | `protocol.target_reads_score` |

### feat: `rci feat submit`

Lists: `stages/feat/floor.py`, `stages/feat/defaults.py`.

| Step | Checks or computes | Role | List | Eval method |
| --- | --- | --- | --- | --- |
| screen | Refuses a missing, example, or unparsable file | blocks | floor | — |
| scores rank candidates | Scores rank candidates at half the times | blocks | floor | — |
| scores read nothing ahead | Same scores with the data cut early | blocks | floor | — |
| decisive test | The decisive effect beats the floor world: z ≥ z_need | decides | floor | — |
| rank ic | Rank correlation of score and label per time | shown | default | `statistics.ic.rank_ic` |
| rank ic without top {clusters} | Rank IC without the top fifth of clusters | shown | default | `statistics.ic.rank_ic` · `structure.filters.{without_top, kept_without_top}` |
| top minus bottom fifth | Top fifth by score less the bottom fifth | shown | default | `statistics.spread.quantile_spread` |
| the idea's evals *(feat.py)* | Each valid idea eval, judged like a diagnostic | shown | floor | — |
| alternatives covered | Every alternative of the contract has a world | warns | floor | — |
| cost | Wall time and peak memory of the features call | shown | floor | — |

## Case evals

A case eval is a function of the idea's own, in a module such as `evals/<name>.py`, with its tests next to it: CASES (required) and PROPERTIES (optional); data.py or feat.py imports it into EVALS (`from evals.<name> import <eval>`). Before anything runs, the engine runs its tests, the cases and properties that run it, and only a case eval that passes them all runs; one without tests is refused, an invalid one is shown with the test it failed and no result. `rci eval check <file>` runs them any time, for free; a validation is kept until the module's files change.

| Layer | Listed in | An eval gets | It returns | Eval(fn, ...) adds | Validated by |
| --- | --- | --- | --- | --- | --- |
| data | `data.py` | the explore data, a FieldTable | the rows it flags: entity, event_time, optionally field | on_fail (mask, drop_entities, ask), covers | its CASES and PROPERTIES; every pit of what it covers that moves the decisive number |
| feat | `feat.py` | the evaluation table of one horizon | t, value per decision time | label, horizons, combine (max, each) | its CASES and PROPERTIES |

| Declaration | Means |
| --- | --- |
| `Eval(of, name='', on_fail=None, covers=(), label=None, horizons=None, combine=None, note='')` | an entry of EVALS with what its layer reads: on_fail, covers (data.py); label, horizons, combine (feat.py) |
| `Case(name, run, expect, tol=1e-09, given=None)` | `run()` must give `expect`; with `given`, `run(given)` must, so a case can name the eval and the frame it reads: `Case("the top two of six", thirds, 0.4, given=SIX)`. |
| `Size(of, market=Market(), markets=100)` | On markets with no information, the statistic must beat the floor world (z ≥ 2) no more often than chance allows, over `markets` seeds. |
| `Power(of, market=Market(), markets=40, rate=0.8)` | On markets with `market.ic` of information, the statistic must beat the floor world at least `rate` of the time. |
| `Recovers(of, market, expect, tol, markets=20)` | Averaged over `markets` seeds and decision times, the statistic must come within `tol` of `expect`. |
| `Market(times=120, entities=100, clusters=20, groups=4, cluster_share=0.3, group_share=0.0, tail=4.0, ic=0.0)` | A synthetic evaluation table: `entities` candidates in `clusters` (and the clusters in `groups`) at `times` hourly decision times. |
| `Source(entities=12, hours=336, settle=8, lag=0, revised=0.0)` | A synthetic field table: `entities` entities with an hourly `price` (a random walk) and a `flow` settled every `settle` hours and known `lag` hours after it; a share `revised` of the flows gets a second vintage one settlement later. |

For example: `CASES = [Case("the top two of six less the bottom two", thirds, 0.4, given=SIX)], where SIX is a polars DataFrame of the evaluation table (t, entity, score, y) and thirds(SIX) returns t, value`.

| Outcome | What happens |
| --- | --- |
| trusted | an eval of the engine's library: its tests run in the engine's CI, never at research time; it runs |
| valid n/n | a case eval that passed all n of its tests: it runs, and its line on the card says so |
| invalid | a case eval that failed a test, or a data eval that missed a pit of what it covers: it does not run and counts as no evidence; its line names the test, what it expected and what it got, and how to fix it. A data check with one is blocked; a feat submission shows it with no numbers |
| no tests | a case eval no case or property of its module runs: refused before anything runs, saying where CASES go |

## Eval library

### protocol

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `protocol` | `adds_up` | Components add up to net; the consumer's own checks pass. | algo, planned | — |
| | `branch_diff` | The branch's extra net over `(start, end]`, and its overlap. | algo, planned | — |
| | `busiest` | A decision time whose next period brings the most fields. | algo, planned | — |
| | `causal` | Changes known after `at` must move only what comes after. | algo, planned | — |
| | `consumer_pit` | Cut data gives the full run's periods before the cut. | algo, planned | — |
| | `deterministic` | Two identical consumer runs must give the same net. | algo, planned | — |
| | `fees_match` | The consumer charges what the cost model says. | algo, planned | — |
| | `label_cases` | Worked examples hold as written and move with their inputs. | contract | — |
| | `label_truncation` | Data stopping at `cut` leaves labels ending by it unchanged. | contract | — |
| | `matches_labels` | The consumer books what the label says, period by period. | algo, planned | — |
| | `max_drawdown` | The consumer's deepest fall in equity from its running peak. | algo, planned | — |
| | `passthrough` | The consumer's own baseline scores must change nothing. | algo, planned | — |
| | `perturb` | An uninformative change to scores: a share dropped or permuted. | algo, planned | — |
| | `target_reads_score` | A target must move with the scores. | contract, data | — |
| | `total` | The consumer's net summed over the run. | algo, planned | — |
| | `truncated` | The field table cut to the rows known by `t`. | inside eval | — |
| | `universe_causal` | Candidates at `t` ignore every value known after `t`. | contract | — |
| | `universe_pit` | Candidates before the cut equal those from all data. | contract | — |

### data

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `baseline` | `baseline` | Generic checks and their anomalies: gaps, near-duplicates, level shifts. | data | 7 ex |
| | `events` | First vintages with each event's spacing and its typical size. | data | 7 ex |
| | `first_vintages` | One row per event: its first vintage. | inside eval | 7 ex |
| | `level_shift_anomalies` | The baseline's level shifts alone, from the declared levels only. | data | — |
| `concentration` | `shares` | Each cluster's share of the noise, between 0 and 1. | contract, data | 7 ex |
| `coverage` | `short_lived` | Entities whose events span less than `horizon_ns`. | data, research code | 2 ex |
| | `thin` | Decision times with some candidates but fewer than `floor`. | data, research code | 1 ex |
| `identity` | `breaks` | Declared levels that come back elsewhere after a long gap. | data, research code | 6 ex |
| | `renames` | What the split calls each new series: `entity`, `at`, `to`. | data, research code | 1 ex |
| `invariants` | `hedge_check` | The card's hedged-legs line, naming the pairs most often apart. | data | — |
| | `hedged_legs` | Per candidate row: the legs' price ratio, correlation, and `ok`. | data | 1 ex |
| `pits` | `duplicate` | A duplicate: one event repeated a millisecond later. | data, research code | 1 ex |
| | `gap` | A gap: ten usual spacings missing from a random time. | data, research code | 1 ex |
| | `leak` | A leak: each event carries the value of the next. | data, research code | 1 ex |
| | `level_break` | A level break: one entity's field half again as large. | data, research code | 1 ex |
| | `shift` | A shift: one entity's events stamped one usual spacing late. | data, research code | 1 ex |
| | `spike` | A spike: one event of one entity's field, ten times larger. | data, research code | 1 ex |
| | `swap` | A swap: one entity's series replaced by another entity's. | data, research code | 1 ex |
| | `target` | An entity whose `field` has enough rows for a pit. | research code | 1 ex |
| | `unit` | A unit error: one entity's field a thousand times larger. | data, research code | 1 ex |
| `reconcile` | `compare` | One result per request: `match`, `mismatch` or `unanswered`. | data | 5 ex |
| | `requests` | Point requests for random rows, and window requests for anomalies. | data | 5 ex |
| | `summarize` | The check, each field's grade, and counts. | data | — |
| `replay` | `cut_points` | Cut points: holes, before new entities, and random times. | data | 3 ex |
| | `replay` | Each cut's fetch must equal the full fetch cut there. | data | 3 ex |
| `sanity` | `nonpositive` | Events of a declared level at zero or below. | data, research code | 3 ex |
| | `spikes` | Events of a declared level that jump and come back. | data, research code | 5 ex |
| | `stale` | Events repeating the value before them in a long run. | data, research code | 3 ex |

### statistics

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `ic` | `rank_ic` | Spearman correlation of score and label at each decision time. | feat, research code | 3 ex · 3 prop |
| `spread` | `quantile_spread` | Mean label of the top `share` less the bottom `share`. | feat, research code | 3 ex · 2 prop |

### structure

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `filters` | `kept_without_top` | Which rows `without_top` keeps. | feat, research code | 3 ex |
| | `without_top` | The table without its `share` of most extreme `column` values. | feat, research code | 2 ex |

### synthetic

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `synthetic` | `hand` | A small evaluation table worked by hand, at one time. | worked examples | — |
| | `market` | A synthetic evaluation table, one row per entity and time. | worked examples | — |
| | `source` | A synthetic field table of hourly prices and settled flows. | worked examples | — |

## Legend

**Role**: `blocks`: stops the stage when it fails: BLOCKED, refused, TAINTED, or FAILED · `decides`: PASS or FAIL; only the decisive test · `treats`: changes the data by default, weighed by its influence · `warns`: a warning on the card, or an item a person decides · `shown`: on the card, decides nothing. **List**: floor steps run on every idea, are part of the engine hash, and no idea can change them; default steps are adjusted from experience, and the engine hash leaves them out. *(levels)*, *(hedge)*, *(reference)*: only when a plugin declares it; *(data.py)* and the other files: only when the idea adds the file. **Used by**: contract, data, feat run it; research code may call it (`rci methods`); inside eval: other eval functions call it; worked examples: other modules' examples build on it; algo, planned: waits for the algo layer; tests only: nothing else calls it. **Backing**: ex worked examples, prop properties on synthetic markets, — none declared (the protocol tests are held to broken toy plugins in `tests/eval/test_protocol.py`).
