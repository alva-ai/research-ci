# Research CI catalog

Generated from the code by `rci methods --markdown`; a test fails when it is out of date: `uv run rci methods --markdown > docs/catalog.md`.

Every layer is the same four slots. **1 floor**: the premises and the decisive test, which alone can pass a claim (z ≥ z_need, rising with the submissions). **2 engine lists**: the default list and the library evals an idea lists; they test generic H0s and are shown, or treat the data. **3 lab evals**: the evals the idea writes for its own case, and its null worlds; they can veto, never pass. **4 tests of lab evals**: what holds a lab eval to its own tests before it judges anything.

## Stages

### contract: `rci contract derive`

Lists: `stages/contract/floor.py`, `stages/contract/defaults.py`.

| Step | Checks or computes | Role | List | Eval method |
| --- | --- | --- | --- | --- |
| **1 floor** | | | | |
| universe point-in-time | Same candidates when the data is cut | blocks | floor | `protocol.universe_pit` |
| universe causal | Later values leave earlier candidates unchanged | blocks | floor | `protocol.universe_causal` |
| universe membership | A static candidate list makes history explore-grade | warns | floor | — |
| label: worked examples | Worked examples hold, and move when inputs move | blocks | floor | `protocol.label_cases` |
| label: fields | Label options name fields; fails when no rows result | blocks | floor | — |
| label: no data after t + h | Labels unchanged when the data stops early | blocks | floor | `protocol.label_truncation` |
| power | MDE per horizon; blocks when no noise is measurable | blocks | floor | `data.concentration.shares` |
| target reads the score | The decisive target moves with the scores | blocks | floor | `protocol.target_reads_score` |
| **2 engine lists** | | | | |
| data is current | Warns when the newest row is too old | warns | default | — |

### data: `rci data check`

Lists: `stages/data/floor.py`, `stages/data/defaults.py`, `stages/data/source.py`, `stages/lab.py`.

| Step | Checks or computes | Role | List | Eval method |
| --- | --- | --- | --- | --- |
| **1 floor: admission, once per source** | | | | |
| baseline | Gaps, near-duplicates, and level shifts in the rows | warns | floor | `data.baseline.baseline` |
| replay as of past times | Fetches as of past times match the full fetch | blocks | floor | `data.replay.{cut_points, replay}` |
| canary: replay | The replay catches a deletion made with hindsight | blocks | floor | `data.baseline.events` · `data.replay.replay` |
| independent source agrees *(reference)* | Another source agrees; grades every field | blocks | floor | `data.reconcile.{summarize, requests, compare}` · `data.baseline.baseline` |
| canary: reconciliation *(reference)* | The comparison catches altered values | blocks | floor | `data.reconcile.{compare, requests}` · `data.baseline.baseline` |
| **1 floor** | | | | |
| row-by-row code *(data.py)* | Row-by-row code in the idea's data files | warns | floor | — |
| field grades | The decisive number reads only verified fields | warns | floor | — |
| treatment influence | Each treatment's move of the decisive number, in MDE | warns | floor | — |
| power | MDE per horizon, before and after the treatments | shown | floor | — |
| decisive test calibration | Stand-ins die and planted signals live | blocks | floor | `protocol.target_reads_score` |
| **2 engine lists** | | | | |
| data is current | Warns when the newest row is too old | warns | default | — |
| level shifts *(levels)* | Masks level jumps the source did not confirm | treats | default | `data.baseline.level_shift_anomalies` |
| level sanity *(levels)* | Masks levels at zero, stuck, or spiking | treats | default | `data.sanity.{nonpositive, stale, spikes}` |
| identity breaks *(levels)* | Splits an entity that returns at another level | treats | default | `data.identity.{breaks, renames}` |
| coverage | Drops entities shorter than the longest horizon | treats | default | `data.coverage.{short_lived, thin}` |
| hedge guard *(hedge)* | Drops candidates whose legs are not one asset | treats | default | `data.invariants.{hedged_legs, hedge_check}` |
| influence concentration | Shows the clusters most of the noise rests on | warns | default | `data.concentration.shares` |
| library evals *(data.py)* | Masks, drops, or asks about what one flags | treats | floor | — |
| **3 lab evals** | | | | |
| lab evals listed *(data.py, feat.py)* | The layer file lists at least one lab eval | blocks | floor | — |
| lab evals *(data.py)* | Treats what one flags; on_fail=block vetoes | vetoes | floor | — |
| **4 tests of lab evals** | | | | |
| lab eval tests *(data.py, feat.py)* | Each lab eval passes its CASES and PROPERTIES | blocks | floor | — |
| pits caught *(data.py)* | Each eval flags the pits it covers | blocks | floor | `data.pits.{unit, level_break, gap, duplicate, shift, leak, spike, swap}` |

### feat: `rci feat submit`

Lists: `stages/feat/floor.py`, `stages/feat/defaults.py`, `stages/lab.py`.

| Step | Checks or computes | Role | List | Eval method |
| --- | --- | --- | --- | --- |
| **1 floor** | | | | |
| screen | Refuses a missing, example, or unparsable file | blocks | floor | — |
| scores rank candidates | Scores rank candidates at half the times | blocks | floor | — |
| scores read nothing ahead | Same scores with the data cut early | blocks | floor | — |
| decisive test | The decisive effect beats the floor world: z ≥ z_need | decides | floor | — |
| cost | Wall time and peak memory of the features call | shown | floor | — |
| **2 engine lists** | | | | |
| rank ic | Rank correlation of score and label per time | shown | default | `statistics.ic.rank_ic` |
| rank ic without top {clusters} | Rank IC without the top fifth of clusters | shown | default | `statistics.ic.rank_ic` · `structure.filters.{without_top, kept_without_top}` |
| top minus bottom fifth | Top fifth by score less the bottom fifth | shown | default | `statistics.spread.quantile_spread` |
| library evals *(feat.py)* | Each library eval feat.py lists, vs the floor | shown | floor | — |
| **3 lab evals** | | | | |
| lab evals listed *(data.py, feat.py)* | The layer file lists at least one lab eval | blocks | floor | — |
| alternatives covered *(nulls/feat.py)* | Every alternative of the contract has a null world | blocks | floor | — |
| lab evals *(feat.py)* | Each lab eval reaches z ≥ 2 where it covers | vetoes | floor | — |
| null worlds *(nulls/feat.py)* | Each world's judgement reaches z ≥ 2 | vetoes | floor | — |
| **4 tests of lab evals** | | | | |
| lab eval tests *(data.py, feat.py)* | Each lab eval passes its CASES and PROPERTIES | blocks | floor | — |

## Slot 3: lab evals

A lab eval is a function of the idea's own, in a module such as `evals/<name>.py`, with its tests next to it: CASES (required), PROPERTIES (optional) and, for a data eval, PITS (optional); data.py or feat.py imports it into EVALS (`from evals.<name> import <eval>`) as `Eval(<eval>, h0='<the H0 it rejects>')`. It is a test of the claim: why a case is wrong is specific to the case, so every layer with EVALS needs at least one, its h0 is required (and refused on a library eval, which tests a generic H0), and it can veto: in feat, it and each null world must reach z ≥ 2.00 at every horizon it covers, or the claim fails there; in data, with on_fail='block', any row it flags in the data after the treatments before it blocks the data. It never passes what the decisive test failed, whose bar rises with the submissions, so it needs no pre-registration. Before anything heavy runs, the engine runs its tests, the cases and properties that run it, and refuses a layer with a lab eval that fails them; `rci eval check <file>` runs them any time, for free, and a validation is kept until the module's files change.

| Layer | Listed in | An eval gets | It returns | Eval(fn, ...) adds | It vetoes when |
| --- | --- | --- | --- | --- | --- |
| data | `data.py` | the explore data, a FieldTable | the rows it flags: entity, event_time, optionally field | h0, on_fail (mask, drop_entities, ask, block), covers | with on_fail block, it flags any row of the data after the treatments before it: BLOCKED |
| feat | `feat.py` | the evaluation table of one horizon | t, value per decision time | h0, label, horizons, combine (max, each) | it, or a null world that judges it, is below z 2.00 at a horizon it covers: the claim fails there |

| Declaration | Means |
| --- | --- |
| `Eval(of, name='', h0='', on_fail=None, covers=(), label=None, horizons=None, combine=None, note='')` | an entry of EVALS with what its layer reads: h0, the H0 a lab eval rejects (required for one, refused on a library eval); on_fail, covers (data.py); label, horizons, combine (feat.py) |
| `Case(name, run, expect, tol=1e-09, given=None)` | `run()` must give `expect`; with `given`, `run(given)` must, so a case can name the eval and the frame it reads: `Case("the top two of six", thirds, 0.4, given=SIX)`. |
| `Size(of, market=Market(), markets=100)` | On markets with no information, the statistic must beat the floor world (z ≥ 2) no more often than chance allows, over `markets` seeds. |
| `Power(of, market=Market(), markets=40, rate=0.8)` | On markets with `market.ic` of information, the statistic must beat the floor world at least `rate` of the time. |
| `Recovers(of, market, expect, tol, markets=20)` | Averaged over `markets` seeds and decision times, the statistic must come within `tol` of `expect`. |
| `Market(times=120, entities=100, clusters=20, groups=4, cluster_share=0.3, group_share=0.0, tail=4.0, ic=0.0)` | A synthetic evaluation table: `entities` candidates in `clusters` (and the clusters in `groups`) at `times` hourly decision times. |
| `Source(entities=12, hours=336, settle=8, lag=0, revised=0.0)` | A synthetic field table: `entities` entities with an hourly `price` (a random walk) and a `flow` settled every `settle` hours and known `lag` hours after it; a share `revised` of the flows gets a second vintage one settlement later. |
| `Pit(category, plant, note='')` | A planted problem of `category` in `field`: `plant(fields, field, rng)` returns a copy with it planted in one entity where the field varies. |

For example: `CASES = [Case("the top two of six less the bottom two", thirds, 0.4, given=SIX)], where SIX is a polars DataFrame of the evaluation table (t, entity, score, y) and thirds(SIX) returns t, value`.

A null world, in `nulls/feat.py`, one for every alternative of the contract:

| Field | Means |
| --- | --- |
| `Null.explains` | the id of the contract's alternative this world answers |
| `Null.world` | a PermuteWithin: how the scores are shuffled while the labels stay |
| `Null.keeps` | the column the alternative lives in; None means the world's bucket_by |
| `Null.evals` | the names of the evals this world judges, any eval of the layer; () means the decisive effect |
| `Null.reps` | the most shuffles it draws, a batch at a time until its votes settle; at least 20, 200 by default |
| `Null.note` | free text kept with the world in the submission's record |
| `PermuteWithin.bucket_by` | shuffle the scores across entities inside each bucket of this column; "time" is one bucket per decision time |
| `PermuteWithin.clusters` | a column whose members move together and are shuffled as one |
| `PermuteWithin.block` | repeat the same shuffle for every decision time inside a block this long; "horizon" uses each label horizon |

## Slot 4: tests of lab evals

| Test | Where | Does |
| --- | --- | --- |
| CASES, required | `validation/tests.py`: `run_tests` | each worked example gives what it expects, within its tol |
| PROPERTIES, optional | `validation/tests.py`: `run_tests` | Size, Power and Recovers on synthetic markets, judged against the floor world as a submission is |
| kept by its files | `validation/load.py`: `validate` | the cases and properties that run the eval, found in its module and kept by the hash of its files |
| no tests, or the example | `validation/load.py`: `refuse_untested` | refused before anything runs, saying what to write |
| PITS, data only | `stages/data/floor.py`: `pits_caught` | every pit of what it covers, built in or in its module's PITS, planted in clean data, is flagged |
| the decisive test's world | `validation/calibration.py`: `calibrate` | stand-ins that carry only what the world keeps die, and planted signals live |

| Outcome | What happens |
| --- | --- |
| trusted | an eval of the engine's library: its tests run in the engine's CI, never at research time; it runs |
| valid n/n | a lab eval that passed all n of its tests: it judges, and its line on the card says so |
| invalid | a lab eval that failed a test, or a data eval that missed a pit of what it covers: it cannot judge, so the layer is refused before anything heavy runs (a missed pit, found on the data, blocks the data check); the refusal names the test, what it expected and what it got, and how to fix it |
| no tests | a lab eval no case or property of its module runs: refused before anything runs, saying where CASES go |
| example | a lab eval whose module is still `evals/example.py` as `rci init` wrote it: refused until rewritten |

### Engine internals

What the engine runs on its own, with worked examples like the library's, that no idea lists: the tests of evals (`validation/`) and the data layer's machinery, its treatments, their influence, and the decisive numbers (`stages/data/treatments.py`).

| Module | Function | Does | Backing |
| --- | --- | --- | --- |
| `validation/tests.py` | `differs` | How `got` differs from `expect`; empty when it does not. | — |
| | `label` | How a test is named in its check. | — |
| | `obligations` | One check per case and property the module declares. | — |
| | `run_tests` | One check per case and property, each named after `name`. | — |
| `validation/calibration.py` | `calibrate` | Stand-ins die as often as chance allows; planted signals live. | 2 ex |
| `stages/data/treatments.py` | `apply_all` | The field table after every field treatment. | — |
| | `apply_candidates` | The candidates after every candidate treatment. | — |
| | `drop_candidates` | Drop candidate rows (`t`, `entity`). | — |
| | `drop_entities` | Drop every row of the named entities. | 1 ex |
| | `drop_spans` | Drop the rows each span covers. | 1 ex |
| | `mask_rows` | Drop the flagged rows. | 1 ex |
| | `mask_windows` | Drop the flagged field's rows within `around_ns` of each window. | 1 ex |
| | `mde` | The smallest effect this horizon can detect; it does not depend on the effect asked for. | — |
| | `renames_rows` | A treatment that changes identities instead of dropping rows, such as a split. | — |
| | `se` | No noise can be measured on a number that is not finite somewhere. | — |
| | `smaller` | The finer of two resolutions. | — |
| | `spans` | When each removed row was current in `fields`. | 6 ex |
| | `split_entities` | Rename each split entity's rows from the split on. | 1 ex |
| | `touched` | Label rows whose window meets a span of a leg. | 6 ex |

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

**Role**: `blocks`: stops the stage when it fails: BLOCKED, refused, TAINTED, or FAILED · `decides`: PASS or FAIL; only the decisive test · `vetoes`: rejects the claim where it does not hold; never passes what the decisive test failed · `treats`: changes the data by default, weighed by its influence · `warns`: a warning on the card, or an item a person decides · `shown`: on the card, decides nothing. **List**: floor steps run on every idea, are part of the engine hash, and no idea can change them; default steps are adjusted from experience, and the engine hash leaves them out. *(levels)*, *(hedge)*, *(reference)*: only when a plugin declares it; *(data.py)* and the other files: only when the idea adds the file. **Used by**: contract, data, feat run it; research code may call it (`rci methods`); inside eval: other eval functions call it; worked examples: other modules' examples build on it; algo, planned: waits for the algo layer; tests only: nothing else calls it. **Backing**: ex worked examples, prop properties on synthetic markets, — none declared (the protocol tests are held to broken toy plugins in `tests/eval/test_protocol.py`).
