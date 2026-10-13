# Research CI catalog

Generated from the code by `rci methods --markdown`; a test fails when it is out of date: `uv run rci methods --markdown > docs/catalog.md`.

Every layer is the same four slots. **1 floor**: the premises and the decisive test, which alone can pass a claim (z ≥ z_need, rising with the submissions). **2 engine lists**: the default list and the library evals an idea lists; they test generic H0s and are shown, or treat the data. **3 lab evals**: the evals the idea writes for its own case, and its null worlds; they can veto, never pass. **4 tests of lab evals**: what holds a lab eval to its own tests before it judges anything.

## Shapes

A universe says which question its scores answer, `Universe(use, clusters, shape=...)`, and the shape sets the floor world of the decisive test, the target that fits it, and the default diagnostics feat shows. For events, the features score only the event rows, every candidate row stays in feat's tables, and statistics and targets judge the rows with a finite score, which move in time with the shift.

| Shape | The question | Floor world | Target | Default diagnostics |
| --- | --- | --- | --- | --- |
| `cross` (the default) | which candidates do better than the others at the same decision time | `PermuteWithin(time, clusters=<cluster column>, block=horizon)` | `Use("ref.mean", {"cluster": "<cluster column>", "top": 0.2})` | rank ic, rank ic without top {clusters}, top minus bottom fifth |
| `series` | when one or a few assets do better: timing, long or short | `ShiftInTime(min_shift=horizon)` | `Use("ref.timing")` | ts ic |
| `events` | what follows an event; the scores exist only at the event rows | `ShiftInTime(min_shift=horizon)` | `Use("ref.mean", {"cluster": "inst"})` | mean at events |

| Target | Role |
| --- | --- |
| `ref.mean` | Which candidates do better (cross), or what follows an event (events): the mean outcome `y` per decision time over the candidates with a score, each cluster weighted equally; with `net`, after each candidate's round trip. |
| `ref.timing` | When to be long or short (series): the mean over the candidates at each decision time of sign(score) × `y`, long where the score is positive, short where it is negative, flat at zero. |

## Stages

### contract: `rci contract derive`

Lists: `stages/contract/floor.py`, `stages/contract/defaults.py`.

| Step | Checks or computes | Role | List | Eval method |
| --- | --- | --- | --- | --- |
| **1 floor** | | | | |
| universe point-in-time | Same candidates when the data is cut | blocks | floor | `protocol.universe_pit` |
| universe causal | Later values leave earlier candidates unchanged | blocks | floor | `protocol.universe_causal` |
| universe membership | A static candidate list makes history explore-grade | warns | floor | — |
| outcome: worked examples | Worked examples hold, and move when inputs move | blocks | floor | `protocol.outcome_cases` |
| outcome: columns | Outcome options name columns; fails when no rows result | blocks | floor | — |
| outcome: no data after t + h | Outcomes unchanged when the data stops early | blocks | floor | `protocol.outcome_truncation` |
| power | MDE per horizon; blocks when no noise is measurable | blocks | floor | `data.concentration.shares` |
| target reads the score | The decisive target moves with the scores | blocks | floor | `protocol.target_reads_score` |
| **2 engine lists** | | | | |
| data is current | Warns when the newest row is too old | warns | default | — |

### data: `rci data check`

Lists: `stages/data/floor.py`, `stages/data/defaults.py`, `stages/data/source.py`, `stages/lab.py`.

| Step | Checks or computes | Role | List | Eval method |
| --- | --- | --- | --- | --- |
| **1 floor: admission, once per source** | | | | |
| baseline | Gaps, near-duplicates, and level shifts in every source | warns | floor | `data.baseline.baseline` |
| replay as of past times | Fetches as of past times match the full fetch | blocks | floor | `data.replay.{cut_points, replay}` |
| canary: replay | The replay catches a deletion made with hindsight | blocks | floor | `data.baseline.{events, rows}` · `data.replay.replay` |
| independent source agrees *(reference)* | Another source agrees; grades every column | blocks | floor | `data.reconcile.{summarize, requests, compare}` · `data.baseline.baseline` |
| canary: reconciliation *(reference)* | The comparison catches altered values | blocks | floor | `data.reconcile.{compare, requests}` · `data.baseline.baseline` |
| **1 floor** | | | | |
| row-by-row code *(data.py)* | Row-by-row code in the idea's data files | warns | floor | — |
| column grades | The decisive number reads only verified columns | warns | floor | — |
| treatment influence | Each treatment's move of the decisive number, in MDE | warns | floor | — |
| power | MDE per horizon, before and after the treatments | shown | floor | — |
| decisive test calibration | Stand-ins die and planted signals live | blocks | floor | `protocol.target_reads_score` |
| **2 engine lists** | | | | |
| data is current | Warns when the newest row is too old | warns | default | — |
| level shifts *(levels)* | Masks level jumps the source did not confirm | treats | default | `data.baseline.{level_shift_anomalies, rows}` |
| level sanity *(levels)* | Masks levels at zero, stuck, or spiking | treats | default | `data.sanity.{nonpositive, stale, spikes}` · `data.baseline.{cells, rows}` |
| identity breaks *(levels)* | Splits an inst that returns at another level | treats | default | `data.identity.{breaks, renames}` · `data.baseline.rows` |
| coverage | Drops insts shorter than the longest horizon | treats | default | `data.coverage.{short_lived, thin}` · `data.baseline.rows` |
| hedge guard *(hedge)* | Drops candidates whose legs are not one asset | treats | default | `data.invariants.{hedged_legs, hedge_check}` |
| influence concentration | Shows the clusters most of the noise rests on | warns | default | `data.concentration.shares` |
| library evals *(data.py)* | Masks, drops, or asks about what one flags | treats | floor | `data.baseline.rows` |
| **3 lab evals** | | | | |
| lab evals listed *(data.py, feat.py)* | The layer file lists at least one lab eval | blocks | floor | — |
| lab evals *(data.py)* | Treats what one flags; on_fail=block vetoes | vetoes | floor | `data.baseline.rows` |
| **4 tests of lab evals** | | | | |
| lab eval tests *(data.py, feat.py)* | Each lab eval passes its CASES and PROPERTIES | blocks | floor | — |
| pits caught *(data.py)* | Each eval flags the pits it covers | blocks | floor | `data.pits.{unit, level_break, gap, duplicate, shift, leak, spike, swap}` |

### feat: `rci feat submit`

Lists: `stages/feat/floor.py`, `stages/feat/defaults.py`, `stages/lab.py`.

| Step | Checks or computes | Role | List | Eval method |
| --- | --- | --- | --- | --- |
| **1 floor** | | | | |
| screen | Refuses a missing, example, or unparsable file | blocks | floor | — |
| scores rank candidates | Scores rank candidates, or times, or mark events | blocks | floor | — |
| scores read nothing ahead | Same scores with the data cut early | blocks | floor | — |
| decisive test | The decisive effect beats the floor world: z ≥ z_need | decides | floor | — |
| cost | Wall time and peak memory of the features call | shown | floor | — |
| **2 engine lists** | | | | |
| rank ic *(cross)* | Rank correlation of score and outcome per time | shown | default | `statistics.ic.rank_ic` |
| rank ic without top {clusters} *(cross)* | Rank IC without the top fifth of clusters | shown | default | `statistics.ic.rank_ic` · `structure.filters.{without_top, kept_without_top}` |
| top minus bottom fifth *(cross)* | Top fifth by score less the bottom fifth | shown | default | `statistics.spread.quantile_spread` |
| ts ic *(series)* | Rank correlation of score and outcome over time | shown | default | `statistics.ic.ts_ic` |
| mean at events *(events)* | Mean outcome at the event rows | shown | default | — |
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
| data | `data.py` | the explore data, a FieldTable | the rows it flags: inst, event_time, optionally field | h0, on_fail (mask, drop_insts, ask, block), covers | with on_fail block, it flags any row of the data after the treatments before it: BLOCKED |
| feat | `feat.py` | the evaluation table of one horizon | t, value per decision time | h0, label, horizons, combine (max, each) | it, or a null world that judges it, is below z 2.00 at a horizon it covers: the claim fails there |

| Declaration | Means |
| --- | --- |
| `Eval(of, name='', h0='', on_fail=None, covers=(), label=None, horizons=None, combine=None, note='')` | an entry of EVALS with what its layer reads: h0, the H0 a lab eval rejects (required for one, refused on a library eval); on_fail, covers (data.py); label (the column read as y), horizons, combine (feat.py) |
| `Case(name, run, expect, tol=1e-09, given=None)` | `run()` must give `expect`; with `given`, `run(given)` must, so a case can name the eval and the frame it reads: `Case("the top two of six", thirds, 0.4, given=SIX)`. |
| `Size(of, market=Market(), markets=100)` | On markets with no information, the statistic must beat the floor world (z ≥ 2) no more often than chance allows, over `markets` seeds. |
| `Power(of, market=Market(), markets=40, rate=0.8)` | On markets with `market.ic` of information, the statistic must beat the floor world at least `rate` of the time. |
| `Recovers(of, market, expect, tol, markets=20)` | Averaged over `markets` seeds and decision times, the statistic must come within `tol` of `expect`. |
| `Market(times=120, insts=100, clusters=20, groups=4, cluster_share=0.3, group_share=0.0, tail=4.0, ic=0.0, shape='cross')` | A synthetic evaluation table: `insts` candidates in `clusters` (and the clusters in `groups`) at `times` hourly decision times. |
| `Source(insts=12, hours=336, settle=8, lag=0, revised=0.0)` | Synthetic sources: `insts` insts with an hourly `price` (a random walk, source `prices`) and a `flow` settled every `settle` hours and known `lag` hours after it (source `flows`); a share `revised` of the flows gets a second vintage one settlement later. |
| `Pit(category, plant, note='')` | A planted problem of `category` in a value column: `plant(sources, column, rng)` returns copies of the sources with it planted in the source that holds the column, in one inst where the column varies. |

For example: `CASES = [Case("the top two of six less the bottom two", thirds, 0.4, given=SIX)], where SIX is a polars DataFrame of the evaluation table (t, inst, score, y) and thirds(SIX) returns t, value`.

A null world, in `nulls/feat.py`, one for every alternative of the contract:

| Field | Means |
| --- | --- |
| `Null.explains` | the id of the contract's alternative this world answers |
| `Null.world` | a PermuteWithin or a ShiftInTime: how the scores are shuffled while the outcomes stay |
| `Null.keeps` | the column the alternative lives in; None means the world's bucket_by, or inst for a ShiftInTime |
| `Null.evals` | the names of the evals this world judges, any eval of the layer; () means the decisive effect |
| `Null.reps` | the most shuffles it draws, a batch at a time until its votes settle; at least 20, 200 by default |
| `Null.note` | free text kept with the world in the submission's record |
| `PermuteWithin.bucket_by` | shuffle the scores across insts inside each bucket of this column; "time" is one bucket per decision time |
| `PermuteWithin.clusters` | a column whose members move together and are shuffled as one |
| `PermuteWithin.block` | repeat the same shuffle for every decision time inside a block this long; "horizon" uses each horizon |
| `ShiftInTime.min_shift` | shift each inst's scores along its decision times, by one offset for all insts, at least this far either way round; "horizon" uses each horizon |

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
| `validation/calibration.py` | `calibrate` | Stand-ins die as often as chance allows; planted signals live. | 4 ex |
| `stages/data/treatments.py` | `apply_all` | The sources after every treatment of the sources: the masks and drops first, each found on the sources as they came, then the splits in order. | — |
| | `apply_candidates` | The candidates after every candidate treatment. | — |
| | `drop_candidates` | Drop the candidate rows at `t` whose `inst`, or one of whose legs (`long`, `short`), is a flagged `inst` at `t`: a flagged candidate, or a flagged row of `wide` a candidate reads. | — |
| | `drop_insts` | Drop every row of the named insts, in every source. | 1 ex |
| | `drop_spans` | Drop the rows each span covers, in every source: every row of the span's `inst` whose event falls in `[start, end]`. | 1 ex |
| | `mask_cells` | Set the flagged cells to null. | 3 ex |
| | `mask_windows` | Mask the window's column within `around_ns` of each window. | 1 ex |
| | `mde` | The smallest effect this horizon can detect; it does not depend on the effect asked for. | — |
| | `renames_rows` | A treatment that changes identities instead of dropping or masking rows, such as a split. | — |
| | `se` | No noise can be measured on a number that is not finite somewhere. | — |
| | `smaller` | The finer of two resolutions. | — |
| | `spans` | When each removed row was current in `sources`. | 6 ex |
| | `split_insts` | Rename each split inst's rows from the split on, in every source. | 1 ex |
| | `touched` | Outcome rows whose window meets a span of a leg. | 6 ex |

## Eval library

### protocol

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `protocol` | `adds_up` | Components add up to net; the consumer's own checks pass. | algo, planned | — |
| | `branch_diff` | The branch's extra net over `(start, end]`, and its overlap. | algo, planned | — |
| | `busiest` | A decision time whose next period brings the most sources. | algo, planned | — |
| | `causal` | Changes known after `at` must move only what comes after. | algo, planned | — |
| | `changed` | The sources with the values of some rows changed. | inside eval | — |
| | `consumer_pit` | Cut data gives the full run's periods before the cut. | algo, planned | — |
| | `deterministic` | Two identical consumer runs must give the same net. | algo, planned | — |
| | `fees_match` | The consumer charges what the cost model says. | algo, planned | — |
| | `matches_outcome` | The consumer books what the outcome says, period by period. | algo, planned | — |
| | `max_drawdown` | The consumer's deepest fall in equity from its running peak. | algo, planned | — |
| | `outcome_cases` | Worked examples hold as written and move with their inputs. | contract | — |
| | `outcome_truncation` | Data stopping at `cut` leaves outcomes ending by it unchanged. | contract | — |
| | `passthrough` | The consumer's own baseline scores must change nothing. | algo, planned | — |
| | `perturb` | An uninformative change to scores: a share dropped or permuted. | algo, planned | — |
| | `target_reads_score` | A target must move with the scores. | contract, data | — |
| | `total` | The consumer's net summed over the run. | algo, planned | — |
| | `universe_causal` | Candidates at `t` ignore every value known after `t`. | contract | — |
| | `universe_pit` | Candidates before the cut equal those from all data. | contract | — |

### data

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `baseline` | `baseline` | Generic checks and their anomalies: gaps, near-duplicates, level shifts. | data | 8 ex |
| | `cells` | The cells that hold a value, one per row and column. | data | 8 ex |
| | `events` | First vintages with each event's spacing, its group's usual spacing and size. | data | 8 ex |
| | `first_vintages` | One row per event of each group `by`: its first vintage. | inside eval | 8 ex |
| | `level_shift_anomalies` | The baseline's level shifts alone, from the declared levels only. | data | — |
| | `rows` | Every row of the sources by its keys. | data | 8 ex |
| `concentration` | `shares` | Each cluster's share of the noise, between 0 and 1. | contract, data | 7 ex |
| `coverage` | `short_lived` | Insts whose events span less than `horizon_ns`. | data, research code | 2 ex |
| | `thin` | Decision times with some candidates but fewer than `floor`. | data, research code | 1 ex |
| `identity` | `breaks` | Declared levels that come back elsewhere after a long gap. | data, research code | 6 ex |
| | `renames` | What the split calls each new series: `inst`, `at`, `to`. | data, research code | 1 ex |
| `invariants` | `hedge_check` | The card's hedged-legs line, naming the pairs most often apart. | data | — |
| | `hedged_legs` | Per candidate row: the legs' price ratio, correlation, and `ok`. | data | 1 ex |
| `pits` | `duplicate` | A duplicate: one row of an inst repeated a millisecond later. | data, research code | 1 ex |
| | `gap` | A gap: ten usual spacings of one inst's rows missing. | data, research code | 1 ex |
| | `holder` | The source that holds `column`. | research code | 1 ex |
| | `leak` | A leak: each event of one inst carries the next one's value. | data, research code | 1 ex |
| | `level_break` | A level break: one inst's column half again as large. | data, research code | 1 ex |
| | `shift` | A shift: one inst's rows stamped one usual spacing late. | data, research code | 1 ex |
| | `spike` | A spike: one event of one inst ten times larger. | data, research code | 1 ex |
| | `swap` | A swap: one inst's column replaced by another inst's. | data, research code | 1 ex |
| | `target` | An inst whose `column` has enough values for a pit. | research code | 1 ex |
| | `unit` | A unit error: one inst's column a thousand times larger. | data, research code | 1 ex |
| `reconcile` | `compare` | One result per request: `match`, `mismatch` or `unanswered`. | data | 5 ex |
| | `requests` | Point requests for random cells, and window requests for anomalies. | data | 5 ex |
| | `summarize` | The check, each column's grade, and counts. | data | — |
| `replay` | `cut_points` | Cut points: holes, before new insts, and random times. | data | 4 ex |
| | `replay` | Each cut's fetch must equal the full fetch cut there. | data | 4 ex |
| `sanity` | `nonpositive` | Events of a declared level at zero or below. | data, research code | 4 ex |
| | `spikes` | Events of a declared level that jump and come back. | data, research code | 5 ex |
| | `stale` | Events repeating the value before them in a long run. | data, research code | 3 ex |

### statistics

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `ic` | `rank_ic` | Spearman correlation of score and outcome at each decision time. | feat, research code | 3 ex · 3 prop |
| | `ts_ic` | Rank correlation of score and outcome over time, per inst. | feat, research code | 3 ex · 3 prop |
| `spread` | `quantile_spread` | Mean outcome of the top `share` less the bottom `share`. | feat, research code | 3 ex · 2 prop |

### structure

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `filters` | `kept_without_top` | Which rows `without_top` keeps. | feat, research code | 3 ex |
| | `without_top` | The table without its `share` of most extreme `column` values. | feat, research code | 2 ex |

### synthetic

| Module | Function | Does | Used by | Backing |
| --- | --- | --- | --- | --- |
| `synthetic` | `hand` | A small evaluation table worked by hand, at one time. | worked examples | — |
| | `market` | A synthetic evaluation table, one row per inst and time. | worked examples | — |
| | `sources` | Synthetic sources of hourly prices and settled flows. | worked examples | — |

## Legend

**Role**: `blocks`: stops the stage when it fails: BLOCKED, refused, TAINTED, or FAILED · `decides`: PASS or FAIL; only the decisive test · `vetoes`: rejects the claim where it does not hold; never passes what the decisive test failed · `treats`: changes the data by default, weighed by its influence · `warns`: a warning on the card, or an item a person decides · `shown`: on the card, decides nothing. **List**: floor steps run on every idea, are part of the engine hash, and no idea can change them; default steps are adjusted from experience, and the engine hash leaves them out. *(levels)*, *(hedge)*, *(reference)*: only when a plugin declares it; *(data.py)* and the other files: only when the idea adds the file; *(cross)*, *(series)*, *(events)*: only for a universe of that shape. **Used by**: contract, data, feat run it; research code may call it (`rci methods`); inside eval: other eval functions call it; worked examples: other modules' examples build on it; algo, planned: waits for the algo layer; tests only: nothing else calls it. **Backing**: ex worked examples, prop properties on synthetic markets, — none declared (the protocol tests are held to broken toy plugins in `tests/eval/test_protocol.py`).
