# TODO

The engine is being reworked from first principles: fewer concepts, wide tables everywhere. Earlier contracts, ledgers and run hashes are not kept compatible. Status: **done**, **decided** (direction set, not started), **open** (to discuss), **known** (a known defect).

## Data shape and interfaces

- **1.** **done**: wide tables everywhere. Each data source is a wide table at its native cadence (`inst | event_time | available_at | vintage | <columns>`), checked for fidelity column by column; research and every slot read `wide`, the as-of join of the sources onto the decision grid (`t | inst | <columns> | age_<source>`), or the sources where they need native rows. The long field table is gone; the ledger format is 3.
- **2.** **open**: feat takes several feature columns at once; how a set of features is judged is not settled.
- **3.** **open**: factor evaluation (screening, a pool, dedupe, increments) is built from evals, researched with agents; the engine hard-codes no rule.
- **4.** **done**: the contract pins an outcome and a horizon grid, not labels (`outcome=Use(...)`, `horizons=(...)`, by default the decision period times 1, 4 and 16); the outcome gives `y`, its parts `y_<part>` and `y_end` per candidate and horizon. Training labels are explored freely in sig (item 7).
- **5.** **decided**: `entity` becomes `inst`. A multi-leg candidate is a synthetic `inst` with columns for its legs (to confirm). An `inst` is never reused; attributes that change over time are columns with their own `available_at`.
- **6.** **decided**: costs once: feat and sig judge the gross outcome; only algo charges costs (`ref.mean` charges them today).
- **18.** **open**: value columns are numbers; an attribute that is text (a listing status, a contract type) has to be coded as a number by its data plugin.
- **19.** **done**: every stage's card freezes its products as parquet in the vault, by content, so the next layer reads them instead of computing them again. The data freeze pins the treated sources, the decision grid, `wide`, the candidates, their outcomes (one table, the horizon in `h`) and the excluded rows, and the DataCard lists each with its hash and folder; feat builds its evaluation tables from them; accepting feat pins the accepted submission's scores (`t | inst | score`).
- **22.** **done**: the data loop may choose how decisions are sampled: `data.py` may define `grid(sources)` (or `grid(sources, wide)`), which returns `t | inst`, such as event-time bars on price moves; without one the grid is the contract's clock. `wide`, the candidates and the outcomes follow the grid, outcome horizons stay in clock time, the grid is a pinned product, a grid that changes when the sources are cut blocks the check, and the DataCard shows its shape and warns when it is uneven (item 21).

## Layers not built

- **7.** **open**: sig: models or rules, training labels, walk-forward, the baseline, memorization guards. Every outcome row records `y_end`, the last time it reads; nothing purges with it yet.
- **8.** **open**: feat screens on the whole explore span, which leaks into sig; split the span in time (decide with sig).
- **9.** algo: filter the signal or process the matrix into positions, then the backtest engine.
- **10.** seal: the real out-of-sample check.
- **20.** **decided**: sig and algo store their products as parquet the same way: sig the accepted predictions, algo the positions.

## Known statistical defects

- **11.** **known**: `ShiftInTime` shifts by rows, not whole days or weeks, so its null is too narrow.
- **12.** **known**: feat keeps at most 1000 decision times (`max_times`), which drops most of an event panel's independent information.
- **13.** **known**: events cannot set a decision time at `available_at` plus a latency; no cooldown; no count of independent events; the contract's MDE for events is optimistic.
- **14.** **known**: hidden beta: no beta-neutral outcome, no null bucketed by beta.
- **17.** **known**: a treatment's influence is weighed by leaving out the outcome rows whose window meets a span in which a dropped or masked source row was current: a mask of one column also leaves out outcome rows that read only the row's other columns, and a treatment that changes the candidates through `wide` is weighed on the candidates as they came (the power after computes everything again).
- **21.** **known**: the shuffles, blocks and MDE assume an even decision grid: blocks are cut by time from the first decision and `ShiftInTime` shifts by rows, so on an uneven grid from a sampler (item 22) the nulls and the power are approximate. Make blocks and shuffles work on uneven grids.

## Product and engineering

- **15.** Sandbox: memory (a large idea's data check peaks near 15 GB), to be met by working in time batches (cross-sectional checks and judging accumulate block by block, per-instrument checks run instrument by instrument, rolling features warm up, `ShiftInTime` keeps one score column whole), categorical strings, float32 prices and time-filtered scans; platform data plugins; a shared data cache. `wide` holds a row per decision time and inst known by then, and the universe and feat lookahead checks build it again on the sources cut at each cut point.
- **16.** `examples/` is not run in CI; card samples in the docs, and the example's transcript, come from polars 1 draws and from before the sources, `wide` and the outcome replaced the field table and labels.

## Ruled out

- A global count of trials across ideas.
- An engine-wide instrument schema: each data plugin declares the attributes its case has.
