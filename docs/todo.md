# TODO

The engine is being reworked from first principles: fewer concepts, wide tables everywhere. Earlier contracts, ledgers and run hashes are not kept compatible. Status: **decided** (direction set, not started), **open** (to discuss), **known** (a known defect).

## Data shape and interfaces

1. **decided**: wide tables everywhere. Each data source is a wide table at its native cadence (`inst | event_time | available_at | vintage | <columns>`), checked for fidelity column by column; research and every slot read `wide`, the as-of join of the sources onto the decision grid (`t | inst | <columns> | age_<source>`). The long field table goes.
2. **open**: feat takes several feature columns at once; how a set of features is judged is not settled.
3. **open**: factor evaluation (screening, a pool, dedupe, increments) is built from evals, researched with agents; the engine hard-codes no rule.
4. **decided**: the contract pins an outcome and a horizon grid, not labels; training labels are explored freely in sig.
5. **decided**: `entity` becomes `inst`. A multi-leg candidate is a synthetic `inst` with columns for its legs (to confirm). An `inst` is never reused; attributes that change over time are columns with their own `available_at`.
6. **decided**: costs once: feat and sig judge the gross outcome; only algo charges costs (`ref.mean` charges them today).

## Layers not built

7. **open**: sig: models or rules, training labels, walk-forward, the baseline, memorization guards.
8. **open**: feat screens on the whole explore span, which leaks into sig; split the span in time (decide with sig).
9. algo: filter the signal or process the matrix into positions, then the backtest engine.
10. seal: the real out-of-sample check.

## Known statistical defects

11. **known**: `ShiftInTime` shifts by rows, not whole days or weeks, so its null is too narrow.
12. **known**: feat keeps at most 1000 decision times (`max_times`), which drops most of an event panel's independent information.
13. **known**: events cannot set a decision time at `available_at` plus a latency; no cooldown; no count of independent events; the contract's MDE for events is optimistic.
14. **known**: hidden beta: no beta-neutral outcome, no null bucketed by beta.

## Product and engineering

15. Sandbox: memory (a large idea's data check peaks near 15 GB), to be met by working in time batches (cross-sectional checks and judging accumulate block by block, per-instrument checks run instrument by instrument, rolling features warm up, `ShiftInTime` keeps one score column whole), categorical strings, float32 prices and time-filtered scans; platform data plugins; a shared data cache.
16. `examples/` is not run in CI; card samples in the docs come from polars 1 draws.

## Ruled out

- A global count of trials across ideas.
- An engine-wide instrument schema: each data plugin declares the attributes its case has.
