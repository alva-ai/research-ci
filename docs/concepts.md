# Concepts

The vocabulary of Research CI, how a verdict is computed, and what a verdict card looks like. Nothing here is implemented yet; this is the target behavior.

## Vocabulary

| Term | Meaning |
| --- | --- |
| **Contract** | The trade hypothesis written before any search: decision time, action, holding period, costs, mechanism, alternative explanations. |
| **Spec** | What the engine derives from the contract: decision times, label definition, event clusters, effective sample size, minimum detectable effect, cost threshold, and how long to wait if the sample is too small. |
| **Reverse derivation** | Thresholds are derived backwards from the final trade; work then runs forwards through the layers. |
| **Angle** | One pre-registered way of testing a claim. `stat` angles are compared with a null distribution; `limit` angles are hard constraints compared directly. |
| **Null world** | A world with the claimed information removed but the alternative explanation kept. The real score must beat the score in these worlds. |
| **Stress world** | A world where an optimistic assumption is broken: thinner depth, failed hedges, a replayed incident. Used by the `algo` layer. |
| **Calibration** | Tests of the tests. Stand-ins must fail, planted signals must pass, mutants must be caught. |
| **Stand-in** | A fake feature or policy that contains only the alternative explanation. If it passes, the null world is too weak. |
| **Planted** | A synthetic signal with known real information. If it fails, the null world is too strict. |
| **Mutant** | A copy of the data with a defect planted in it: a shifted `available_at`, a level break, a merged holiday week. Data tests must catch it. |
| **Challenger** | A separate LLM process with no access to the research agent's conversation. It writes null worlds and tests, never verdicts. |
| **Tier** | Explore (numbers visible), sealed (pass/fail only, one use per idea), forward (data created after freeze). |
| **Freeze** | A hash over the contract, evaluators, null worlds, kernel SHA, plugin hashes, lock file, interpreter, and data snapshot. |
| **Ledger** | Append-only, hash-chained record of every submission. Trial counts come from here, not from the agent. |
| **Lineage** | Each verdict records the hashes of the upstream outputs it used. If upstream reopens, downstream becomes STALE. |

## How a verdict is computed

```text
research code                     labels (built by the engine)
      |                                     |
stepper: asof t only                        |
      |                                     |
output at each t                            |
      +-----------------+-------------------+
                        |
          evaluator (returns one number)
             /                      \
       real world               200 null worlds
           |                           |
       real score               null distribution
             \                       /
          z = (real - null mean) / null sd        ledger: trial count n
                        |                                  |
                        v                                  v
             z > z_need for every angle?  <--  z_need = z_base * sqrt(1 + ln n)
                        |
                        v
             verdict + evidence level
```

- The engine never reads what an evaluator means. It only compares the real score with the same evaluator run in null worlds, so any metric works, and any search hidden inside an evaluator is repeated in the null worlds too.
- An evidence level requires **all** of its angles to pass (an intersection-union test), so angles need no extra multiple-testing correction. What does need correction is the number of versions tried, and the trial count handles that.
- A parameter grid is declared as a family. Each null world runs the whole grid and keeps its maximum (White's Reality Check), so the threshold grows with the grid size automatically.

### Null worlds map to alternative explanations

| "It is only..." | Null world |
| --- | --- |
| chance | `BlockPermute`: permute labels by time blocks or event clusters |
| seasonality | `ShiftTime`: shift the feature by whole years |
| the obvious variable X | `PermuteWithin(bucket_by=X)`: shuffle within buckets of X, moving whole clusters |
| clock skew | `ShiftTime` on one venue within the measured skew |
| any news at all | `RandomEvents`: unselected events from the same source |
| trading less often | `RandomizeTiming(keep="rate")`: same decision rate, random timing |
| the claimed mechanism is not needed | `ReplaceField`: scramble the mechanism inputs |

## Evidence levels

| Level | Requires |
| --- | --- |
| L0 | Results on the explore tier only |
| L1 | Predictive power, increment, and stability under walk-forward |
| L2 | Plus mechanism ablation and a secondary prediction |
| L3 | Plus economic value after the consumer and costs |
| L4 | Plus at least one new-data angle: an independent source, the sealed tier, forward data, or live trading |

Caps: DEV runs stop at L1; explore-grade data stops at L1; `Quotes` that pass only under the optimistic fill model stop at L2; strategies whose labels depend on their own actions stop at L3 until live trading.

Verdict states: `PASS`, `FAIL`, `INSUFFICIENT`, `TAINTED`, `STALE`, `DEV`.

## Example verdict card

An illustrative card for a maker-taker toxicity feature. All numbers are made up.

```text
Feat · toxicity score (B lead + book imbalance + staleness)     trial 3 / 20   z_need 2.9
decision time   trade time − 35 ms (p95 cancel latency)          stepper: 0 mismatches
sample          41,200 taker trades → 3,100 independent clusters
calibration     stand-ins 5/5 killed, planted 5/5 kept
────────────────────────────────────────────────────────────────────────
angle                    null world                      z      result
predictive               block permute by cluster        4.1    pass
increment over B move    permute within B-move buckets   3.2    pass
economic (drop top 20%)  permute within B-move buckets   3.0    pass   −0.6 → +0.4 bp
horizon shape            permute within hour             2.9    pass
mechanism ablation       scramble B inputs               —      pass   IC → 0.01
latency robustness       p50 / p95 / p99                 —      p99 fails (signal gone at 80 ms)
────────────────────────────────────────────────────────────────────────
PASS · L3 · caps: none · requirement for algo: cancel latency ≤ 40 ms
```

The card is rendered from `verdict.json` without any LLM. A narrated summary may be attached, but every sentence must cite a field in `verdict.json`, and those citations are checked by code.
