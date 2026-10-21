# Stage Kit

The Stage Kit is an optional construction library for external Labs. It does
not define stage policy, run eval slots, access networks, or write artifacts.
An external builder may use these helpers, combine them with case code, or
ignore them entirely as long as it returns the stage protocol's expected
object.

The construction modules deliberately have different shapes:

- `data.py` contains table-in/table-out assembly helpers: timestamp
  normalization, deduplication, point-in-time as-of joins, namespaced full
  joins, entity stacking, and complete-cross-section filtering. Source
  downloads, caches, source-specific schemas, and sampling cadence stay in the
  Lab.
- `feat.py` contains name-free Polars expressions. The Lab supplies input
  columns, output aliases, and sequential expression layers. It includes
  first- and second-order differences, simple and log returns, signed-log and
  imbalance transforms, guarded finite ratios, true range/ATR, row-clock
  sums/minimums/maximums/correlation/RSI, and duration-clock
  sums/minimums/maximums/quantiles/correlation/RSI. Rolling z-scores, range positions,
  moving-average and volatility ratios, median/IQR robust scaling, and an
  explicitly scaled tanh transform are composed from those primitives.
- `sig.py` contains explicit target expressions, simple signal composition,
  chronological and walk-forward splits, group-tail purging, fitted score
  calibration, and `SignalBundle`. Target helpers distinguish simple return,
  log return, and forward difference so an already-log-transformed spread is
  not logged twice.
- `model.py` contains a small sklearn adapter owned by Sig. `ModelSpec` records
  the task, model family, feature and target columns, scaling, imputation,
  random seed, estimator parameters, score method, and selected class label.
  It supports ridge and logistic linear models plus histogram gradient
  boosting for regression and classification. Candidate grids, hyperparameter
  choices, refit schedules, ensembles, and promotion decisions remain in the
  Lab.
- `algo.py` contains bounded-score and dollar-neutral pair expressions plus
  `AlgoBundle` and `BacktestResult`. The Lab still owns sizing, orders, costs,
  fills, and the backtest implementation.
- `viz.py` contains optional, stage-agnostic helpers for aligned panels from a
  wide Polars table and heatmaps from row/column/value evidence. A case-owned
  renderer selects the artifacts and columns.

Raw helpers do not fill nulls, replace NaNs, add epsilon, infer time units,
sort or partition tables, or choose output column names. Explicitly guarded
helpers advertise their treatment in their names and signatures. Those choices
change research meaning and must remain visible in the Lab recipe.

## Fit and evaluation boundary

The walk-forward helpers divide every declared fold into three half-open
tables: train, calibration, and evaluation. The intended order is:

1. Purge the last label-horizon rows of each ordered entity from any fit or
   selection partition when a forward target could cross its boundary.
2. Fit `ModelSpec` preprocessing and estimator state on train only.
3. Score calibration and fit z-score statistics or absolute-quantile trading
   thresholds on calibration only.
4. Apply the already-fitted model and calibration objects to evaluation.

The split helper does not infer dates, rolling widths, or expanding-window
policy. Every fold boundary is supplied by the Lab. Calibration application
does not recompute statistics from evaluation rows, and an entity absent from
the calibration set receives a null fitted value rather than an implicit
global fallback. Z-score fitting rejects non-finite or non-positive scales.

The model adapter similarly has no automatic model search or winner selection.
`make_model` only translates one complete `ModelSpec` into a pipeline;
`fit_model` sees only the table passed as train; and `score_model` reuses that
pipeline unchanged. Probability and decision scores require an explicit class
label so class ordering cannot silently reverse a signal.

The Kit still leaves higher-level factor definitions in the Lab. For example,
RSI leaves a flat window undefined unless `flat_value` is supplied, and
duration-clock correlation is pairwise non-null but deliberately does not
clean NaNs or add a variance floor. Directional divergences, field-specific
interactions, clipping thresholds, grouping, and warm-up policy remain Lab
decisions.

`BuildInfo` and `BuildStep` are optional recipe metadata. When a builder
exports `BUILD_INFO`, the CLI records it as `recipe.json`. The Kit catalog is
for discovery only; unlike the eval menu, it is not an allowlist.

Plotting dependencies are lazy and live in the `visualization` extra. Stage
runs do not import them. `rci visualize --stage ...` calls an external Lab
renderer explicitly and confines its files to that stage's `plots/` directory.
Visualization does not participate in the four eval slots or stage approval.
