# Stage Kit

The Stage Kit is an optional construction library for external Labs. It does
not define stage policy, run eval slots, access networks, or write artifacts.
An external builder may use these helpers, combine them with case code, or
ignore them entirely as long as it returns the stage protocol's expected
object.

The three modules deliberately have different shapes:

- `data.py` contains table-in/table-out assembly helpers. Source downloads and
  caches stay in the Lab.
- `feat.py` contains name-free Polars expressions. The Lab supplies input
  columns, output aliases, and sequential expression layers. The first set
  includes lag/difference/ratio/log transforms, row- and duration-clock
  rolling statistics, rolling z-scores, rolling quantiles, and median/IQR
  robust scaling, plus an explicitly scaled tanh transform.
- `sig.py` contains explicit target expressions, simple signal composition,
  chronological splits, and `SignalBundle`. Target helpers distinguish simple
  return, log return, and forward difference so an already-log-transformed
  spread is not logged twice. Models remain case-owned until repeated cases
  justify a stable adapter.
- `viz.py` contains optional, stage-agnostic helpers for aligned panels from a
  wide Polars table and heatmaps from row/column/value evidence. A case-owned
  renderer selects the artifacts and columns.

Helpers do not fill nulls, replace NaNs, add epsilon, infer time units, sort
tables, or choose output column names. Those choices change research meaning
and must remain visible in the Lab recipe.

`BuildInfo` and `BuildStep` are optional recipe metadata. When a builder
exports `BUILD_INFO`, the CLI records it as `recipe.json`. The Kit catalog is
for discovery only; unlike the eval menu, it is not an allowlist.

Plotting dependencies are lazy and live in the `visualization` extra. Stage
runs do not import them. `rci visualize --stage ...` calls an external Lab
renderer explicitly and confines its files to that stage's `plots/` directory.
Visualization does not participate in the four eval slots or stage approval.
