# Evaluation layout

`protocol.py` defines the four fixed slots. `runner.py` executes an external
plan against a read-only Polars table and writes sidecar records. `baseline.py`
contains the Engine floor that runs for every stage.

The methods in `library/` are reusable general methodology. They are grouped
by what they measure, not by stage. Their metadata declares which stages and
input shapes are meaningful. `DataSet` or `FeatureSet` describes a stage table;
an external case agent then creates `LibrarySelection` with the method, column
scope, parameters, and reason for every choice. `selection.py` validates and
expands only those choices. The Engine does not choose methods for the case.

The current families are `quality`, `temporal`, `structure`, `association`,
`information`, and `stability`. Structure methods include PCA and VIF;
information methods use scikit-learn k-nearest-neighbour mutual information;
association methods include IC, ICIR, cumulative IC, and bucket spreads.
Target-dependent methods can use `EvalContext` and a named sidecar table, so a
label or forward return never becomes an untracked feature column.

Method location and stage ownership are separate. Feat selects correctness,
coverage, feature-feature structure, redundancy, and stability methods. Sig
selects target-aware association, target mutual information, out-of-sample,
null-world, and signal-stability methods. The implementation may be shared,
but its interpretation and default proposal belong to the stage that owns the
question.

`FeatureSet.targets` is the currently implemented low-level target-sidecar
bridge. The planned `SignalSet` will become the canonical declaration for
features, targets, horizons, targetless hypotheses, and signal recipes. Until
then, documentation and new orchestration should treat target-dependent
evaluation as Sig work rather than Feat structural screening.

Case-specific rules remain in the Lab's `evals/<stage>.py` files.

Each record states its actual execution path: `polars_expr`, `polars_select`,
`polars_join_group`, `matrix_backend`, or `custom_runner`. Matrix methods first
select, cast, filter nulls, and filter non-finite values in Polars before
crossing into NumPy/scikit-learn.
