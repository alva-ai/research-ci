# Evaluation layout

`protocol.py` defines the four fixed slots. `runner.py` executes an external
plan against a read-only Polars table and writes sidecar records. `baseline.py`
contains the Engine floor that runs for every stage.

The methods in `library/` are reusable general methodology. They are grouped
by what they measure, not by stage. Their metadata declares their purpose,
input roles, requirements, fit policy, and valid stages. `policies/` defines
each stage's hard semantic boundary, while `defaults/` marks recommended
starting methods. `menu.py` combines all three with the case declaration and
reports which methods are selectable or unavailable, including the reason.

`DataSet`, target-free `FeatureSet`, `SignalSet`, and `AlgoSet` describe the
Data, Feat, Sig, and Algo inputs. An external case agent creates
`LibrarySelection` with the
method, column scope, parameters, and reason for every choice. `selection.py`
checks the compiled menu before it expands those choices. A method that is
incompatible with the stage or declaration is rejected before table
execution. The Engine recommends defaults but does not choose case semantics.

The current families are `quality`, `temporal`, `structure`, `association`,
`information`, `stability`, and `economic`. Structure methods include PCA and VIF;
information methods use scikit-learn k-nearest-neighbour mutual information;
association methods include IC, ICIR, cumulative IC, and bucket spreads.
Target-dependent methods use `SignalSet` together with `EvalContext` and a
named sidecar table, so a label or forward return never becomes an untracked
feature column.

`AlgoSet` declares whether its main table represents weights or trades, the
source prices and signals, position columns grouped by strategy, and the
external backtest return/equity/turnover schema. Economic methods read the
named backtest sidecar; they evaluate it but do not create a backtest.

Method location and stage ownership are separate. Feat selects correctness,
coverage, feature-feature structure, redundancy, and stability methods. Sig
selects target-aware association, target mutual information, out-of-sample,
null-world, and signal-stability methods. The implementation may be shared,
but its interpretation and default proposal belong to the stage that owns the
question.

`FeatureSet` cannot declare targets. `SignalSet` is the canonical declaration
for the complete approved feature universe, candidate columns, final signal
columns, targets keyed by horizon, targetless hypotheses, fit policy, and an
optional `evaluation_by` sidecar column for time-block evidence. IC, ICIR,
cumulative IC, bucket spread, and feature-target mutual information therefore
cannot be selected in Feat. A targetless Sig declaration must state its
hypothesis explicitly, and the menu keeps target-dependent methods unavailable
for that declaration.

`ValidationPlan` separately describes how a Sig case develops and selects its
final recipe. Its `ValidationMethod` entries are deliberately open-ended:
rolling folds, PBO, cluster stability, or a case-owned method can all be named
without expanding an Engine enum. An optional `ConfirmationPlan` names a
one-time context and the already-declared function that evaluates final signals
after the development attempt is frozen. If confirmation is omitted, the Card
marks the evidence as development-only.

For target-aware Sig, the Engine floor requires exact feature-universe
declaration, library screening coverage for every feature-horizon pair, a
saved screening table, one explained disposition per feature, and acyclic
lineage for every candidate and final signal. These checks enforce research
accounting; they do not choose a model or automatically promote the candidate
with the largest metric.

Case-specific rules remain in the Lab's `evals/<stage>.py` files. Slot 2 only
accepts registered Engine methods; custom Polars expressions or runners belong
in slot 3 and their valid and counterexample fixtures belong in slot 4.

Each record states its actual execution path: `polars_expr`, `polars_select`,
`polars_join_group`, `matrix_backend`, or `custom_runner`. Matrix methods first
select, cast, filter nulls, and filter non-finite values in Polars before
crossing into NumPy/scikit-learn.
