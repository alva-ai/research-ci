# Research CI

Research CI is the execution engine for agent-owned quantitative research
workspaces. The first implementation establishes the protocol boundary:

- `Contract` describes the research boundary and stage order;
- every data-producing stage supplies four eval slots: `baseline`, `library`,
  `case`, and `case_eval`; the small Contract is structurally validated instead;
- evals read a Polars main table without modifying it;
- results are written to an eval sidecar and may be booleans, metrics,
  findings, or no business output.

The research case lives outside this repository. Its `lab/` code builds the
tables and assembles the stage-specific `EvalPlan`; Research CI executes and
audits that plan.

The project uses `uv` for environments and dependency locking.

## Design documents

- [Design and product requirements](docs/design.md) defines the normative
  stage boundaries, shared four-slot protocol, artifact contracts, and the
  detailed Sig research loop.
- [Interactive design map](docs/research-ci-viz-en.html) shows the current MVP
  workflow, stage ownership, Sig internals, and the guarantees still planned.
- [TODO](TODO.md) records implementation status and open design work in
  Chinese; it is not the normative architecture document.

## Stage boundary

| stage | owns |
| --- | --- |
| Data | source facts, alignment, and the research-ready data table |
| Feat | reusable point-in-time measurements and structural screening |
| Sig | target-aware selection and a forecast, ranking, direction, or strength |
| Algo | capital allocation, positions, orders, risk, costs, and execution |

Sig may use a rule, a weighted feature combination, a matrix method, a fitted
model, or an ensemble. A model is therefore a Sig backend rather than a fifth
top-level stage. Weights that produce a prediction score belong to Sig; weights
that allocate capital or positions belong to Algo.

Feature selection occurs twice for different reasons. Feat rejects invalid or
unusable columns and records structural redundancy such as exact duplicates,
high correlation, VIF, PCA concentration, and feature-feature mutual
information. Sig then makes a recipe-specific selection with an explicit
target and horizon, using evidence such as IC, rank IC, ICIR, target mutual
information, bucket behavior, out-of-sample stability, and ablation. A
targetless rule-based Sig is also valid; target metrics are selected only when
they match the case.

A target-aware Sig must declare the complete approved Feat universe. Every
feature is screened for every declared horizon and receives one explicit
disposition: retained directly, used by a candidate component, retained as
context, or rejected with evidence. Candidate construction and final signal
materialization are separate outputs, so an agent cannot silently skip most of
the Feat table and jump to one convenient rule.

## Minimal folder boundary

```text
research-ci/
  src/research_ci/
    contract.py   Engine Contract protocol and validation
    eval/         four-slot protocol, runner, baseline, and library
      protocol.py four-slot plan model
      runner.py   Polars eval execution and sidecar records
      baseline.py Engine-owned floor checks
      assessment.py execution/research triage summary for Cards
      menu.py     declaration-aware stage menu compiler
      policies/   hard Data, Feat, Sig, and Algo method boundaries
      library/    reusable methods grouped by methodology
        quality.py coverage, finite values, dispersion
        structure.py correlation, VIF, PCA, autocorrelation
        association.py IC, ICIR, cumulative IC, bucket spread
        information.py mutual information and target-sidecar MI
        stability.py split stability, rank persistence, drift
        economic.py external-backtest return, risk, and turnover summary
      defaults/   stage-specific method proposals
    kit/          optional construction helpers for external Labs
      common.py   recipe and tool metadata
      data.py     table-in/table-out alignment helpers
      feat.py     name-free Polars expressions and explicit layers
      sig.py      targets, signal composition, time splits, and bundles
      algo.py     bounded sizing, pair weights, and Algo/backtest bundles
      viz.py      optional wide-table time-series and heatmap helpers
    cards.py      Markdown cards and stable card IDs
    workflow.py   stage states and human/agent approvals
    cli.py        command line orchestration
  docs/
    design.md              normative requirements and stage contracts
    research-ci-viz-en.html   interactive visual design map
research-ci-lab/
  lab/<case>/
    contract.py    external research boundary
    data.py        source exploration and Data table builder
    feat.py        feature exploration and feature table builder
    sig.py         signal builder and explicit label sidecars
    algo.py        position or trade builder
    backtest.py    case-owned external backtester
    visualize.py   case-owned selection of stage plots
    evals/data.py  Data four-slot plan
    evals/feat.py  Feat four-slot plan
    evals/sig.py   Sig four-slot plan and selected target methods
    evals/algo.py  Algo four-slot plan and selected economic methods
    raw/           source snapshots owned by the lab
  .rci/runs/<run>/workflow.json
                  stage states and approvals
  .rci/runs/<run>/<stage>/
    attempts/<id>/ Agent-owned iterations; never a downstream input
    main.parquet   submitted artifact; readable downstream only after approval
    card.md        submitted review Card
    candidate.json selected attempt and recommendation
                  plus stage-specific sidecar evidence
```

The Engine library is organized by methodology rather than by stage. For
example, correlation and PCA live in `eval/library/structure.py`, while IC
methods live in `eval/library/association.py`. `eval/policies/` supplies hard
stage boundaries, while `eval/defaults/` supplies recommendations. This keeps
one implementation reusable while retaining stage-specific selection. A
method declares its purpose, supported stages, input roles, requirements, and
fit policy. The menu rejects a semantically incompatible choice before data
execution; a compatible calculation can still report `not_applicable` when
the actual data is insufficient.

The Stage Kit has a different role from the eval library. It is an optional
construction library that an external Lab may import while building a stage
artifact. It is neither an allowlist nor another workflow stage:

- Data helpers accept and return in-memory Polars tables; downloads, caches,
  and source credentials remain in the Lab.
- Feat helpers accept `pl.Expr` inputs and return an unnamed `pl.Expr`. The Lab
  chooses source columns, output aliases, and expression layers.
- Sig helpers construct explicit label sidecars, simple signal combinations,
  chronological splits, and a `SignalBundle`. The bundle separates candidate
  scores from final signals and records `FeatureDisposition` and
  `SignalLineage`; case-specific estimators remain in the Lab.
- Algo helpers provide bounded scores, dollar-neutral pair sizing,
  `AlgoBundle`, and the standardized `BacktestResult` interface. Portfolio
  rules, cost assumptions, and the backtest implementation remain in the Lab.
- Visualization helpers draw aligned panels from a wide Polars table or a
  row/column/value heatmap table. The Lab chooses the stage artifacts, columns,
  plot names, and interpretation.

Kit functions do not silently sort, fill nulls, replace NaNs, add epsilon,
infer timestamp units, or choose output column names. A builder may export
optional `BUILD_INFO` metadata so the CLI can save the construction meaning in
`recipe.json`; `exploration.json`, `manifest.json`, and eval evidence retain
their separate purposes.

Visualization is an optional Kit capability, not a workflow stage or a fifth
eval slot. Normal stage runs neither import plotting libraries nor write plots.
Install the `visualization` extra and invoke the case renderer explicitly when
a reviewer needs figures:

```text
uv run --extra visualization rci visualize --stage sig \
  --renderer lab/<case>/visualize.py --out .rci/runs/001
```

The same command works for Data, Feat, Sig, and Algo. The external renderer
decides which stages it supports and writes only under `<stage>/plots/`.

For example, a Feat builder can declare independent expressions and apply
dependent groups in order:

```python
import polars as pl

from research_ci.kit.feat import apply_layers, difference, rolling_zscore_rows

FEATURE_LAYERS = (
    (
        rolling_zscore_rows(pl.col("spread"), window_size=24, min_samples=24).alias(
            "spread_z_24h"
        ),
        difference(pl.col("spread"), periods=6).alias("spread_change_6h"),
    ),
)

feature_table = apply_layers(data_table, FEATURE_LAYERS)
```

The run directory is the handoff boundary. `rci <stage> run` writes a numbered
attempt and leaves the stage `ready`, so an Agent can diagnose, revise, and run
again without interrupting a person. `rci <stage> submit` selects one valid
attempt, publishes its formal artifacts, records the Agent's recommendation,
and moves the stage to `review`. The next stage remains locked until a reviewer
accepts the Card with a `continue` decision.

The Engine always contributes the general baseline floor. A lab may add
stage-specific baseline rules, but an empty external `baseline` tuple is valid;
the other three slots must still be supplied by the lab.

The library has one registry and a different compiled menu for each stage.
`DataSet`, `FeatureSet`, `SignalSet`, and `AlgoSet` describe the current inputs;
a stage
policy filters the registry before an external agent can select methods. The
agent creates a `LibrarySelection`, records why each method applies, and
supplies its scope and parameters. The Engine validates that choice, expands
it across declared columns, and records the method identity and evidence
without appending eval columns to the main table. Agents can inspect a static
stage menu or compile it against a case declaration:

```text
rci eval methods --stage feat
rci eval methods --stage feat --declaration lab/<case>/evals/feat.py
rci eval methods --family information --json
```

The menu reports `selectable` and `unavailable` methods with reasons, so an
agent can see that IC belongs to Sig while working in Feat. `FeatureSet` has no
target fields. `SignalSet` owns the complete feature universe, candidate and
final signal columns, targets and horizons, or an explicit targetless
hypothesis. `EvalContext` carries named label Parquet sidecars without
modifying the main table. The Engine never infers a target or horizon from
column names.

Every stage has one workflow state and a submitted Card has one Agent
recommendation:

- workflow state: `locked`, `ready`, `review`, `approved`, `stopped`, or
  `disabled`;
- recommendation: `continue`, `stop`, or `uncertain`.

Attempts separately report whether mandatory checks are `valid` or `invalid`.
Draft Cards therefore use `workflow_state: ready` with
`artifact_role: attempt`; submitted Cards use `workflow_state: review` with
`artifact_role: candidate`. An invalid attempt cannot be submitted. Detailed
assessments still separate
two evidence questions:

- `execution_status`: did the requested checks run and did any stop-policy
  correctness rule halt the stage?
- `research_disposition`: what evidence category should be reviewed next,
  such as `incomplete`, `weak_data`, `redundant`, `unstable`, or
  `candidate_for_review`?

The second field is triage evidence, not an automatic alpha verdict. The Card
also lists reason codes such as missing context, low coverage, high VIF,
missing values, and failed case counterexamples. A metric can pass execution
while the Card still asks for more target, stability, or coverage evidence.
Feat Cards additionally list each feature's structural status, usable coverage,
and redundancy evidence so a sparse or constant column is visible without
turning the entire wide table into an eval table.

## Minimal example

```python
import polars as pl

from research_ci import (
    CaseEvalTest,
    Contract,
    DataSet,
    EvalPlan,
    EvalSpec,
    LibrarySelection,
    SelectedMethod,
    expand_library,
    run_eval_plan,
)

contract = Contract(
    mission="check whether two venue prices are aligned",
    stages=("data", "feat", "sig", "algo"),
    invariants=("point_in_time", "deterministic"),
)

table = pl.DataFrame(
    {
        "inst": ["a", "b"],
        "long_price": [101.0, 99.0],
        "short_price": [100.0, 98.0],
    }
)

data_set = DataSet(
    keys=("inst",),
    required=("inst", "long_price", "short_price"),
    numeric=("long_price", "short_price"),
    groups={"prices": ("long_price", "short_price")},
)

selected_methods = LibrarySelection(
    methods=(
        SelectedMethod(
            "library.quality.required_columns",
            reason="the assembled table must match the declared schema",
            failure_policy="stop",
        ),
        SelectedMethod(
            "library.quality.finite_columns",
            reason="the case computes a price spread from both columns",
            on="prices",
            failure_policy="stop",
        ),
    )
)

plan = EvalPlan(
    stage="data",
    baseline=(),
    library=expand_library(data_set, selected_methods),
    case=(
        EvalSpec(
            "spread_positive",
            expr=pl.col("long_price") > pl.col("short_price"),
        ),
    ),
    case_eval=(
        CaseEvalTest.from_rows(
            name="spread_positive_has_counterexample",
            case="spread_positive",
            valid={"long_price": [101.0], "short_price": [100.0]},
            invalid={"long_price": [99.0], "short_price": [100.0]},
        ),
    ),
)

result = run_eval_plan(table, plan)
assert result.passed
assert table.columns == ["inst", "long_price", "short_price"]
```

The command line entry point is `rci`:

```text
rci contract check path/to/contract.py
rci contract derive path/to/contract.py --out .rci/runs/001
rci approve --stage contract --out .rci/runs/001 --by human \
  --decision continue --card <card_id>
rci dataloop run --contract path/to/contract.py --builder path/to/data.py \
  --plan path/to/evals/data.py --out .rci/runs/001
rci data submit --out .rci/runs/001 --attempt 0001 --recommendation continue
rci approve --stage data --out .rci/runs/001 --by human \
  --decision continue --card <card_id>
rci feat run --builder path/to/feat.py --plan path/to/evals/feat.py \
  --out .rci/runs/001
rci feat submit --out .rci/runs/001 --attempt 0001 --recommendation continue
rci approve --stage feat --out .rci/runs/001 --by human \
  --decision continue --card <card_id>
rci sig run --builder path/to/sig.py --plan path/to/evals/sig.py \
  --out .rci/runs/001
rci sig freeze --out .rci/runs/001 --attempt 0001 \
  --context path/to/confirmation-context  # when ConfirmationPlan is declared
rci sig submit --out .rci/runs/001 --attempt 0001 --recommendation uncertain
rci approve --stage sig --out .rci/runs/001 --by human \
  --decision continue --card <card_id>
rci algo run --builder path/to/algo.py --plan path/to/evals/algo.py \
  --backtester path/to/backtest.py --out .rci/runs/001
rci algo submit --out .rci/runs/001 --attempt 0001 --recommendation stop
rci approve --stage algo --out .rci/runs/001 --by human \
  --decision stop --card <card_id>
rci next --out .rci/runs/001 --json
rci guide --stage sig
rci visualize --stage sig --renderer path/to/visualize.py --out .rci/runs/001
rci status --out .rci/runs/001
rci eval run --stage data --table data.parquet --plan evals/data.py --out .rci/runs/001
```

`rci contract derive` writes the Contract Card. After approval,
`rci dataloop run` writes the Data table, exploration summary, recipe,
four-slot report, and
draft Card under `data/attempts/<id>/`. A hard-check failure remains an invalid
attempt in `ready`; it cannot reach review. A valid attempt becomes the formal
Data candidate only through `rci data submit`.

`rci feat run` reads only the approved `data/main.parquet`, calls the external
`build_features(table)` or `build(table)`, and first writes:

```text
feat/attempts/<id>/main.parquet
feat/attempts/<id>/exploration.json
feat/attempts/<id>/recipe.json
feat/attempts/<id>/eval/report.jsonl
feat/attempts/<id>/card.md
feat/attempts/<id>/manifest.json
```

The Engine verifies that the feature output preserves the Data key grid and
that the builder did not modify its input. Warm-up nulls remain visible to the
general missing-value baseline and can be reviewed without being written back
as eval columns. Feat evaluates correctness, coverage, structure, redundancy,
and stability. It leaves target-aware screening to Sig. Feat remains `ready`
while the Agent edits and reruns its builder or plan. Submission moves one
valid attempt to `review`; the stage cannot run again until the reviewer either
rejects it back to `ready` or accepts it.

## Sig handoff

`SignalSet`, the declaration-aware Sig menu, and SigLoop are implemented.
SigLoop reads only an approved `feat/main.parquet`. The external case agent
declares a target and horizon or a targetless hypothesis, chooses applicable
library methods, and returns a `SignalBundle` containing candidate and final
scores, one disposition per approved feature, ordered signal lineage, named
sidecars, metadata, research tables, and optional fitted artifacts. Runtime
outputs are:

```text
sig/main.parquet          direct Algo input: keys, market context, and signals
sig/research/candidates.parquet
                          keys and all candidate signal columns
sig/research/screening.parquet
                          feature-target evidence by sample split
sig/research/disposition.parquet
                          one retained/rejected decision per Feat column
sig/research/lineage.json ordered candidate and final signal recipes
sig/research/*.parquet    optional redundancy, cluster, or ablation tables
sig/context/*.parquet     explicit labels and other named sidecars
sig/context/metadata.json sidecar and research metadata
sig/selection.json        retained, excluded, and clustered features
sig/recipe.json           rules, weights, normalization, and fit policy
sig/validation_plan.json  open-ended development methods and selection rule
sig/provenance.json       direct builder and plan source hashes
sig/artifacts/            optional fitted model or matrix artifacts
sig/eval/report.jsonl     four-slot evidence
sig/eval/assessment.json  review disposition and reason codes
sig/confirmation/         one-time final-signal holdout evidence, when declared
sig/card.md               human/agent review entry point
sig/manifest.json         provenance and hashes
sig/plots/                optional case-owned review figures
```

The Engine checks that candidate scores and final signals preserve the approved
Feat key grid. A normal label sidecar preserves that grid too. When a
confirmation holdout is declared, the development label sidecar must instead
contain a non-empty subset of approved keys and omit all holdout keys. For
target-aware Sig the Engine also
requires exact declaration of the complete Feat universe, library screening
coverage for every feature-horizon pair, a complete screening artifact, one
valid disposition per feature, and acyclic lineage for every candidate and
final output. It builds a transient evaluation table from declared Feat,
candidate, and final columns, then joins each label from its sidecar only
inside the selected method. The library slot remains agent-selected and may
include IC, rank IC, ICIR, cumulative IC, target mutual information, bucket
behavior, split stability, rank persistence, and drift. The case slot defines
the actual recipe; `case_eval` proves it with valid fixtures and
counterexamples.

An optional `ValidationPlan` records the case's development procedure without
forcing every case into one method. Its method names are open-ended, so a Lab
may declare rolling K-fold, purged walk-forward, PBO, clustering stability, or
a case-specific procedure. The declaration also records the deterministic
selection rule. Without a `ConfirmationPlan`, the Card explicitly labels the
result `development_only`.

With a `ConfirmationPlan`, a valid development attempt cannot be submitted
directly. `rci sig freeze` verifies the saved plan and builder source hashes,
hashes the attempt products, consumes the named holdout once, and runs the
predeclared confirmation function. That function receives approved features
and final signals only; candidate columns are excluded. The holdout keys must
be disjoint from the saved development label keys. A successful confirmation
freezes the attempt for submission, and any later change to its recipe,
selection, research evidence, fitted artifacts, or eval products is rejected.
The workflow also refuses another holdout with the same ID or byte-content
hash. These are protocol guarantees for a cooperating external Lab; the CLI
does not isolate a malicious process that reads files outside the run.

For a fixed set of instruments, a case exposes `build_main_table(data,
signals)`. SigLoop writes that result directly to `sig/main.parquet`: one wide
row per signal key, with paired `<name>_inst` and `<name>_price` columns,
market context, and the final signals. Algo reads this table as its sole stage
input. The Engine checks the key grid, instrument/price pairs, and positive
finite prices. A cross-sectional case may define a different downstream table
when its row grain is instrument-by-time.

The Sig Card presents final and candidate signal evidence, feature disposition
counts, signal-by-horizon association, early/late stability, block IC evidence,
cumulative paths, and feature rankings. The complete per-method records stay in
`sig/eval/report.jsonl`. Sig approval means the final signal set is defined and
evidenced well enough for Algo research. It does not automatically promote the
highest metric or claim profitability. Position sizing, stop rules, leverage,
costs, slippage, capacity, and execution remain Algo concerns.

## Algo handoff

AlgoLoop reads only an approved `sig/main.parquet`. The Lab returns an
`AlgoBundle` whose main table follows an `AlgoSet` contract. The declaration
chooses `weights` or `trades`, names source prices and signals, partitions
position columns by strategy, and declares standardized backtest return,
equity, and turnover columns.

The backtest implementation is injected explicitly with `--backtester` and
must return a `BacktestResult`. The Engine does not currently calculate fills,
costs, or PnL. It validates the position table, external backtest paths, and
one summary row per strategy, then exposes those artifacts to all four eval
slots. Runtime outputs are:

```text
algo/main.parquet                 prices, signals, and positions/trades
algo/backtest/timeseries.parquet standardized economic paths
algo/backtest/summary.parquet    one review row per strategy
algo/backtest/trades.parquet     optional trade ledger
algo/backtest/metadata.json      external assumptions and backend identity
algo/recipe.json                 construction and backtester provenance
algo/artifacts/                  optional sizing or execution artifacts
algo/eval/report.jsonl           four-slot evidence
algo/eval/assessment.json        review disposition and reason codes
algo/card.md                     human/agent review entry point
algo/manifest.json               artifact index and declared schema
algo/plots/                      optional case-owned review figures
```

The first economic library method recomputes total return, annualized
volatility, Sharpe, maximum drawdown, and average turnover from the declared
external timeseries. Case rules still verify the actual position formula and
backtest accounting, while `case_eval` proves those rules against mutated
counterexamples. A built-in default backtester remains future work.
