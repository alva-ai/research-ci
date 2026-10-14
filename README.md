# Research CI

Research CI is the execution engine for agent-owned quantitative research
workspaces. The first implementation establishes the protocol boundary:

- `Contract` describes the research boundary and stage order;
- every stage supplies four eval slots: `baseline`, `library`, `case`, and
  `case_eval`;
- evals read a Polars main table without modifying it;
- results are written to an eval sidecar and may be booleans, metrics,
  findings, or no business output.

The research case lives outside this repository. Its `lab/` code builds the
tables and assembles the stage-specific `EvalPlan`; Research CI executes and
audits that plan.

The project uses `uv` for environments and dependency locking.

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
      library/    reusable methods grouped by methodology
        quality.py coverage, finite values, dispersion
        structure.py correlation, VIF, PCA, autocorrelation
        association.py IC, ICIR, cumulative IC, bucket spread
        information.py mutual information and target-sidecar MI
        stability.py split stability, rank persistence, drift
      defaults/   stage-specific method proposals
    cards.py      Markdown cards and stable card IDs
    workflow.py   stage states and human/agent approvals
    cli.py        command line orchestration
research-ci-lab/
  lab/<case>/
    contract.py    external research boundary
    data.py        source exploration and Data table builder
    feat.py        feature exploration and feature table builder
    evals/data.py  Data four-slot plan
    evals/feat.py  Feat four-slot plan
    raw/           source snapshots owned by the lab
  .rci/runs/<run>/workflow.json
                  stage states and approvals
  .rci/runs/<run>/<stage>/
                  stage Card, main table, and sidecar evidence
```

The Engine library is organized by methodology rather than by stage. For
example, correlation and PCA live in `eval/library/structure.py`, while IC
methods live in `eval/library/association.py`. Stage proposal lists belong to
`eval/defaults/` once that stage's input contract is explicit. This keeps one
implementation reusable while retaining stage-specific selection. A method
declares its supported stage and input shape; an inapplicable method reports
`not_applicable` with a reason.

The run directory is the handoff boundary. A stage writes its Card and
evidence, moves to `review`, and waits for an explicit human or agent approval.
The next stage remains locked until that approval is recorded.

The Engine always contributes the general baseline floor. A lab may add
stage-specific baseline rules, but an empty external `baseline` tuple is valid;
the other three slots must still be supplied by the lab.

The library is a menu. `DataSet` and `FeatureSet` describe the current Data and
Feat tables, their keys, columns, and column groups. They do not choose
methodology. An external agent creates a `LibrarySelection`, records why each
method applies, and supplies its scope and parameters. The Engine validates
that choice, expands it across the declared columns, and records the evidence
without appending eval columns to the main table. Agents can inspect the menu
with:

```text
rci eval methods --stage feat
rci eval methods --family information --json
```

`FeatureSet.targets` and `EvalContext` currently provide a low-level way to
join named target Parquet sidecars without modifying a feature table. The
canonical stage boundary is now Sig: the planned `SignalSet` will declare
features, targets, horizons, or a targetless hypothesis and will own
target-dependent selection. The Engine never infers a target or horizon from
column names.

Every Data and Feat Card now separates two questions:

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
rci approve --stage contract --out .rci/runs/001 --by agent --card <card_id>
rci dataloop run --contract path/to/contract.py --builder path/to/data.py \
  --plan path/to/evals/data.py --out .rci/runs/001
rci approve --stage data --out .rci/runs/001 --by human --card <card_id>
rci feat run --builder path/to/feat.py --plan path/to/evals/feat.py \
  --out .rci/runs/001
rci status --out .rci/runs/001
rci eval run --stage data --table data.parquet --plan evals/data.py --out .rci/runs/001
```

`rci contract derive` writes the Contract card. `rci dataloop run` then reads
that approved boundary and writes the Data main table, the external
exploration summary, the four-slot eval report, and a Data result card. It
stops after Data so a person can review the result before Feat begins.

`rci feat run` reads only the approved `data/main.parquet`, calls the external
`build_features(table)` or `build(table)`, and writes:

```text
feat/main.parquet
feat/exploration.json
feat/eval/report.jsonl
feat/card.md
feat/manifest.json
```

The Engine verifies that the feature output preserves the Data key grid and
that the builder did not modify its input. Warm-up nulls remain visible to the
general missing-value baseline and can be reviewed without being written back
as eval columns. Feat evaluates correctness, coverage, structure, redundancy,
and stability. It leaves target-aware screening to Sig. Feat remains in
`review` until its active Card is approved, so Sig stays locked while an agent
edits and reruns the feature builder or plan.

## Planned Sig handoff

SigLoop will read only an approved `feat/main.parquet`. The external case agent
will declare a target and horizon or a targetless hypothesis, choose the
applicable library methods, and define a signal recipe. Planned outputs are:

```text
sig/main.parquet          keys and signal columns
sig/selection.json        retained, excluded, and clustered features
sig/recipe.json           rules, weights, normalization, and fit policy
sig/artifacts/            optional fitted model or matrix artifacts
sig/eval/report.jsonl     four-slot evidence
sig/eval/assessment.json  review disposition and reason codes
sig/card.md               human/agent review entry point
sig/manifest.json         provenance and hashes
```

The Sig baseline will check schema, key alignment, finite values, point-in-time
behavior, fit/predict isolation, and reproducibility. The library slot remains
agent-selected and may include IC, rank IC, ICIR, cumulative IC, target mutual
information, bucket behavior, horizon decay, out-of-sample stability, null or
permutation evidence, turnover, and concentration. The case slot defines the
actual recipe; `case_eval` proves it with valid fixtures and counterexamples.
Sig approval means the signal is defined and evidenced well enough for Algo
research. Profitability, position sizing, stop rules, leverage, costs,
slippage, capacity, and execution remain Algo concerns.
