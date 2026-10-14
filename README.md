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

## Minimal folder boundary

```text
research-ci/
  src/research_ci/
    contract.py   Engine Contract protocol and validation
    eval/         four-slot protocol, runner, baseline, and library
      protocol.py four-slot plan model
      runner.py   Polars eval execution and sidecar records
      baseline.py Engine-owned floor checks
      library/    reusable methods grouped by methodology
      defaults/   stage-specific method proposals
    cards.py      Markdown cards and stable card IDs
    workflow.py   stage states and human/agent approvals
    cli.py        command line orchestration
research-ci-lab/
  lab/<case>/     external contract, stage builders, eval plans, and raw files
  .rci/runs/<run>/workflow.json
                  contract/ and data/ cards and evidence
```

The Engine library is organized by methodology rather than by stage. For
example, correlation and IC methods belong to `eval/library/association.py`,
while the stage proposal list belongs to `eval/defaults/` when that
stage's input contract is explicit. This keeps one implementation reusable by
Feat and Sig while retaining stage-specific selection. A method declares its
supported stage and input shape; an inapplicable method reports
`not_applicable` with a reason.

The run directory is the handoff boundary. A stage writes its Card and
evidence, moves to `review`, and waits for an explicit human or agent approval.
The next stage remains locked until that approval is recorded.

The Engine always contributes the general baseline floor. A lab may add
stage-specific baseline rules, but an empty external `baseline` tuple is valid;
the other three slots must still be supplied by the lab.

## Minimal example

```python
import polars as pl

from research_ci import CaseEvalTest, Contract, EvalPlan, EvalSpec, run_eval_plan

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

plan = EvalPlan(
    stage="data",
    baseline=(EvalSpec("key_present", expr=pl.col("inst").is_not_null()),),
    library=(
        EvalSpec(
            "prices_finite",
            expr=pl.all_horizontal(
                pl.col("long_price").is_finite(),
                pl.col("short_price").is_finite(),
            ),
        ),
    ),
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
rci status --out .rci/runs/001
rci eval run --stage data --table data.parquet --plan evals/data.py --out .rci/runs/001
```

`rci dataloop run` writes a Contract card, the Data main table, the external
exploration summary, the four-slot eval report, and a Data result card. It
stops after Data so a person can review the result before Feat begins.
