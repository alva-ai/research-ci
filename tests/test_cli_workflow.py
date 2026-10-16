from __future__ import annotations

import json
import shlex

from research_ci.cli import main
from research_ci.workflow import read_workflow


def test_cli_drives_agent_attempts_and_refuses_skipped_transitions(
    tmp_path, capsys
) -> None:
    lab = tmp_path / "lab" / "case"
    evals = lab / "evals"
    evals.mkdir(parents=True)
    out = tmp_path / ".rci" / "runs" / "case"

    contract = lab / "contract.py"
    contract.write_text(
        """
from research_ci import Contract

CONTRACT = Contract(
    mission="check one positive source series",
    stages=("data", "feat"),
    scope=("one synthetic source",),
    invariants=("point_in_time", "review_each_stage"),
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    builder = lab / "data.py"
    builder.write_text(
        """
import polars as pl

def build_main_table():
    return pl.DataFrame({"event_time": [1, 2], "value": [10.0, 11.0]})
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plan = evals / "data.py"
    plan.write_text(
        """
import polars as pl
from research_ci import CaseEvalTest, EvalPlan, EvalSpec

CASE = EvalSpec(
    "data.case.positive_value",
    expr=pl.col("value") > 0,
    failure_policy="stop",
)

def make_plan(table, stage="data", context=None):
    del table, context
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=(EvalSpec.noop(
            "data.library.not_applicable",
            status="passed",
            reason="the minimal fixture needs no optional method",
            library_method="library.not_applicable",
        ),),
        case=(CASE,),
        case_eval=(CaseEvalTest.from_rows(
            name="data.case_eval.positive_value_has_counterexample",
            case=CASE.name,
            valid={"event_time": [1], "value": [1.0]},
            invalid={"event_time": [1], "value": [-1.0]},
        ),),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert main(["contract", "derive", str(contract), "--out", str(out)]) == 0
    contract_card = read_workflow(out)["cards"]["contract"]
    capsys.readouterr()

    assert main(["contract", "derive", str(contract), "--out", str(out)]) == 1
    frozen_contract = capsys.readouterr().out
    assert "REFUSED [RCI-STAGE-FROZEN]" in frozen_contract
    assert "actor: reviewer" in frozen_contract

    refused = main(
        [
            "data",
            "run",
            "--contract",
            str(contract),
            "--builder",
            str(builder),
            "--plan",
            str(plan),
            "--out",
            str(out),
        ]
    )
    refusal = capsys.readouterr().out
    assert refused == 1
    assert "REFUSED [RCI-STAGE-NOT-READY]" in refusal
    assert "actor: reviewer" in refusal
    assert "rci approve --stage contract" in refusal

    assert (
        main(
            [
                "approve",
                "--stage",
                "contract",
                "--out",
                str(out),
                "--by",
                "human",
                "--decision",
                "continue",
                "--card",
                contract_card,
            ]
        )
        == 0
    )
    capsys.readouterr()

    abandoned = out / "data" / "attempts" / "0001"
    abandoned.mkdir(parents=True)
    (abandoned / "partial.txt").write_text("builder crashed\n", encoding="utf-8")

    assert (
        main(
            [
                "data",
                "run",
                "--contract",
                str(contract),
                "--builder",
                str(builder),
                "--plan",
                str(plan),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    capsys.readouterr()

    state = read_workflow(out)
    assert state["stages"]["data"] == "ready"
    assert state["attempts"]["data"][-1]["valid"] is True
    assert (out / "data" / "attempts" / "0001" / "partial.txt").exists()
    assert (out / "data" / "attempts" / "0002" / "main.parquet").exists()
    assert (out / "data" / "attempts" / "0002" / "catalog.json").exists()
    assert (out / "data" / "attempts" / "0002" / "source_coverage.parquet").exists()
    assert not (out / "data" / "main.parquet").exists()
    attempt_card = (out / "data" / "attempts" / "0002" / "card.md").read_text()
    assert "- workflow_state: ready" in attempt_card
    assert "- artifact_role: attempt" in attempt_card

    assert main(["next", "--out", str(out), "--json"]) == 0
    action = json.loads(capsys.readouterr().out)
    assert action["actor"] == "agent"
    assert action["action"] == "prepare"
    next_args = shlex.split(action["command"])
    assert next_args[next_args.index("--attempt") + 1] == "0002"

    brief = lab / "data-review.json"
    brief.write_text(
        json.dumps(
            {
                "question": "Is the source table structurally usable?",
                "conclusion": "The two-row source is aligned and ready for Feat.",
                "evidence": ["All mandatory checks passed on both rows."],
                "counterevidence": [],
                "limitations": ["This is a synthetic fixture."],
                "changes": ["Completed the first valid Data attempt."],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "data",
                "prepare",
                "--out",
                str(out),
                "--attempt",
                "0002",
                "--brief",
                str(brief),
            ]
        )
        == 0
    )
    capsys.readouterr()
    prepared_card = (out / "data" / "attempts" / "0002" / "card.md").read_text()
    assert "## Result" in prepared_card
    assert "## Audit trail" in prepared_card
    assert "## Exploration" not in prepared_card
    assert "data.case.positive_value" not in prepared_card

    assert main(["next", "--out", str(out), "--json"]) == 0
    action = json.loads(capsys.readouterr().out)
    assert action["action"] == "submit"

    assert (
        main(
            [
                "data",
                "submit",
                "--out",
                str(out),
                "--attempt",
                "0002",
                "--recommendation",
                "continue",
            ]
        )
        == 0
    )
    capsys.readouterr()
    state = read_workflow(out)
    assert state["stages"]["data"] == "review"
    assert state["recommendations"]["data"] == "continue"
    assert (out / "data" / "main.parquet").exists()
    submitted_card = (out / "data" / "card.md").read_text()
    assert "- workflow_state: review" in submitted_card
    assert "- artifact_role: candidate" in submitted_card
    assert "- recommendation: continue" in submitted_card

    assert main(["next", "--out", str(out), "--json"]) == 0
    review_action = json.loads(capsys.readouterr().out)
    assert review_action["actor"] == "reviewer"
    assert review_action["allowed_actors"] == ["human", "agent"]
    review_args = shlex.split(review_action["command"])
    assert review_args[review_args.index("--by") + 1] == "agent"

    refused_rerun = main(
        [
            "data",
            "run",
            "--contract",
            str(contract),
            "--builder",
            str(builder),
            "--plan",
            str(plan),
            "--out",
            str(out),
        ]
    )
    rerun_output = capsys.readouterr().out
    assert refused_rerun == 1
    assert "Its submitted Card must be reviewed first" in rerun_output
