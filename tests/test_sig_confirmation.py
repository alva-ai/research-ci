from __future__ import annotations

import json

import polars as pl

from research_ci.cli import main
from research_ci.workflow import (
    new_workflow,
    next_action,
    read_workflow,
    write_workflow,
)


def test_sig_freezes_open_ended_validation_before_one_time_confirmation(
    tmp_path, capsys
) -> None:
    out = tmp_path / "run"
    (out / "feat").mkdir(parents=True)
    (out / "data").mkdir(parents=True)
    feature_table = pl.DataFrame(
        {"event_time": [1, 2, 3, 4, 5, 6], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    feature_table.write_parquet(out / "feat" / "main.parquet")
    pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4, 5, 6],
            "left_price": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "right_price": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
        }
    ).write_parquet(out / "data" / "main.parquet")
    state = new_workflow(("data", "feat", "sig", "algo"), "contract-card")
    state["stages"].update(
        {
            "contract": "approved",
            "data": "approved",
            "feat": "approved",
            "sig": "ready",
        }
    )
    write_workflow(out, state)

    builder = tmp_path / "sig_builder.py"
    builder.write_text(
        """
import polars as pl
from research_ci.kit import FeatureDisposition, SignalBundle, SignalLineage

def build_signals(table):
    return SignalBundle(
        main=table.select("event_time", pl.col("x").alias("signal")),
        candidates=table.select("event_time", pl.col("x").alias("candidate")),
        dispositions=(FeatureDisposition(
            feature="x",
            action="keep_direct",
            reason="the full feature universe is retained",
            evidence=("screening:x:1h",),
            selected_for=("candidate",),
        ),),
        lineage=(
            SignalLineage(
                signal="candidate",
                role="candidate",
                method="case_search",
                inputs=("x",),
                fit_policy="purged_walk_forward",
            ),
            SignalLineage(
                signal="signal",
                role="final",
                method="frozen_selection_rule",
                inputs=("candidate",),
                fit_policy="rolling_refit",
            ),
        ),
        sidecars={
            "label": table.head(4).select(
                "event_time", (pl.col("x") * 0.1).alias("return_1h")
            )
        },
        research={
            "screening": pl.DataFrame(
                {"feature": ["x"], "horizon": ["1h"], "rank_ic": [1.0]}
            )
        },
        metadata={"selection": {"signals": ["signal"], "rule": "frozen_rule"}},
        artifacts={"model": {"refit": "rolling", "window": 3}},
    )

def build_main_table(data, signals):
    return data.select(
        "event_time",
        pl.lit("left").alias("left_inst"),
        "left_price",
        pl.lit("right").alias("right_inst"),
        "right_price",
    ).join(signals, on="event_time", how="inner")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plan = tmp_path / "sig_plan.py"
    plan.write_text(
        """
import polars as pl
from research_ci import (
    CaseEvalTest,
    ConfirmationPlan,
    EvalPlan,
    EvalSpec,
    LibrarySelection,
    SelectedMethod,
    SignalSet,
    ValidationMethod,
    ValidationPlan,
    expand_library,
)

SIGNAL_SET = SignalSet(
    keys=("event_time",),
    features=("x",),
    candidates=("candidate",),
    signals=("signal",),
    targets={"1h": "return_1h"},
    fit_policy="walk_forward",
)

VALIDATION_PLAN = ValidationPlan(
    development=(
        ValidationMethod(
            "purged_walk_forward",
            purpose="measure temporal stability without overlapping labels",
            parameters={"purge": 1, "step": 1},
        ),
        ValidationMethod(
            "pbo",
            purpose="estimate selection overfit across candidate paths",
            parameters={"partitions": 4},
        ),
        ValidationMethod(
            "cluster_stability",
            purpose="check whether redundant groups persist across folds",
        ),
    ),
    selection_rule="freeze the declared rolling recipe before confirmation",
    confirmation=ConfirmationPlan(
        holdout_id="future_block_001",
        context_table="label",
        description="one untouched chronological block",
    ),
)

LIBRARY_SELECTION = LibrarySelection((
    SelectedMethod(
        "library.association.spearman_ic",
        reason="screen every feature against the development target",
        on="features",
    ),
))

def _case(name):
    return EvalSpec(name, expr=pl.col("signal") == pl.col("x"), failure_policy="stop")

def _plan(stage, case):
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=expand_library(SIGNAL_SET, LIBRARY_SELECTION),
        case=(case,),
        case_eval=(CaseEvalTest.from_rows(
            name=f"{case.name}.has_counterexample",
            case=case.name,
            valid={"event_time": [1], "x": [1.0], "signal": [1.0]},
            invalid={"event_time": [1], "x": [1.0], "signal": [2.0]},
        ),),
    )

def make_plan(table, stage="sig", context=None):
    del table, context
    return _plan(stage, _case("sig.case.development_signal_definition"))

def make_confirmation_plan(table, stage="sig", context=None):
    assert "candidate" not in table.columns
    assert context.table("label").get_column("event_time").to_list() == [5, 6]
    case = _case("sig.case.confirmation_signal_definition")
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=(EvalSpec.noop(
            "sig.library.confirmation_not_applicable",
            status="passed",
            reason="the fixture only verifies the confirmation boundary",
            library_method="library.not_applicable",
        ),),
        case=(case,),
        case_eval=(CaseEvalTest.from_rows(
            name="sig.case_eval.confirmation_signal_definition_has_counterexample",
            case=case.name,
            valid={"event_time": [5], "x": [5.0], "signal": [5.0]},
            invalid={"event_time": [5], "x": [5.0], "signal": [4.0]},
        ),),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "sig",
                "run",
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
    attempt = state["attempts"]["sig"][0]
    assert attempt["confirmation_required"] is True
    assert attempt["confirmation_status"] == "pending"
    assert next_action(state)["action"] == "freeze"

    assert (
        main(
            [
                "sig",
                "submit",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--recommendation",
                "continue",
            ]
        )
        == 1
    )
    assert "RCI-SIG-CONFIRMATION-REQUIRED" in capsys.readouterr().out

    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pl.DataFrame({"event_time": [5, 6], "return_1h": [0.5, 0.6]}).write_parquet(
        holdout / "label.parquet"
    )
    assert (
        main(
            [
                "sig",
                "freeze",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--context",
                str(holdout),
            ]
        )
        == 0
    )
    capsys.readouterr()

    state = read_workflow(out)
    attempt = state["attempts"]["sig"][0]
    assert attempt["frozen"] is True
    assert attempt["confirmation_status"] == "revealed_valid"
    assert attempt["evidence_scope"] == "confirmed_holdout"
    assert len(state["holdouts"]) == 1
    assert next_action(state)["action"] == "prepare"
    attempt_dir = out / "sig" / "attempts" / "0001"
    holdout_record = json.loads(
        (attempt_dir / "confirmation" / "holdout.json").read_text()
    )
    assert holdout_record["scope"]["candidate_columns_exposed"] == []
    assert holdout_record["scope"]["development_overlap_rows"] == 0
    assert "confirmed_holdout" in (attempt_dir / "card.md").read_text()

    assert (
        main(
            [
                "sig",
                "freeze",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--context",
                str(holdout),
            ]
        )
        == 1
    )
    assert "RCI-SIG-CONFIRMATION-CONSUMED" in capsys.readouterr().out

    selection_path = attempt_dir / "selection.json"
    original_selection = selection_path.read_bytes()
    selection_path.write_text('{"signals": ["changed_after_holdout"]}\n')
    assert (
        main(
            [
                "sig",
                "submit",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--recommendation",
                "stop",
            ]
        )
        == 1
    )
    assert "RCI-SIG-FREEZE-MISMATCH" in capsys.readouterr().out
    selection_path.write_bytes(original_selection)

    brief = tmp_path / "sig-confirmed-review.json"
    brief.write_text(
        json.dumps(
            {
                "question": "Did the frozen signal survive its confirmation block?",
                "conclusion": "The frozen final signal passed confirmation.",
                "evidence": ["The one-time holdout checks passed."],
                "limitations": ["The confirmation block contains two rows."],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "sig",
                "prepare",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--brief",
                str(brief),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "sig",
                "submit",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--recommendation",
                "continue",
            ]
        )
        == 0
    )
    assert (out / "sig" / "confirmation" / "holdout.json").exists()
    assert read_workflow(out)["stages"]["sig"] == "review"


def test_validation_plan_accepts_case_specific_methods() -> None:
    from research_ci import ConfirmationPlan, ValidationMethod, ValidationPlan

    plan = ValidationPlan(
        development=(
            ValidationMethod("rolling_k_fold", "temporal resampling"),
            ValidationMethod("custom_pbo", "selection-overfit evidence"),
            ValidationMethod("case_cluster_graph", "candidate grouping"),
        ),
        selection_rule="case-owned deterministic promotion procedure",
        confirmation=ConfirmationPlan(
            "future_snapshot",
            description="a later sample unavailable during development",
        ),
    )

    assert [item["name"] for item in plan.as_dict()["development"]] == [
        "rolling_k_fold",
        "custom_pbo",
        "case_cluster_graph",
    ]
    assert plan.evidence_scope == "confirmation_pending"
