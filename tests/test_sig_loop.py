from __future__ import annotations

import json

import polars as pl

from research_ci.cli import main
from research_ci.workflow import new_workflow, read_workflow, write_workflow


def test_sig_cli_writes_clean_main_label_sidecar_and_review_card(tmp_path) -> None:
    out = tmp_path / "run"
    (out / "feat").mkdir(parents=True)
    (out / "data").mkdir(parents=True)
    feature_table = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    feature_table.write_parquet(out / "feat" / "main.parquet")
    pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "left_price": [10.0, 11.0, 12.0, 13.0],
            "right_price": [5.0, 5.5, 6.0, 6.5],
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
        dispositions=(
            FeatureDisposition(
                feature="x",
                action="keep_direct",
                reason="the complete one-feature universe is retained",
                evidence=("screening:x:1h",),
                selected_for=("candidate",),
            ),
        ),
        lineage=(
            SignalLineage(
                signal="candidate",
                role="candidate",
                method="identity",
                inputs=("x",),
                fit_policy="none",
            ),
            SignalLineage(
                signal="signal",
                role="final",
                method="promote_candidate",
                inputs=("candidate",),
                fit_policy="none",
                evidence=("screening:x:1h",),
            ),
        ),
        sidecars={
            "label": table.select(
                "event_time",
                (pl.col("x").shift(-1) - pl.col("x")).alias("return_1h"),
            )
        },
        research={
            "screening": pl.DataFrame(
                {"feature": ["x"], "horizon": ["1h"], "rank_ic": [1.0]}
            )
        },
        metadata={"selection": {"signals": ["signal"]}},
        artifacts={"weights": {"signal": 1.0}},
    )

def build_main_table(data, signals):
    return data.select(
        "event_time",
        pl.lit("left").alias("left_inst"),
        pl.col("left_price"),
        pl.lit("right").alias("right_inst"),
        pl.col("right_price"),
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
    EvalPlan,
    EvalSpec,
    LibrarySelection,
    SelectedMethod,
    SignalSet,
    expand_library,
)

SIGNAL_SET = SignalSet(
    keys=("event_time",),
    features=("x",),
    candidates=("candidate",),
    signals=("signal",),
    targets={"1h": "return_1h"},
)

LIBRARY_SELECTION = LibrarySelection((
    SelectedMethod(
        "library.quality.required_columns",
        reason="the signal evaluation table needs its declared inputs",
        failure_policy="stop",
    ),
    SelectedMethod(
        "library.association.spearman_ic",
        reason="screen the complete feature universe against every target",
        on="features",
    ),
))

def make_plan(table, stage="sig", context=None):
    del table, context
    case = EvalSpec(
        "sig.case.signal_definition",
        expr=pl.col("signal") == pl.col("x"),
        failure_policy="stop",
    )
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=expand_library(SIGNAL_SET, LIBRARY_SELECTION),
        case=(case,),
        case_eval=(
            CaseEvalTest.from_rows(
                name="sig.case_eval.signal_definition_has_counterexample",
                case=case.name,
                valid={"x": [1.0], "signal": [1.0]},
                invalid={"x": [1.0], "signal": [2.0]},
            ),
        ),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
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

    assert exit_code == 0
    assert read_workflow(out)["stages"]["sig"] == "ready"
    assert (out / "sig" / "attempts" / "0001" / "main.parquet").exists()
    assert not (out / "sig" / "main.parquet").exists()

    brief = tmp_path / "sig-review.json"
    brief.write_text(
        json.dumps(
            {
                "question": "Does the candidate preserve and explain the signal?",
                "conclusion": (
                    "The declared signal passed the selected evidence checks."
                ),
                "evidence": ["The full feature universe has a disposition."],
                "limitations": ["Evidence is development-only."],
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

    submit_exit = main(
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

    assert submit_exit == 0
    main_table = pl.read_parquet(out / "sig" / "main.parquet")
    label = pl.read_parquet(out / "sig" / "context" / "label.parquet")
    assert main_table.columns == [
        "event_time",
        "left_inst",
        "left_price",
        "right_inst",
        "right_price",
        "signal",
    ]
    assert label.columns == ["event_time", "return_1h"]
    assert "return_1h" not in main_table.columns
    assert main_table.height == feature_table.height
    assert not (out / "sig" / "algo_input.parquet").exists()
    assert (out / "sig" / "selection.json").exists()
    assert (out / "sig" / "research" / "candidates.parquet").exists()
    assert (out / "sig" / "research" / "disposition.parquet").exists()
    assert (out / "sig" / "research" / "lineage.json").exists()
    assert (out / "sig" / "research" / "screening.parquet").exists()
    assert (out / "sig" / "artifacts" / "weights.json").exists()
    manifest = json.loads((out / "sig" / "manifest.json").read_text())
    assert manifest["signals"] == ["signal"]
    assert manifest["candidates"] == ["candidate"]
    assert manifest["context_tables"] == ["label"]
    assert manifest["main_table"] == str(out / "sig" / "main.parquet")
    assert manifest["main_layout"] == "wide"
    assert manifest["columns"] == main_table.columns
    assert manifest["workflow_state"] == "review"
    assert manifest["recommendation"] == "continue"
    assert manifest["evidence_scope"] == "development_only"
    assert manifest["confirmation_status"] == "not_declared"
    assert manifest["input_tables"] == {
        "data": str(out / "data" / "main.parquet"),
        "feat": str(out / "feat" / "main.parquet"),
    }
    assert not (out / "sig" / "plots").exists()
    assert read_workflow(out)["stages"]["sig"] == "review"

    renderer = tmp_path / "sig_renderer.py"
    renderer.write_text(
        """
def render_visualizations(run_dir, stage, output_dir):
    path = output_dir / "overview.txt"
    path.write_text(f"rendered {stage} from {run_dir}\\n", encoding="utf-8")
    return {"overview": path}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    visualize_exit = main(
        [
            "visualize",
            "--stage",
            "sig",
            "--renderer",
            str(renderer),
            "--out",
            str(out),
        ]
    )

    assert visualize_exit == 0
    assert (out / "sig" / "plots" / "overview.txt").exists()
    manifest = json.loads((out / "sig" / "manifest.json").read_text())
    assert manifest["visualizations"] == {
        "overview": str(out / "sig" / "plots" / "overview.txt")
    }


def test_sig_cli_writes_non_trading_assessment_without_fake_market_context(
    tmp_path,
) -> None:
    out = tmp_path / "assessment-run"
    (out / "feat").mkdir(parents=True)
    (out / "data").mkdir(parents=True)
    feature_table = pl.DataFrame(
        {
            "event_time": [1, 2, 3],
            "market_return": [0.01, -0.02, 0.03],
        }
    )
    feature_table.write_parquet(out / "feat" / "main.parquet")
    feature_table.write_parquet(out / "data" / "main.parquet")
    state = new_workflow(("data", "feat", "sig"), "contract-card")
    state["stages"].update(
        {"contract": "approved", "data": "approved", "feat": "approved", "sig": "ready"}
    )
    write_workflow(out, state)

    builder = tmp_path / "assessment_builder.py"
    builder.write_text(
        """
import polars as pl
from research_ci.kit import FeatureDisposition, SignalBundle, SignalLineage

def build_signals(table):
    return SignalBundle(
        main=pl.DataFrame({
            "assessment_id": ["1001:2001:2026-10-15"],
            "overall_status": ["narrow"],
        }),
        candidates=pl.DataFrame({"assessment_id": ["1001:2001:2026-10-15"]}),
        dispositions=(
            FeatureDisposition(
                feature="market_return",
                action="keep_as_context",
                reason="used as market context for the premise assessment",
                evidence=("quantitative_findings:market_return",),
            ),
        ),
        lineage=(
            SignalLineage(
                signal="overall_status",
                role="final",
                method="premise_evidence_synthesis",
                inputs=("market_return",),
                fit_policy="diagnostic_only",
            ),
        ),
        research={
            "quantitative_findings": pl.DataFrame({
                "feature": ["market_return"],
                "finding": ["mixed"],
            }),
        },
        metadata={"assessment": True, "selection": {"signals": ["overall_status"]}},
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plan = tmp_path / "assessment_plan.py"
    plan.write_text(
        """
import polars as pl
from research_ci import (
    CaseEvalTest,
    EvalPlan,
    EvalSpec,
    LibrarySelection,
    SelectedMethod,
    SignalSet,
    expand_library,
)

SIGNAL_SET = SignalSet(
    keys=("assessment_id",),
    features=("market_return",),
    signals=("overall_status",),
    hypothesis=(
        "Assess a frozen Thesis version from quantitative and documentary evidence."
    ),
)
LIBRARY_SELECTION = LibrarySelection((
    SelectedMethod(
        "library.quality.required_columns",
        reason="the evaluation table needs all declared inputs and assessment outputs",
        on=("overall_status",),
        failure_policy="stop",
    ),
))

def make_plan(table, stage="sig", context=None):
    del table, context
    case = EvalSpec(
        "sig.case.assessment_status_domain",
        expr=pl.col("overall_status").is_in(
            ["maintain", "narrow", "revise", "reject", "insufficient"]
        ),
        failure_policy="stop",
    )
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=expand_library(SIGNAL_SET, LIBRARY_SELECTION),
        case=(case,),
        case_eval=(
            CaseEvalTest.from_rows(
                name="sig.case_eval.assessment_status_has_counterexample",
                case=case.name,
                valid={"overall_status": ["narrow"]},
                invalid={"overall_status": ["buy"]},
            ),
        ),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
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

    assert exit_code == 0
    attempt = out / "sig" / "attempts" / "0001"
    main_table = pl.read_parquet(attempt / "main.parquet")
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert main_table.to_dicts() == [
        {
            "assessment_id": "1001:2001:2026-10-15",
            "overall_status": "narrow",
        }
    ]
    assert not any(
        column.endswith(("_inst", "_price")) for column in main_table.columns
    )
    assert manifest["main_layout"] == "assessment"
    assert manifest["assessment_mode"] is True


def test_sig_assessment_target_methods_use_internal_feature_context(tmp_path) -> None:
    out = tmp_path / "assessment-target-run"
    (out / "feat").mkdir(parents=True)
    (out / "data").mkdir(parents=True)
    feature_table = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    feature_table.write_parquet(out / "feat" / "main.parquet")
    feature_table.write_parquet(out / "data" / "main.parquet")
    state = new_workflow(("data", "feat", "sig"), "contract-card")
    state["stages"].update(
        {"contract": "approved", "data": "approved", "feat": "approved", "sig": "ready"}
    )
    write_workflow(out, state)

    builder = tmp_path / "assessment_target_builder.py"
    builder.write_text(
        """
import polars as pl
from research_ci.kit import FeatureDisposition, SignalBundle, SignalLineage

def build_signals(table):
    return SignalBundle(
        main=pl.DataFrame({
            "thesis_id": [1001],
            "author_version_id": [2001],
            "as_of": ["2026-10-15"],
            "overall_status": ["narrow"],
        }),
        candidates=pl.DataFrame({
            "thesis_id": [1001],
            "author_version_id": [2001],
            "as_of": ["2026-10-15"],
        }),
        dispositions=(
            FeatureDisposition(
                feature="x",
                action="keep_as_context",
                reason="retain the approved feature for assessment research",
                evidence=("screening:x:1h",),
            ),
        ),
        lineage=(
            SignalLineage(
                signal="overall_status",
                role="final",
                method="premise_evidence_synthesis",
                inputs=("x",),
                fit_policy="diagnostic_only",
            ),
        ),
        sidecars={
            "label": table.select(
                "event_time",
                (pl.col("x") * 0.1).alias("return_1h"),
            ),
        },
        research={
            "screening": pl.DataFrame({
                "feature": ["x"],
                "horizon": ["1h"],
            }),
        },
        metadata={
            "assessment": True,
            "thesis_id": 1001,
            "author_version_id": 2001,
            "as_of": "2026-10-15",
            "selection": {"signals": ["overall_status"]},
        },
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plan = tmp_path / "assessment_target_plan.py"
    plan.write_text(
        """
import polars as pl
from research_ci import (
    CaseEvalTest,
    EvalPlan,
    EvalSpec,
    LibrarySelection,
    SelectedMethod,
    SignalSet,
    expand_library,
)

SIGNAL_SET = SignalSet(
    keys=("thesis_id", "author_version_id", "as_of"),
    features=("x",),
    signals=("overall_status",),
    targets={"1h": "return_1h"},
    label_table="label",
    hypothesis="Assess a frozen thesis from the approved Feat panel.",
)
LIBRARY_SELECTION = LibrarySelection((
    SelectedMethod(
        "library.association.pearson_ic",
        reason="screen the approved feature against the declared target",
        on="features",
    ),
))

def make_plan(table, stage="sig", context=None):
    del table, context
    case = EvalSpec(
        "sig.case.assessment_status_domain",
        expr=pl.col("overall_status").is_in(
            ["maintain", "narrow", "revise", "reject", "insufficient"]
        ),
        failure_policy="stop",
    )
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=expand_library(SIGNAL_SET, LIBRARY_SELECTION),
        case=(case,),
        case_eval=(
            CaseEvalTest.from_rows(
                name="sig.case_eval.status_has_counterexample",
                case=case.name,
                valid={"overall_status": ["narrow"]},
                invalid={"overall_status": ["buy"]},
            ),
        ),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
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

    assert exit_code == 0
    attempt = out / "sig" / "attempts" / "0001"
    report = [
        json.loads(line)
        for line in (attempt / "eval" / "report.jsonl").read_text().splitlines()
    ]
    target_record = next(record for record in report if "pearson_ic" in record["name"])
    assert target_record["status"] == "passed"
    assert target_record["details"]["context_source"] == "__rci_feature_table"
    assert target_record["details"]["join_on"] == ["event_time"]
    assert target_record["details"]["joined_rows"] == 4

    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["context_tables"] == ["label"]
    assert not (attempt / "context" / "__rci_feature_table.parquet").exists()
