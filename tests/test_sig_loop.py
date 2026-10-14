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
        {"contract": "passed", "data": "passed", "feat": "passed", "sig": "ready"}
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

def build_algo_input(data, signals):
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
    signal_table = pl.read_parquet(out / "sig" / "main.parquet")
    label = pl.read_parquet(out / "sig" / "context" / "label.parquet")
    algo_input = pl.read_parquet(out / "sig" / "algo_input.parquet")
    assert signal_table.columns == ["event_time", "signal"]
    assert label.columns == ["event_time", "return_1h"]
    assert "return_1h" not in signal_table.columns
    assert algo_input.columns == [
        "event_time",
        "left_inst",
        "left_price",
        "right_inst",
        "right_price",
        "signal",
    ]
    assert algo_input.height == signal_table.height
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
    assert manifest["algo_input"] == str(out / "sig" / "algo_input.parquet")
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
