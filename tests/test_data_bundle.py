from __future__ import annotations

import json

import polars as pl
import pytest

from research_ci import DataBundle, DataTableSpec, ReviewBrief
from research_ci.cli import main
from research_ci.workflow import new_workflow, write_workflow


def test_data_bundle_round_trip_preserves_formal_tables_and_audit_files(
    tmp_path,
) -> None:
    raw = tmp_path / "download.json"
    raw.write_text('{"source": "fixture"}\n', encoding="utf-8")
    main = pl.DataFrame(
        {
            "event_time": [2, 3],
            "price": [101.0, 102.0],
            "macro_value": [5.0, 5.0],
        }
    )
    native_price = pl.DataFrame(
        {"event_time": [1, 2, 3], "price": [100.0, 101.0, 102.0]}
    )
    native_macro = pl.DataFrame(
        {
            "event_time": [1],
            "available_at": [2],
            "macro_value": [5.0],
        }
    )
    bundle = DataBundle(
        main=main,
        native={"price_1h": native_price, "macro_release": native_macro},
        catalog={
            "main": DataTableSpec(
                keys=("event_time",),
                frequency="1h",
                source="assembled",
                units={"price": "USD", "macro_value": "index"},
            ),
            "price_1h": DataTableSpec(
                keys=("event_time",),
                frequency="1h",
                source="market_fixture",
                units={"price": "USD"},
            ),
            "macro_release": DataTableSpec(
                keys=("event_time",),
                frequency="event",
                source="macro_fixture",
                available_at="available_at",
                units={"macro_value": "index"},
            ),
        },
        request_log=({"source": "market_fixture", "status": 200},),
        metadata={"as_of": "2026-10-16"},
        raw_files={"market/download.json": raw},
    )

    root = tmp_path / "attempt"
    manifest = bundle.write(root)
    loaded = DataBundle.read(root)

    assert loaded.main.equals(main)
    assert loaded.native["price_1h"].equals(native_price)
    assert loaded.native["macro_release"].equals(native_macro)
    assert loaded.catalog["macro_release"].available_at == "available_at"
    assert set(loaded.eval_tables()) == {
        "data.native.price_1h",
        "data.native.macro_release",
    }
    assert manifest["native"]["price_1h"]["rows"] == 3
    assert (root / "source_coverage.parquet").exists()
    assert (root / "raw" / "market" / "download.json").read_text() == raw.read_text()
    request = json.loads((root / "request_log.jsonl").read_text())
    assert request == {"source": "market_fixture", "status": 200}
    assert loaded.raw_files == {}


def test_data_bundle_requires_catalog_for_every_formal_table() -> None:
    table = pl.DataFrame({"event_time": [1], "value": [2.0]})

    with pytest.raises(ValueError, match="catalog must describe"):
        DataBundle(
            main=table,
            native={"source": table},
            catalog={
                "main": DataTableSpec(
                    keys=("event_time",), frequency="1h", source="fixture"
                )
            },
        )


def test_review_brief_rejects_missing_evidence_and_unsafe_plot_paths() -> None:
    with pytest.raises(ValueError, match="evidence"):
        ReviewBrief(question="question", conclusion="conclusion", evidence=())

    with pytest.raises(ValueError, match="key plot"):
        ReviewBrief(
            question="question",
            conclusion="conclusion",
            evidence=("evidence",),
            key_plots=("../outside.png",),
        )


def test_feat_builder_and_eval_can_select_approved_native_tables(tmp_path) -> None:
    out = tmp_path / "run"
    data_dir = out / "data"
    DataBundle(
        main=pl.DataFrame({"event_time": [1, 2], "value": [10.0, 11.0]}),
        native={
            "source_1h": pl.DataFrame(
                {"event_time": [1, 2], "source_value": [20.0, 22.0]}
            )
        },
        catalog={
            "main": DataTableSpec(
                keys=("event_time",), frequency="1h", source="assembled"
            ),
            "source_1h": DataTableSpec(
                keys=("event_time",), frequency="1h", source="fixture"
            ),
        },
    ).write(data_dir)
    state = new_workflow(("data", "feat"), "contract-card")
    state["stages"].update(
        {"contract": "approved", "data": "approved", "feat": "ready"}
    )
    write_workflow(out, state)

    builder = tmp_path / "feat.py"
    builder.write_text(
        """
import polars as pl

FEATURE_KEYS = ("event_time",)

def build_features_from_bundle(bundle):
    return bundle.main.select(
        "event_time",
        (pl.col("value") * 2.0).alias("feature"),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plan = tmp_path / "feat_eval.py"
    plan.write_text(
        """
import polars as pl
from research_ci import CaseEvalTest, EvalPlan, EvalSpec

CASE = EvalSpec(
    "feat.case.formula",
    expr=pl.col("feature") == pl.col("event_time").replace({1: 20.0, 2: 22.0}),
    failure_policy="stop",
)

def make_plan(table, stage="feat", context=None):
    del table
    assert context.table("data.native.source_1h").height == 2
    return EvalPlan(
        stage=stage,
        baseline=(),
        library=(EvalSpec.noop(
            "feat.library.not_applicable",
            status="passed",
            reason="the fixture only verifies DataBundle access",
            library_method="library.not_applicable",
        ),),
        case=(CASE,),
        case_eval=(CaseEvalTest.from_rows(
            name="feat.case_eval.formula_has_counterexample",
            case=CASE.name,
            valid={"event_time": [1], "feature": [20.0]},
            invalid={"event_time": [1], "feature": [21.0]},
        ),),
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "feat",
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
    manifest = json.loads(
        (out / "feat" / "attempts" / "0001" / "manifest.json").read_text()
    )
    assert manifest["input_data_bundle"]["native_tables"] == ["source_1h"]
