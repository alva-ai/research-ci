import polars as pl

from research_ci import LibrarySelection, SelectedMethod, SignalSet
from research_ci.eval.baseline import (
    feature_output_shape,
    sidecar_output_shape,
    signal_disposition_coverage,
    signal_feature_universe,
    signal_lineage_coverage,
    signal_main_output_shape,
    signal_screening_artifact_coverage,
    signal_screening_plan_coverage,
)
from research_ci.kit import FeatureDisposition, SignalLineage


def test_signal_main_output_shape_requires_one_wide_row_per_time() -> None:
    signals = pl.DataFrame({"event_time": [1, 2], "signal": [0.5, -0.25]})
    wide = pl.DataFrame(
        {
            "event_time": [1, 2],
            "skhx_inst": ["xyz:SKHX", "xyz:SKHX"],
            "skhx_price": [100.0, 101.0],
            "skhy_inst": ["xyz:SKHY", "xyz:SKHY"],
            "skhy_price": [10.0, 10.5],
            "signal": [0.5, -0.25],
        }
    )

    passed = signal_main_output_shape(signals, wide, ("event_time",), ("signal",))
    invalid = signal_main_output_shape(
        signals,
        wide.drop("skhy_price"),
        ("event_time",),
        ("signal",),
    )

    assert passed.status == "passed"
    assert passed.details["layout"] == "wide"
    assert passed.details["inst_columns"] == ("skhx_inst", "skhy_inst")
    assert invalid.status == "failed"
    assert invalid.details["unmatched_inst_prefixes"] == ["skhy"]


def test_feature_output_shape_preserves_keys_and_allows_warmup_nulls() -> None:
    data = pl.DataFrame(
        {
            "event_time": [1, 2, 3],
            "source_value": [10.0, 11.0, 12.0],
        }
    )
    features = pl.DataFrame(
        {
            "event_time": [1, 2, 3],
            "rolling_feature": [None, None, 1.5],
        }
    )

    observation = feature_output_shape(data, features, ("event_time",))

    assert observation.status == "passed"
    assert observation.details["feature_columns"] == ["rolling_feature"]
    assert observation.details["missing_key_rows"] == 0
    assert observation.details["extra_key_rows"] == 0
    assert observation.details["duplicate_key_rows"] == 0


def test_feature_output_shape_rejects_changed_key_grid() -> None:
    data = pl.DataFrame({"event_time": [1, 2, 3]})
    features = pl.DataFrame(
        {
            "event_time": [1, 2, 2],
            "feature": [0.1, 0.2, 0.3],
        }
    )

    observation = feature_output_shape(data, features, ("event_time",))

    assert observation.status == "failed"
    assert observation.details["missing_key_rows"] == 1
    assert observation.details["duplicate_key_rows"] == 1


def test_sidecar_output_shape_requires_an_exact_unique_key_grid() -> None:
    source = pl.DataFrame({"event_time": [1, 2], "x": [3.0, 4.0]})
    valid = pl.DataFrame({"event_time": [1, 2], "target": [0.1, None], "block": [0, 0]})
    duplicate = pl.DataFrame(
        {"event_time": [1, 1], "target": [0.1, 0.2], "block": [0, 0]}
    )

    passed = sidecar_output_shape(
        source,
        valid,
        ("event_time",),
        ("target", "block"),
    )
    failed = sidecar_output_shape(
        source,
        duplicate,
        ("event_time",),
        ("target", "block"),
    )

    assert passed.status == "passed"
    assert failed.status == "failed"
    assert failed.details["reason_code"] == "CONTEXT_DUPLICATE_KEYS"


def test_signal_research_protocol_requires_the_complete_feature_universe() -> None:
    features = pl.DataFrame({"event_time": [1, 2], "x": [1.0, 2.0], "y": [3.0, 4.0]})
    declaration = SignalSet(
        keys=("event_time",),
        features=("x", "y"),
        candidates=("candidate",),
        signals=("signal",),
        targets={"1h": "target_1h", "6h": "target_6h"},
    )
    complete_selection = LibrarySelection(
        (
            SelectedMethod(
                "library.association.spearman_ic",
                reason="screen every feature and horizon",
                on="features",
            ),
        )
    )
    incomplete_selection = LibrarySelection(
        (
            SelectedMethod(
                "library.association.spearman_ic",
                reason="intentionally incomplete fixture",
                on=("x",),
                targets=("1h",),
            ),
        )
    )
    screening = pl.DataFrame(
        {
            "feature": ["x", "x", "y", "y"],
            "horizon": ["1h", "6h", "1h", "6h"],
        }
    )

    assert (
        signal_feature_universe(features, ("event_time",), declaration.features).status
        == "passed"
    )
    assert (
        signal_feature_universe(
            features,
            ("event_time",),
            (feature for feature in ("x", "y")),
        ).status
        == "passed"
    )
    assert (
        signal_feature_universe(features, ("event_time",), ("x",)).details[
            "reason_code"
        ]
        == "INCOMPLETE_FEATURE_UNIVERSE"
    )
    assert (
        signal_screening_plan_coverage(declaration, complete_selection).status
        == "passed"
    )
    assert (
        signal_screening_plan_coverage(declaration, incomplete_selection).details[
            "reason_code"
        ]
        == "INCOMPLETE_TARGET_SCREENING"
    )
    assert (
        signal_screening_artifact_coverage(
            screening, declaration.features, declaration.targets
        ).status
        == "passed"
    )
    assert (
        signal_screening_artifact_coverage(
            screening.filter(pl.col("feature") == "x"),
            declaration.features,
            declaration.targets,
        ).details["reason_code"]
        == "INCOMPLETE_SCREENING_ARTIFACT"
    )
    assert (
        signal_screening_artifact_coverage(
            None,
            declaration.features,
            {},
        ).status
        == "passed"
    )


def test_signal_research_protocol_requires_disposition_and_lineage() -> None:
    features = pl.DataFrame({"event_time": [1, 2], "x": [1.0, 2.0], "y": [3.0, 4.0]})
    dispositions = (
        FeatureDisposition(
            feature="x",
            action="keep_direct",
            reason="x builds the candidate",
            evidence=("screening:x:1h",),
            selected_for=("candidate",),
        ),
        FeatureDisposition(
            feature="y",
            action="reject_weak",
            reason="y is target-weak",
            evidence=("screening:y:1h",),
        ),
    )
    lineage = (
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
        ),
    )

    disposition = signal_disposition_coverage(
        features,
        ("event_time",),
        dispositions,
        ("candidate", "signal"),
    )
    lineage_result = signal_lineage_coverage(
        ("x", "y"),
        ("candidate",),
        ("signal",),
        lineage,
    )

    assert disposition.status == "passed"
    assert disposition.details["coverage"] == 1.0
    assert lineage_result.status == "passed"
    incomplete_disposition = signal_disposition_coverage(
        features,
        ("event_time",),
        dispositions[:1],
        ("candidate", "signal"),
    )
    assert incomplete_disposition.details["reason_code"] == (
        "INCOMPLETE_FEATURE_DISPOSITION"
    )
    unconnected_disposition = signal_disposition_coverage(
        features,
        ("event_time",),
        (
            FeatureDisposition(
                feature="x",
                action="keep_direct",
                reason="invalid fixture lacks a candidate connection",
                evidence=("screening:x:1h",),
            ),
            dispositions[1],
        ),
        ("candidate", "signal"),
    )
    assert unconnected_disposition.details["selected_without_outputs"] == ["x"]
    assert (
        signal_lineage_coverage(
            ("x", "y"),
            ("candidate",),
            ("signal",),
            lineage[:1],
        ).details["reason_code"]
        == "INCOMPLETE_SIGNAL_LINEAGE"
    )
