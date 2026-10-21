import math
from datetime import datetime, timedelta, timezone
from types import ModuleType

import polars as pl
import pytest

from research_ci.cli import _builder_recipe
from research_ci.kit import (
    BuildInfo,
    BuildStep,
    FeatureDisposition,
    SignalBundle,
    SignalLineage,
    catalog,
)
from research_ci.kit.data import (
    asof_join,
    deduplicate,
    filter_complete_cross_sections,
    full_join_sources,
    normalize_timestamp,
    prefix_columns,
    stack_entity_tables,
)
from research_ci.kit.feat import (
    apply_layers,
    average_true_range_rows,
    difference,
    finite_ratio_or_null,
    imbalance,
    positive_log_return,
    relative_strength_index_rows,
    relative_strength_index_time,
    relative_to_rolling_mean_rows,
    rolling_correlation_rows,
    rolling_correlation_time,
    rolling_max_time,
    rolling_mean_ratio_rows,
    rolling_min_time,
    rolling_quantile_time,
    rolling_range_position_rows,
    rolling_range_position_time,
    rolling_robust_scale_rows,
    rolling_std_ratio_rows,
    rolling_sum_rows,
    rolling_sum_time,
    rolling_zscore_rows,
    second_difference,
    signed_log1p,
    simple_return,
    tanh,
    true_range,
)
from research_ci.kit.model import ModelSpec, fit_model, make_model, score_model
from research_ci.kit.sig import (
    WalkForwardWindow,
    apply_absolute_quantile_calibration,
    apply_zscore_calibration,
    fit_absolute_quantile_calibration,
    fit_zscore_calibration,
    forward_difference,
    forward_return,
    mean_signal,
    purge_group_tail,
    split_by_time,
    split_walk_forward,
    weighted_sum,
)


def test_build_info_serializes_as_a_recipe_without_becoming_policy() -> None:
    info = BuildInfo(
        stage="feat",
        summary="build one lagged feature",
        steps=(
            BuildStep(
                tool="kit.feat.difference",
                inputs=("price",),
                outputs=("price_change",),
                parameters={"periods": 1},
            ),
        ),
    )
    module = ModuleType("builder")
    module.BUILD_INFO = info

    assert _builder_recipe(module, "feat") == {
        "declared": True,
        "stage": "feat",
        "summary": "build one lagged feature",
        "steps": [
            {
                "tool": "kit.feat.difference",
                "inputs": ["price"],
                "outputs": ["price_change"],
                "parameters": {"periods": 1},
            }
        ],
    }
    assert catalog("feat")
    assert all(tool.stage == "feat" for tool in catalog("feat"))
    assert any(tool.kind == "model" for tool in catalog("sig"))


def test_builder_recipe_is_optional_but_rejects_the_wrong_stage() -> None:
    module = ModuleType("builder")
    assert _builder_recipe(module, "data")["declared"] is False

    module.BUILD_INFO = BuildInfo(
        stage="feat",
        summary="one step",
        steps=(BuildStep(tool="lab.custom"),),
    )
    with pytest.raises(ValueError, match="expected 'data'"):
        _builder_recipe(module, "data")


def test_data_helpers_keep_source_semantics_explicit() -> None:
    raw = pl.DataFrame({"t": [0, 3_600_000], "value": [1.0, 2.0]})
    normalized = normalize_timestamp(raw, "t")
    assert normalized.get_column("event_time").to_list() == [
        datetime(1970, 1, 1, tzinfo=timezone.utc),
        datetime(1970, 1, 1, 1, tzinfo=timezone.utc),
    ]

    prefixed = prefix_columns(normalized, "left_", exclude=("event_time",))
    assert prefixed.columns == ["left_t", "left_value", "event_time"]

    duplicated = pl.DataFrame({"key": [1, 1, 2], "value": [10, 11, 20]})
    assert deduplicate(duplicated, "key").to_dict(as_series=False) == {
        "key": [1, 2],
        "value": [11, 20],
    }

    left = pl.DataFrame({"time": [1, 3], "price": [10.0, 30.0]})
    right = pl.DataFrame({"seen": [1, 2], "rate": [0.1, 0.2]})
    joined = asof_join(left, right, left_on="time", right_on="seen")
    assert joined.get_column("rate").to_list() == [0.1, 0.2]


def test_data_helpers_assemble_namespaced_sources_and_entity_tables() -> None:
    merged = full_join_sources(
        (
            pl.DataFrame({"time": [1, 2], "value": [10.0, 20.0]}),
            pl.DataFrame({"time": [2, 3], "value": [200.0, 300.0]}),
        ),
        ("price_", "volume_"),
        keys="time",
        validate="1:1",
    ).sort("time")

    assert merged.to_dict(as_series=False) == {
        "time": [1, 2, 3],
        "price_value": [10.0, 20.0, None],
        "volume_value": [None, 200.0, 300.0],
    }

    stacked = stack_entity_tables(
        {
            "alpha": pl.DataFrame({"time": [1], "x": [1]}),
            "beta": pl.DataFrame({"time": [2], "y": [2.5]}),
        },
        entity_key="entity",
    ).sort("entity")
    assert stacked.select("entity", "time", "x", "y").to_dict(as_series=False) == {
        "entity": ["alpha", "beta"],
        "time": [1, 2],
        "x": [1, None],
        "y": [None, 2.5],
    }


def test_complete_cross_sections_reject_missing_null_and_duplicate_entities() -> None:
    table = pl.DataFrame(
        {
            "time": [1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5],
            "entity": ["a", "b", "a", "a", "b", "b", "a", "b", "a", "b", "c"],
            "value": [1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, None, 1.0, 2.0, 9.0],
        }
    )

    complete = filter_complete_cross_sections(
        table,
        time_keys="time",
        entity_key="entity",
        entities=("a", "b"),
        required_columns=("value",),
    )

    assert complete.select("time", "entity").to_dict(as_series=False) == {
        "time": [1, 1, 5, 5],
        "entity": ["a", "b", "a", "b"],
    }


def test_feat_helpers_are_name_free_exprs_and_layers_are_explicit() -> None:
    source = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    layers = (
        ((pl.col("x") * 2).alias("twice"),),
        (
            difference(pl.col("twice"), periods=1).alias("change"),
            rolling_zscore_rows(pl.col("x"), window_size=3, min_samples=3).alias(
                "zscore"
            ),
            rolling_robust_scale_rows(pl.col("x"), window_size=4, min_samples=4).alias(
                "robust"
            ),
        ),
    )

    result = apply_layers(source, layers)

    assert result.columns == ["x", "twice", "change", "zscore", "robust"]
    assert result.get_column("change").to_list() == [None, 2.0, 2.0, 2.0]
    assert result.get_column("zscore").null_count() == 2
    assert result.get_column("zscore")[2] == pytest.approx(1.0)
    assert result.get_column("robust").null_count() == 3
    assert result.get_column("robust")[3] == pytest.approx(1.0)


def test_tanh_is_bounded_and_requires_an_explicit_positive_scale() -> None:
    source = pl.DataFrame({"value": [-2.0, 0.0, 2.0]})

    result = source.select(tanh(pl.col("value"), scale=2.0).alias("bounded"))

    assert result.get_column("bounded").to_list() == pytest.approx(
        [-0.7615941559557649, 0.0, 0.7615941559557649]
    )
    with pytest.raises(ValueError, match="scale must be positive"):
        tanh(pl.col("value"), scale=0.0)


def test_market_exprs_preserve_undefined_values_for_the_lab_to_treat() -> None:
    source = pl.DataFrame(
        {
            "x": [1.0, 2.0, 4.0, 8.0],
            "buy": [3.0, 0.0, 4.0, 1.0],
            "sell": [1.0, 0.0, 2.0, 3.0],
            "high": [11.0, 13.0, 12.0, 15.0],
            "low": [9.0, 10.0, 9.0, 12.0],
            "close": [10.0, 12.0, 10.0, 14.0],
        }
    )

    result = source.select(
        simple_return(pl.col("x")).alias("return"),
        second_difference(pl.col("x")).alias("second_difference"),
        signed_log1p(pl.col("sell") - pl.col("buy")).alias("signed_log"),
        imbalance(pl.col("buy"), pl.col("sell")).alias("imbalance"),
        true_range(pl.col("high"), pl.col("low"), pl.col("close")).alias("true_range"),
    )

    assert result.get_column("return").to_list() == [None, 1.0, 1.0, 1.0]
    assert result.get_column("second_difference").to_list() == [
        None,
        None,
        1.0,
        2.0,
    ]
    assert result.get_column("signed_log").to_list() == pytest.approx(
        [-math.log(3.0), 0.0, -math.log(3.0), math.log(3.0)]
    )
    assert result.get_column("imbalance")[0] == pytest.approx(0.5)
    assert math.isnan(result.get_column("imbalance")[1])
    assert result.get_column("true_range").to_list() == [2.0, 3.0, 3.0, 5.0]


def test_guarded_ratio_and_log_return_make_invalid_policy_explicit() -> None:
    source = pl.DataFrame(
        {
            "numerator": [1.0, 2.0, float("inf"), 4.0],
            "denominator": [2.0, 0.0, 2.0, float("nan")],
            "price": [1.0, 2.0, 0.0, 4.0],
        }
    )

    result = source.select(
        finite_ratio_or_null(pl.col("numerator"), pl.col("denominator")).alias("ratio"),
        positive_log_return(pl.col("price")).alias("log_return"),
    )

    assert result.get_column("ratio").to_list() == [0.5, None, None, None]
    assert result.get_column("log_return").to_list() == [
        None,
        pytest.approx(math.log(2.0)),
        None,
        None,
    ]


def test_row_clock_helpers_keep_windows_and_constant_ranges_explicit() -> None:
    source = pl.DataFrame(
        {
            "x": [1.0, 2.0, 4.0, 8.0],
            "y": [2.0, 4.0, 1.0, 3.0],
        }
    )

    result = source.select(
        rolling_sum_rows(pl.col("x"), window_size=3, min_samples=2).alias("sum"),
        rolling_range_position_rows(pl.col("x"), window_size=3, min_samples=3).alias(
            "position"
        ),
        rolling_correlation_rows(
            pl.col("x"), pl.col("y"), window_size=3, min_samples=3
        ).alias("correlation"),
    )

    assert result.get_column("sum").to_list() == [None, 3.0, 7.0, 14.0]
    assert result.get_column("position").to_list() == [None, None, 1.0, 1.0]
    assert result.get_column("correlation")[2] == pytest.approx(-0.5)
    assert result.get_column("correlation")[3] == pytest.approx(-1.0 / 7.0)


def test_composite_row_clock_helpers_keep_parameters_visible() -> None:
    source = pl.DataFrame(
        {
            "x": [1.0, 2.0, 4.0, 8.0],
            "oscillator": [1.0, 2.0, 1.0, 2.0],
            "flat": [1.0, 1.0, 1.0, 1.0],
            "high": [11.0, 13.0, 12.0, 15.0],
            "low": [9.0, 10.0, 9.0, 12.0],
            "close": [10.0, 12.0, 10.0, 14.0],
        }
    )

    result = source.select(
        relative_to_rolling_mean_rows(pl.col("x"), window_size=2, min_samples=2).alias(
            "relative_mean"
        ),
        rolling_mean_ratio_rows(
            pl.col("x"),
            short_window_size=1,
            long_window_size=2,
            short_min_samples=1,
            long_min_samples=2,
        ).alias("mean_ratio"),
        rolling_std_ratio_rows(
            pl.col("x"),
            short_window_size=2,
            long_window_size=3,
            short_min_samples=2,
            long_min_samples=3,
            ddof=0,
        ).alias("std_ratio"),
        average_true_range_rows(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            window_size=2,
            min_samples=2,
        ).alias("atr"),
        relative_strength_index_rows(
            pl.col("oscillator"), window_size=2, min_samples=2
        ).alias("rsi"),
        relative_strength_index_rows(
            pl.col("flat"), window_size=1, min_samples=1
        ).alias("flat_raw"),
        relative_strength_index_rows(
            pl.col("flat"), window_size=1, min_samples=1, flat_value=50.0
        ).alias("flat_explicit"),
    )

    expected_relative = [None, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0]
    assert result.get_column("relative_mean").to_list() == pytest.approx(
        expected_relative
    )
    assert result.get_column("mean_ratio").to_list() == pytest.approx(expected_relative)
    assert result.get_column("std_ratio").to_list() == pytest.approx(
        [None, None, 3.0 / math.sqrt(14.0), 3.0 / math.sqrt(14.0)]
    )
    assert result.get_column("atr").to_list() == [None, 2.5, 3.0, 4.0]
    assert result.get_column("rsi").to_list() == [None, None, 50.0, 50.0]
    assert math.isnan(result.get_column("flat_raw")[1])
    assert result.get_column("flat_explicit").to_list() == [
        None,
        50.0,
        50.0,
        50.0,
    ]


def test_duration_clock_helpers_respect_right_closed_elapsed_windows() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    source = pl.DataFrame(
        {
            "event_time": [
                start,
                start + timedelta(seconds=1),
                start + timedelta(seconds=3),
                start + timedelta(seconds=4),
            ],
            "x": [1.0, 2.0, 4.0, 8.0],
            "oscillator": [1.0, 2.0, 1.0, 2.0],
            "y": [2.0, 4.0, 1.0, 3.0],
        }
    )
    options = {
        "by": pl.col("event_time"),
        "window_size": "2s",
        "min_samples": 1,
    }

    result = source.select(
        rolling_sum_time(pl.col("x"), **options).alias("sum"),
        rolling_min_time(pl.col("x"), **options).alias("minimum"),
        rolling_max_time(pl.col("x"), **options).alias("maximum"),
        rolling_quantile_time(pl.col("x"), quantile=0.5, **options).alias("median"),
        rolling_range_position_time(pl.col("x"), **options).alias("position"),
        rolling_correlation_time(
            pl.col("oscillator"),
            pl.col("y"),
            by=pl.col("event_time"),
            window_size="4s",
            min_samples=3,
        ).alias("correlation"),
        relative_strength_index_time(
            pl.col("oscillator"),
            by=pl.col("event_time"),
            window_size="4s",
            min_samples=2,
        ).alias("rsi"),
    )

    assert result.get_column("sum").to_list() == [1.0, 3.0, 4.0, 12.0]
    assert result.get_column("minimum").to_list() == [1.0, 1.0, 4.0, 4.0]
    assert result.get_column("maximum").to_list() == [1.0, 2.0, 4.0, 8.0]
    assert result.get_column("median").to_list() == [1.0, 1.5, 4.0, 6.0]
    assert math.isnan(result.get_column("position")[0])
    assert result.get_column("position")[1] == pytest.approx(1.0)
    assert math.isnan(result.get_column("position")[2])
    assert result.get_column("position")[3] == pytest.approx(1.0)
    assert result.get_column("correlation").to_list() == pytest.approx(
        [None, None, 0.944911182523068, 0.944911182523068]
    )
    assert result.get_column("rsi").to_list() == pytest.approx(
        [None, None, 50.0, 200.0 / 3.0]
    )


def test_sig_helpers_keep_targets_sidecars_and_time_splits_explicit() -> None:
    feature_table = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4, 5],
            "price": [100.0, 110.0, 121.0, 120.0, 125.0],
            "x": [1.0, 2.0, None, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    main = feature_table.select(
        "event_time",
        weighted_sum((pl.col("x"), pl.col("y")), (0.5, 0.5)).alias("signal"),
        mean_signal((pl.col("x"), pl.col("y"))).alias("strict_mean"),
    )
    target = feature_table.select(
        "event_time",
        forward_return(pl.col("price")).alias("return_1h"),
        forward_difference(pl.col("x"), periods=2).alias("change_2h"),
    )
    bundle = SignalBundle(
        main=main,
        candidates=feature_table.select("event_time", pl.col("x").alias("candidate")),
        dispositions=(
            FeatureDisposition(
                feature="x",
                action="keep_direct",
                reason="used by the direct candidate",
                evidence=("screening:x:1h",),
                selected_for=("candidate",),
            ),
            FeatureDisposition(
                feature="y",
                action="reject_weak",
                reason="the example does not select y",
                evidence=("screening:y:1h",),
            ),
            FeatureDisposition(
                feature="price",
                action="keep_as_context",
                reason="price defines the explicit return target",
                evidence=("target:return_1h",),
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
                method="weighted_sum",
                inputs=("x", "y"),
                fit_policy="none",
            ),
            SignalLineage(
                signal="strict_mean",
                role="final",
                method="mean_signal",
                inputs=("x", "y"),
                fit_policy="none",
            ),
        ),
        sidecars={"label": target},
        research={
            "screening": pl.DataFrame(
                {
                    "feature": ["x", "y", "price"],
                    "horizon": ["1h", "1h", "1h"],
                }
            )
        },
        metadata={"horizon": "1h"},
    )
    split = split_by_time(
        feature_table,
        time_key="event_time",
        train_end=3,
        validation_end=5,
    )

    assert bundle.main.columns == ["event_time", "signal", "strict_mean"]
    assert bundle.sidecars["label"].get_column("return_1h")[0] == pytest.approx(0.1)
    assert bundle.sidecars["label"].get_column("return_1h")[-1] is None
    assert bundle.sidecars["label"].get_column("change_2h").to_list() == [
        None,
        2.0,
        None,
        None,
        None,
    ]
    assert bundle.main.get_column("strict_mean")[2] is None
    assert (split.train.height, split.validation.height, split.test.height) == (2, 2, 1)


def test_walk_forward_splits_use_declared_half_open_boundaries() -> None:
    table = pl.DataFrame({"time": list(range(12)), "value": list(range(12))})
    window = WalkForwardWindow(
        name="fold_1",
        train_start=1,
        train_end=5,
        calibration_end=7,
        evaluation_end=10,
    )

    (fold,) = split_walk_forward(table, time_key="time", windows=(window,))

    assert fold.window == window
    assert fold.train.get_column("time").to_list() == [1, 2, 3, 4]
    assert fold.calibration.get_column("time").to_list() == [5, 6]
    assert fold.evaluation.get_column("time").to_list() == [7, 8, 9]
    with pytest.raises(ValueError, match="train_start < train_end"):
        WalkForwardWindow(
            name="invalid",
            train_start=0,
            train_end=5,
            calibration_end=5,
            evaluation_end=10,
        )


def test_group_tail_purge_uses_explicit_order_and_preserves_input_order() -> None:
    table = pl.DataFrame(
        {
            "entity": ["a", "a", "a", "b", "b"],
            "time": [2, 1, 3, 2, 1],
            "value": [20, 10, 30, 20, 10],
        }
    )

    purged = purge_group_tail(
        table,
        group_keys="entity",
        order_keys="time",
        rows=1,
    )

    assert purged.to_dict(as_series=False) == {
        "entity": ["a", "a", "b"],
        "time": [2, 1, 1],
        "value": [20, 10, 10],
    }
    assert purge_group_tail(
        table,
        group_keys="entity",
        order_keys="time",
        rows=10,
    ).is_empty()
    duplicates = pl.DataFrame({"entity": ["a", "a"], "time": [1, 1], "value": [1, 2]})
    with pytest.raises(ValueError, match="uniquely order"):
        purge_group_tail(
            duplicates,
            group_keys="entity",
            order_keys="time",
            rows=1,
        )


def test_calibration_state_is_fitted_once_and_reused_on_evaluation() -> None:
    calibration = pl.DataFrame(
        {
            "entity": ["a", "a", "b", "b"],
            "score": [1.0, 3.0, 10.0, 14.0],
        }
    )
    evaluation = pl.DataFrame(
        {"entity": ["a", "b", "unseen"], "score": [4.0, 8.0, 1_000.0]}
    )

    zscore = fit_zscore_calibration(
        calibration,
        value_column="score",
        group_keys="entity",
        ddof=0,
    )
    scored = apply_zscore_calibration(
        evaluation,
        zscore,
        output_column="score_z",
    )
    assert scored.get_column("score_z").to_list() == [2.0, -2.0, None]
    assert zscore.statistics.sort("entity").to_dict(as_series=False) == {
        "entity": ["a", "b"],
        "mean": [2.0, 12.0],
        "scale": [1.0, 2.0],
    }

    absolute = fit_absolute_quantile_calibration(
        calibration,
        value_column="score",
        quantile=0.5,
        group_keys="entity",
    )
    thresholded = apply_absolute_quantile_calibration(
        evaluation,
        absolute,
        output_column="trade_threshold",
    )
    assert thresholded.get_column("trade_threshold").to_list() == [
        2.0,
        12.0,
        None,
    ]

    with pytest.raises(ValueError, match="finite positive scale"):
        fit_zscore_calibration(
            pl.DataFrame({"score": [1.0, 1.0]}),
            value_column="score",
        )


def test_model_adapter_keeps_preprocessing_and_scoring_choices_explicit() -> None:
    regression = ModelSpec(
        task="regression",
        family="ridge",
        feature_columns=("x",),
        target_column="target",
        scaling="standard",
        imputation="median",
        seed=7,
        score_method="predict",
        parameters={"alpha": 0.0},
    )
    train = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "target": [1.0, 3.0, 5.0, 7.0]})

    fitted = fit_model(regression, train)
    scored = score_model(
        fitted,
        pl.DataFrame({"x": [4.0]}),
        output_column="prediction",
    )

    assert fitted.training_rows == 4
    assert list(fitted.estimator.named_steps) == ["imputer", "scaler", "model"]
    assert scored.get_column("prediction")[0] == pytest.approx(9.0)
    assert regression.as_dict()["feature_columns"] == ["x"]

    classification = ModelSpec(
        task="classification",
        family="logistic",
        feature_columns=("x",),
        target_column="direction",
        scaling="standard",
        imputation="none",
        seed=11,
        score_method="predict_proba",
        class_label="up",
        parameters={"max_iter": 500},
    )
    classifier = fit_model(
        classification,
        pl.DataFrame(
            {
                "x": [-2.0, -1.0, 1.0, 2.0],
                "direction": ["down", "down", "up", "up"],
            }
        ),
    )
    probabilities = score_model(
        classifier,
        pl.DataFrame({"x": [-3.0, 3.0]}),
        output_column="up_probability",
    ).get_column("up_probability")
    assert probabilities[0] < 0.5 < probabilities[1]


@pytest.mark.parametrize(
    ("task", "family", "score_method", "class_label", "estimator_name"),
    [
        ("regression", "ridge", "predict", None, "Ridge"),
        ("classification", "logistic", "predict_proba", 1, "LogisticRegression"),
        (
            "regression",
            "hist_gradient_boosting_regressor",
            "predict",
            None,
            "HistGradientBoostingRegressor",
        ),
        (
            "classification",
            "hist_gradient_boosting_classifier",
            "predict_proba",
            1,
            "HistGradientBoostingClassifier",
        ),
    ],
)
def test_model_factory_supports_the_declared_families(
    task: str,
    family: str,
    score_method: str,
    class_label: int | None,
    estimator_name: str,
) -> None:
    spec = ModelSpec(
        task=task,
        family=family,
        feature_columns=("x",),
        target_column="target",
        scaling="none",
        imputation="none",
        seed=19,
        score_method=score_method,
        class_label=class_label,
    )

    pipeline = make_model(spec)

    assert type(pipeline.named_steps["model"]).__name__ == estimator_name
    assert pipeline.named_steps["model"].random_state == 19


def test_model_spec_rejects_implicit_or_conflicting_choices() -> None:
    with pytest.raises(ValueError, match="does not support task"):
        ModelSpec(
            task="regression",
            family="logistic",
            feature_columns=("x",),
            target_column="target",
            scaling="none",
            imputation="none",
            seed=1,
            score_method="predict",
        )
    with pytest.raises(ValueError, match="ModelSpec.seed"):
        ModelSpec(
            task="regression",
            family="ridge",
            feature_columns=("x",),
            target_column="target",
            scaling="none",
            imputation="none",
            seed=1,
            score_method="predict",
            parameters={"random_state": 9},
        )
    with pytest.raises(ValueError, match="only by logistic"):
        ModelSpec(
            task="classification",
            family="hist_gradient_boosting_classifier",
            feature_columns=("x",),
            target_column="target",
            scaling="none",
            imputation="none",
            seed=1,
            score_method="decision_function",
            class_label=1,
        )
    classification = ModelSpec(
        task="classification",
        family="logistic",
        feature_columns=("x",),
        target_column="target",
        scaling="none",
        imputation="none",
        seed=1,
        score_method="predict_proba",
        class_label=1,
    )
    with pytest.raises(ValueError, match="at least two classes"):
        fit_model(
            classification,
            pl.DataFrame({"x": [1.0, 2.0], "target": [1, 1]}),
        )
