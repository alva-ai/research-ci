from datetime import datetime, timezone
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
    normalize_timestamp,
    prefix_columns,
)
from research_ci.kit.feat import (
    apply_layers,
    difference,
    rolling_robust_scale_rows,
    rolling_zscore_rows,
    tanh,
)
from research_ci.kit.sig import (
    forward_difference,
    forward_return,
    mean_signal,
    split_by_time,
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
        candidates=feature_table.select(
            "event_time", pl.col("x").alias("candidate")
        ),
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
