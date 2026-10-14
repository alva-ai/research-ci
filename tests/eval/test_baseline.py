import polars as pl

from research_ci.eval.baseline import feature_output_shape


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
