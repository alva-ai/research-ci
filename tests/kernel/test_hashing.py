import math

import pandas as pd
import polars as pl
import pytest

from research_ci.kernel.hashing import canonical_json, hash_frame, hash_obj
from research_ci.kernel.tables import field_table

T0 = pd.Timestamp("2026-01-01", tz="UTC")


def _frame() -> pl.DataFrame:
    return field_table(
        pd.DataFrame(
            {
                "entity": ["a", "b", "a"],
                "field": ["x", "x", "y"],
                "event_time": [T0, T0, T0],
                "available_at": [T0, T0, T0 + pd.Timedelta(hours=1)],
                "value": [1.0, 2.0, float("nan")],
            }
        )
    )


def test_canonical_json_ignores_key_order() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert hash_obj({"b": 1, "a": 2}) == hash_obj({"a": 2, "b": 1})


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


def test_frame_hash_ignores_row_and_column_order() -> None:
    frame = _frame()
    shuffled = frame.reverse().select(list(reversed(frame.columns)))
    assert hash_frame(shuffled) == hash_frame(frame)


def test_frame_hash_sees_a_changed_value() -> None:
    frame = _frame()
    changed = frame.with_columns(pl.when(pl.col("entity") == "b").then(2.5).otherwise(pl.col("value")).alias("value"))
    assert hash_frame(changed) != hash_frame(frame)


def test_frame_hash_sees_a_changed_timestamp_by_one_nanosecond() -> None:
    frame = _frame()
    shifted = frame.with_columns(pl.col("available_at") + pl.duration(nanoseconds=1))
    assert hash_frame(shifted) != hash_frame(frame)


def test_frame_hash_distinguishes_null_from_nan_and_zero() -> None:
    base = pl.DataFrame({"v": [0.0]})
    assert (
        len(
            {
                hash_frame(base),
                hash_frame(pl.DataFrame({"v": [None]}, schema={"v": pl.Float64})),
                hash_frame(pl.DataFrame({"v": [math.nan]})),
            }
        )
        == 3
    )


def test_frame_hash_treats_negative_zero_as_zero() -> None:
    assert hash_frame(pl.DataFrame({"v": [-0.0]})) == hash_frame(pl.DataFrame({"v": [0.0]}))


def test_frame_hash_rejects_unsupported_dtypes() -> None:
    with pytest.raises(TypeError):
        hash_frame(pl.DataFrame({"v": [[1, 2]]}))
