import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl

from research_ci.kernel.time.clock import TIME

_TAGS: list[tuple[pl.DataType, str]] = [
    (TIME, "ts_ns_utc"),
    (pl.Duration("ns"), "dur_ns"),
    (pl.Int64(), "i64"),
    (pl.Float64(), "f64"),
    (pl.String(), "str"),
    (pl.Boolean(), "bool"),
]


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def hash_frame(df: pl.DataFrame) -> str:
    """Content hash that ignores row and column order and does not depend on the polars version."""
    columns = sorted(df.columns)
    tags = {c: _tag(df.schema[c]) for c in columns}
    zeroed = [pl.when(pl.col(c) == 0.0).then(0.0).otherwise(pl.col(c)).alias(c) for c in columns if tags[c] == "f64"]
    ordered = df.select(columns).with_columns(zeroed).sort(columns, nulls_last=True)
    digest = hashlib.sha256(canonical_json({"rows": ordered.height, "columns": [[c, tags[c]] for c in columns]}))
    for column in columns:
        series = ordered[column]
        digest.update(np.packbits(series.is_null().to_numpy()).tobytes())
        digest.update(_payload(series, tags[column]))
    return digest.hexdigest()


def hash_tables(tables: Mapping[str, pl.DataFrame]) -> str:
    """Content hash of named tables, such as an idea's sources: each table's ``hash_frame`` by name."""
    return hash_obj({name: hash_frame(table) for name, table in tables.items()})


def _tag(dtype: pl.DataType) -> str:
    for known, tag in _TAGS:
        if dtype == known:
            return tag
    raise TypeError(f"cannot hash a column of type {dtype}")


def _payload(series: pl.Series, tag: str) -> bytes:
    if tag == "str":
        text = series.fill_null("")
        lengths = text.str.len_bytes().cast(pl.Int64).to_numpy().astype("<i8").tobytes()
        return lengths + "".join(text.to_list()).encode()
    if tag == "f64":
        values = series.fill_null(0.0).to_numpy().astype("<f8")
        values[np.isnan(values)] = np.nan
        return values.tobytes()
    if tag == "bool":
        return series.fill_null(False).to_numpy().astype("u1").tobytes()
    return series.cast(pl.Int64).fill_null(0).to_numpy().astype("<i8").tobytes()
