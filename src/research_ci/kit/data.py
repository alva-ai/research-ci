"""Table-in/table-out helpers for external Data builders.

These functions do not download data or read and write files. Source access and
case-specific assembly remain in the external Lab.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any, Literal

import polars as pl

from .common import ToolInfo


def normalize_timestamp(
    table: pl.DataFrame,
    source: str,
    *,
    target: str = "event_time",
    time_unit: Literal["s", "ms", "us", "ns"] = "ms",
    time_zone: str = "UTC",
) -> pl.DataFrame:
    """Convert one explicit epoch column into a timezone-aware timestamp."""

    return table.with_columns(
        pl.from_epoch(source, time_unit=time_unit)
        .dt.replace_time_zone(time_zone)
        .alias(target)
    )


def deduplicate(
    table: pl.DataFrame,
    keys: str | Sequence[str],
    *,
    keep: Literal["first", "last", "any", "none"] = "last",
) -> pl.DataFrame:
    """Remove duplicate keys while preserving the input row order."""

    return table.unique(subset=keys, keep=keep, maintain_order=True)


def prefix_columns(
    table: pl.DataFrame,
    prefix: str,
    *,
    exclude: Sequence[str] = (),
) -> pl.DataFrame:
    """Prefix all columns except explicit join keys or shared identifiers."""

    excluded = set(exclude)
    return table.rename(
        {
            column: f"{prefix}{column}"
            for column in table.columns
            if column not in excluded
        }
    )


def asof_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_on: str,
    right_on: str,
    by: str | Sequence[str] | None = None,
    strategy: Literal["backward", "forward", "nearest"] = "backward",
    tolerance: str | int | float | timedelta | None = None,
    suffix: str = "_right",
) -> pl.DataFrame:
    """Join already-sorted tables without adding sorting or fill behavior."""

    return left.join_asof(
        right,
        left_on=left_on,
        right_on=right_on,
        by=by,
        strategy=strategy,
        tolerance=tolerance,
        suffix=suffix,
    )


def full_join_sources(
    tables: Sequence[pl.DataFrame],
    prefixes: Sequence[str],
    *,
    keys: str | Sequence[str],
    validate: Literal["m:m", "m:1", "1:m", "1:1"] = "m:m",
) -> pl.DataFrame:
    """Full-join source tables after explicitly namespacing non-key columns."""

    if not tables:
        raise ValueError("tables must contain at least one DataFrame")
    if len(tables) != len(prefixes):
        raise ValueError("tables and prefixes must have the same length")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("prefixes must be unique")
    if any(not prefix for prefix in prefixes):
        raise ValueError("prefixes must not be empty")
    join_keys = (keys,) if isinstance(keys, str) else tuple(keys)
    if not join_keys or any(not key for key in join_keys):
        raise ValueError("keys must contain at least one non-empty column name")
    if len(join_keys) != len(set(join_keys)):
        raise ValueError("keys must not contain duplicates")
    for table in tables:
        missing = sorted(set(join_keys) - set(table.columns))
        if missing:
            raise ValueError(f"source table is missing join keys: {missing}")

    namespaced = [
        prefix_columns(table, prefix, exclude=join_keys)
        for table, prefix in zip(tables, prefixes)
    ]
    output_columns = [
        column
        for table in namespaced
        for column in table.columns
        if column not in join_keys
    ]
    if len(output_columns) != len(set(output_columns)):
        raise ValueError("prefixes do not produce unique non-key column names")
    merged = namespaced[0]
    for table in namespaced[1:]:
        merged = merged.join(
            table,
            on=list(join_keys),
            how="full",
            coalesce=True,
            validate=validate,
        )
    return merged


def stack_entity_tables(
    tables: Mapping[Any, pl.DataFrame],
    *,
    entity_key: str,
    how: Literal["vertical", "diagonal", "diagonal_relaxed"] = "diagonal_relaxed",
) -> pl.DataFrame:
    """Stack named entity tables after adding one explicit entity column."""

    if not tables:
        raise ValueError("tables must contain at least one entity")
    if not entity_key:
        raise ValueError("entity_key must not be empty")
    frames: list[pl.DataFrame] = []
    for entity, table in tables.items():
        if entity_key in table.columns:
            raise ValueError(f"entity_key already exists in table: {entity_key!r}")
        frames.append(table.with_columns(pl.lit(entity).alias(entity_key)))
    return pl.concat(frames, how=how)


def filter_complete_cross_sections(
    table: pl.DataFrame,
    *,
    time_keys: str | Sequence[str],
    entity_key: str,
    entities: Sequence[Any],
    required_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Keep time groups with exactly one valid row for every expected entity."""

    keys = (time_keys,) if isinstance(time_keys, str) else tuple(time_keys)
    expected = tuple(entities)
    if not keys or any(not key for key in keys):
        raise ValueError("time_keys must contain at least one non-empty column name")
    if not entity_key:
        raise ValueError("entity_key must not be empty")
    if not expected:
        raise ValueError("entities must contain at least one value")
    if len(set(expected)) != len(expected):
        raise ValueError("entities must not contain duplicates")

    valid = pl.col(entity_key).is_in(expected)
    for column in required_columns:
        valid &= pl.col(column).is_not_null()
    eligible = table.filter(valid)
    complete = (
        eligible.group_by(*keys, maintain_order=True)
        .agg(
            pl.len().alias("__row_count"),
            pl.col(entity_key).n_unique().alias("__entity_count"),
        )
        .filter(
            (pl.col("__row_count") == len(expected))
            & (pl.col("__entity_count") == len(expected))
        )
        .select(*keys)
    )
    return eligible.join(complete, on=list(keys), how="inner", maintain_order="left")


TOOLS = (
    ToolInfo(
        "kit.data.normalize_timestamp",
        "data",
        "table",
        "convert an explicit epoch column to a timezone-aware timestamp",
    ),
    ToolInfo(
        "kit.data.deduplicate",
        "data",
        "table",
        "remove duplicate keys with an explicit keep policy",
    ),
    ToolInfo(
        "kit.data.prefix_columns",
        "data",
        "table",
        "namespace source columns while preserving explicit keys",
    ),
    ToolInfo(
        "kit.data.asof_join",
        "data",
        "table",
        "align already-sorted event and observation tables",
    ),
    ToolInfo(
        "kit.data.full_join_sources",
        "data",
        "table",
        "full-join explicitly namespaced source tables on shared keys",
    ),
    ToolInfo(
        "kit.data.stack_entity_tables",
        "data",
        "table",
        "stack entity tables with one explicit entity identifier",
    ),
    ToolInfo(
        "kit.data.filter_complete_cross_sections",
        "data",
        "table",
        "retain time groups containing every expected valid entity exactly once",
    ),
)

__all__ = [
    "TOOLS",
    "asof_join",
    "deduplicate",
    "filter_complete_cross_sections",
    "full_join_sources",
    "normalize_timestamp",
    "prefix_columns",
    "stack_entity_tables",
]
