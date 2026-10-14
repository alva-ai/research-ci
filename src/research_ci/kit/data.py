"""Table-in/table-out helpers for external Data builders.

These functions do not download data or read and write files. Source access and
case-specific assembly remain in the external Lab.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Literal

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
)

__all__ = [
    "TOOLS",
    "asof_join",
    "deduplicate",
    "normalize_timestamp",
    "prefix_columns",
]
