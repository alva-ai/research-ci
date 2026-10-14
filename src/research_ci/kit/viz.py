"""Small, stage-agnostic visualization helpers for external Labs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class TimeSeriesPanel:
    """Declare one panel drawn from aligned columns in a wide table."""

    title: str
    columns: tuple[str, ...]
    ylabel: str = ""
    rebase: bool = False
    zero_line: bool = False

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("TimeSeriesPanel.title must not be empty")
        if not self.columns or any(not column.strip() for column in self.columns):
            raise ValueError("TimeSeriesPanel.columns needs named columns")


def _plot_modules() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError(
            "visualization requires the optional research-ci[visualization] extra"
        ) from exc
    return plt, sns


def _require_columns(table: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"visualization table is missing columns: {missing}")


def _rebase(values: np.ndarray) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(values))
    if not finite.size:
        return values
    base = values[finite[0]]
    if base == 0:
        raise ValueError("cannot rebase a series whose first finite value is zero")
    return values / base * 100.0


def save_time_series_panels(
    table: pl.DataFrame,
    *,
    x: str,
    panels: Sequence[TimeSeriesPanel],
    path: str | Path,
    title: str = "",
    width: float = 14.0,
    panel_height: float = 3.0,
) -> Path:
    """Save aligned time-series panels from one wide Polars table."""

    if not panels:
        raise ValueError("panels must contain at least one TimeSeriesPanel")
    columns = (x, *(column for panel in panels for column in panel.columns))
    _require_columns(table, columns)
    if table.is_empty():
        raise ValueError("visualization table must not be empty")

    plt, _ = _plot_modules()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(width, panel_height * len(panels)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    x_values = table.get_column(x).to_list()
    for axis, panel in zip(axes[:, 0], panels):
        for column in panel.columns:
            series = table.get_column(column)
            if not series.dtype.is_numeric():
                raise ValueError(f"time-series column {column!r} must be numeric")
            values = series.cast(pl.Float64).to_numpy()
            if panel.rebase:
                values = _rebase(values)
            axis.plot(x_values, values, linewidth=1.0, label=column)
        if panel.zero_line:
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
        axis.set_title(panel.title, loc="left", fontsize=10)
        axis.set_ylabel(panel.ylabel)
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8, ncols=min(3, len(panel.columns)))
    axes[-1, 0].set_xlabel(x)
    if title:
        figure.suptitle(title, fontsize=13)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output


def save_heatmap(
    table: pl.DataFrame,
    *,
    row: str,
    column: str,
    value: str,
    path: str | Path,
    title: str = "",
    row_order: Sequence[Any] | None = None,
    column_order: Sequence[Any] | None = None,
    center: float | None = None,
    annotate: bool = True,
    value_format: str = ".3f",
) -> Path:
    """Save a heatmap from a unique row/column/value Polars table."""

    _require_columns(table, (row, column, value))
    if table.is_empty():
        raise ValueError("heatmap table must not be empty")
    cells = table.select(row, column, value)
    if cells.select(row, column).n_unique() != cells.height:
        raise ValueError("heatmap row and column pairs must be unique")
    if not cells.schema[value].is_numeric():
        raise ValueError(f"heatmap value column {value!r} must be numeric")

    rows = list(row_order) if row_order is not None else cells[row].unique().to_list()
    columns = (
        list(column_order)
        if column_order is not None
        else cells[column].unique().to_list()
    )
    observed_rows = set(cells[row].to_list())
    observed_columns = set(cells[column].to_list())
    if observed_rows - set(rows) or observed_columns - set(columns):
        raise ValueError("heatmap order omits observed row or column values")

    row_index = {name: index for index, name in enumerate(rows)}
    column_index = {name: index for index, name in enumerate(columns)}
    matrix = np.full((len(rows), len(columns)), np.nan, dtype=float)
    for row_value, column_value, metric in cells.iter_rows():
        if metric is not None:
            matrix[row_index[row_value], column_index[column_value]] = float(metric)

    plt, sns = _plot_modules()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(
        figsize=(max(7.0, len(columns) * 1.5), max(4.0, len(rows) * 0.48)),
        constrained_layout=True,
    )
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="vlag" if center is not None else "viridis",
        center=center,
        annot=annotate,
        fmt=value_format,
        mask=np.isnan(matrix),
        xticklabels=[str(item) for item in columns],
        yticklabels=[str(item) for item in rows],
        linewidths=0.35,
    )
    axis.set_xlabel(column)
    axis.set_ylabel(row)
    axis.set_title(title, loc="left", fontsize=12)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output


__all__ = ["TimeSeriesPanel", "save_heatmap", "save_time_series_panels"]
