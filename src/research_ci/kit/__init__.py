"""Optional construction and visualization helpers for external Labs."""

from __future__ import annotations

from .common import BuildInfo, BuildStep, StageName, ToolInfo, ToolKind
from .data import TOOLS as DATA_TOOLS
from .feat import TOOLS as FEAT_TOOLS
from .sig import TOOLS as SIG_TOOLS
from .sig import FeatureDisposition, SignalBundle, SignalLineage, TimeSplit
from .viz import TimeSeriesPanel, save_heatmap, save_time_series_panels

CATALOG = (*DATA_TOOLS, *FEAT_TOOLS, *SIG_TOOLS)


def catalog(stage: StageName | None = None) -> tuple[ToolInfo, ...]:
    """List optional helpers; this catalog is advisory rather than a gate."""

    if stage is None:
        return CATALOG
    return tuple(tool for tool in CATALOG if tool.stage == stage)


__all__ = [
    "BuildInfo",
    "BuildStep",
    "CATALOG",
    "FeatureDisposition",
    "SignalBundle",
    "SignalLineage",
    "StageName",
    "TimeSplit",
    "TimeSeriesPanel",
    "ToolInfo",
    "ToolKind",
    "catalog",
    "save_heatmap",
    "save_time_series_panels",
]
