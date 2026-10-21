"""Optional construction and visualization helpers for external Labs."""

from __future__ import annotations

from .algo import TOOLS as ALGO_TOOLS
from .algo import AlgoBundle, BacktestResult, bounded_score, dollar_neutral_pair
from .common import BuildInfo, BuildStep, StageName, ToolInfo, ToolKind
from .data import TOOLS as DATA_TOOLS
from .feat import TOOLS as FEAT_TOOLS
from .model import TOOLS as MODEL_TOOLS
from .model import FittedModel, ModelSpec, fit_model, make_model, score_model
from .sig import TOOLS as SIG_TOOLS
from .sig import (
    AbsoluteQuantileCalibration,
    FeatureDisposition,
    SignalBundle,
    SignalLineage,
    TimeSplit,
    WalkForwardSplit,
    WalkForwardWindow,
    ZScoreCalibration,
)
from .viz import TimeSeriesPanel, save_heatmap, save_time_series_panels

CATALOG = (*DATA_TOOLS, *FEAT_TOOLS, *SIG_TOOLS, *MODEL_TOOLS, *ALGO_TOOLS)


def catalog(stage: StageName | None = None) -> tuple[ToolInfo, ...]:
    """List optional helpers; this catalog is advisory rather than a gate."""

    if stage is None:
        return CATALOG
    return tuple(tool for tool in CATALOG if tool.stage == stage)


__all__ = [
    "BuildInfo",
    "BuildStep",
    "CATALOG",
    "AbsoluteQuantileCalibration",
    "AlgoBundle",
    "BacktestResult",
    "FeatureDisposition",
    "FittedModel",
    "ModelSpec",
    "SignalBundle",
    "SignalLineage",
    "StageName",
    "TimeSplit",
    "TimeSeriesPanel",
    "ToolInfo",
    "ToolKind",
    "WalkForwardSplit",
    "WalkForwardWindow",
    "ZScoreCalibration",
    "catalog",
    "bounded_score",
    "dollar_neutral_pair",
    "fit_model",
    "make_model",
    "save_heatmap",
    "save_time_series_panels",
    "score_model",
]
