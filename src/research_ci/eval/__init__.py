"""Four-slot evaluation protocol, runner, floor checks, and method library."""

from .assessment import assess_features, assess_records
from .baseline import feature_output_shape, general_missing_values
from .declarations import DataSet, FeatureSet, LibrarySelection, SelectedMethod
from .protocol import (
    CaseEvalTest,
    EvalContext,
    EvalObservation,
    EvalPlan,
    EvalSpec,
    EvalStatus,
    ExecutionMode,
    FailurePolicy,
    OutputKind,
    Slot,
)
from .registry import InputKind, MethodInfo, StageName
from .runner import EvalRecord, EvalRunResult, run_eval_plan
from .selection import LibrarySelectionError, expand_library

__all__ = [
    "CaseEvalTest",
    "DataSet",
    "FeatureSet",
    "LibrarySelection",
    "LibrarySelectionError",
    "SelectedMethod",
    "assess_records",
    "assess_features",
    "EvalContext",
    "EvalObservation",
    "EvalPlan",
    "EvalRecord",
    "EvalRunResult",
    "EvalSpec",
    "EvalStatus",
    "ExecutionMode",
    "FailurePolicy",
    "InputKind",
    "MethodInfo",
    "OutputKind",
    "StageName",
    "Slot",
    "feature_output_shape",
    "general_missing_values",
    "expand_library",
    "run_eval_plan",
]
