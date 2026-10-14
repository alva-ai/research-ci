"""Four-slot evaluation protocol, runner, floor checks, and method library."""

from .assessment import assess_features, assess_records, assess_signal_evidence
from .baseline import feature_output_shape, general_missing_values
from .declarations import (
    DataSet,
    FeatureSet,
    LibrarySelection,
    SelectedMethod,
    SignalSet,
)
from .menu import (
    MethodMenuEntry,
    StageMenu,
    compile_stage_menu,
    require_selectable_method,
)
from .policies import stage_policy
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
from .registry import (
    FitPolicy,
    InputKind,
    InputRole,
    MethodInfo,
    MethodPurpose,
    MethodRequirement,
    StageName,
    StagePolicy,
)
from .runner import EvalRecord, EvalRunResult, run_eval_plan
from .selection import LibrarySelectionError, expand_library

__all__ = [
    "CaseEvalTest",
    "DataSet",
    "FeatureSet",
    "LibrarySelection",
    "LibrarySelectionError",
    "MethodMenuEntry",
    "SelectedMethod",
    "SignalSet",
    "StageMenu",
    "assess_records",
    "assess_features",
    "assess_signal_evidence",
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
    "InputRole",
    "FitPolicy",
    "MethodInfo",
    "MethodPurpose",
    "MethodRequirement",
    "OutputKind",
    "StageName",
    "StagePolicy",
    "Slot",
    "feature_output_shape",
    "general_missing_values",
    "compile_stage_menu",
    "expand_library",
    "require_selectable_method",
    "run_eval_plan",
    "stage_policy",
]
