"""Four-slot evaluation protocol, runner, floor checks, and method library."""

from .baseline import general_missing_values
from .protocol import (
    CaseEvalTest,
    EvalObservation,
    EvalPlan,
    EvalSpec,
    EvalStatus,
    FailurePolicy,
    OutputKind,
    Slot,
)
from .registry import InputKind, MethodInfo, StageName
from .runner import EvalRecord, EvalRunResult, run_eval_plan

__all__ = [
    "CaseEvalTest",
    "EvalObservation",
    "EvalPlan",
    "EvalRecord",
    "EvalRunResult",
    "EvalSpec",
    "EvalStatus",
    "FailurePolicy",
    "InputKind",
    "MethodInfo",
    "OutputKind",
    "StageName",
    "Slot",
    "general_missing_values",
    "run_eval_plan",
]
