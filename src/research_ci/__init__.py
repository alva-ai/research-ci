"""The first protocol layer of the Research CI research engine."""

from .contract import Contract, ContractError, validate_contract
from .eval import (
    CaseEvalTest,
    EvalObservation,
    EvalPlan,
    EvalRecord,
    EvalRunResult,
    EvalSpec,
    EvalStatus,
    FailurePolicy,
    OutputKind,
    Slot,
    run_eval_plan,
)

__all__ = [
    "CaseEvalTest",
    "Contract",
    "ContractError",
    "EvalObservation",
    "EvalPlan",
    "EvalRecord",
    "EvalRunResult",
    "EvalSpec",
    "FailurePolicy",
    "EvalStatus",
    "OutputKind",
    "Slot",
    "run_eval_plan",
    "validate_contract",
]
