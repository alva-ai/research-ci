"""The first protocol layer of the Research CI research engine."""

from .contract import Contract, ContractError, validate_contract
from .engine import EvalRecord, EvalRunResult, run_eval_plan
from .evals import (
    CaseEvalTest,
    EvalObservation,
    EvalPlan,
    EvalSpec,
    EvalStatus,
    FailurePolicy,
    OutputKind,
    Slot,
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
