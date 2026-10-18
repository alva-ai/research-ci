"""Execution of four-slot eval plans without mutating main tables."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .baseline import general_missing_values
from .protocol import (
    CaseEvalTest,
    EvalContext,
    EvalObservation,
    EvalPlan,
    EvalSpec,
    FailurePolicy,
    Slot,
)


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """Serializable sidecar record for one eval or case-eval test."""

    stage: str
    slot: str
    name: str
    status: str
    value: Any = None
    output_kind: str = "none"
    failure_policy: FailurePolicy = "record"
    details: dict[str, Any] | None = None
    execution_mode: str = "custom_runner"
    selection_reason: str = ""
    library_method: str = ""
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "slot": self.slot,
            "name": self.name,
            "status": self.status,
            "value": _json_value(self.value),
            "output_kind": self.output_kind,
            "failure_policy": self.failure_policy,
            "details": _json_value(self.details or {}),
            "execution_mode": self.execution_mode,
            "selection_reason": self.selection_reason,
            "library_method": self.library_method,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    """Run output; ``table`` is the original main table object."""

    table: pl.DataFrame
    records: tuple[EvalRecord, ...]

    @property
    def passed(self) -> bool:
        return not self.halted

    @property
    def halted(self) -> bool:
        return any(
            record.failure_policy == "stop" and record.status in {"failed", "error"}
            for record in self.records
        )

    def write_report(self, path: str | Path) -> None:
        """Write only eval records; never rewrite the main table."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.as_dict(), sort_keys=True))
                handle.write("\n")


def run_eval_plan(
    table: pl.DataFrame,
    plan: EvalPlan,
    context: EvalContext | None = None,
) -> EvalRunResult:
    """Run all four slots against a read-only table and explicit sidecars."""

    plan.validate()
    context = context or EvalContext()
    records: list[EvalRecord] = []
    by_name = {spec.name: spec for spec in plan.case}

    # This is an engine-owned floor check. It is deliberately a sidecar
    # observation: the engine reports malformed/missing values without
    # inventing an eval column in the caller's main table.
    records.append(
        _record(
            plan.stage,
            Slot.BASELINE.value,
            "engine.baseline.general_missing_values",
            "record",
            general_missing_values(table),
            execution_mode="polars_select",
            description=(
                "Report null, NaN, infinite, and string-sentinel values across "
                "the complete table."
            ),
        )
    )

    for slot in (Slot.BASELINE, Slot.LIBRARY, Slot.CASE):
        specs = getattr(plan, slot.value)
        for spec in specs:
            records.append(_run_spec(table, plan.stage, slot, spec, context))

    for test in plan.case_eval:
        records.append(_run_case_test(table, plan.stage, by_name, test, context))

    return EvalRunResult(table=table, records=tuple(records))


def _run_spec(
    table: pl.DataFrame,
    stage: str,
    slot: Slot,
    spec: EvalSpec,
    context: EvalContext,
) -> EvalRecord:
    try:
        raw = _evaluate_value(table, spec, context)
        observation = _normalize(raw, spec)
    except Exception as exc:  # noqa: BLE001 - an eval must leave an audit record
        observation = EvalObservation(
            status="error",
            output_kind="none",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
    return _record(
        stage,
        slot.value,
        spec.name,
        spec.failure_policy,
        observation,
        execution_mode=_execution_mode(spec),
        selection_reason=spec.selection_reason,
        library_method=spec.library_method,
        description=spec.description,
    )


def _run_case_test(
    main_table: pl.DataFrame,
    stage: str,
    case_rules: dict[str, EvalSpec],
    test: CaseEvalTest,
    context: EvalContext,
) -> EvalRecord:
    spec = case_rules[test.case]
    try:
        valid = _normalize(
            _evaluate_value(_fixture(main_table, test.valid), spec, context), spec
        )
        invalid = _normalize(
            _evaluate_value(_fixture(main_table, test.invalid), spec, context), spec
        )
        passed = valid.status == "passed" and invalid.status == "failed"
        observation = EvalObservation(
            status="passed" if passed else "failed",
            value=passed,
            output_kind="summary",
            details={
                "case": test.case,
                "valid_status": valid.status,
                "invalid_status": invalid.status,
            },
        )
    except Exception as exc:  # noqa: BLE001 - preserve the failed test record
        observation = EvalObservation(
            status="error",
            output_kind="none",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
    return _record(
        stage,
        Slot.CASE_EVAL.value,
        test.name,
        test.failure_policy,
        observation,
        execution_mode=_execution_mode(spec),
        selection_reason=spec.selection_reason,
        description=(
            "The valid fixture must pass and the counterexample must be rejected."
        ),
    )


def _fixture(main_table: pl.DataFrame, fixture: Any) -> pl.DataFrame:
    if callable(fixture):
        result = fixture(main_table)
        if not isinstance(result, pl.DataFrame):
            raise TypeError("case fixture callable must return a Polars DataFrame")
        return result
    if not isinstance(fixture, pl.DataFrame):
        raise TypeError("case fixture must be a Polars DataFrame or callable")
    return fixture


def _evaluate_value(table: pl.DataFrame, spec: EvalSpec, context: EvalContext) -> Any:
    if spec.column is not None:
        return table.get_column(spec.column)
    if spec.expr is not None:
        return table.select(spec.expr.alias("__rci_eval_value")).to_series()
    if spec.runner is not None:
        return spec.runner(table)
    if spec.context_runner is not None:
        return spec.context_runner(table, context)
    raise ValueError(f"eval {spec.name!r} has no executable source")


def _normalize(raw: Any, spec: EvalSpec) -> EvalObservation:
    if isinstance(raw, EvalObservation):
        return raw
    if raw is None:
        return EvalObservation(status="passed", output_kind=spec.output_kind)

    if isinstance(raw, pl.Series):
        if spec.reduction == "none":
            return EvalObservation(
                status="passed",
                output_kind=spec.output_kind,
                details={"length": len(raw)},
            )
        if raw.dtype != pl.Boolean:
            raise TypeError(
                f"eval {spec.name!r} reduction {spec.reduction!r} requires "
                "boolean values"
            )
        if spec.reduction == "all":
            value = raw.all()
        elif spec.reduction == "any":
            value = raw.any()
        elif spec.reduction == "count_true":
            value = raw.sum()
        else:
            raise ValueError(f"unsupported boolean reduction: {spec.reduction}")
    else:
        value = raw

    if isinstance(value, bool):
        status = "passed" if value else "failed"
    else:
        status = "passed"
    return EvalObservation(status=status, value=value, output_kind=spec.output_kind)


def _record(
    stage: str,
    slot: str,
    name: str,
    failure_policy: FailurePolicy,
    observation: EvalObservation,
    *,
    execution_mode: str = "custom_runner",
    selection_reason: str = "",
    library_method: str = "",
    description: str = "",
) -> EvalRecord:
    return EvalRecord(
        stage=stage,
        slot=slot,
        name=name,
        status=observation.status,
        value=observation.value,
        output_kind=observation.output_kind,
        failure_policy=failure_policy,
        details=observation.details,
        execution_mode=execution_mode,
        selection_reason=selection_reason,
        library_method=library_method,
        description=description,
    )


def _execution_mode(spec: EvalSpec) -> str:
    if spec.execution_mode is not None:
        return spec.execution_mode
    if spec.expr is not None or spec.column is not None:
        return "polars_expr"
    return "custom_runner"


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
