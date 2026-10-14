"""Data structures for externally assembled four-slot eval plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Literal

import polars as pl


class Slot(StrEnum):
    """The fixed execution order for every stage."""

    BASELINE = "baseline"
    LIBRARY = "library"
    CASE = "case"
    CASE_EVAL = "case_eval"


SLOT_ORDER = tuple(Slot)
EvalStatus = Literal["passed", "failed", "error", "not_applicable"]
OutputKind = Literal["none", "summary", "metric", "findings", "full"]
Reduction = Literal["all", "any", "count_true", "mean", "none"]
FailurePolicy = Literal["stop", "record"]
ExecutionMode = Literal[
    "polars_expr",
    "polars_select",
    "polars_join_group",
    "matrix_backend",
    "custom_runner",
]
Runner = Callable[[pl.DataFrame], Any]
ContextRunner = Callable[[pl.DataFrame, "EvalContext"], Any]
Fixture = pl.DataFrame | Callable[[pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Explicit sidecar inputs available to an evaluation plan.

    The stage main table stays clean.  A lab can provide labels, targets,
    horizons, panel metadata, or other research context as named sidecar
    tables.  The Engine only passes the context through; it never invents a
    target or joins one implicitly.
    """

    tables: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def table(self, name: str) -> pl.DataFrame:
        """Return a named sidecar table with a clear missing-context error."""

        try:
            return self.tables[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.tables)) or "<none>"
            raise KeyError(
                f"evaluation context table {name!r} is not available "
                f"(available: {available})"
            ) from exc


@dataclass(frozen=True, slots=True)
class EvalObservation:
    """An eval may have a status without producing a business column."""

    status: EvalStatus
    value: Any = None
    output_kind: OutputKind = "none"
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def passed_without_output(cls, **details: Any) -> "EvalObservation":
        return cls("passed", output_kind="none", details=details)


@dataclass(frozen=True, slots=True)
class EvalSpec:
    """One externally named rule over the stage's main table.

    Exactly one of ``expr``, ``column``, ``runner`` or ``context_runner`` is
    required. The
    expression or column is evaluated in a scratch result and never appended
    to the caller's main table.
    """

    name: str
    expr: pl.Expr | None = None
    column: str | None = None
    runner: Runner | None = None
    reduction: Reduction = "all"
    output_kind: OutputKind = "none"
    failure_policy: FailurePolicy = "record"
    description: str = ""
    context_runner: ContextRunner | None = None
    execution_mode: ExecutionMode | None = None
    selection_reason: str = ""

    @classmethod
    def noop(
        cls,
        name: str,
        *,
        status: EvalStatus = "not_applicable",
        reason: str = "",
        failure_policy: FailurePolicy = "record",
    ) -> "EvalSpec":
        """Declare a filled slot when no rule is applicable."""

        def run(_: pl.DataFrame) -> EvalObservation:
            return EvalObservation(
                status=status,
                output_kind="none",
                details={"reason": reason} if reason else {},
            )

        return cls(name=name, runner=run, failure_policy=failure_policy)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("eval name must not be empty")
        provided = sum(
            value is not None
            for value in (self.expr, self.column, self.runner, self.context_runner)
        )
        if provided != 1:
            raise ValueError(
                f"eval {self.name!r} must provide exactly one of expr, column, "
                "runner, context_runner"
            )
        if self.column is not None and not self.column.strip():
            raise ValueError(f"eval {self.name!r} has an empty column")
        if self.failure_policy not in {"stop", "record"}:
            raise ValueError(
                f"eval {self.name!r} has an invalid failure policy: "
                f"{self.failure_policy!r}"
            )
        if self.execution_mode not in {
            None,
            "polars_expr",
            "polars_select",
            "polars_join_group",
            "matrix_backend",
            "custom_runner",
        }:
            raise ValueError(
                f"eval {self.name!r} has an invalid execution mode: "
                f"{self.execution_mode!r}"
            )


@dataclass(frozen=True, slots=True)
class CaseEvalTest:
    """A slot-four test with a valid case and an explicit counterexample."""

    name: str
    case: str
    valid: Fixture
    invalid: Fixture
    failure_policy: FailurePolicy = "stop"

    @classmethod
    def from_rows(
        cls,
        *,
        name: str,
        case: str,
        valid: dict[str, list[Any]],
        invalid: dict[str, list[Any]],
        failure_policy: FailurePolicy = "stop",
    ) -> "CaseEvalTest":
        return cls(
            name=name,
            case=case,
            valid=pl.DataFrame(valid),
            invalid=pl.DataFrame(invalid),
            failure_policy=failure_policy,
        )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("case eval test name must not be empty")
        if not self.case.strip():
            raise ValueError(f"case eval test {self.name!r} has no case rule")
        if self.failure_policy not in {"stop", "record"}:
            raise ValueError(
                f"case eval test {self.name!r} has an invalid failure policy: "
                f"{self.failure_policy!r}"
            )


@dataclass(frozen=True, slots=True)
class EvalPlan:
    """A four-slot plan assembled by a research workspace.

    The Engine always fills the general baseline floor. ``baseline`` may be
    empty when a stage has no additional baseline rules.
    """

    stage: str
    baseline: tuple[EvalSpec, ...]
    library: tuple[EvalSpec, ...]
    case: tuple[EvalSpec, ...]
    case_eval: tuple[CaseEvalTest, ...]

    def slots(self) -> tuple[tuple[Slot, tuple[EvalSpec | CaseEvalTest, ...]], ...]:
        return (
            (Slot.BASELINE, self.baseline),
            (Slot.LIBRARY, self.library),
            (Slot.CASE, self.case),
            (Slot.CASE_EVAL, self.case_eval),
        )

    def validate(self) -> None:
        if not self.stage.strip():
            raise ValueError("eval plan stage must not be empty")

        names: set[str] = set()
        for slot, items in self.slots():
            # The engine always contributes the general baseline floor. A lab
            # may add stage-specific baseline rules, but it does not have to
            # duplicate the engine rule just to fill slot one.
            if not items and slot is not Slot.BASELINE:
                raise ValueError(f"eval slot {slot.value!r} must be filled")
            for item in items:
                item.validate()
                if item.name in names:
                    raise ValueError(f"duplicate eval name: {item.name!r}")
                names.add(item.name)

        case_names = {item.name for item in self.case}
        for test in self.case_eval:
            if test.case not in case_names:
                raise ValueError(
                    f"case eval {test.name!r} references unknown case {test.case!r}"
                )
