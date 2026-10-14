"""General diagnostics over standardized external backtest timeseries."""

from __future__ import annotations

import math
from collections.abc import Mapping

import polars as pl

from ..protocol import EvalContext, EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo

METHODS = (
    MethodInfo(
        name="library.economic.backtest_summary",
        family="economic",
        stages=("algo",),
        input_kind="evaluation_table",
        purpose="economic",
        input_roles=("pnl",),
        requires=("pnl",),
        description=(
            "Recompute return, volatility, Sharpe, drawdown, and turnover "
            "diagnostics from the declared external backtest timeseries."
        ),
    ),
)


def backtest_summary(
    *,
    table_name: str,
    return_columns: Mapping[str, str],
    equity_columns: Mapping[str, str],
    turnover_columns: Mapping[str, str],
    annualization_periods: float,
    name: str = "library.economic.backtest_summary",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build an economic diagnostic over a named backtest sidecar."""

    if annualization_periods <= 0:
        raise ValueError("annualization_periods must be positive")
    returns = dict(return_columns)
    equities = dict(equity_columns)
    turnovers = dict(turnover_columns)

    def measure(_: pl.DataFrame, context: EvalContext) -> EvalObservation:
        try:
            table = context.table(table_name)
        except KeyError as exc:
            return EvalObservation(
                "not_applicable",
                details={
                    "reason": str(exc),
                    "reason_code": "MISSING_CONTEXT",
                    "context_table": table_name,
                },
            )
        required = (
            set(returns.values()) | set(equities.values()) | set(turnovers.values())
        )
        missing = sorted(required - set(table.columns))
        if missing:
            return EvalObservation(
                "failed",
                output_kind="findings",
                details={
                    "reason_code": "MISSING_COLUMNS",
                    "missing_columns": missing,
                    "context_table": table_name,
                },
            )

        evidence: dict[str, dict[str, float | int | None]] = {}
        invalid: list[str] = []
        for strategy, return_column in returns.items():
            values = _finite_series(table, return_column)
            equity = _finite_series(table, equities[strategy])
            if values.is_empty() or equity.is_empty() or bool((equity <= 0).any()):
                invalid.append(strategy)
                continue
            rows = len(values)
            mean = float(values.mean())
            raw_std = values.std(ddof=1)
            std = float(raw_std) if raw_std is not None else 0.0
            annualized_volatility = std * math.sqrt(annualization_periods)
            sharpe = mean / std * math.sqrt(annualization_periods) if std > 0 else None
            drawdown = equity / equity.cum_max() - 1.0
            turnover_column = turnovers.get(strategy)
            average_turnover = (
                float(_finite_series(table, turnover_column).mean())
                if turnover_column is not None
                else None
            )
            evidence[strategy] = {
                "rows": rows,
                "total_return": float(equity[-1] - 1.0),
                "annualized_volatility": annualized_volatility,
                "sharpe": sharpe,
                "max_drawdown": abs(float(drawdown.min())),
                "average_turnover": average_turnover,
            }
        return EvalObservation(
            "failed" if invalid else "passed",
            value=evidence,
            output_kind="findings" if invalid else "metric",
            details={
                "context_table": table_name,
                "annualization_periods": annualization_periods,
                "strategies": evidence,
                "invalid_strategies": invalid,
                "reason_code": "INVALID_BACKTEST_PATH" if invalid else None,
            },
        )

    return EvalSpec(
        name=name,
        context_runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_select",
        library_method="library.economic.backtest_summary",
    )


def _finite_series(table: pl.DataFrame, column: str) -> pl.Series:
    values = table.get_column(column).cast(pl.Float64, strict=False).drop_nulls()
    return values.filter(values.is_finite())


__all__ = ["METHODS", "backtest_summary"]
