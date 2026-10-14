"""General feature/target association methods.

These functions deliberately return observations rather than a verdict about
an alpha.  A lab chooses the target, horizon, grouping column, and any
thresholds in its plan; the Engine computes the requested evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import polars as pl

from ..protocol import EvalContext, EvalObservation, EvalSpec, FailurePolicy
from ..registry import MethodInfo
from ._utils import (
    column_names,
    mean,
    missing_columns,
    not_applicable,
    sample_std,
)

METHODS = (
    MethodInfo(
        name="library.association.pearson_ic",
        family="association",
        stages=("feat", "sig"),
        input_kind="evaluation_table",
        description=(
            "Measure Pearson score/target association when a target is available."
        ),
    ),
    MethodInfo(
        name="library.association.context_pearson_ic",
        family="association",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description="Measure Pearson IC against an explicitly named target sidecar.",
    ),
    MethodInfo(
        name="library.association.context_spearman_ic",
        family="association",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description="Measure rank IC against an explicitly named target sidecar.",
    ),
    MethodInfo(
        name="library.association.spearman_ic",
        family="association",
        stages=("feat", "sig"),
        input_kind="evaluation_table",
        description=(
            "Measure rank based score/target association when a target is available."
        ),
    ),
    MethodInfo(
        name="library.association.icir",
        family="association",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description=(
            "Summarise grouped cross-sectional IC as mean divided by its dispersion."
        ),
    ),
    MethodInfo(
        name="library.association.cumulative_ic",
        family="association",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description="Return grouped IC observations and their cumulative path.",
    ),
    MethodInfo(
        name="library.association.bucket_spread",
        family="association",
        stages=("feat", "sig"),
        input_kind="feature_panel",
        description="Compare the average target of high and low feature buckets.",
    ),
)


def pearson_ic(
    score: str,
    target: str,
    *,
    name: str = "library.association.pearson_ic",
    minimum_abs: float | None = None,
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a Pearson association metric over explicit score and target columns."""

    return _correlation_spec(
        score,
        target,
        method="pearson",
        name=name,
        minimum_abs=minimum_abs,
        failure_policy=failure_policy,
    )


def spearman_ic(
    score: str,
    target: str,
    *,
    name: str = "library.association.spearman_ic",
    minimum_abs: float | None = None,
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a rank based association metric over explicit columns."""

    return _correlation_spec(
        score,
        target,
        method="spearman",
        name=name,
        minimum_abs=minimum_abs,
        failure_policy=failure_policy,
    )


def context_pearson_ic(
    score: str,
    target: str,
    *,
    context_table: str = "target",
    join_on: Iterable[str] = ("event_time",),
    name: str = "library.association.context_pearson_ic",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build Pearson IC from a named sidecar table joined on explicit keys.

    The target sidecar is never appended to the feature main table.  This is
    the normal path once a lab starts exploring labels and horizons.
    """

    return _context_correlation_spec(
        score,
        target,
        method="pearson",
        context_table=context_table,
        join_on=join_on,
        name=name,
        failure_policy=failure_policy,
    )


def context_spearman_ic(
    score: str,
    target: str,
    *,
    context_table: str = "target",
    join_on: Iterable[str] = ("event_time",),
    name: str = "library.association.context_spearman_ic",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build rank IC from a named sidecar target table."""

    return _context_correlation_spec(
        score,
        target,
        method="spearman",
        context_table=context_table,
        join_on=join_on,
        name=name,
        failure_policy=failure_policy,
    )


def _context_correlation_spec(
    score: str,
    target: str,
    *,
    method: str,
    context_table: str,
    join_on: Iterable[str],
    name: str,
    failure_policy: FailurePolicy,
) -> EvalSpec:
    keys = column_names(join_on)

    def measure(table: pl.DataFrame, context: EvalContext) -> EvalObservation:
        try:
            sidecar = context.table(context_table)
        except KeyError as exc:
            return not_applicable(str(exc), context_table=context_table)
        missing = missing_columns(table, keys) + missing_columns(sidecar, keys)
        if target not in sidecar.columns:
            missing.append(target)
        if score not in table.columns:
            missing.append(score)
        if missing:
            return not_applicable(
                "target context is missing required columns",
                missing_columns=sorted(set(missing)),
                context_table=context_table,
            )
        sidecar_keys = sidecar.select(keys)
        duplicate_keys = sidecar_keys.height - sidecar_keys.unique().height
        if duplicate_keys:
            return EvalObservation(
                status="failed",
                output_kind="findings",
                details={
                    "reason": "target context has duplicate join keys",
                    "reason_code": "CONTEXT_DUPLICATE_KEYS",
                    "context_table": context_table,
                    "join_on": keys,
                    "duplicate_key_rows": duplicate_keys,
                },
            )
        joined = table.select([*keys, score]).join(
            sidecar.select([*keys, target]), on=list(keys), how="inner"
        )
        observation = _measure_correlation(joined, score, target, method)
        details = dict(observation.details)
        details.update({"context_table": context_table, "join_on": keys})
        return EvalObservation(
            status=observation.status,
            value=observation.value,
            output_kind=observation.output_kind,
            details=details,
        )

    return EvalSpec(
        name=name,
        context_runner=measure,
        failure_policy=failure_policy,
        output_kind="metric",
        execution_mode="polars_join_group",
    )


def icir(
    score: str,
    target: str,
    *,
    by: str | None = None,
    annualization: float | None = None,
    min_group_size: int = 3,
    name: str = "library.association.icir",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build grouped ICIR evidence.

    ``by`` must identify the period/group over which cross-sectional IC is
    calculated.  Without it, a single full-table correlation cannot be called
    ICIR, so the method reports ``not_applicable`` explicitly.
    """

    if min_group_size < 2:
        raise ValueError("icir min_group_size must be at least 2")

    def measure(table: pl.DataFrame) -> EvalObservation:
        if by is None:
            return not_applicable("icir requires an explicit group column")
        missing = missing_columns(table, (score, target, by))
        if missing:
            return not_applicable("icir columns are missing", missing_columns=missing)
        groups = _group_correlations(table, score, target, by, min_group_size)
        return _ic_sequence_observation(
            groups,
            name="icir",
            annualization=annualization,
            extra={"score": score, "target": target, "group": by},
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_join_group",
    )


def cumulative_ic(
    score: str,
    target: str,
    *,
    by: str | None = None,
    min_group_size: int = 3,
    name: str = "library.association.cumulative_ic",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Build a cumulative grouped-IC path and its drawdown summary."""

    def measure(table: pl.DataFrame) -> EvalObservation:
        if by is None:
            return not_applicable("cumulative IC requires an explicit group column")
        missing = missing_columns(table, (score, target, by))
        if missing:
            return not_applicable(
                "cumulative IC columns are missing", missing_columns=missing
            )
        groups = _group_correlations(table, score, target, by, min_group_size)
        if not groups:
            return not_applicable("fewer than two valid groups", groups=0)
        cumulative: list[float] = []
        running = 0.0
        for _, value in groups:
            running += value
            cumulative.append(running)
        peak = cumulative[0]
        max_drawdown = 0.0
        for value in cumulative:
            peak = max(peak, value)
            max_drawdown = max(max_drawdown, peak - value)
        return EvalObservation(
            status="passed",
            value=float(cumulative[-1]),
            output_kind="metric",
            details={
                "score": score,
                "target": target,
                "group": by,
                "groups": len(groups),
                "ic_values": [value for _, value in groups],
                "cumulative_ic": cumulative,
                "max_drawdown": max_drawdown,
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_join_group",
    )


def bucket_spread(
    score: str,
    target: str,
    *,
    buckets: int = 5,
    by: str | None = None,
    min_group_size: int = 10,
    name: str = "library.association.bucket_spread",
    failure_policy: FailurePolicy = "record",
) -> EvalSpec:
    """Compare average target in the highest and lowest score buckets."""

    if buckets < 2:
        raise ValueError("bucket_spread requires at least two buckets")
    if min_group_size < buckets * 2:
        raise ValueError("bucket_spread min_group_size must cover at least two buckets")

    def measure(table: pl.DataFrame) -> EvalObservation:
        required = (score, target) if by is None else (score, target, by)
        missing = missing_columns(table, required)
        if missing:
            return not_applicable(
                "bucket spread columns are missing", missing_columns=missing
            )
        if by is None:
            spreads = [_bucket_spread(table, score, target, buckets, min_group_size)]
        else:
            spreads = [
                _bucket_spread(group, score, target, buckets, min_group_size)
                for _, group in table.group_by(by, maintain_order=True)
            ]
        values = [value for value in spreads if value is not None]
        if not values:
            return not_applicable(
                "no group has enough complete observations",
                min_group_size=min_group_size,
            )
        return EvalObservation(
            status="passed",
            value=float(mean(values) or 0.0),
            output_kind="metric",
            details={
                "score": score,
                "target": target,
                "buckets": buckets,
                "group": by,
                "group_spreads": values,
                "groups": len(values),
            },
        )

    return EvalSpec(
        name=name,
        runner=measure,
        output_kind="metric",
        failure_policy=failure_policy,
        execution_mode="polars_join_group",
    )


def _correlation_spec(
    score: str,
    target: str,
    *,
    method: str,
    name: str,
    minimum_abs: float | None,
    failure_policy: FailurePolicy,
) -> EvalSpec:
    if not score.strip() or not target.strip():
        raise ValueError("association evals need named score and target columns")
    if minimum_abs is not None and minimum_abs < 0:
        raise ValueError("association minimum_abs cannot be negative")

    def measure(table: pl.DataFrame) -> EvalObservation:
        missing = missing_columns(table, (score, target))
        if missing:
            return EvalObservation(
                status="not_applicable",
                details={"missing_columns": missing},
            )
        observation = _measure_correlation(table, score, target, method)
        if (
            observation.status == "passed"
            and minimum_abs is not None
            and abs(float(observation.value)) < minimum_abs
        ):
            details = dict(observation.details)
            details.update(
                {"minimum_abs": minimum_abs, "reason_code": "WEAK_ASSOCIATION"}
            )
            return EvalObservation(
                status="failed",
                value=observation.value,
                output_kind="findings",
                details=details,
            )
        return observation

    return EvalSpec(
        name=name,
        runner=measure,
        failure_policy=failure_policy,
        output_kind="metric",
        execution_mode="polars_select",
    )


def _measure_correlation(
    table: pl.DataFrame, score: str, target: str, method: str
) -> EvalObservation:
    pairs = _complete_numeric_frame(table, (score, target))
    if pairs.height < 2:
        return not_applicable(
            "fewer than two complete rows",
            rows=pairs.height,
            score=score,
            target=target,
        )
    stats = pairs.select(
        pl.col(score).n_unique().alias("score_unique"),
        pl.col(target).n_unique().alias("target_unique"),
        pl.corr(score, target, method=method).alias("correlation"),
    ).row(0, named=True)
    if stats["score_unique"] < 2 or stats["target_unique"] < 2:
        return not_applicable(
            "score or target has no variation",
            rows=pairs.height,
            score=score,
            target=target,
        )
    value = stats["correlation"]
    if value is None or not math.isfinite(float(value)):
        return not_applicable("correlation is undefined", rows=pairs.height)
    return EvalObservation(
        status="passed",
        value=float(value),
        output_kind="metric",
        details={
            "score": score,
            "target": target,
            "rows": pairs.height,
            "method": method,
        },
    )


def _group_correlations(
    table: pl.DataFrame,
    score: str,
    target: str,
    by: str,
    min_group_size: int,
) -> list[tuple[Any, float]]:
    selected = table.select(
        pl.col(by),
        pl.col(score).cast(pl.Float64, strict=False).alias(score),
        pl.col(target).cast(pl.Float64, strict=False).alias(target),
    ).drop_nulls()
    selected = selected.filter(pl.col(score).is_finite() & pl.col(target).is_finite())
    grouped = (
        selected.group_by(by, maintain_order=True)
        .agg(
            pl.len().alias("__rows"),
            pl.col(score).n_unique().alias("__score_unique"),
            pl.col(target).n_unique().alias("__target_unique"),
            pl.corr(score, target, method="spearman").alias("__ic"),
        )
        .filter(
            (pl.col("__rows") >= min_group_size)
            & (pl.col("__score_unique") >= 2)
            & (pl.col("__target_unique") >= 2)
            & pl.col("__ic").is_finite()
        )
    )
    return [
        (row[by], float(row["__ic"]))
        for row in grouped.select(by, "__ic").iter_rows(named=True)
    ]


def _ic_sequence_observation(
    groups: list[tuple[Any, float]],
    *,
    name: str,
    annualization: float | None,
    extra: dict[str, Any],
) -> EvalObservation:
    if len(groups) < 2:
        return not_applicable("fewer than two valid groups", groups=len(groups))
    values = [value for _, value in groups]
    average = mean(values)
    dispersion = sample_std(values)
    if average is None or dispersion in (None, 0.0):
        return not_applicable("IC dispersion is undefined", groups=len(groups))
    ratio = average / dispersion
    if annualization is not None:
        if annualization <= 0:
            raise ValueError("annualization must be positive")
        ratio *= math.sqrt(annualization)
    return EvalObservation(
        status="passed",
        value=float(ratio),
        output_kind="metric",
        details={
            **extra,
            "method": name,
            "groups": len(groups),
            "ic_mean": average,
            "ic_std": dispersion,
            "positive_ic_ratio": sum(value > 0 for value in values) / len(values),
            "group_values": values,
            "annualization": annualization,
        },
    )


def _bucket_spread(
    table: pl.DataFrame,
    score: str,
    target: str,
    buckets: int,
    min_group_size: int,
) -> float | None:
    pairs = _complete_numeric_frame(table, (score, target)).sort(score)
    if pairs.height < max(buckets * 2, min_group_size):
        return None
    if pairs.select(pl.col(score).n_unique()).item() < 2:
        return None
    grouped = (
        pairs.with_row_index("__rank")
        .with_columns(
            ((pl.col("__rank") * buckets) // pairs.height)
            .clip(upper_bound=buckets - 1)
            .alias("__bucket")
        )
        .group_by("__bucket")
        .agg(pl.col(target).mean().alias("__target_mean"))
        .sort("__bucket")
    )
    if grouped.height < buckets:
        return None
    return float(
        grouped.get_column("__target_mean")[-1] - grouped.get_column("__target_mean")[0]
    )


def _complete_numeric_frame(
    table: pl.DataFrame, columns: tuple[str, ...]
) -> pl.DataFrame:
    selected = table.select(
        pl.col(column).cast(pl.Float64, strict=False).alias(column)
        for column in columns
    )
    return selected.drop_nulls().filter(
        pl.all_horizontal([pl.col(column).is_finite() for column in columns])
    )


__all__ = [
    "METHODS",
    "bucket_spread",
    "context_pearson_ic",
    "context_spearman_ic",
    "cumulative_ic",
    "icir",
    "pearson_ic",
    "spearman_ic",
]
