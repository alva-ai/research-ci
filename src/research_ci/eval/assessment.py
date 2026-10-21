"""Evidence-oriented review summaries for stage Cards.

Assessment is deliberately separate from execution.  A passed calculation
means the requested method ran; it does not mean that a feature is useful.
This module only produces a transparent triage label and reason codes from
the recorded evidence.  A human or the case agent still decides what to keep.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import polars as pl

from .runner import EvalRecord

_MISSING_CONTEXT_WORDS = ("target", "context", "label", "horizon")


def assess_records(records: Iterable[EvalRecord]) -> dict[str, Any]:
    """Return execution and research triage summaries for a set of records."""

    records = tuple(records)
    reasons: dict[str, dict[str, Any]] = {}
    hard_failures: list[str] = []
    soft_findings: list[str] = []
    not_applicable: list[str] = []
    metric_count = 0

    for record in records:
        if record.output_kind == "metric" and record.status == "passed":
            metric_count += 1
        details = record.details or {}
        codes = _reason_codes(record, details)
        for code in codes:
            entry = reasons.setdefault(
                code,
                {"code": code, "records": [], "details": []},
            )
            if record.name not in entry["records"]:
                entry["records"].append(record.name)
            if details and details not in entry["details"]:
                entry["details"].append(details)

        if record.status in {"failed", "error"}:
            if record.failure_policy == "stop":
                hard_failures.append(record.name)
            else:
                soft_findings.append(record.name)
        elif record.status == "not_applicable":
            not_applicable.append(record.name)

    if hard_failures:
        execution_status = "halted"
    elif soft_findings or not_applicable:
        execution_status = "review"
    else:
        execution_status = "ready"

    reason_codes = sorted(reasons)
    if hard_failures:
        disposition = "invalid"
    elif _contains_any(
        reason_codes,
        {
            "MISSING_CONTEXT",
            "NOT_APPLICABLE",
            "MISSING_VALUES_REVIEW",
            "INCOMPLETE_CROSS_SECTION",
            "MISSING_REQUIRED_VALUES",
            "UNEXPECTED_ENTITIES",
            "DUPLICATE_KEYS",
        },
    ):
        disposition = "incomplete"
    elif _contains_any(
        reason_codes,
        {
            "LOW_COVERAGE",
            "LOW_CROSS_SECTION_COMPLETENESS",
            "NO_VARIATION",
            "INSUFFICIENT_SAMPLE",
            "INSUFFICIENT_GROUPS",
            "WEAK_ASSOCIATION",
            "MODEL_PERFORMANCE_BELOW_THRESHOLD",
            "MODEL_CALIBRATION_BELOW_THRESHOLD",
            "INVALID_PROBABILITY",
        },
    ):
        disposition = "weak_data"
    elif _contains_any(reason_codes, {"HIGH_REDUNDANCY", "STRUCTURE_CONCENTRATED"}):
        disposition = "redundant"
    elif _contains_any(
        reason_codes,
        {"UNSTABLE", "DISTRIBUTION_SHIFT", "WALK_FORWARD_UNSTABLE"},
    ):
        disposition = "unstable"
    elif metric_count:
        disposition = "candidate_for_review"
    else:
        disposition = "no_research_evidence"

    return {
        "execution_status": execution_status,
        "research_disposition": disposition,
        "reason_codes": reason_codes,
        "hard_failures": sorted(set(hard_failures)),
        "soft_findings": sorted(set(soft_findings)),
        "not_applicable": sorted(set(not_applicable)),
        "metric_count": metric_count,
        "reasons": [reasons[code] for code in reason_codes],
        "interpretation": (
            "Execution status describes whether the plan ran. Research "
            "disposition is triage evidence, not an automatic alpha approval."
        ),
    }


def assess_features(
    table: pl.DataFrame,
    keys: Iterable[str],
    records: Iterable[EvalRecord],
    *,
    minimum_coverage: float = 0.8,
) -> dict[str, dict[str, Any]]:
    """Create a transparent per-feature structural triage table.

    ``candidate_for_review`` means that the feature is structurally usable for
    the next research question. It never means that the feature predicts a
    target; target-dependent methods must provide that separate evidence.
    """

    if not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be between zero and one")
    key_set = set(keys)
    feature_columns = [column for column in table.columns if column not in key_set]
    rows = table.height
    vif_values: dict[str, float] = {}
    for record in records:
        details = record.details or {}
        if "max_vif" in details and isinstance(record.value, dict):
            for column, value in record.value.items():
                if isinstance(value, (int, float)):
                    vif_values[column] = float(value)

    aliases: dict[str, tuple[str, str, str]] = {}
    expressions: list[pl.Expr] = []
    for index, column in enumerate(feature_columns):
        numeric = pl.col(column).cast(pl.Float64, strict=False)
        usable = numeric.is_not_null() & numeric.is_finite()
        usable_alias = f"__rci_feature_{index}_usable"
        unique_alias = f"__rci_feature_{index}_unique"
        null_alias = f"__rci_feature_{index}_null"
        aliases[column] = (usable_alias, unique_alias, null_alias)
        expressions.extend(
            (
                usable.sum().alias(usable_alias),
                numeric.filter(usable).n_unique().alias(unique_alias),
                pl.col(column).null_count().alias(null_alias),
            )
        )
    aggregate = table.select(expressions).row(0, named=True) if expressions else {}

    result: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        usable_alias, unique_alias, null_alias = aliases[column]
        usable = int(aggregate[usable_alias])
        coverage = usable / rows if rows else 0.0
        unique = int(aggregate[unique_alias])
        reasons: list[str] = []
        if coverage == 0:
            reasons.append("NO_USABLE_VALUES")
        elif coverage < minimum_coverage:
            reasons.append("LOW_COVERAGE")
        if unique < 2 and usable:
            reasons.append("NO_VARIATION")
        vif_value = vif_values.get(column)
        display_vif: float | str | None = vif_value
        if vif_value is not None and not math.isfinite(vif_value):
            display_vif = "inf"
        if vif_value is not None and vif_value >= 10:
            reasons.append("HIGH_REDUNDANCY")
        if not reasons:
            status = "candidate_for_review"
        elif "NO_USABLE_VALUES" in reasons:
            status = "invalid"
        elif "HIGH_REDUNDANCY" in reasons and len(reasons) == 1:
            status = "redundant"
        elif "LOW_COVERAGE" in reasons:
            status = "incomplete"
        elif "NO_VARIATION" in reasons:
            status = "weak_data"
        else:
            status = "incomplete"
        result[column] = {
            "status": status,
            "coverage": coverage,
            "usable_rows": usable,
            "rows": rows,
            "unique_values": unique,
            "null_rows": int(aggregate[null_alias]),
            "vif": display_vif,
            "reason_codes": reasons,
        }
    return result


def assess_signal_evidence(
    records: Iterable[EvalRecord],
    signals: Iterable[str],
    candidates: Iterable[str] = (),
    *,
    ranking_size: int = 5,
) -> dict[str, Any]:
    """Collect target evidence without turning it into an alpha verdict."""

    signal_names = tuple(dict.fromkeys(signals))
    candidate_names = set(candidates)
    evidence: dict[str, dict[str, Any]] = {
        signal: {
            "targets": {},
            "block_stability": {},
            "rank_autocorrelation": None,
            "distribution_shift": None,
            "reason_codes": [],
        }
        for signal in signal_names
    }
    candidate_rank_ic: dict[str, list[dict[str, Any]]] = {}
    candidate_mi: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        details = record.details or {}
        method = record.library_method
        score = details.get("score")
        target = details.get("target")
        if score in evidence and isinstance(target, str):
            target_evidence = evidence[score]["targets"].setdefault(target, {})
            if method == "library.association.pearson_ic":
                target_evidence["pearson_ic"] = record.value
            elif method == "library.association.spearman_ic":
                target_evidence["rank_ic"] = record.value
            elif method == "library.association.bucket_spread":
                target_evidence["bucket_spread"] = record.value
            elif method == "library.stability.split_association":
                target_evidence["split"] = {
                    "first_half": details.get("first_half"),
                    "second_half": details.get("second_half"),
                    "same_sign": details.get("same_sign"),
                }
                if details.get("same_sign") is False:
                    evidence[score]["reason_codes"].append(
                        f"SPLIT_SIGN_CHANGE:{target}"
                    )
            elif method == "library.stability.walk_forward_association":
                target_evidence["walk_forward"] = {
                    "mean_association": details.get("mean_association"),
                    "median_association": details.get("median_association"),
                    "association_std": details.get("association_std"),
                    "sign_consistency": details.get("sign_consistency"),
                    "worst_directional_fold": details.get("worst_directional_fold"),
                    "valid_folds": details.get("valid_folds"),
                }
            elif method == "library.model.regression_diagnostics":
                target_evidence["regression_diagnostics"] = record.value
            elif method == "library.model.classification_probability_diagnostics":
                target_evidence["classification_probability_diagnostics"] = record.value
            elif method == "library.association.icir":
                evidence[score]["block_stability"].setdefault(target, {}).update(
                    {
                        "icir": record.value,
                        "ic_mean": details.get("ic_mean"),
                        "ic_std": details.get("ic_std"),
                        "positive_ic_ratio": details.get("positive_ic_ratio"),
                        "groups": details.get("groups"),
                    }
                )
            elif method == "library.association.cumulative_ic":
                evidence[score]["block_stability"].setdefault(target, {}).update(
                    {
                        "cumulative_ic": record.value,
                        "max_drawdown": details.get("max_drawdown"),
                    }
                )
        column = details.get("column")
        if column in evidence:
            if method == "library.stability.rank_autocorrelation":
                evidence[column]["rank_autocorrelation"] = record.value
            elif method == "library.stability.distribution_shift":
                evidence[column]["distribution_shift"] = record.value

        if (
            method == "library.association.spearman_ic"
            and score in candidate_names
            and isinstance(target, str)
            and isinstance(record.value, (int, float))
            and math.isfinite(float(record.value))
        ):
            candidate_rank_ic.setdefault(target, []).append(
                {"feature": score, "value": float(record.value)}
            )
        left = details.get("left")
        right = details.get("right")
        if (
            method == "library.information.feature_target_mutual_information"
            and left in candidate_names
            and isinstance(right, str)
            and isinstance(record.value, (int, float))
            and math.isfinite(float(record.value))
        ):
            candidate_mi.setdefault(right, []).append(
                {"feature": left, "value": float(record.value)}
            )

    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for target, values in candidate_rank_ic.items():
        ordered = sorted(values, key=lambda item: item["value"])
        rankings[target] = {
            "most_negative": ordered[:ranking_size],
            "most_positive": list(reversed(ordered[-ranking_size:])),
        }
    mi_rankings = {
        target: sorted(values, key=lambda item: item["value"], reverse=True)[
            :ranking_size
        ]
        for target, values in candidate_mi.items()
    }
    for item in evidence.values():
        item["reason_codes"] = sorted(set(item["reason_codes"]))
    return {
        "signals": evidence,
        "candidate_rank_ic": rankings,
        "candidate_mutual_information": mi_rankings,
        "interpretation": (
            "These are association and stability observations. They do not include "
            "fees, execution, position sizing, or an automatic keep/drop decision."
        ),
    }


def _reason_codes(record: EvalRecord, details: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    explicit = details.get("reason_codes")
    if isinstance(explicit, (list, tuple, set)):
        codes.update(str(code) for code in explicit)
    explicit_one = details.get("reason_code")
    if explicit_one and not (
        explicit_one == "NOT_APPLICABLE" and details.get("missing_columns")
    ):
        codes.add(str(explicit_one))

    if record.name == "engine.baseline.general_missing_values":
        totals = details.get("totals", {})
        if any(int(value or 0) for value in totals.values()):
            codes.add("MISSING_VALUES_REVIEW")
    elif record.status == "error":
        codes.add("EVAL_ERROR")
    elif record.status == "failed":
        if record.slot == "case_eval":
            codes.add("CASE_RULE_COUNTEREXAMPLE_FAILED")
        elif record.slot in {"baseline", "case"}:
            codes.add("CORRECTNESS_FAILURE")
        else:
            if not codes:
                codes.add("LIBRARY_CHECK_FAILED")
            if "finite" in record.name.lower():
                codes.add("NON_FINITE_VALUES")
    elif record.status == "not_applicable":
        reason = str(details.get("reason", "")).lower()
        if any(word in reason for word in _MISSING_CONTEXT_WORDS) or details.get(
            "missing_columns"
        ):
            codes.add("MISSING_CONTEXT")
        elif not codes:
            codes.add("NOT_APPLICABLE")

    if "cumulative_ic" in details and details.get("max_drawdown", 0) > 0:
        codes.add("PATH_REQUIRES_REVIEW")
    if details.get("same_sign") is False:
        codes.add("UNSTABLE")
    if (
        details.get("absolute_shift", 0) is not None
        and details.get("absolute_shift", 0) > 2
    ):
        codes.add("DISTRIBUTION_SHIFT")
    if details.get("max_vif", 0) is not None and details.get("max_vif", 0) >= 10:
        codes.add("HIGH_REDUNDANCY")
    effective_rank = details.get("effective_rank")
    columns = details.get("columns")
    if (
        isinstance(effective_rank, (int, float))
        and isinstance(columns, (list, tuple))
        and len(columns) >= 3
        and effective_rank <= len(columns) * 0.5
    ):
        codes.add("STRUCTURE_CONCENTRATED")
    return codes


def _contains_any(values: Iterable[str], choices: set[str]) -> bool:
    return any(value in choices for value in values)


__all__ = ["assess_features", "assess_records", "assess_signal_evidence"]
