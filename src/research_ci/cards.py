"""Readable Markdown cards emitted by the command line runner."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from .artifacts import ReviewBrief
from .eval.assessment import (
    assess_features,
    assess_records,
    assess_signal_evidence,
)
from .workflow import render_card


def render_terminal_card(
    stage: str,
    card_text: str,
    *,
    card_identifier: str,
    recommendation: str | None = None,
    dependencies: tuple[tuple[str, str], ...] = (),
) -> str:
    """Render a submitted Markdown Card as a compact terminal review.

    The Markdown Card remains the durable artifact. This view changes only the
    presentation so a person can review the current Card without opening a
    file or remembering a long command.
    """

    title, metadata, sections = _parse_terminal_markdown(card_text)
    header = (
        title.replace("Contract Card", "ContractCard", 1)
        if title.startswith("Contract Card")
        else f"{stage.title()}Card · {card_identifier}"
    )
    lines = [header]
    add = _terminal_adder(lines)

    labels = {
        "workflow_state": "state",
        "artifact_role": "artifact",
        "research_ready": "ready",
        "previous_attempt": "previous",
        "eval_execution": "eval",
        "stage_readiness": "readiness",
        "evidence_disposition": "triage",
    }
    for key, values in metadata:
        if key in {"card_id", "recommendation"} or not values:
            continue
        label = labels.get(key, key.replace("_", " "))
        rendered = tuple(
            _terminal_status(key, _terminal_value(value)) for value in values
        )
        add(label, *rendered)

    for name, values in sections:
        _render_terminal_section(add, name, values)

    if recommendation:
        add("verdict", recommendation.upper())
    lines.extend(("-" * 100, f"card {card_identifier}"))
    if dependencies:
        lines.extend(("", "  frozen dependencies:"))
        lines.extend(f"    {name:<16} {digest}" for name, digest in dependencies)
    return "\n".join(lines)


def _parse_terminal_markdown(
    card_text: str,
) -> tuple[str, list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Keep Markdown hierarchy while extracting the small terminal view."""

    title = ""
    metadata: list[tuple[str, list[str]]] = []
    sections: list[tuple[str, list[str]]] = []
    section: list[str] | None = None
    metadata_values: list[str] | None = None
    in_fence = False
    for raw in card_text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            continue
        if stripped.startswith("## "):
            sections.append((stripped[3:].strip(), []))
            section = sections[-1][1]
            metadata_values = None
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            continue
        if section is None:
            match = re.match(r"^-\s+([a-zA-Z][\w-]*):\s*(.*)$", raw)
            if match:
                key, value = match.groups()
                metadata_values = []
                if value:
                    metadata_values.append(value)
                metadata.append((key, metadata_values))
                continue
            if raw.startswith("  - ") and metadata_values is not None:
                metadata_values.append(raw[4:].strip())
            continue
        if stripped.startswith("### "):
            section.append(stripped[4:].strip())
        elif stripped.startswith("| ---"):
            continue
        elif stripped.startswith("| "):
            section.append(stripped.replace("|", " ").strip())
        elif stripped.startswith("- "):
            section.append(stripped[2:].strip())
        else:
            section.append(stripped)
    return title, metadata, sections


def _terminal_adder(lines: list[str]):
    """Use the compact key/continuation layout of the original RCI Cards."""

    def add(key: str, *values: str) -> None:
        body = [value for value in values if value]
        for index, value in enumerate(body):
            lines.append(f"{key if index == 0 else '':<11}{value}")

    return add


def _render_terminal_section(add, name: str, values: list[str]) -> None:
    """Map reviewer-facing Markdown sections onto old RCI-style rows."""

    if (
        name
        in {
            "Review",
            "Audit trail",
            "Evidence references",
            "Exploration",
        }
        or not values
    ):
        return
    cleaned = [_terminal_value(value) for value in values]
    if name == "Result":
        questions = [
            value.removeprefix("Question:").strip()
            for value in cleaned
            if value.startswith("Question:")
        ]
        results = [value for value in cleaned if not value.startswith("Question:")]
        add("question", *questions)
        add("result", *results)
        return
    if name == "Evidence":
        add("evidence", *(f"ok   {value}" for value in cleaned))
        return
    if name == "Counterevidence and limits":
        warnings = []
        for value in cleaned:
            value = value.removeprefix("Counterevidence:")
            value = value.removeprefix("Limitation:").strip()
            warnings.append(f"[!] {value}")
        add("limits", *warnings)
        return
    if name == "Four slots":
        current_slot = "slots"
        groups: dict[str, list[str]] = {}
        for value in cleaned:
            match = re.match(r"^(baseline|library|case|case_eval):\s*(.*)$", value)
            if match:
                current_slot = match.group(1)
                groups.setdefault(current_slot, []).append(match.group(2))
            else:
                groups.setdefault(current_slot, []).append(value)
        for slot, slot_values in groups.items():
            add(slot, *slot_values)
        return
    if name == "Eval legend":
        add("legend", *cleaned)
        return
    if name == "Key figures" and values == [
        "No key figure was selected for this review."
    ]:
        return
    label = {
        "Iteration delta": "changes",
        "Engine facts": "facts",
        "Key figures": "figures",
    }.get(name, name.lower())
    add(label, *cleaned)


def _terminal_value(value: str) -> str:
    """Remove Markdown decoration while retaining the Card's words."""

    return value.replace("**", "").replace("`", "").replace(" -> ", " → ")


def _terminal_status(key: str, value: str) -> str:
    """Use the compact status vocabulary familiar from the old RCI Card."""

    if key in {"checks", "research_ready"}:
        lowered = value.lower()
        if lowered in {"valid", "passed", "true", "ok"}:
            return "ok"
        if lowered in {"failed", "error", "false", "invalid"}:
            return "FAIL"
        if lowered in {"not_applicable", "uncertain"}:
            return "[!]"
    return value


def render_eval_terminal(
    stage: str,
    records: Sequence[Any],
    assessment: Mapping[str, Any],
) -> str:
    """Render the evidence that determines whether an attempt may advance."""

    normalized = tuple(_eval_record(record) for record in records)
    lines = [f"Eval · {stage}"]
    add = _terminal_adder(lines)
    execution = str(assessment.get("execution_status", "undeclared"))
    add("execution", execution.upper())
    add("readiness", _stage_readiness(execution))
    for slot in ("baseline", "library", "case", "case_eval"):
        selected = tuple(record for record in normalized if record["slot"] == slot)
        if slot == "library":
            groups = _library_groups(selected)
            add(
                slot,
                f"{_slot_marker(selected)} {_counted(len(groups), 'method')} · "
                f"{_counted(len(selected), 'record')}",
            )
            for _, group in groups:
                add(
                    "",
                    f"{_slot_marker(group)} {_eval_name(group[0])} · "
                    f"{_eval_group_evidence(group)}",
                )
                for detail in _eval_group_details(group):
                    add("", f"    {detail}")
            continue
        marker = _slot_marker(selected)
        passed = sum(record["status"] == "passed" for record in selected)
        add(slot, f"{marker} {passed}/{len(selected)} passed")
        for record in selected:
            add(
                "",
                f"{_eval_marker(record)} {_eval_name(record)} · "
                f"{_eval_evidence(record)}",
            )
    hard_failures = assessment.get("hard_failures", ())
    soft_findings = assessment.get("soft_findings", ())
    not_applicable = assessment.get("not_applicable", ())
    if hard_failures:
        add("blocked", *[str(value) for value in hard_failures])
    if soft_findings or not_applicable:
        add(
            "review",
            f"{len(soft_findings)} non-blocking findings · "
            f"{len(not_applicable)} unavailable · see eval/report.jsonl",
        )
    add("meaning", _stage_meaning(stage, execution))
    add(
        "legend",
        "[OK] completed without configured finding · [!] review finding · "
        "[FAIL] blocks attempt",
    )
    return "\n".join(lines)


def _append_eval_results(
    lines: list[str], records: Sequence[Mapping[str, Any]]
) -> None:
    normalized = tuple(_eval_record(record) for record in records)
    if not normalized:
        lines.append("- eval records unavailable")
        return
    for slot in ("baseline", "library", "case", "case_eval"):
        selected = tuple(record for record in normalized if record["slot"] == slot)
        if slot == "library":
            groups = _library_groups(selected)
            lines.append(
                f"- {slot}: {_slot_marker(selected)} "
                f"{_counted(len(groups), 'method')} · "
                f"{_counted(len(selected), 'record')}"
            )
            for _, group in groups:
                lines.append(
                    f"  - {_slot_marker(group)} `{_eval_name(group[0])}`: "
                    f"{_eval_group_evidence(group)}"
                )
                lines.extend(f"    - {detail}" for detail in _eval_group_details(group))
            continue
        passed = sum(record["status"] == "passed" for record in selected)
        lines.append(
            f"- {slot}: {_slot_marker(selected)} {passed}/{len(selected)} passed"
        )
        for record in selected:
            lines.append(
                f"  - {_eval_marker(record)} `{_eval_name(record)}`: "
                f"{_eval_evidence(record)}"
            )


def _eval_record(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    as_dict = getattr(record, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    raise TypeError("eval terminal renderer requires EvalRecord-like values")


def _library_groups(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    """Group expanded metric records by the method selected by the Agent."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        method = str(record.get("library_method") or record.get("name") or "unknown")
        grouped.setdefault(method, []).append(record)
    return tuple((method, tuple(group)) for method, group in grouped.items())


def _counted(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _eval_marker(record: Mapping[str, Any]) -> str:
    status = record.get("status")
    if status == "passed":
        details = record.get("details")
        if isinstance(details, Mapping) and details.get("reason_code"):
            return "[!]"
        return "[OK]"
    if status in {"failed", "error"} and record.get("failure_policy") == "stop":
        return "[FAIL]"
    return "[!]"


def _slot_marker(records: Sequence[Mapping[str, Any]]) -> str:
    markers = {_eval_marker(record) for record in records}
    if "[FAIL]" in markers or not records:
        return "[FAIL]"
    if "[!]" in markers:
        return "[!]"
    return "[OK]"


def _eval_name(record: Mapping[str, Any]) -> str:
    method = str(record.get("library_method") or "")
    name = method or str(record.get("name") or "unnamed eval")
    for prefix in (
        "engine.baseline.",
        "library.",
        "data.library.",
        "feat.library.",
        "sig.library.",
        "algo.library.",
        "data.case_eval.",
        "feat.case_eval.",
        "sig.case_eval.",
        "algo.case_eval.",
        "data.case.",
        "feat.case.",
        "sig.case.",
        "algo.case.",
    ):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    for suffix in (".default", ".numeric", ".keys"):
        name = name.removesuffix(suffix)
    return name.replace(".", " / ").replace("_", " ")


def _eval_evidence(record: Mapping[str, Any]) -> str:
    name = str(record.get("name") or "")
    details = record.get("details")
    details = dict(details) if isinstance(details, Mapping) else {}
    value = record.get("value")
    status = str(record.get("status") or "unknown")

    if name == "engine.baseline.general_missing_values":
        totals = details.get("totals", {})
        return " · ".join(
            (
                f"null={totals.get('null', 0)}",
                f"NaN={totals.get('nan', 0)}",
                f"non-finite={totals.get('non_finite', 0)}",
                f"sentinel={totals.get('string_sentinel', 0)}",
            )
        )
    if name == "engine.baseline.data_bundle_shape":
        tables = details.get("tables", {})
        duplicate_rows = sum(
            int(summary.get("duplicate_key_rows", 0))
            for summary in tables.values()
            if isinstance(summary, Mapping)
        )
        return (
            f"tables={len(tables)} · duplicate keys={duplicate_rows} · "
            f"missing coverage={len(details.get('missing_coverage_tables', ()))}"
        )
    if name == "engine.baseline.feature_output_shape":
        return " · ".join(
            (
                f"input={details.get('input_rows', '-')}",
                f"output={details.get('output_rows', '-')}",
                f"missing keys={details.get('missing_key_rows', '-')}",
                f"extra keys={details.get('extra_key_rows', '-')}",
                f"duplicate keys={details.get('duplicate_key_rows', '-')}",
            )
        )
    if record.get("library_method") == "library.quality.required_columns":
        missing = details.get("missing_columns", ())
        return "all required columns present" if not missing else f"missing={missing}"
    if record.get("library_method") == "library.quality.finite_columns":
        findings = value if isinstance(value, Mapping) else {}
        return (
            f"columns={len(details.get('columns', ()))} · "
            f"non-finite={sum(int(count) for count in findings.values())}"
        )
    if record.get("library_method") == "library.temporal.key_grid":
        return " · ".join(
            (
                f"rows={details.get('rows', '-')}",
                f"sorted={str(details.get('sorted', '-')).lower()}",
                f"duplicate keys={details.get('duplicate_key_rows', '-')}",
                f"irregular gaps={details.get('irregular_gaps', '-')}",
            )
        )
    if record.get("slot") == "case_eval":
        valid_status = details.get("valid_status", "-")
        invalid_status = details.get("invalid_status", "-")
        valid_result = "accepted" if valid_status == "passed" else valid_status
        invalid_result = "rejected" if invalid_status == "failed" else invalid_status
        return f"valid fixture={valid_result} · counterexample={invalid_result}"
    if details.get("reason_code"):
        return str(details["reason_code"])
    if details.get("reason"):
        return str(details["reason"])
    description = str(record.get("description") or "").strip()
    if status == "passed" and isinstance(value, bool):
        outcome = "all evaluated rows satisfied the rule"
        return f"{outcome}: {description}" if description else outcome
    if status in {"failed", "error"}:
        return str(details.get("error") or "rule was not satisfied")
    if record.get("output_kind") == "metric":
        return f"value={value}"
    if description:
        return description
    return status


def _eval_group_evidence(records: Sequence[Mapping[str, Any]]) -> str:
    """Summarize one selected library method without hiding its raw records."""

    first = records[0]
    method = str(first.get("library_method") or "")
    if method == "library.quality.dispersion":
        details = first.get("details")
        details = dict(details) if isinstance(details, Mapping) else {}
        constants = details.get("constant_columns", ())
        return (
            f"columns={len(details.get('columns', ()))} · "
            f"constant={len(constants) if isinstance(constants, Sequence) else '-'}"
        )
    if method == "library.quality.coverage":
        details = first.get("details")
        details = dict(details) if isinstance(details, Mapping) else {}
        values = details.get("coverage", {})
        values = dict(values) if isinstance(values, Mapping) else {}
        numeric = [float(value) for value in values.values() if _finite_number(value)]
        minimum = details.get("minimum")
        below = details.get("below_minimum", {})
        below = dict(below) if isinstance(below, Mapping) else {}
        return " · ".join(
            (
                f"columns={len(values)}",
                f"lowest={min(numeric):.3f}" if numeric else "lowest=-",
                (
                    f"minimum={float(minimum):.3f}"
                    if _finite_number(minimum)
                    else "minimum=not set"
                ),
                f"below={len(below)}",
            )
        )
    if method == "library.structure.pairwise_correlation":
        thresholds = [
            float(details["maximum_absolute"])
            for record in records
            if isinstance((details := record.get("details")), Mapping)
            and isinstance(details.get("maximum_absolute"), (int, float))
        ]
        unavailable = sum(
            record.get("status") == "not_applicable" for record in records
        )
        if thresholds:
            threshold = thresholds[0]
            findings = sum(
                isinstance(record.get("value"), (int, float))
                and not isinstance(record.get("value"), bool)
                and abs(float(record["value"])) >= threshold
                for record in records
            )
            return (
                f"pairs={len(records)} · |r|>={threshold:.2f}: {findings} · "
                f"unavailable={unavailable}"
            )
        return f"pairs={len(records)} · unavailable={unavailable}"
    if method == "library.structure.vif":
        details = first.get("details")
        details = dict(details) if isinstance(details, Mapping) else {}
        maximum = details.get("maximum")
        return " · ".join(
            (
                f"columns={len(details.get('columns', ()))}",
                _number_summary("max VIF", details.get("max_vif")),
                (
                    _number_summary("limit", maximum)
                    if maximum is not None
                    else "limit=not set"
                ),
                f"rows={details.get('rows', '-')}",
            )
        )
    if method == "library.structure.pca_diagnostics":
        details = first.get("details")
        details = dict(details) if isinstance(details, Mapping) else {}
        value = first.get("value")
        value = dict(value) if isinstance(value, Mapping) else {}
        ratios = value.get("explained_variance_ratio", ())
        ratios = ratios if isinstance(ratios, Sequence) else ()
        first_ratio = ratios[0] if ratios else None
        minimum_rank = details.get("minimum_effective_rank")
        return " · ".join(
            (
                f"columns={len(details.get('columns', ()))}",
                _number_summary("effective rank", details.get("effective_rank")),
                _number_summary("PC1", first_ratio),
                (
                    _number_summary("minimum rank", minimum_rank)
                    if minimum_rank is not None
                    else "minimum rank=not set"
                ),
            )
        )
    if method == "library.information.feature_pair_mutual_information":
        values = [
            float(record["value"])
            for record in records
            if _finite_number(record.get("value"))
        ]
        unavailable = sum(
            record.get("status") == "not_applicable" for record in records
        )
        return " · ".join(
            (
                f"pairs={len(records)}",
                f"max NMI={max(values):.3f}" if values else "max NMI=-",
                f"unavailable={unavailable}",
            )
        )
    if method == "library.stability.distribution_shift":
        values = [
            abs(float(record["value"]))
            for record in records
            if _finite_number(record.get("value"))
        ]
        limits = [
            float(details["max_absolute_shift"])
            for record in records
            if isinstance((details := record.get("details")), Mapping)
            and _finite_number(details.get("max_absolute_shift"))
        ]
        findings = sum(
            record.get("status") in {"failed", "error"} for record in records
        )
        return " · ".join(
            (
                f"columns={len(records)}",
                f"max |shift|={max(values):.3f}" if values else "max |shift|=-",
                f"limit={limits[0]:.3f}" if limits else "limit=not set",
                f"findings={findings}",
            )
        )
    if method in {
        "library.structure.autocorrelation",
        "library.stability.rank_autocorrelation",
    }:
        values = [
            abs(float(record["value"]))
            for record in records
            if _finite_number(record.get("value"))
        ]
        return (
            f"columns={len(records)} · max |rho|={max(values):.3f}"
            if values
            else f"columns={len(records)}"
        )
    if len(records) == 1:
        return _eval_evidence(first)
    counts = Counter(str(record.get("status") or "unknown") for record in records)
    findings = counts["failed"] + counts["error"]
    return (
        f"records={len(records)} · passed={counts['passed']} · "
        f"findings={findings} · unavailable={counts['not_applicable']}"
    )


def _eval_group_details(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Show only the decision-relevant tail of an expanded method."""

    first = records[0]
    method = first.get("library_method")
    if method == "library.structure.vif":
        values = first.get("value")
        values = dict(values) if isinstance(values, Mapping) else {}
        ranked = sorted(
            (
                (str(column), float(value))
                for column, value in values.items()
                if _finite_number(value)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return tuple(f"{column}: VIF {value:.2f}" for column, value in ranked[:3])
    if method in {
        "library.information.feature_pair_mutual_information",
        "library.stability.distribution_shift",
        "library.structure.autocorrelation",
        "library.stability.rank_autocorrelation",
    }:
        ranked = [record for record in records if _finite_number(record.get("value"))]
        ranked.sort(key=lambda record: abs(float(record["value"])), reverse=True)
        summaries: list[str] = []
        for record in ranked[:3]:
            metadata = record.get("details")
            metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            if method == "library.information.feature_pair_mutual_information":
                label = f"{metadata.get('left', '-')} ↔ {metadata.get('right', '-')}"
                metric = "NMI"
            else:
                label = str(metadata.get("column", "-"))
                metric = "shift" if "distribution_shift" in str(method) else "rho"
            summaries.append(f"{label}: {metric} {float(record['value']):.3f}")
        return tuple(summaries)
    if method != "library.structure.pairwise_correlation":
        return ()
    ranked = [
        record
        for record in records
        if isinstance(record.get("value"), (int, float))
        and not isinstance(record.get("value"), bool)
        and math.isfinite(float(record["value"]))
    ]
    ranked.sort(key=lambda record: abs(float(record["value"])), reverse=True)
    thresholds = [
        float(details["maximum_absolute"])
        for record in records
        if isinstance((details := record.get("details")), Mapping)
        and isinstance(details.get("maximum_absolute"), (int, float))
    ]
    if thresholds:
        threshold = thresholds[0]
        ranked = [
            record for record in ranked if abs(float(record["value"])) >= threshold
        ]
    else:
        ranked = ranked[:3]
    details: list[str] = []
    for record in ranked[:5]:
        metadata = record.get("details")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        details.append(
            f"{metadata.get('left', '-')} ↔ {metadata.get('right', '-')}: "
            f"{float(record['value']):.3f}"
        )
    if len(ranked) > 5:
        details.append(f"{len(ranked) - 5} more in eval/report.jsonl")
    return tuple(details)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number_summary(label: str, value: Any) -> str:
    return f"{label}={float(value):.3f}" if _finite_number(value) else f"{label}=-"


def _stage_readiness(execution: str) -> str:
    return {
        "ready": "ELIGIBLE",
        "review": "REVIEW_REQUIRED",
        "halted": "BLOCKED",
    }.get(execution, "UNDECLARED")


def _stage_meaning(stage: str, execution: str) -> str:
    if execution == "halted":
        return "mandatory correctness failed; fix the attempt before review"
    if execution == "review":
        return "no stop rule failed, but findings require explicit review"
    if stage == "data":
        return (
            "structurally admissible for Feat review; no research claim is established"
        )
    return (
        "mechanically eligible for research review; usefulness is a separate decision"
    )


def _stage_disposition(stage: str, assessment: Mapping[str, Any]) -> str:
    execution = str(assessment.get("execution_status", "undeclared"))
    if stage == "data":
        return {
            "ready": "admissible_for_feat_review",
            "review": "data_findings_require_review",
            "halted": "invalid_data_attempt",
        }.get(execution, "undeclared")
    return str(assessment.get("research_disposition", "undeclared"))


def review_card(
    stage: str,
    brief: ReviewBrief,
    manifest: dict[str, Any],
    assessment: dict[str, Any],
    *,
    attempt: str,
    previous_attempt: str | None,
    eval_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, str]:
    """Render the concise result page shown to an external reviewer."""

    execution_status = str(assessment.get("execution_status", "undeclared"))
    lines = [
        f"# {stage.title()} Review Card",
        "",
        "- workflow_state: ready",
        "- artifact_role: attempt",
        "- checks: valid",
        "- research_ready: true",
        f"- attempt: {attempt}",
        f"- previous_attempt: {previous_attempt or '-'}",
        f"- eval_execution: {execution_status}",
        f"- stage_readiness: {_stage_readiness(execution_status)}",
        f"- evidence_disposition: {_stage_disposition(stage, assessment)}",
        "",
        "## Result",
        "",
        f"**Question:** {brief.question}",
        "",
        brief.conclusion,
        "",
        "## Evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in brief.evidence)
    lines.extend(("", "## Counterevidence and limits", ""))
    if brief.counterevidence:
        lines.extend(f"- Counterevidence: {item}" for item in brief.counterevidence)
    if brief.limitations:
        lines.extend(f"- Limitation: {item}" for item in brief.limitations)
    if not brief.counterevidence and not brief.limitations:
        lines.append("- None declared.")

    lines.extend(("", "## Evidence references", ""))
    if brief.evidence_refs:
        lines.extend(f"- `{path}`" for path in brief.evidence_refs)
    else:
        lines.append("- No machine-readable evidence reference was selected.")

    lines.extend(("", "## Iteration delta", ""))
    if brief.changes:
        lines.extend(f"- {item}" for item in brief.changes)
    elif previous_attempt is None:
        lines.append("- First completed attempt.")
    else:
        lines.append("- No material change was declared from the previous attempt.")

    lines.extend(("", "## Engine facts", ""))
    lines.append(f"- rows: {manifest.get('rows', '-')}")
    columns = manifest.get("columns", ())
    lines.append(
        f"- columns: {len(columns) if isinstance(columns, list) else columns or '-'}"
    )
    data_bundle = manifest.get("data_bundle")
    if isinstance(data_bundle, dict):
        native = data_bundle.get("native", {})
        lines.append(f"- native source tables: {len(native)}")
        for name, summary in native.items():
            lines.append(
                f"  - `{name}`: {summary.get('rows', '-')} rows, "
                f"{summary.get('frequency', 'unspecified')}"
            )
    execution = assessment.get("execution_status", manifest.get("checks", "-"))
    lines.append(f"- eval execution: {execution}")
    lines.append(f"- evidence triage: {_stage_disposition(stage, assessment)}")
    reason_codes = assessment.get("reason_codes", ())
    lines.append(f"- reason codes: {', '.join(reason_codes) if reason_codes else '-'}")

    lines.extend(("", "## Four slots", ""))
    _append_eval_results(lines, eval_records)
    lines.extend(
        (
            "",
            "## Eval legend",
            "",
            "- [OK] check or method completed without a configured finding",
            "- [!] non-blocking finding or not applicable; reviewer must decide",
            "- [FAIL] stop rule failed or could not be evaluated",
            "- Stage readiness measures data admissibility for the next stage; "
            "it does not prove research value or causality.",
        )
    )

    lines.extend(("", "## Key figures", ""))
    if brief.key_plots:
        lines.extend(f"- [{Path(path).stem}]({path})" for path in brief.key_plots)
    else:
        lines.append("- No key figure was selected for this review.")

    lines.extend(
        (
            "",
            "## Review",
            "",
            "Review the result, evidence, counterevidence, limitations, and key "
            "figures. Detailed methods remain available for audit but are not "
            "required for this decision.",
            "",
            f"- Approve and continue: `rci approve --stage {stage} --out <run> "
            "--by human --decision continue --card <card_id>`",
            f"- Approve and stop: `rci approve --stage {stage} --out <run> "
            "--by human --decision stop --card <card_id>`",
            f"- Reject with feedback: `rci reject --stage {stage} --out <run> "
            "--by human --message <reason> --card <card_id>`",
            "- A designated external review Agent uses `--by agent` for the same "
            "three decisions.",
            "",
            "## Audit trail",
            "",
            "- complete eval records: `eval/report.jsonl`",
            "- eval assessment: `eval/assessment.json`",
            "- construction recipe: `recipe.json`",
            "- artifact index: `manifest.json`",
        )
    )
    return render_card("\n".join(lines))


def contract_card(
    name: str,
    payload: dict[str, Any],
    *,
    state: str = "review",
) -> tuple[str, str]:
    """Render a Contract card and return ``(markdown, card_id)``."""

    lines = [f"# Contract Card · {name}", "", f"- state: {state}"]
    lines.append(f"- mission: {payload['mission']}")
    lines.append(f"- stages: {' -> '.join(payload['stages'])}")
    lines.append("- scope:")
    lines.extend(f"  - {item}" for item in payload.get("scope", []))
    lines.append("- invariants:")
    lines.extend(f"  - {item}" for item in payload.get("invariants", []))
    lines.extend(
        (
            "",
            "## Review",
            "",
            "Approve this card before the Data stage can run.",
            "",
            "`rci approve --stage contract --out <run> --by agent "
            "--decision continue|stop --card <card_id>`",
            "",
            "Or reject it with `rci reject --stage contract --out <run> "
            "--by agent --message <reason> --card <card_id>`. A human reviewer "
            "uses `--by human` instead.",
        )
    )
    return render_card("\n".join(lines))


def data_card(
    table: pl.DataFrame,
    result: Any,
    exploration: dict[str, Any],
    *,
    result_status: str,
    workflow_state: str = "review",
    artifact_role: str = "candidate",
) -> tuple[str, str]:
    """Render a Data card and return ``(markdown, card_id)``."""

    lines = ["# Data Card", "", f"- workflow_state: {workflow_state}"]
    lines.append(f"- artifact_role: {artifact_role}")
    lines.append(f"- checks: {result_status}")
    lines.append(f"- rows: {table.height}")
    lines.append(f"- columns: {', '.join(table.columns)}")
    if "event_time" in table.columns and table.height:
        times = table.get_column("event_time")
        lines.append(f"- event_time: {times.min()} -> {times.max()}")
    _append_assessment(lines, result)
    lines.extend(("", "## Four slots", ""))
    _append_slot_contract(lines)
    _append_eval_results(lines, result.records)
    lines.extend(("", "## Exploration", "", "```json"))
    lines.append(json.dumps(exploration, indent=2, sort_keys=True, default=str))
    review_message, review_command = _review_prompt("data", artifact_role)
    lines.extend(
        (
            "```",
            "",
            "## Review",
            "",
            review_message,
            "",
            review_command,
            "",
            "main table: `main.parquet` (relative to this Card)",
            "construction recipe: `recipe.json`",
            "eval report: `eval/report.jsonl`",
        )
    )
    return render_card("\n".join(lines))


def feat_card(
    table: pl.DataFrame,
    result: Any,
    exploration: dict[str, Any],
    *,
    input_rows: int,
    keys: tuple[str, ...],
    result_status: str,
    workflow_state: str = "review",
    artifact_role: str = "candidate",
) -> tuple[str, str]:
    """Render a Feat card and return ``(markdown, card_id)``."""

    feature_columns = [column for column in table.columns if column not in keys]
    lines = ["# Feat Card", "", f"- workflow_state: {workflow_state}"]
    lines.append(f"- artifact_role: {artifact_role}")
    lines.append(f"- checks: {result_status}")
    lines.append(f"- input_rows: {input_rows}")
    lines.append(f"- output_rows: {table.height}")
    lines.append(f"- keys: {', '.join(keys) or '-'}")
    lines.append(f"- feature_columns: {', '.join(feature_columns) or '-'}")
    _append_assessment(lines, result)
    _append_feature_triage(lines, table, keys, result.records)
    lines.extend(("", "## Four slots", ""))
    _append_slot_contract(lines)
    _append_eval_results(lines, result.records)
    lines.extend(("", "## Exploration", "", "```json"))
    lines.append(json.dumps(exploration, indent=2, sort_keys=True, default=str))
    review_message, review_command = _review_prompt("feat", artifact_role)
    lines.extend(
        (
            "```",
            "",
            "## Review",
            "",
            review_message,
            "",
            review_command,
            "",
            "feature table: `main.parquet` (relative to this Card)",
            "construction recipe: `recipe.json`",
            "eval report: `eval/report.jsonl`",
        )
    )
    return render_card("\n".join(lines))


def sig_card(
    table: pl.DataFrame,
    result: Any,
    metadata: dict[str, Any],
    *,
    input_rows: int,
    evaluation_columns: tuple[str, ...],
    keys: tuple[str, ...],
    signals: tuple[str, ...],
    sidecars: tuple[str, ...],
    candidates: tuple[str, ...],
    dispositions: pl.DataFrame,
    result_status: str,
    validation_plan: dict[str, Any] | None = None,
    assessment_research: dict[str, pl.DataFrame] | None = None,
    confirmation_status: str = "not_declared",
    workflow_state: str = "review",
    artifact_role: str = "candidate",
) -> tuple[str, str]:
    """Render a Sig card while keeping targets in named sidecars."""

    signal_columns = list(signals)
    handoff_columns = [
        column for column in table.columns if column not in {*keys, *signals}
    ]
    lines = ["# Sig Card", "", f"- workflow_state: {workflow_state}"]
    lines.append(f"- artifact_role: {artifact_role}")
    lines.append(f"- checks: {result_status}")
    lines.append(f"- input_rows: {input_rows}")
    lines.append(f"- output_rows: {table.height}")
    lines.append(f"- keys: {', '.join(keys) or '-'}")
    lines.append(f"- signal_columns: {', '.join(signal_columns) or '-'}")
    lines.append(f"- handoff_columns: {', '.join(handoff_columns) or '-'}")
    lines.append(f"- candidate_signals: {len(candidates)}")
    lines.append(f"- evaluation_columns: {len(evaluation_columns)}")
    lines.append(f"- sidecars: {', '.join(sidecars) or '-'}")
    evidence_scope = (
        validation_plan.get("evidence_scope", "development_only")
        if validation_plan is not None
        else "development_only"
    )
    lines.append(f"- evidence_scope: {evidence_scope}")
    lines.append(f"- confirmation_status: {confirmation_status}")
    if metadata.get("assessment"):
        _append_thesis_assessment_summary(lines, metadata, result)
        _append_assessment_output(lines, metadata, assessment_research or {})
    else:
        _append_assessment(lines, result)
        signal_table = table.select(*keys, *signals)
        triage = assess_features(signal_table, keys, result.records)
        lines.extend(("", "## Signal output triage", ""))
        for column, evidence in triage.items():
            reasons = ", ".join(evidence["reason_codes"]) or "-"
            lines.append(
                f"- `{column}`: {evidence['status']} "
                f"(coverage={evidence['coverage']:.3f}, reasons={reasons})"
            )
        feature_columns = tuple(
            column
            for column in evaluation_columns
            if column not in {*keys, *signal_columns, *candidates}
        )
        signal_evidence = assess_signal_evidence(
            result.records,
            signal_columns,
            feature_columns,
        )
        _append_signal_evidence(lines, signal_evidence, heading="Final signal evidence")
        if candidates:
            candidate_evidence = assess_signal_evidence(result.records, candidates)
            _append_signal_evidence(
                lines,
                candidate_evidence,
                heading="Candidate signal evidence",
                include_feature_ranking=False,
            )
    lines.extend(("", "## Feature disposition", ""))
    if dispositions.is_empty():
        lines.append("- no feature disposition was supplied")
    else:
        for row in (
            dispositions.group_by("action").len().sort("action").iter_rows(named=True)
        ):
            lines.append(f"- `{row['action']}`: {row['len']}")
        lines.append(f"- covered features: {dispositions.height}")
    lines.extend(("", "## Four slots", ""))
    _append_slot_contract(lines)
    _append_compact_eval_summary(lines, result.records)
    lines.extend(("", "## Validation design", ""))
    if validation_plan is None:
        lines.append("- no `VALIDATION_PLAN` declared; evidence is development-only")
    else:
        lines.extend(("```json", json.dumps(validation_plan, indent=2, sort_keys=True)))
        lines.append("```")
    lines.extend(("", "## Signal recipe metadata", "", "```json"))
    lines.append(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    if artifact_role == "attempt" and confirmation_status == "pending":
        review_message = (
            "This development attempt declares a confirmation holdout. Freeze its "
            "recipe before submission."
        )
        review_command = (
            "`rci sig freeze --out <run> --attempt <id> "
            "--context <confirmation-context>`"
        )
    else:
        review_message, review_command = _review_prompt("sig", artifact_role)
    lines.extend(
        (
            "```",
            "",
            "## Review",
            "",
            review_message,
            "",
            review_command,
            "",
            "signal table: `main.parquet` (relative to this Card)",
            "candidate and screening tables: `research/`",
            "feature decisions: `research/disposition.parquet`",
            "signal lineage: `research/lineage.json`",
            "label sidecars: `context/`",
            "selection: `selection.json`",
            "construction recipe: `recipe.json`",
            "eval report: `eval/report.jsonl`",
        )
    )
    return render_card("\n".join(lines))


def algo_card(
    table: pl.DataFrame,
    result: Any,
    metadata: dict[str, Any],
    summary: pl.DataFrame,
    *,
    input_rows: int,
    keys: tuple[str, ...],
    positions: tuple[str, ...],
    representation: str,
    strategies: tuple[str, ...],
    result_status: str,
    workflow_state: str = "review",
    artifact_role: str = "candidate",
) -> tuple[str, str]:
    """Render an Algo review card with external backtest evidence."""

    lines = ["# Algo Card", "", f"- workflow_state: {workflow_state}"]
    lines.append(f"- artifact_role: {artifact_role}")
    lines.append(f"- checks: {result_status}")
    lines.append(f"- input_rows: {input_rows}")
    lines.append(f"- output_rows: {table.height}")
    lines.append(f"- representation: {representation}")
    lines.append(f"- keys: {', '.join(keys)}")
    lines.append(f"- strategies: {', '.join(strategies)}")
    lines.append(f"- position_columns: {', '.join(positions)}")
    _append_assessment(lines, result)

    lines.extend(("", "## External backtest summary", ""))
    if summary.is_empty():
        lines.append("- no summary rows")
    else:
        columns = summary.columns
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in summary.iter_rows(named=True):
            values = [_card_value(row[column]) for column in columns]
            lines.append("| " + " | ".join(values) + " |")

    lines.extend(("", "## Four slots", ""))
    _append_slot_contract(lines)
    _append_compact_eval_summary(lines, result.records, stage="algo")
    lines.extend(("", "## Algo recipe metadata", "", "```json"))
    lines.append(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    review_message, review_command = _review_prompt("algo", artifact_role)
    lines.extend(
        (
            "```",
            "",
            "## Review",
            "",
            review_message,
            "",
            review_command,
            "",
            "position table: `main.parquet` (relative to this Card)",
            "backtest timeseries: `backtest/timeseries.parquet`",
            "backtest summary: `backtest/summary.parquet`",
            "construction recipe: `recipe.json`",
            "eval report: `eval/report.jsonl`",
        )
    )
    return render_card("\n".join(lines))


def _append_assessment_output(
    lines: list[str],
    metadata: dict[str, Any],
    research: dict[str, pl.DataFrame],
) -> None:
    """Render Thesis Assessment evidence without treating text status as a signal."""

    lines.extend(("", "## Thesis Assessment", ""))
    for field in (
        "thesis_id",
        "author_version_id",
        "as_of",
        "assessment_status",
        "uncertainty",
        "stopping_rule",
    ):
        if field in metadata:
            lines.append(f"- {field}: {_card_value(metadata[field])}")
    premise_statuses = metadata.get("premise_statuses")
    if isinstance(premise_statuses, dict):
        lines.append(
            "- premise_statuses: "
            + ", ".join(f"`{key}`={value}" for key, value in premise_statuses.items())
        )

    premise_table = research.get("premise_assessment")
    if premise_table is not None and not premise_table.is_empty():
        lines.extend(("", "### Premise assessment", ""))
        columns = [
            column
            for column in (
                "premise_id",
                "status",
                "effect_size",
                "effective_n",
                "uncertainty",
            )
            if column in premise_table.columns
        ]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in premise_table.select(columns).iter_rows(named=True):
            lines.append(
                "| " + " | ".join(_card_value(row[column]) for column in columns) + " |"
            )

    findings = research.get("quantitative_findings")
    if findings is not None and not findings.is_empty():
        lines.extend(("", "### Quantitative findings", ""))
        columns = [
            column
            for column in ("premise_id", "test", "estimate", "unit", "n", "method")
            if column in findings.columns
        ]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in findings.select(columns).iter_rows(named=True):
            lines.append(
                "| " + " | ".join(_card_value(row[column]) for column in columns) + " |"
            )

    ledger = research.get("evidence_ledger")
    if ledger is not None and not ledger.is_empty():
        lines.extend(("", "### Evidence ledger", ""))
        lines.append(f"- evidence rows: {ledger.height}")
        if "stance" in ledger.columns:
            counts = ledger.group_by("stance").len().sort("stance").iter_rows()
            lines.append(
                "- stance counts: "
                + ", ".join(f"{stance}={count}" for stance, count in counts)
            )
        if "independence_cluster" in ledger.columns:
            lines.append(
                f"- independent evidence clusters: "
                f"{ledger.get_column('independence_cluster').n_unique()}"
            )


def _append_signal_evidence(
    lines: list[str],
    evidence: dict[str, Any],
    *,
    heading: str = "Signal evidence",
    include_feature_ranking: bool = True,
) -> None:
    lines.extend(
        (
            "",
            f"## {heading}",
            "",
            "Association is oriented so a positive value supports the declared "
            "signal direction. It is evidence for review, not an automatic decision.",
            "",
            "| signal | label | Pearson IC | Rank IC | bucket spread | "
            "early → late Rank IC |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        )
    )
    for signal, item in evidence["signals"].items():
        for target, target_item in item["targets"].items():
            split = target_item.get("split", {})
            split_text = (
                f"{_metric(split.get('first_half'))} → "
                f"{_metric(split.get('second_half'))}"
            )
            if split.get("same_sign") is False:
                split_text += " ⚠ sign change"
            lines.append(
                f"| `{signal}` | `{target}` | "
                f"{_metric(target_item.get('pearson_ic'))} | "
                f"{_metric(target_item.get('rank_ic'))} | "
                f"{_metric(target_item.get('bucket_spread'))} | {split_text} |"
            )

    lines.extend(
        (
            "",
            "### Weekly-block stability",
            "",
            "| signal | label | ICIR | mean block IC | positive blocks | "
            "cumulative IC | max drawdown | groups |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for signal, item in evidence["signals"].items():
        for target, block in item["block_stability"].items():
            lines.append(
                f"| `{signal}` | `{target}` | {_metric(block.get('icir'))} | "
                f"{_metric(block.get('ic_mean'))} | "
                f"{_metric(block.get('positive_ic_ratio'))} | "
                f"{_metric(block.get('cumulative_ic'))} | "
                f"{_metric(block.get('max_drawdown'))} | "
                f"{block.get('groups', '-')} |"
            )

    if not include_feature_ranking:
        return
    lines.extend(("", "### Candidate feature ranking", ""))
    for target, ranking in evidence["candidate_rank_ic"].items():
        positive = _ranking_text(ranking["most_positive"])
        negative = _ranking_text(ranking["most_negative"])
        lines.append(f"- `{target}` strongest positive Rank IC: {positive}")
        lines.append(f"- `{target}` strongest negative Rank IC: {negative}")
    for target, ranking in evidence["candidate_mutual_information"].items():
        lines.append(
            f"- `{target}` highest mutual information: {_ranking_text(ranking)}"
        )


def _append_compact_eval_summary(
    lines: list[str],
    records: Any,
    *,
    stage: str = "sig",
) -> None:
    records = tuple(records)
    counts = Counter((record.slot, record.status) for record in records)
    for slot in ("baseline", "library", "case", "case_eval"):
        values = [
            f"{status}={counts[(slot, status)]}"
            for status in ("passed", "failed", "error", "not_applicable")
            if counts[(slot, status)]
        ]
        lines.append(f"- `{slot}` totals: {', '.join(values) or 'none'}")

    lines.extend(("", "### Correctness rules", ""))
    for record in records:
        if record.slot in {"case", "case_eval"} or (
            record.slot == "baseline"
            and record.name != "engine.baseline.general_missing_values"
        ):
            lines.append(f"- `{record.name}`: {record.status}")

    exceptions = [record for record in records if record.status != "passed"]
    lines.extend(("", "### Findings requiring review", ""))
    if not exceptions:
        lines.append("- none")
    for record in exceptions:
        reason = (record.details or {}).get("reason_code") or (
            "expected warm-up nulls"
            if record.name == "engine.baseline.general_missing_values"
            else (record.details or {}).get("reason", "review the full report")
        )
        lines.append(f"- `{record.name}`: {record.status} ({reason})")
    lines.append(
        f"- Complete per-method details remain in `{stage}/eval/report.jsonl`."
    )


def _card_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _metric(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.4f}"


def _ranking_text(items: list[dict[str, Any]]) -> str:
    return (
        ", ".join(f"`{item['feature']}` {_metric(item['value'])}" for item in items)
        or "-"
    )


def _append_assessment(lines: list[str], result: Any) -> None:
    """Add a compact evidence triage without turning metrics into verdicts."""

    assessment = assess_records(result.records)
    lines.extend(
        (
            "",
            "## Assessment",
            "",
            f"- eval_execution: {assessment['execution_status']}",
            f"- evidence_summary: {assessment['research_disposition']}",
            f"- metric_count: {assessment['metric_count']}",
            "- reason_codes: " + (", ".join(assessment["reason_codes"]) or "-"),
            "- interpretation: execution describes whether the plan ran; "
            "research disposition is triage evidence and needs review.",
        )
    )
    for reason in assessment["reasons"]:
        affected = ", ".join(reason["records"])
        lines.append(f"- reason `{reason['code']}`: {affected}")


def _append_thesis_assessment_summary(
    lines: list[str], metadata: dict[str, Any], result: Any
) -> None:
    """Render execution status separately from a non-trading assessment verdict."""

    assessment = assess_records(result.records)
    lines.extend(
        (
            "",
            "## Assessment",
            "",
            f"- eval_execution: {assessment['execution_status']}",
            "- thesis_status: "
            + _card_value(metadata.get("assessment_status", "undeclared")),
            f"- uncertainty: {_card_value(metadata.get('uncertainty', 'undeclared'))}",
            f"- metric_count: {assessment['metric_count']}",
            "- interpretation: this is a Thesis Assessment; it is not a trading signal "
            "and has no signal-output triage.",
        )
    )
    statuses = metadata.get("premise_statuses")
    if isinstance(statuses, dict):
        lines.append(
            "- premise_statuses: "
            + ", ".join(f"`{key}`={value}" for key, value in statuses.items())
        )


def _review_prompt(stage: str, artifact_role: str) -> tuple[str, str]:
    if artifact_role == "attempt":
        return (
            "This is an Agent attempt. Inspect it, run another attempt, or submit it "
            "for review.",
            f"`rci {stage} submit --out <run> --attempt <id> "
            "--recommendation continue|stop|uncertain`",
        )
    return (
        "The submitted artifact and eval evidence are ready for reviewer action.",
        f"`rci approve --stage {stage} --out <run> --by agent "
        "--decision continue|stop --card <card_id>` (use `--by human` when the "
        "reviewer is human)",
    )


def _append_slot_contract(lines: list[str]) -> None:
    """Explain the fixed responsibilities before listing individual records."""

    lines.extend(
        (
            "- `baseline`: Engine and stage correctness floor",
            "- `library`: selected general methodology and diagnostics",
            "- `case`: task-specific definitions and research assumptions",
            "- `case_eval`: valid fixtures and counterexamples for case rules",
        )
    )


def _append_feature_triage(
    lines: list[str],
    table: pl.DataFrame,
    keys: tuple[str, ...],
    records: Any,
) -> None:
    """Show which columns are structurally usable and why."""

    triage = assess_features(table, keys, records)
    lines.extend(("", "## Feature triage", ""))
    for column, evidence in triage.items():
        reasons = ", ".join(evidence["reason_codes"]) or "-"
        lines.append(
            f"- `{column}`: {evidence['status']} "
            f"(coverage={evidence['coverage']:.3f}, reasons={reasons})"
        )
