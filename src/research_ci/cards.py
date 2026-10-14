"""Readable Markdown cards emitted by the command line runner."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from .eval.assessment import assess_features, assess_records
from .workflow import render_card


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
            "`rci approve --stage contract --out <run> --by human|agent "
            "--card <card_id>`",
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
) -> tuple[str, str]:
    """Render a Data card and return ``(markdown, card_id)``."""

    lines = ["# Data Card", "", f"- workflow_state: {workflow_state}"]
    lines.append(f"- result: {result_status}")
    lines.append(f"- rows: {table.height}")
    lines.append(f"- columns: {', '.join(table.columns)}")
    if "event_time" in table.columns and table.height:
        times = table.get_column("event_time")
        lines.append(f"- event_time: {times.min()} -> {times.max()}")
    _append_assessment(lines, result)
    lines.extend(("", "## Four slots", ""))
    _append_slot_contract(lines)
    for record in result.records:
        lines.append(
            f"- `{record.slot}` `{record.name}`: {record.status}"
            f" (failure_policy={record.failure_policy}, "
            f"execution_mode={record.execution_mode})"
        )
        if record.selection_reason:
            lines.append(f"  selected because: {record.selection_reason}")
        if record.details:
            lines.append(
                "  details: " + json.dumps(record.details, sort_keys=True, default=str)
            )
    lines.extend(("", "## Exploration", "", "```json"))
    lines.append(json.dumps(exploration, indent=2, sort_keys=True, default=str))
    lines.extend(
        (
            "```",
            "",
            "## Review",
            "",
            "The main table and eval report are ready for review.",
            "",
            "`rci approve --stage data --out <run> --by human|agent --card <card_id>`",
            "",
            "main table: `data/main.parquet`",
            "eval report: `data/eval/report.jsonl`",
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
) -> tuple[str, str]:
    """Render a Feat card and return ``(markdown, card_id)``."""

    feature_columns = [column for column in table.columns if column not in keys]
    lines = ["# Feat Card", "", f"- workflow_state: {workflow_state}"]
    lines.append(f"- result: {result_status}")
    lines.append(f"- input_rows: {input_rows}")
    lines.append(f"- output_rows: {table.height}")
    lines.append(f"- keys: {', '.join(keys) or '-'}")
    lines.append(f"- feature_columns: {', '.join(feature_columns) or '-'}")
    _append_assessment(lines, result)
    _append_feature_triage(lines, table, keys, result.records)
    lines.extend(("", "## Four slots", ""))
    _append_slot_contract(lines)
    for record in result.records:
        lines.append(
            f"- `{record.slot}` `{record.name}`: {record.status}"
            f" (failure_policy={record.failure_policy}, "
            f"execution_mode={record.execution_mode})"
        )
        if record.selection_reason:
            lines.append(f"  selected because: {record.selection_reason}")
        if record.details:
            lines.append(
                "  details: " + json.dumps(record.details, sort_keys=True, default=str)
            )
    lines.extend(("", "## Exploration", "", "```json"))
    lines.append(json.dumps(exploration, indent=2, sort_keys=True, default=str))
    lines.extend(
        (
            "```",
            "",
            "## Review",
            "",
            "The feature table and eval report are ready for review.",
            "",
            "`rci approve --stage feat --out <run> --by human|agent --card <card_id>`",
            "",
            "feature table: `feat/main.parquet`",
            "eval report: `feat/eval/report.jsonl`",
        )
    )
    return render_card("\n".join(lines))


def _append_assessment(lines: list[str], result: Any) -> None:
    """Add a compact evidence triage without turning metrics into verdicts."""

    assessment = assess_records(result.records)
    lines.extend(
        (
            "",
            "## Assessment",
            "",
            f"- execution_status: {assessment['execution_status']}",
            f"- research_disposition: {assessment['research_disposition']}",
            f"- metric_count: {assessment['metric_count']}",
            "- reason_codes: " + (", ".join(assessment["reason_codes"]) or "-"),
            "- interpretation: execution describes whether the plan ran; "
            "research disposition is triage evidence and needs review.",
        )
    )
    for reason in assessment["reasons"]:
        affected = ", ".join(reason["records"])
        lines.append(f"- reason `{reason['code']}`: {affected}")


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
