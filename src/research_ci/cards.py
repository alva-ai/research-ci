"""Readable Markdown cards emitted by the command line runner."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

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
    lines.extend(("", "## Four slots", ""))
    for record in result.records:
        lines.append(
            f"- `{record.slot}` `{record.name}`: {record.status}"
            f" (failure_policy={record.failure_policy})"
        )
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
