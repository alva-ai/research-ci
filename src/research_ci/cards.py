"""Readable Markdown cards emitted by the command line runner."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import polars as pl

from .eval.assessment import (
    assess_features,
    assess_records,
    assess_signal_evidence,
)
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
        if record.library_method:
            lines.append(f"  library method: `{record.library_method}`")
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
            "construction recipe: `data/recipe.json`",
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
        if record.library_method:
            lines.append(f"  library method: `{record.library_method}`")
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
            "construction recipe: `feat/recipe.json`",
            "eval report: `feat/eval/report.jsonl`",
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
    workflow_state: str = "review",
) -> tuple[str, str]:
    """Render a Sig card while keeping targets in named sidecars."""

    signal_columns = list(signals)
    handoff_columns = [
        column for column in table.columns if column not in {*keys, *signals}
    ]
    lines = ["# Sig Card", "", f"- workflow_state: {workflow_state}"]
    lines.append(f"- result: {result_status}")
    lines.append(f"- input_rows: {input_rows}")
    lines.append(f"- output_rows: {table.height}")
    lines.append(f"- keys: {', '.join(keys) or '-'}")
    lines.append(f"- signal_columns: {', '.join(signal_columns) or '-'}")
    lines.append(f"- handoff_columns: {', '.join(handoff_columns) or '-'}")
    lines.append(f"- candidate_signals: {len(candidates)}")
    lines.append(f"- evaluation_columns: {len(evaluation_columns)}")
    lines.append(f"- sidecars: {', '.join(sidecars) or '-'}")
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
        for row in dispositions.group_by("action").len().sort("action").iter_rows(
            named=True
        ):
            lines.append(f"- `{row['action']}`: {row['len']}")
        lines.append(f"- covered features: {dispositions.height}")
    lines.extend(("", "## Four slots", ""))
    _append_slot_contract(lines)
    _append_compact_eval_summary(lines, result.records)
    lines.extend(("", "## Signal recipe metadata", "", "```json"))
    lines.append(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    lines.extend(
        (
            "```",
            "",
            "## Review",
            "",
            "The signal table, label sidecars, selection, and eval evidence are "
            "ready for review.",
            "",
            "`rci approve --stage sig --out <run> --by human|agent --card <card_id>`",
            "",
            "signal table: `sig/main.parquet`",
            "candidate and screening tables: `sig/research/`",
            "feature decisions: `sig/research/disposition.parquet`",
            "signal lineage: `sig/research/lineage.json`",
            "label sidecars: `sig/context/`",
            "selection: `sig/selection.json`",
            "construction recipe: `sig/recipe.json`",
            "eval report: `sig/eval/report.jsonl`",
        )
    )
    return render_card("\n".join(lines))


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


def _append_compact_eval_summary(lines: list[str], records: Any) -> None:
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
    lines.append("- Complete per-method details remain in `sig/eval/report.jsonl`.")


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
