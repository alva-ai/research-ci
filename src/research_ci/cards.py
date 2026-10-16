"""Readable Markdown cards emitted by the command line runner."""

from __future__ import annotations

import json
import re
from collections import Counter
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

    lines = [f"{stage.title()}Card · {card_identifier}"]
    metadata: list[tuple[str, str]] = []
    body: list[str] = []
    section: str | None = None
    for raw in card_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("# ") or line.startswith("- card_id:"):
            continue
        if line.startswith("## "):
            section = line[3:].strip()
            if section not in {"Review", "Audit trail"}:
                body.extend(("", section))
            continue
        if line.startswith("### "):
            body.extend(("", line[4:].strip()))
            continue
        match = re.match(r"^-\s+([a-zA-Z][\w-]*):\s*(.*)$", line)
        if match and section is None:
            key, value = match.groups()
            metadata.append(
                (
                    key.replace("_", " "),
                    _terminal_status(key, _terminal_value(value)),
                )
            )
            continue
        if line.startswith("```") or line.startswith("| ---"):
            continue
        if line.startswith("| "):
            body.append(line.replace("|", " ").strip())
            continue
        if line.startswith("- "):
            if section not in {"Review", "Audit trail"}:
                body.append("  " + _terminal_value(line[2:]))
            continue
        if section not in {"Review", "Audit trail"}:
            body.append(_terminal_value(line))

    for key, value in metadata:
        lines.append(f"{key:<18} {value}")
    if recommendation:
        lines.append(f"recommendation      {recommendation.upper()}")
    if dependencies:
        lines.extend(("", "frozen dependencies:"))
        lines.extend(f"  {name:<16} {digest}" for name, digest in dependencies)
    if body:
        lines.extend(("", *body))
    lines.extend(("", "-" * 88, f"card {card_identifier}"))
    return "\n".join(lines)


def _terminal_value(value: str) -> str:
    """Remove Markdown decoration while retaining the Card's words."""

    return value.replace("**", "").replace("`", "")


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


def review_card(
    stage: str,
    brief: ReviewBrief,
    manifest: dict[str, Any],
    assessment: dict[str, Any],
    *,
    attempt: str,
    previous_attempt: str | None,
) -> tuple[str, str]:
    """Render the concise result page shown to an external reviewer."""

    lines = [
        f"# {stage.title()} Review Card",
        "",
        "- workflow_state: ready",
        "- artifact_role: attempt",
        "- checks: valid",
        "- research_ready: true",
        f"- attempt: {attempt}",
        f"- previous_attempt: {previous_attempt or '-'}",
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
    if assessment.get("research_disposition"):
        lines.append(f"- evidence triage: {assessment['research_disposition']}")
    reason_codes = assessment.get("reason_codes", ())
    lines.append(f"- reason codes: {', '.join(reason_codes) if reason_codes else '-'}")

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
