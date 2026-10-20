from __future__ import annotations

import json
from pathlib import Path

from research_ci.cli import main
from research_ci.workflow import (
    new_workflow,
    read_workflow,
    record_attempt,
    render_card,
    write_workflow,
)


def _make_run(tmp_path: Path) -> Path:
    out = tmp_path / ".rci" / "runs" / "full"
    card, card_id = render_card(
        "\n".join(
            (
                "# Contract Card · acceptance",
                "",
                "- state: review",
                "- mission: exercise every stage boundary",
                "",
                "## Review",
                "",
                "Approve the contract before Data runs.",
            )
        )
    )
    state = new_workflow(("data", "feat", "sig", "algo"), card_id)
    write_workflow(out, state)
    (out / "contract").mkdir(parents=True)
    (out / "contract" / "card.md").write_text(card + "\n", encoding="utf-8")
    return out


def _attempt(out: Path, stage: str, number: int) -> tuple[str, Path]:
    attempt = f"{number:04d}"
    path = out / stage / "attempts" / attempt
    (path / "eval").mkdir(parents=True)
    (path / "eval" / "report.jsonl").write_text(
        '{"status":"passed","slot":"baseline","name":"fixture"}\n',
        encoding="utf-8",
    )
    (path / "eval" / "assessment.json").write_text(
        json.dumps({"execution_status": "passed", "reason_codes": []}) + "\n",
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "rows": 4,
                "columns": ["event_time", "value"],
                "checks": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    card, card_id = render_card(
        "\n".join(
            (
                f"# {stage.title()} Attempt",
                "",
                "- workflow_state: ready",
                "- artifact_role: attempt",
                f"- attempt: {attempt}",
                "",
                "## Result",
                "",
                "The deterministic fixture passed its stage checks.",
            )
        )
    )
    (path / "card.md").write_text(card + "\n", encoding="utf-8")
    record_attempt(
        out,
        stage=stage,
        attempt=attempt,
        path=str(path),
        valid=True,
        card=card_id,
        checks="ok",
    )
    return attempt, path


def _prepare_submit_approve(
    out: Path,
    tmp_path: Path,
    stage: str,
    attempt: str,
    *,
    decision: str = "continue",
) -> None:
    brief = tmp_path / f"{stage}-{attempt}-brief.json"
    brief.write_text(
        json.dumps(
            {
                "question": f"Is the {stage} attempt reviewable?",
                "conclusion": (
                    "The fixture is complete and its limitations are explicit."
                ),
                "evidence": ["The four-slot fixture report passed."],
                "counterevidence": ["The fixture is synthetic."],
                "limitations": ["It is not a market conclusion."],
                "changes": (
                    [] if attempt == "0001" else ["addressed reviewer feedback"]
                ),
                "evidence_refs": ["eval/report.jsonl"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                stage,
                "prepare",
                "--out",
                str(out),
                "--attempt",
                attempt,
                "--brief",
                str(brief),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                stage,
                "submit",
                "--out",
                str(out),
                "--attempt",
                attempt,
                "--recommendation",
                decision,
            ]
        )
        == 0
    )
    card = read_workflow(out)["cards"][stage]
    assert (
        main(
            [
                "approve",
                "--stage",
                stage,
                "--out",
                str(out),
                "--by",
                "agent",
                "--decision",
                decision,
                "--card",
                card,
            ]
        )
        == 0
    )


def test_full_protocol_acceptance_covers_reject_revision_and_terminal_algo(
    tmp_path, capsys
) -> None:
    """Exercise the complete state machine with two independent test layers.

    Stage runners have focused tests elsewhere. This acceptance test supplies a
    deterministic artifact fixture and verifies the protocol that connects
    them: explicit review, rejection, a new attempt, upstream revision, and a
    terminal Algo decision.
    """

    out = _make_run(tmp_path)
    contract_card = read_workflow(out)["cards"]["contract"]
    assert (
        main(
            [
                "approve",
                "--stage",
                "contract",
                "--out",
                str(out),
                "--by",
                "human",
                "--decision",
                "continue",
                "--card",
                contract_card,
            ]
        )
        == 0
    )

    # Data attempt 0001 is deliberately rejected; the rejection is part of
    # the durable history and cannot be resubmitted.
    attempt, _ = _attempt(out, "data", 1)
    brief = tmp_path / "data-rejected.json"
    brief.write_text(
        json.dumps(
            {
                "question": "Is the first Data result sufficient?",
                "conclusion": "It is structurally valid but misses a required source.",
                "evidence": ["The fixture passed the mechanical floor."],
                "limitations": ["A source is intentionally absent."],
                "evidence_refs": ["eval/report.jsonl"],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "data",
                "prepare",
                "--out",
                str(out),
                "--attempt",
                attempt,
                "--brief",
                str(brief),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "data",
                "submit",
                "--out",
                str(out),
                "--attempt",
                attempt,
                "--recommendation",
                "uncertain",
            ]
        )
        == 0
    )
    card = read_workflow(out)["cards"]["data"]
    assert (
        main(
            [
                "reject",
                "--stage",
                "data",
                "--out",
                str(out),
                "--by",
                "agent",
                "--message",
                "add the missing source",
                "--card",
                card,
            ]
        )
        == 0
    )
    state = read_workflow(out)
    assert state["stages"]["data"] == "ready"
    assert state["attempts"]["data"][0]["rejected"] is True

    _attempt(out, "data", 2)
    _prepare_submit_approve(out, tmp_path, "data", "0002")

    # Create a downstream attempt, then revise Data. The Engine must mark the
    # downstream attempt stale and lock Feat until the replacement is approved.
    _attempt(out, "feat", 1)
    assert (
        main(
            [
                "revise",
                "--stage",
                "data",
                "--out",
                str(out),
                "--message",
                "include the missing source",
            ]
        )
        == 0
    )
    state = read_workflow(out)
    assert state["stages"]["data"] == "ready"
    assert state["stages"]["feat"] == "locked"
    assert state["attempts"]["feat"][0]["stale"] is True

    _attempt(out, "data", 3)
    _prepare_submit_approve(out, tmp_path, "data", "0003")
    _attempt(out, "feat", 2)
    _prepare_submit_approve(out, tmp_path, "feat", "0002")

    # Sig is reviewed twice; Algo is reviewed twice and then accepted as a
    # terminal no-go, proving that both decisions are represented explicitly.
    _attempt(out, "sig", 1)
    sig_brief = tmp_path / "sig-reject.json"
    sig_brief.write_text(
        json.dumps(
            {
                "question": "Is the first signal candidate ready?",
                "conclusion": "The candidate needs another comparison.",
                "evidence": ["Candidate diagnostics are present."],
                "limitations": ["Selection is not stable yet."],
                "evidence_refs": ["eval/report.jsonl"],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "sig",
                "prepare",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--brief",
                str(sig_brief),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "sig",
                "submit",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--recommendation",
                "uncertain",
            ]
        )
        == 0
    )
    card = read_workflow(out)["cards"]["sig"]
    assert (
        main(
            [
                "reject",
                "--stage",
                "sig",
                "--out",
                str(out),
                "--by",
                "agent",
                "--message",
                "compare the full feature universe",
                "--card",
                card,
            ]
        )
        == 0
    )
    _attempt(out, "sig", 2)
    _prepare_submit_approve(out, tmp_path, "sig", "0002")

    _attempt(out, "algo", 1)
    # The first Algo Card is rejected before the second attempt is accepted.
    # Use a dedicated brief so the rejection is visible in the test trace.
    algo_brief = tmp_path / "algo-reject.json"
    algo_brief.write_text(
        json.dumps(
            {
                "question": "Is the execution mapping complete?",
                "conclusion": "The mapping is missing cost assumptions.",
                "evidence": ["The position table exists."],
                "limitations": ["No slippage model yet."],
                "evidence_refs": ["eval/report.jsonl"],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "algo",
                "prepare",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--brief",
                str(algo_brief),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "algo",
                "submit",
                "--out",
                str(out),
                "--attempt",
                "0001",
                "--recommendation",
                "uncertain",
            ]
        )
        == 0
    )
    card = read_workflow(out)["cards"]["algo"]
    assert (
        main(
            [
                "reject",
                "--stage",
                "algo",
                "--out",
                str(out),
                "--by",
                "agent",
                "--message",
                "add costs and lag",
                "--card",
                card,
            ]
        )
        == 0
    )
    _attempt(out, "algo", 2)
    _prepare_submit_approve(out, tmp_path, "algo", "0002", decision="stop")

    final = read_workflow(out)
    assert final["stages"]["algo"] == "stopped"
    assert len(final["rejections"]) == 3
    assert len(final["revisions"]) == 1
    assert [item["id"] for item in final["attempts"]["data"]] == [
        "0001",
        "0002",
        "0003",
    ]
    assert final["approvals"][-1]["decision"] == "stop"
    capsys.readouterr()
    assert main(["next", "--out", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "completed_no_go"
