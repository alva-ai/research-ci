"""Small CLI for the first protocol layer."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl

from .cards import contract_card, data_card
from .contract import ContractError, validate_contract
from .engine import run_eval_plan
from .workflow import (
    STAGE_ORDER,
    WorkflowError,
    new_workflow,
    next_action,
    read_workflow,
    require_stage_ready,
    submit_for_review,
    write_workflow,
)
from .workflow import (
    approve as approve_workflow,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rci")
    commands = parser.add_subparsers(dest="command", required=True)

    contract = commands.add_parser("contract", help="check an external contract.py")
    contract_commands = contract.add_subparsers(dest="contract_command", required=True)
    contract_check = contract_commands.add_parser("check")
    contract_check.add_argument("path", type=Path)
    contract_derive = contract_commands.add_parser(
        "derive", help="write a readable contract card"
    )
    contract_derive.add_argument("path", type=Path)
    contract_derive.add_argument("--out", required=True, type=Path)

    for command_name in ("data", "dataloop"):
        data = commands.add_parser(command_name, help="run an external data builder")
        data_commands = data.add_subparsers(dest="data_command", required=True)
        data_run = data_commands.add_parser("run")
        data_run.add_argument("--contract", required=True, type=Path)
        data_run.add_argument("--builder", required=True, type=Path)
        data_run.add_argument("--plan", required=True, type=Path)
        data_run.add_argument("--out", required=True, type=Path)

    evaluate = commands.add_parser("eval", help="run an external four-slot plan")
    eval_commands = evaluate.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_commands.add_parser("run")
    eval_run.add_argument("--stage", required=True)
    eval_run.add_argument("--table", required=True, type=Path)
    eval_run.add_argument("--plan", required=True, type=Path)
    eval_run.add_argument("--out", required=True, type=Path)

    status = commands.add_parser("status", help="show stage states and approvals")
    status.add_argument("--out", required=True, type=Path)

    next_command = commands.add_parser("next", help="show the next workflow action")
    next_command.add_argument("--out", required=True, type=Path)

    approve = commands.add_parser("approve", help="approve a reviewed stage card")
    approve.add_argument(
        "--stage", required=True, choices=("contract", "data", "feat", "sig", "algo")
    )
    approve.add_argument("--out", required=True, type=Path)
    approve.add_argument("--by", required=True, choices=("human", "agent"))
    approve.add_argument("--message", default="")
    approve.add_argument("--card", dest="expected_card")

    args = parser.parse_args(argv)
    if args.command == "contract":
        if args.contract_command == "derive":
            return _contract_derive(args.path, args.out)
        return _contract_check(args.path)
    if args.command in {"data", "dataloop"}:
        return _data_run(args.contract, args.builder, args.plan, args.out)
    if args.command == "eval":
        return _eval_run(args.stage, args.table, args.plan, args.out)
    if args.command == "status":
        return _status(args.out)
    if args.command == "next":
        return _next(args.out)
    if args.command == "approve":
        return _approve(
            args.stage,
            args.out,
            args.by,
            args.message,
            args.expected_card,
        )
    return 2


def _contract_check(path: Path) -> int:
    try:
        module = _load_module(path)
        validate_contract(module.CONTRACT)
    except (AttributeError, ContractError, OSError, SyntaxError, ValueError) as exc:
        print(f"contract failed: {exc}")
        return 1
    print("contract passed")
    return 0


def _contract_derive(path: Path, out: Path) -> int:
    try:
        module = _load_module(path)
        contract = module.CONTRACT
        validate_contract(contract)
        payload = asdict(contract)
        name = path.parent.name
        card, card_identifier = contract_card(name, payload)
        (out / "contract").mkdir(parents=True, exist_ok=True)
        (out / "contract" / "contract.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out / "contract" / "card.md").write_text(card + "\n", encoding="utf-8")
        write_workflow(out, new_workflow(contract.stages, card_identifier))
    except Exception as exc:  # noqa: BLE001 - CLI reports external workspace errors
        print(f"contract derive failed: {exc}")
        return 1
    print(card)
    print(f"contract card: {out / 'contract' / 'card.md'}")
    print(
        "next: rci approve "
        f"--stage contract --out {out} --by human --card {card_identifier}"
    )
    print(
        "   or: rci approve "
        f"--stage contract --out {out} --by agent --card {card_identifier}"
    )
    return 0


def _eval_run(stage: str, table_path: Path, plan_path: Path, out: Path) -> int:
    try:
        module = _load_module(plan_path)
        table = pl.read_parquet(table_path)
        plan = module.make_plan(table, stage=stage)
        result = run_eval_plan(table, plan)
        result.write_report(out / "eval" / "report.jsonl")
        (out / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "status": _result_status(result),
                    "records": len(result.records),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (AttributeError, OSError, SyntaxError, TypeError, ValueError) as exc:
        print(f"eval failed: {exc}")
        return 1
    print(f"eval {_result_status(result)}")
    return 0 if result.passed else 1


def _data_run(
    contract_path: Path, builder_path: Path, plan_path: Path, out: Path
) -> int:
    try:
        contract_module = _load_module(contract_path)
        validate_contract(contract_module.CONTRACT)
        if "data" not in contract_module.CONTRACT.stages:
            raise ValueError("contract does not enable the data stage")
        state = read_workflow(out)
        require_stage_ready(state, "data")

        builder = _load_module(builder_path)
        table = _build_table(builder)
        if not isinstance(table, pl.DataFrame):
            raise TypeError("data builder must return a Polars DataFrame")

        out_data = out / "data"
        out_data.mkdir(parents=True, exist_ok=True)
        table.write_parquet(out_data / "main.parquet")

        exploration = {}
        explore = getattr(builder, "explore", None)
        if explore is not None:
            exploration = explore()
            if not isinstance(exploration, dict):
                raise TypeError("data explore() must return a dict")
        (out_data / "exploration.json").write_text(
            json.dumps(exploration, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        plan_module = _load_module(plan_path)
        plan = plan_module.make_plan(table, stage="data")
        result = run_eval_plan(table, plan)
        result.write_report(out_data / "eval" / "report.jsonl")
        result_status = _result_status(result)
        card, card_identifier = data_card(
            table,
            result,
            exploration,
            result_status=result_status,
        )
        (out_data / "card.md").write_text(card + "\n", encoding="utf-8")
        state = submit_for_review(
            out,
            stage="data",
            card=card_identifier,
            result_status=result_status,
        )
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "data",
                    "status": result_status,
                    "workflow_state": state["stages"]["data"],
                    "card_id": card_identifier,
                    "rows": table.height,
                    "columns": table.columns,
                    "records": len(result.records),
                    "main_table": str(out_data / "main.parquet"),
                    "eval_report": str(out_data / "eval" / "report.jsonl"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports external workspace errors
        print(f"data run failed: {exc}")
        return 1
    print(card)
    print(f"data card: {out_data / 'card.md'}")
    print(
        "next: rci approve "
        f"--stage data --out {out} --by human --card {card_identifier}"
    )
    print(
        "   or: rci approve "
        f"--stage data --out {out} --by agent --card {card_identifier}"
    )
    return 0 if result.passed else 1


def _status(out: Path) -> int:
    try:
        state = read_workflow(out)
    except WorkflowError as exc:
        print(f"status failed: {exc}")
        return 1
    print(f"workflow: {out / 'workflow.json'}")
    for stage in STAGE_ORDER:
        status = state["stages"].get(stage, "disabled")
        card = state.get("cards", {}).get(stage, "-")
        result = state.get("results", {}).get(stage, "-")
        print(f"{stage}: {status} (card={card}, result={result})")
    approvals = state.get("approvals", [])
    print(f"approvals: {len(approvals)}")
    for item in approvals:
        print(f"- {item['stage']} by {item['by']} ({item['card']})")
    action = next_action(state)
    if action["stage"] is None:
        print("next: complete")
    else:
        print(
            f"next: {action['action']} {action['stage']}"
            f" ({action.get('card') or 'no card'})"
        )
    return 0


def _next(out: Path) -> int:
    try:
        state = read_workflow(out)
    except WorkflowError as exc:
        print(f"next failed: {exc}")
        return 1
    action = next_action(state)
    if action["stage"] is None:
        print("complete: all enabled stages are approved")
    else:
        print(f"stage: {action['stage']}")
        print(f"action: {action['action']}")
        print(f"card: {action.get('card') or '-'}")
        print(f"reason: {action['reason']}")
    return 0


def _approve(
    stage: str,
    out: Path,
    by: str,
    message: str,
    expected_card: str | None,
) -> int:
    try:
        _, next_stage = approve_workflow(
            out,
            stage=stage,
            by=by,
            message=message,
            expected_card=expected_card,
        )
    except WorkflowError as exc:
        print(f"approval failed: {exc}")
        return 1
    if next_stage is None:
        print(f"approved: {stage} by {by}; workflow complete")
    else:
        print(f"approved: {stage} by {by}; next stage ready: {next_stage}")
    return 0


def _build_table(module: ModuleType) -> pl.DataFrame:
    for name in ("build_main_table", "build"):
        builder = getattr(module, name, None)
        if builder is not None:
            return builder()
    raise AttributeError("data builder must define build_main_table() or build()")


def _result_status(result: Any) -> str:
    if not result.passed:
        return "halted"
    if any(record.status in {"failed", "error"} for record in result.records):
        return "review"
    return "passed"


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("rci_external_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
