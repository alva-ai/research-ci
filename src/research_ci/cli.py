"""Small CLI for the first protocol layer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import shlex
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl

from .artifacts import DataBundle, ReviewBrief
from .cards import (
    algo_card,
    contract_card,
    data_card,
    feat_card,
    render_terminal_card,
    review_card,
    sig_card,
)
from .contract import ContractError, validate_contract
from .eval import (
    AlgoSet,
    DataSet,
    EvalContext,
    EvalObservation,
    EvalSpec,
    FeatureSet,
    LibrarySelection,
    SignalSet,
    ValidationPlan,
    assess_features,
    assess_records,
    assess_signal_evidence,
    compile_stage_menu,
)
from .eval.baseline import (
    algo_backtest_output_shape,
    algo_backtest_summary_shape,
    algo_main_output_shape,
    assessment_output_shape,
    feature_output_shape,
    sidecar_output_shape,
    signal_disposition_coverage,
    signal_feature_universe,
    signal_lineage_coverage,
    signal_main_output_shape,
    signal_screening_artifact_coverage,
    signal_screening_plan_coverage,
)
from .eval.library import METHODS as LIBRARY_METHODS
from .eval.runner import run_eval_plan
from .kit import AlgoBundle, BacktestResult, BuildInfo, SignalBundle
from .workflow import (
    STAGE_ORDER,
    WorkflowError,
    begin_confirmation,
    finish_confirmation,
    latest_attempt,
    next_action,
    prepare_attempt,
    read_workflow,
    record_attempt,
    render_card,
    require_stage_ready,
    submit_contract_for_review,
    submit_for_review,
)
from .workflow import (
    approve as approve_workflow,
)
from .workflow import (
    reject as reject_workflow,
)
from .workflow import (
    reopen as reopen_workflow,
)
from .workflow import (
    revise as revise_workflow,
)

_INTERNAL_FEATURE_TABLE = "__rci_feature_table"
_INTERNAL_CONTEXT_PREFIX = "__rci_"

_STAGE_GUIDES: dict[str, dict[str, Any]] = {
    "data": {
        "objective": "assemble point-in-time source facts into a formal DataBundle",
        "input": "the approved Contract plus case-owned raw sources",
        "output": (
            "one aligned main table, normalized native-frequency tables, catalog, "
            "coverage, request log, and audit-only raw snapshots"
        ),
        "tools": [
            "Polars joins, backward as-of joins, alignment, and resampling",
            "case-owned downloaders and source reconciliation",
            "research_ci.kit.data helpers when they match the source shape",
        ],
        "required": [
            "all four eval slots are populated",
            "mandatory baseline and case rules pass",
            "catalog metadata covers main and every native-frequency table",
            "no feature, label, signal, position, or PnL semantics enter the table",
        ],
        "completion_checklist": [
            (
                "identify every source, frequency, event-time and "
                "availability-time meaning"
            ),
            "record coverage, freshness, proxy status and unresolved gaps",
            "preserve native source tables and explain every alignment or treatment",
            (
                "state whether the result is sufficient, insufficient, or needs "
                "another data attempt"
            ),
        ],
        "common_failure_modes": [
            "using a revised value before its publication time",
            "silently forward-filling stale or missing observations",
            "mixing features, labels, signals, or outcomes into the Data table",
            "claiming a source is available when only a proxy was downloaded",
        ],
    },
    "feat": {
        "objective": "build reusable point-in-time, target-free measurements",
        "input": (
            "the approved Data main table and normalized native-frequency tables; "
            "raw snapshots are inspection-only"
        ),
        "output": "the preserved key grid plus declared feature columns",
        "tools": [
            "name-free Polars expressions from research_ci.kit.feat",
            "rolling, normalization, robust transforms, and structural combinations",
            "optional matrix diagnostics such as PCA and VIF in eval sidecars",
        ],
        "required": [
            "the approved Data key grid is preserved",
            "any raw field used by Feat was first promoted through Data validation",
            "features are causal and target-free",
            "all four eval slots are populated",
        ],
        "completion_checklist": [
            "declare the complete feature universe and formula lineage",
            "replay every rolling, lagged, normalized, and transformed expression",
            "show coverage, warm-up behavior, dispersion, redundancy, and stability",
            "return to Data when a required raw field is missing instead of guessing",
        ],
        "common_failure_modes": [
            "filling warm-up rows without declaring a boundary policy",
            "using a future row or full-sample fit in a historical feature",
            "dropping inconvenient features without a disposition or reason",
            "using target-aware IC or labels before the Sig stage",
        ],
    },
    "sig": {
        "objective": "turn the complete approved feature universe into signals",
        "input": "the approved Feat table and explicit label sidecars",
        "output": "wide signal handoff plus research evidence and feature dispositions",
        "tools": [
            "rules, ranks, quantiles, composites, PCA, fitted models, or ensembles",
            "time splits and forward labels from research_ci.kit.sig",
            "target-aware methods selected from the Sig eval menu",
            "open-ended ValidationPlan methods with optional one-time confirmation",
        ],
        "required": [
            "every approved feature has one evidenced disposition",
            "candidate and final signal lineage is complete",
            "confirmation, when declared, freezes development before holdout reveal",
            "all four eval slots are populated",
        ],
        "completion_checklist": [
            (
                "declare labels, horizons, time splits, and the complete approved "
                "Feat universe"
            ),
            "screen every approved feature and give each one an evidenced disposition",
            (
                "compare candidates with stability, counterevidence, and a "
                "predeclared selection rule"
            ),
            "separate development evidence from confirmation or holdout evidence",
        ],
        "common_failure_modes": [
            "jumping directly to one convenient rule and skipping most features",
            "tuning on the validation or holdout period",
            "treating correlation or IC as causal proof",
            "publishing a signal without an explicit label sidecar and lineage",
        ],
    },
    "algo": {
        "objective": "map approved signals into executable positions or trades",
        "input": "only the approved Sig handoff plus declared execution context",
        "output": "positions or trades and explicit external backtest evidence",
        "tools": [
            "bounded sizing and pair-weight helpers from research_ci.kit.algo",
            "case-owned risk, execution, cost, and backtest implementations",
            "optional visualization after a candidate is submitted",
        ],
        "required": [
            "source prices and signals are preserved",
            "position and backtest accounting satisfy the declared contract",
            "all four eval slots are populated",
        ],
        "completion_checklist": [
            "declare how each signal becomes a position, order, or trade",
            (
                "measure exposure, leverage, turnover, costs, drawdown, and "
                "liquidity constraints"
            ),
            "run the external backtest and preserve its assumptions and time series",
            "state whether the result is executable, insufficient, or rejected",
        ],
        "common_failure_modes": [
            "treating a signal as a position without sizing or execution semantics",
            "omitting fees, slippage, funding, or trading lag",
            "using prices or signals that do not come from the approved Sig handoff",
            "reporting return without risk, turnover, or drawdown evidence",
        ],
    },
}


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
        data_run.add_argument("--context", type=Path)
        _add_submit_parser(data_commands)

    feat = commands.add_parser("feat", help="run an external feature builder")
    feat_commands = feat.add_subparsers(dest="feat_command", required=True)
    feat_run = feat_commands.add_parser("run")
    feat_run.add_argument("--builder", required=True, type=Path)
    feat_run.add_argument("--plan", required=True, type=Path)
    feat_run.add_argument("--out", required=True, type=Path)
    feat_run.add_argument("--context", type=Path)
    _add_submit_parser(feat_commands)

    sig = commands.add_parser("sig", help="run an external signal builder")
    sig_commands = sig.add_subparsers(dest="sig_command", required=True)
    sig_run = sig_commands.add_parser("run")
    sig_run.add_argument("--builder", required=True, type=Path)
    sig_run.add_argument("--plan", required=True, type=Path)
    sig_run.add_argument("--out", required=True, type=Path)
    sig_run.add_argument("--context", type=Path)
    sig_freeze = sig_commands.add_parser(
        "freeze", help="freeze a Sig recipe and reveal one confirmation holdout"
    )
    sig_freeze.add_argument("--out", required=True, type=Path)
    sig_freeze.add_argument("--attempt")
    sig_freeze.add_argument("--context", required=True, type=Path)
    _add_submit_parser(sig_commands)

    algo = commands.add_parser("algo", help="run an external portfolio builder")
    algo_commands = algo.add_subparsers(dest="algo_command", required=True)
    algo_run = algo_commands.add_parser("run")
    algo_run.add_argument("--builder", required=True, type=Path)
    algo_run.add_argument("--plan", required=True, type=Path)
    algo_run.add_argument("--backtester", required=True, type=Path)
    algo_run.add_argument("--out", required=True, type=Path)
    algo_run.add_argument("--context", type=Path)
    _add_submit_parser(algo_commands)

    evaluate = commands.add_parser("eval", help="run an external four-slot plan")
    eval_commands = evaluate.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_commands.add_parser("run")
    eval_run.add_argument("--stage", required=True)
    eval_run.add_argument("--table", required=True, type=Path)
    eval_run.add_argument("--plan", required=True, type=Path)
    eval_run.add_argument("--out", required=True, type=Path)
    eval_run.add_argument("--context", type=Path)
    eval_methods = eval_commands.add_parser(
        "methods", help="list reusable methodology methods"
    )
    eval_methods.add_argument("--stage", choices=("data", "feat", "sig", "algo"))
    eval_methods.add_argument("--family")
    eval_methods.add_argument("--declaration", type=Path)
    eval_methods.add_argument("--available-only", action="store_true")
    eval_methods.add_argument("--json", action="store_true", dest="as_json")

    visualize = commands.add_parser(
        "visualize", help="run an optional external stage renderer"
    )
    visualize.add_argument(
        "--stage", required=True, choices=("data", "feat", "sig", "algo")
    )
    visualize.add_argument("--renderer", required=True, type=Path)
    visualize.add_argument("--out", required=True, type=Path)

    status = commands.add_parser("status", help="show stage states and approvals")
    status.add_argument("--out", required=True, type=Path)

    next_command = commands.add_parser("next", help="show the next workflow action")
    next_command.add_argument("--out", required=True, type=Path)
    next_command.add_argument("--json", action="store_true", dest="as_json")

    approve = commands.add_parser("approve", help="approve a reviewed stage card")
    approve.add_argument(
        "--stage", required=True, choices=("contract", "data", "feat", "sig", "algo")
    )
    approve.add_argument("--out", required=True, type=Path)
    approve.add_argument(
        "--by",
        choices=("human", "agent"),
        help="reviewer identity; omit in an interactive terminal for human review",
    )
    approve.add_argument(
        "--decision",
        choices=("continue", "stop"),
        help="review decision; omit in an interactive terminal to see a yes/no prompt",
    )
    approve.add_argument("--message", default="")
    approve.add_argument(
        "--card",
        dest="expected_card",
        help="Card id; omit in an interactive terminal to bind the displayed Card",
    )

    reject = commands.add_parser("reject", help="return a reviewed stage to its Agent")
    reject.add_argument(
        "--stage", required=True, choices=("contract", "data", "feat", "sig", "algo")
    )
    reject.add_argument("--out", required=True, type=Path)
    reject.add_argument(
        "--by",
        choices=("human", "agent"),
        help="reviewer identity; omit in an interactive terminal for human review",
    )
    reject.add_argument(
        "--message",
        help="feedback; in a terminal, omit it to enter feedback after confirmation",
    )
    reject.add_argument(
        "--card",
        dest="expected_card",
        help="Card id; omit in an interactive terminal to bind the displayed Card",
    )

    reopen = commands.add_parser("reopen", help="reopen an accepted stage")
    reopen.add_argument(
        "--stage", required=True, choices=("contract", "data", "feat", "sig", "algo")
    )
    reopen.add_argument("--out", required=True, type=Path)
    reopen.add_argument("--by", required=True, choices=("human", "agent"))
    reopen.add_argument("--message", required=True)

    revise = commands.add_parser(
        "revise", help="return to an approved upstream stage from an active loop"
    )
    revise.add_argument("--stage", required=True, choices=("data", "feat", "sig"))
    revise.add_argument("--out", required=True, type=Path)
    revise.add_argument("--message", required=True)

    guide = commands.add_parser("guide", help="show optional stage guidance")
    guide.add_argument(
        "--stage", required=True, choices=("data", "feat", "sig", "algo")
    )
    guide.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.command == "contract":
        if args.contract_command == "derive":
            return _contract_derive(args.path, args.out)
        return _contract_check(args.path)
    if args.command in {"data", "dataloop"}:
        if args.data_command == "prepare":
            return _stage_prepare("data", args.out, args.attempt, args.brief)
        if args.data_command == "submit":
            return _stage_submit("data", args.out, args.recommendation, args.attempt)
        return _data_run(args.contract, args.builder, args.plan, args.out, args.context)
    if args.command == "feat":
        if args.feat_command == "prepare":
            return _stage_prepare("feat", args.out, args.attempt, args.brief)
        if args.feat_command == "submit":
            return _stage_submit("feat", args.out, args.recommendation, args.attempt)
        return _feat_run(args.builder, args.plan, args.out, args.context)
    if args.command == "sig":
        if args.sig_command == "freeze":
            return _sig_freeze(args.out, args.attempt, args.context)
        if args.sig_command == "submit":
            return _stage_submit("sig", args.out, args.recommendation, args.attempt)
        if args.sig_command == "prepare":
            return _stage_prepare("sig", args.out, args.attempt, args.brief)
        return _sig_run(args.builder, args.plan, args.out, args.context)
    if args.command == "algo":
        if args.algo_command == "prepare":
            return _stage_prepare("algo", args.out, args.attempt, args.brief)
        if args.algo_command == "submit":
            return _stage_submit("algo", args.out, args.recommendation, args.attempt)
        return _algo_run(
            args.builder,
            args.plan,
            args.backtester,
            args.out,
            args.context,
        )
    if args.command == "eval":
        if args.eval_command == "methods":
            return _eval_methods(
                args.stage,
                args.family,
                args.declaration,
                args.available_only,
                args.as_json,
            )
        return _eval_run(args.stage, args.table, args.plan, args.out, args.context)
    if args.command == "visualize":
        return _visualize(args.stage, args.renderer, args.out)
    if args.command == "status":
        return _status(args.out)
    if args.command == "next":
        return _next(args.out, args.as_json)
    if args.command == "approve":
        return _approve(
            args.stage,
            args.out,
            args.by,
            args.decision,
            args.message,
            args.expected_card,
        )
    if args.command == "reject":
        return _reject(
            args.stage,
            args.out,
            args.by,
            args.message,
            args.expected_card,
        )
    if args.command == "reopen":
        return _reopen(args.stage, args.out, args.by, args.message)
    if args.command == "revise":
        return _revise(args.stage, args.out, args.message)
    if args.command == "guide":
        return _guide(args.stage, args.as_json)
    return 2


def _add_submit_parser(commands: argparse._SubParsersAction[Any]) -> None:
    prepare = commands.add_parser(
        "prepare", help="declare a valid attempt ready for external review"
    )
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--attempt")
    prepare.add_argument("--brief", required=True, type=Path)
    submit = commands.add_parser("submit", help="submit a valid attempt for review")
    submit.add_argument("--out", required=True, type=Path)
    submit.add_argument("--attempt")
    submit.add_argument(
        "--recommendation",
        required=True,
        choices=("continue", "stop", "uncertain"),
    )


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
        submit_contract_for_review(
            out,
            card=card_identifier,
            enabled_stages=contract.stages,
            commands=_workflow_commands(path, out),
        )
        (out / "contract").mkdir(parents=True, exist_ok=True)
        (out / "contract" / "contract.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out / "contract" / "card.md").write_text(card + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - CLI reports external workspace errors
        print(f"contract derive failed: {exc}")
        return 1
    print(card)
    print(f"contract card: {out / 'contract' / 'card.md'}")
    print(
        "next (agent reviewer): rci approve "
        f"--stage contract --out {out} --by agent --decision continue "
        f"--card {card_identifier}"
    )
    print(
        "   or (human reviewer): rci approve "
        f"--stage contract --out {out} --by human --decision continue "
        f"--card {card_identifier}"
    )
    print(
        "   or: rci reject "
        f"--stage contract --out {out} --by agent --message <reason> "
        f"--card {card_identifier}"
    )
    return 0


def _eval_run(
    stage: str,
    table_path: Path,
    plan_path: Path,
    out: Path,
    context_path: Path | None = None,
) -> int:
    try:
        module = _load_module(plan_path)
        table = pl.read_parquet(table_path)
        context = _load_context(context_path)
        plan = _make_plan(module, table, stage, context)
        result = run_eval_plan(table, plan, context)
        result.write_report(out / "eval" / "report.jsonl")
        (out / "eval" / "assessment.json").write_text(
            json.dumps(assess_records(result.records), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "status": _result_status(result),
                    "records": len(result.records),
                    "assessment": str(out / "eval" / "assessment.json"),
                    "context_tables": sorted(context.tables),
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


def _eval_methods(
    stage: str | None,
    family: str | None,
    declaration_path: Path | None,
    available_only: bool,
    as_json: bool,
) -> int:
    """Expose the methodology catalog to external research agents."""

    try:
        declaration = (
            _load_stage_declaration(declaration_path, stage)
            if declaration_path is not None
            else None
        )
        if stage is None and declaration is not None:
            stage = declaration.stage
        if stage is None:
            methods = [
                method
                for method in LIBRARY_METHODS
                if family is None or method.family == family
            ]
            payload: object = [
                {
                    "name": method.name,
                    "family": method.family,
                    "purpose": method.purpose,
                    "stages": method.stages,
                    "input_kind": method.input_kind,
                    "input_roles": method.input_roles,
                    "requires": method.requires,
                    "fit_policy": method.fit_policy,
                    "description": method.description,
                }
                for method in methods
            ]
            menu = None
        else:
            menu = compile_stage_menu(
                stage,
                declaration,
                family=family,
                include_unavailable=not available_only,
            )
            payload = menu.as_dict()
    except (AttributeError, ImportError, OSError, SyntaxError, ValueError) as exc:
        print(f"eval methods failed: {exc}")
        return 1
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif menu is None:
        for method in payload:  # type: ignore[union-attr]
            stages = ",".join(method["stages"])  # type: ignore[index]
            print(
                f"{method['name']} | family={method['family']} "  # type: ignore[index]
                f"| purpose={method['purpose']} "  # type: ignore[index]
                f"| input={method['input_kind']} | stages={stages}"  # type: ignore[index]
            )
            print(f"  {method['description']}")  # type: ignore[index]
    else:
        policy = menu.policy
        print(f"stage: {policy.stage}")
        print(f"objective: {policy.objective}")
        print(f"input: {policy.input_boundary}")
        print(f"output: {policy.output_boundary}")
        print(f"allowed purposes: {', '.join(policy.allowed_purposes)}")
        print(f"next stage: {policy.next_stage or '-'}")
        for entry in menu.entries:
            labels = [entry.availability]
            if entry.recommended:
                labels.append("recommended")
            method = entry.method
            print(
                f"[{','.join(labels)}] {method.name} | family={method.family} "
                f"| purpose={method.purpose} | input={method.input_kind}"
            )
            print(f"  {method.description}")
            if method.requires:
                print(f"  requires: {', '.join(method.requires)}")
            if entry.reasons:
                print(f"  unavailable because: {'; '.join(entry.reasons)}")
    return 0


def _load_stage_declaration(
    path: Path,
    stage: str | None,
) -> DataSet | FeatureSet | SignalSet | AlgoSet:
    module = _load_module(path)
    declarations = [
        value
        for name in (
            "DECLARATION",
            "DATA_SET",
            "FEATURE_SET",
            "SIGNAL_SET",
            "ALGO_SET",
        )
        if isinstance(
            (value := getattr(module, name, None)),
            (DataSet, FeatureSet, SignalSet, AlgoSet),
        )
    ]
    if stage is not None:
        declarations = [item for item in declarations if item.stage == stage]
    unique = list({id(item): item for item in declarations}.values())
    if not unique:
        raise ValueError(
            "declaration module must expose DECLARATION, DATA_SET, FEATURE_SET, "
            "SIGNAL_SET, or ALGO_SET"
        )
    if len(unique) > 1:
        raise ValueError("declaration module exposes more than one matching stage set")
    return unique[0]


def _data_run(
    contract_path: Path,
    builder_path: Path,
    plan_path: Path,
    out: Path,
    context_path: Path | None = None,
) -> int:
    try:
        contract_module = _load_module(contract_path)
        validate_contract(contract_module.CONTRACT)
        if "data" not in contract_module.CONTRACT.stages:
            raise ValueError("contract does not enable the data stage")
        state = read_workflow(out)
        require_stage_ready(state, "data")
        attempt, out_data = _allocate_attempt(out, state, "data")

        builder = _load_module(builder_path)
        built = _build_table(builder)
        if isinstance(built, pl.DataFrame):
            bundle = DataBundle.from_main(built)
        elif isinstance(built, DataBundle):
            bundle = built
        else:
            raise TypeError(
                "data builder must return a Polars DataFrame or research_ci.DataBundle"
            )
        table = bundle.main
        bundle_manifest = bundle.write(out_data)

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
        recipe = _builder_recipe(builder, "data")
        (out_data / "recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        plan_module = _load_module(plan_path)
        context = _with_data_bundle_context(_load_context(context_path), bundle)
        plan = _make_plan(plan_module, table, "data", context)
        bundle_rule = EvalSpec(
            "engine.baseline.data_bundle_shape",
            runner=lambda _: _data_bundle_shape(bundle),
            failure_policy="stop",
            description=(
                "validate catalog coverage and unique keys for main and native tables"
            ),
            execution_mode="polars_select",
        )
        plan = replace(plan, baseline=(bundle_rule, *plan.baseline))
        result = run_eval_plan(table, plan, context)
        result.write_report(out_data / "eval" / "report.jsonl")
        assessment = assess_records(result.records)
        (out_data / "eval" / "assessment.json").write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checks_status = _checks_status(result)
        card, card_identifier = data_card(
            table,
            result,
            exploration,
            result_status=checks_status,
            workflow_state="ready",
            artifact_role="attempt",
        )
        (out_data / "card.md").write_text(card + "\n", encoding="utf-8")
        (out_data / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "data",
                    "attempt": attempt,
                    "checks": checks_status,
                    "workflow_state": "ready",
                    "card_id": card_identifier,
                    "rows": table.height,
                    "columns": table.columns,
                    "records": len(result.records),
                    "data_bundle": bundle_manifest,
                    "main_table": str(out_data / "main.parquet"),
                    "recipe": str(out_data / "recipe.json"),
                    "eval_report": str(out_data / "eval" / "report.jsonl"),
                    "assessment": str(out_data / "eval" / "assessment.json"),
                    "context_tables": sorted(context.tables),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record_attempt(
            out,
            stage="data",
            attempt=attempt,
            path=str(out_data),
            valid=result.passed,
            card=card_identifier,
            checks=checks_status,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports external workspace errors
        print(f"data run failed: {exc}")
        return 1
    print(f"data attempt {attempt}: {checks_status}")
    print(f"attempt report: {out_data / 'card.md'}")
    if result.passed:
        print(
            "next: inspect the evidence, run another attempt, or prepare this "
            f"result with `rci data prepare --out {out} --attempt {attempt} "
            "--brief <review-brief.json>`"
        )
    else:
        print("next: fix the reported mandatory checks and run data again")
    return 0 if result.passed else 1


def _feat_run(
    builder_path: Path,
    plan_path: Path,
    out: Path,
    context_path: Path | None = None,
) -> int:
    try:
        state = read_workflow(out)
        require_stage_ready(state, "feat")
        attempt, out_feat = _allocate_attempt(out, state, "feat")
        data_dir = out / "data"
        data_path = data_dir / "main.parquet"
        data_bundle = _read_data_bundle(data_dir)
        data_table = data_bundle.main
        data_before = {
            name: table.clone() for name, table in data_bundle.tables().items()
        }
        catalog_before = data_bundle.as_catalog()
        metadata_before = dict(data_bundle.metadata)
        coverage_before = data_bundle.coverage().clone()

        builder = _load_module(builder_path)
        feature_table = _build_features(builder, data_bundle)
        if not isinstance(feature_table, pl.DataFrame):
            raise TypeError("feature builder must return a Polars DataFrame")
        data_after = data_bundle.tables()
        changed = sorted(set(data_before) ^ set(data_after))
        changed.extend(
            name
            for name in sorted(set(data_before) & set(data_after))
            if not data_after[name].equals(data_before[name])
        )
        if data_bundle.as_catalog() != catalog_before:
            changed.append("catalog")
        if dict(data_bundle.metadata) != metadata_before:
            changed.append("metadata")
        if not data_bundle.coverage().equals(coverage_before):
            changed.append("source_coverage")
        if changed:
            raise ValueError(f"feature builder modified DataBundle tables: {changed}")

        keys = _feature_keys(builder, data_table, feature_table)
        plan_module = _load_module(plan_path)
        context = _with_data_bundle_context(_load_context(context_path), data_bundle)
        plan = _make_plan(plan_module, feature_table, "feat", context)
        shape_rule = EvalSpec(
            "engine.baseline.feature_output_shape",
            runner=lambda _: feature_output_shape(data_table, feature_table, keys),
            failure_policy="stop",
            description="preserve the approved Data key grid",
            execution_mode="polars_join_group",
        )
        plan = replace(plan, baseline=(shape_rule, *plan.baseline))
        result = run_eval_plan(feature_table, plan, context)

        feature_table.write_parquet(out_feat / "main.parquet")

        exploration = {}
        explore = getattr(builder, "explore", None)
        if explore is not None:
            exploration = explore()
            if not isinstance(exploration, dict):
                raise TypeError("feature explore() must return a dict")
        (out_feat / "exploration.json").write_text(
            json.dumps(exploration, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        recipe = _builder_recipe(builder, "feat")
        (out_feat / "recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        result.write_report(out_feat / "eval" / "report.jsonl")
        assessment = assess_records(result.records)
        assessment["feature_triage"] = assess_features(
            feature_table, keys, result.records
        )
        (out_feat / "eval" / "assessment.json").write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checks_status = _checks_status(result)
        card, card_identifier = feat_card(
            feature_table,
            result,
            exploration,
            input_rows=data_table.height,
            keys=keys,
            result_status=checks_status,
            workflow_state="ready",
            artifact_role="attempt",
        )
        (out_feat / "card.md").write_text(card + "\n", encoding="utf-8")
        feature_columns = [
            column for column in feature_table.columns if column not in keys
        ]
        (out_feat / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "feat",
                    "attempt": attempt,
                    "checks": checks_status,
                    "workflow_state": "ready",
                    "card_id": card_identifier,
                    "input_table": str(data_path),
                    "input_data_bundle": {
                        "catalog": data_bundle.as_catalog(),
                        "native_tables": sorted(data_bundle.native),
                    },
                    "main_table": str(out_feat / "main.parquet"),
                    "recipe": str(out_feat / "recipe.json"),
                    "eval_report": str(out_feat / "eval" / "report.jsonl"),
                    "assessment": str(out_feat / "eval" / "assessment.json"),
                    "context_tables": sorted(context.tables),
                    "rows": feature_table.height,
                    "columns": feature_table.columns,
                    "keys": keys,
                    "feature_columns": feature_columns,
                    "records": len(result.records),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record_attempt(
            out,
            stage="feat",
            attempt=attempt,
            path=str(out_feat),
            valid=result.passed,
            card=card_identifier,
            checks=checks_status,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports external workspace errors
        print(f"feat run failed: {exc}")
        return 1
    print(f"feat attempt {attempt}: {checks_status}")
    print(f"attempt report: {out_feat / 'card.md'}")
    if result.passed:
        print(
            "next: inspect the evidence, run another attempt, or prepare this "
            f"result with `rci feat prepare --out {out} --attempt {attempt} "
            "--brief <review-brief.json>`"
        )
    else:
        print("next: fix the reported mandatory checks and run feat again")
    return 0 if result.passed else 1


def _sig_run(
    builder_path: Path,
    plan_path: Path,
    out: Path,
    context_path: Path | None = None,
) -> int:
    try:
        state = read_workflow(out)
        require_stage_ready(state, "sig")
        attempt, out_sig = _allocate_attempt(out, state, "sig")
        feature_path = out / "feat" / "main.parquet"
        feature_table = pl.read_parquet(feature_path)
        feature_before = feature_table.clone()

        builder = _load_module(builder_path)
        bundle = _build_signals(builder, feature_table)
        if not feature_table.equals(feature_before):
            raise ValueError("signal builder modified the input Feature DataFrame")

        plan_module = _load_module(plan_path)
        declaration = _signal_declaration(plan_module)
        validation_plan = _signal_validation_plan(plan_module)
        validation_payload = (
            validation_plan.as_dict() if validation_plan is not None else None
        )
        confirmation_required = bool(
            validation_plan is not None and validation_plan.confirmation is not None
        )
        library_selection = getattr(plan_module, "LIBRARY_SELECTION", None)
        if not isinstance(library_selection, LibrarySelection):
            raise TypeError(
                "signal plan must expose a LibrarySelection as LIBRARY_SELECTION"
            )
        assessment_mode = bool(dict(bundle.metadata).get("assessment", False))
        keys = declaration.keys
        assessment_metadata = dict(bundle.metadata)
        frozen_assessment_keys = ("thesis_id", "author_version_id", "as_of")
        if assessment_mode and all(
            field in assessment_metadata for field in frozen_assessment_keys
        ):
            # A frozen Thesis Assessment is keyed by its immutable research
            # identity.  Keeping event_time or an instrument key here would
            # make a one-row verdict look like a daily signal and breaks the
            # separation between research sidecars and the final assessment.
            if tuple(keys) != frozen_assessment_keys:
                raise ValueError(
                    "assessment mode with thesis/version/as-of metadata requires "
                    "SignalSet.keys=('thesis_id', 'author_version_id', 'as_of'); "
                    f"got {tuple(keys)!r}"
                )
        # An assessment output has its own immutable identifier (for example
        # thesis/version/as-of), while its research inputs still come from the
        # approved Feat table's native key grid.  Keep those two key spaces
        # separate so timestamp columns are not mistaken for undeclared
        # research features.
        feature_keys = (
            _feature_keys(builder, feature_table, feature_table)
            if assessment_mode
            else keys
        )
        expected_signal_columns = (*keys, *declaration.signals)
        missing_main = sorted(set(expected_signal_columns) - set(bundle.main.columns))
        extra_main = sorted(set(bundle.main.columns) - set(expected_signal_columns))
        if missing_main or extra_main:
            raise ValueError(
                "SignalBundle.main must contain exactly declared keys and signals; "
                f"missing={missing_main}, extra={extra_main}"
            )
        signal_table = bundle.main.select(expected_signal_columns)
        expected_candidates = (*keys, *declaration.candidates)
        missing_candidates = sorted(
            set(expected_candidates) - set(bundle.candidates.columns)
        )
        extra_candidates = sorted(
            set(bundle.candidates.columns) - set(expected_candidates)
        )
        if missing_candidates or extra_candidates:
            raise ValueError(
                "signal candidate table must contain exactly declared keys and "
                f"candidates; missing={missing_candidates}, extra={extra_candidates}"
            )
        candidate_table = bundle.candidates.select(expected_candidates)
        required_feature_columns = (
            declaration.features if assessment_mode else (*keys, *declaration.features)
        )
        missing_features = sorted(
            set(required_feature_columns) - set(feature_table.columns)
        )
        if missing_features:
            raise ValueError(
                f"approved Feat table is missing Sig inputs: {missing_features}"
            )
        data_path = out / "data" / "main.parquet"
        if assessment_mode:
            # A Thesis Assessment is a Sig research product, not an Algo input.
            # Preserve the declared output exactly and avoid inventing an
            # instrument or price column merely to satisfy trading semantics.
            main_table = signal_table
        else:
            main_table = _build_signal_main(
                builder,
                data_path,
                signal_table,
            )

        external_context = _load_context(context_path)
        duplicate_sidecars = sorted(set(external_context.tables) & set(bundle.sidecars))
        if duplicate_sidecars:
            raise ValueError(
                f"signal sidecars collide with external context: {duplicate_sidecars}"
            )
        if assessment_mode and (
            _INTERNAL_FEATURE_TABLE in external_context.tables
            or _INTERNAL_FEATURE_TABLE in bundle.sidecars
        ):
            raise ValueError(
                f"{_INTERNAL_FEATURE_TABLE!r} is reserved for the assessment "
                "Engine context"
            )
        context_tables = {**external_context.tables, **bundle.sidecars}
        if assessment_mode:
            internal_feature_columns = tuple(
                dict.fromkeys((*feature_keys, *declaration.features))
            )
            context_tables[_INTERNAL_FEATURE_TABLE] = feature_table.select(
                *internal_feature_columns
            )
        context = EvalContext(
            tables=context_tables,
            metadata={**external_context.metadata, **bundle.metadata},
        )
        public_context_tables = tuple(
            sorted(
                name
                for name in context.tables
                if not name.startswith(_INTERNAL_CONTEXT_PREFIX)
            )
        )
        if assessment_mode:
            evaluation_table = signal_table
            if declaration.candidates:
                evaluation_table = evaluation_table.join(
                    candidate_table,
                    on=list(keys),
                    how="left",
                )
        else:
            evaluation_table = feature_table.select(
                *keys,
                *declaration.features,
            )
            if declaration.candidates:
                evaluation_table = evaluation_table.join(
                    candidate_table,
                    on=list(keys),
                    how="left",
                )
            evaluation_table = evaluation_table.join(
                signal_table,
                on=list(keys),
                how="left",
            )

        plan = _make_plan(plan_module, evaluation_table, "sig", context)
        shape_rule = EvalSpec(
            "engine.baseline.assessment_output_shape"
            if assessment_mode
            else "engine.baseline.signal_output_shape",
            runner=(
                (
                    lambda _: assessment_output_shape(
                        signal_table, keys, declaration.signals
                    )
                )
                if assessment_mode
                else (lambda _: feature_output_shape(feature_table, signal_table, keys))
            ),
            failure_policy="stop",
            description=(
                "validate one unique row per declared Thesis Assessment key"
                if assessment_mode
                else "preserve the approved Feat key grid in the signal output"
            ),
            execution_mode="polars_select" if assessment_mode else "polars_join_group",
        )
        all_outputs = (*declaration.candidates, *declaration.signals)
        baseline_rules = [
            EvalSpec(
                "engine.baseline.signal_feature_universe",
                runner=lambda _: signal_feature_universe(
                    feature_table,
                    feature_keys,
                    declaration.features,
                ),
                failure_policy="stop",
                description="declare every feature in the approved Feat table",
                execution_mode="polars_select",
            ),
            EvalSpec(
                "engine.baseline.signal_screening_plan_coverage",
                runner=lambda _: signal_screening_plan_coverage(
                    declaration,
                    library_selection,
                ),
                failure_policy="stop",
                description=(
                    "screen every approved feature against every declared label"
                ),
                execution_mode="custom_runner",
            ),
            EvalSpec(
                "engine.baseline.signal_screening_artifact_coverage",
                runner=lambda _: signal_screening_artifact_coverage(
                    bundle.research.get("screening"),
                    declaration.features,
                    declaration.targets,
                ),
                failure_policy="stop",
                description="save reusable screening evidence for the full universe",
                execution_mode="polars_select",
            ),
            EvalSpec(
                "engine.baseline.signal_feature_disposition_coverage",
                runner=lambda _: signal_disposition_coverage(
                    feature_table,
                    feature_keys,
                    bundle.dispositions,
                    all_outputs,
                ),
                failure_policy="stop",
                description="give every approved feature one explained disposition",
                execution_mode="custom_runner",
            ),
            EvalSpec(
                "engine.baseline.signal_lineage_coverage",
                runner=lambda _: signal_lineage_coverage(
                    declaration.features,
                    declaration.candidates,
                    declaration.signals,
                    bundle.lineage,
                ),
                failure_policy="stop",
                description="trace every candidate and final signal to named inputs",
                execution_mode="custom_runner",
            ),
            shape_rule,
        ]
        if declaration.candidates and not assessment_mode:
            baseline_rules.insert(
                -1,
                EvalSpec(
                    "engine.baseline.signal_candidate_output_shape",
                    runner=lambda _: feature_output_shape(
                        feature_table,
                        candidate_table,
                        keys,
                    ),
                    failure_policy="stop",
                    description="preserve the approved Feat key grid in candidates",
                    execution_mode="polars_join_group",
                ),
            )
        if declaration.targets:
            baseline_rules.append(
                EvalSpec(
                    "engine.baseline.label_sidecar_shape",
                    context_runner=lambda _, active_context: _label_sidecar_shape(
                        feature_table,
                        declaration,
                        active_context,
                        key_columns=feature_keys if assessment_mode else None,
                        allow_subset=confirmation_required,
                    ),
                    failure_policy="stop",
                    description="preserve keys and declared labels in the sidecar",
                    execution_mode="polars_join_group",
                )
            )
        if not assessment_mode:
            baseline_rules.append(
                EvalSpec(
                    "engine.baseline.signal_main_output_shape",
                    runner=lambda _: signal_main_output_shape(
                        signal_table,
                        main_table,
                        keys,
                        declaration.signals,
                    ),
                    failure_policy="stop",
                    description=(
                        "write the direct Algo input as one wide row per signal key "
                        "with named instrument and price columns"
                    ),
                    execution_mode="polars_join_group",
                )
            )
        plan = replace(plan, baseline=(*baseline_rules, *plan.baseline))
        result = run_eval_plan(evaluation_table, plan, context)

        main_table.write_parquet(out_sig / "main.parquet")
        (out_sig / "algo_input.parquet").unlink(missing_ok=True)
        research_dir = out_sig / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        candidate_table.write_parquet(research_dir / "candidates.parquet")
        disposition_table = bundle.disposition_frame()
        disposition_table.write_parquet(research_dir / "disposition.parquet")
        (research_dir / "lineage.json").write_text(
            json.dumps(
                bundle.lineage_records(),
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        research_manifest = {
            "candidates": str(research_dir / "candidates.parquet"),
            "disposition": str(research_dir / "disposition.parquet"),
            "lineage": str(research_dir / "lineage.json"),
        }
        for name, research_table in sorted(bundle.research.items()):
            _safe_artifact_name(name)
            path = research_dir / f"{name}.parquet"
            research_table.write_parquet(path)
            research_manifest[name] = str(path)
        context_dir = out_sig / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        for name in public_context_tables:
            sidecar = context.tables[name]
            _safe_artifact_name(name)
            sidecar.write_parquet(context_dir / f"{name}.parquet")
        (context_dir / "metadata.json").write_text(
            json.dumps(dict(context.metadata), indent=2, sort_keys=True, default=str)
            + "\n",
            encoding="utf-8",
        )

        recipe = _builder_recipe(builder, "sig")
        (out_sig / "recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        selection = dict(bundle.metadata).get("selection", {})
        (out_sig / "selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (out_sig / "validation_plan.json").write_text(
            json.dumps(
                validation_payload
                or {
                    "development": [],
                    "selection_rule": "legacy_unspecified",
                    "confirmation": None,
                    "evidence_scope": "development_only",
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        source_manifest = {
            "builder": _source_identity(builder_path),
            "plan": _source_identity(plan_path),
        }
        (out_sig / "provenance.json").write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_manifest = _write_signal_artifacts(out_sig, bundle.artifacts)

        result.write_report(out_sig / "eval" / "report.jsonl")
        assessment = assess_records(result.records)
        if assessment_mode:
            # Assessment output is a research verdict keyed by thesis/version/as-of.
            # Do not attach the trading-oriented signal triage fields to it.
            assessment["assessment_mode"] = True
            assessment["assessment_status"] = dict(bundle.metadata).get(
                "assessment_status", "undeclared"
            )
            assessment["premise_statuses"] = dict(bundle.metadata).get(
                "premise_statuses", {}
            )
            assessment["research_tables"] = sorted(bundle.research)
        else:
            assessment["signal_triage"] = assess_features(
                signal_table,
                keys,
                result.records,
            )
            assessment["signal_evidence"] = assess_signal_evidence(
                result.records,
                declaration.signals,
                declaration.features,
            )
            assessment["candidate_signal_evidence"] = assess_signal_evidence(
                result.records,
                declaration.candidates,
            )
        assessment["feature_disposition"] = {
            "features": disposition_table.height,
            "actions": disposition_table.group_by("action")
            .len()
            .sort("action")
            .to_dicts(),
        }
        (out_sig / "eval" / "assessment.json").write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checks_status = _checks_status(result)
        card, card_identifier = sig_card(
            main_table,
            result,
            dict(bundle.metadata),
            input_rows=feature_table.height,
            evaluation_columns=tuple(evaluation_table.columns),
            keys=keys,
            signals=declaration.signals,
            sidecars=public_context_tables,
            candidates=declaration.candidates,
            dispositions=disposition_table,
            result_status=checks_status,
            validation_plan=validation_payload,
            assessment_research=bundle.research if assessment_mode else None,
            confirmation_status=(
                "pending" if confirmation_required else "not_declared"
            ),
            workflow_state="ready",
            artifact_role="attempt",
        )
        (out_sig / "card.md").write_text(card + "\n", encoding="utf-8")
        (out_sig / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "sig",
                    "attempt": attempt,
                    "checks": checks_status,
                    "workflow_state": "ready",
                    "card_id": card_identifier,
                    "input_tables": {
                        "data": str(data_path),
                        "feat": str(feature_path),
                    },
                    "main_table": str(out_sig / "main.parquet"),
                    "main_layout": "assessment" if assessment_mode else "wide",
                    "assessment_mode": assessment_mode,
                    "recipe": str(out_sig / "recipe.json"),
                    "selection": str(out_sig / "selection.json"),
                    "validation_plan": str(out_sig / "validation_plan.json"),
                    "evidence_scope": (
                        validation_plan.evidence_scope
                        if validation_plan is not None
                        else "development_only"
                    ),
                    "confirmation_status": (
                        "pending" if confirmation_required else "not_declared"
                    ),
                    "sources": source_manifest,
                    "eval_report": str(out_sig / "eval" / "report.jsonl"),
                    "assessment": str(out_sig / "eval" / "assessment.json"),
                    "context_tables": list(public_context_tables),
                    "research": research_manifest,
                    "artifacts": artifact_manifest,
                    "rows": main_table.height,
                    "columns": main_table.columns,
                    "keys": keys,
                    "signals": declaration.signals,
                    "candidates": declaration.candidates,
                    "evaluation_columns": evaluation_table.columns,
                    "records": len(result.records),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record_attempt(
            out,
            stage="sig",
            attempt=attempt,
            path=str(out_sig),
            valid=result.passed,
            card=card_identifier,
            checks=checks_status,
            confirmation_required=confirmation_required,
            evidence_scope=(
                validation_plan.evidence_scope
                if validation_plan is not None
                else "development_only"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports external workspace errors
        print(f"sig run failed: {exc}")
        return 1
    print(f"sig attempt {attempt}: {checks_status}")
    print(f"attempt report: {out_sig / 'card.md'}")
    if result.passed:
        if confirmation_required:
            print(
                "next: inspect the development evidence, run another attempt, or "
                "freeze this recipe with "
                f"`rci sig freeze --out {out} --attempt {attempt} "
                "--context <confirmation-context>`"
            )
        else:
            print(
                "next: inspect the evidence, run another attempt, or prepare this "
                f"result with `rci sig prepare --out {out} --attempt {attempt} "
                "--brief <review-brief.json>`"
            )
    else:
        print("next: fix the reported mandatory checks and run sig again")
    return 0 if result.passed else 1


def _sig_freeze(
    out: Path,
    requested_attempt: str | None,
    context_path: Path,
) -> int:
    """Freeze one Sig procedure, then reveal and evaluate one holdout."""

    started = False
    selected: dict[str, Any] | None = None
    attempt_dir: Path | None = None
    evidence_path: Path | None = None
    try:
        state = read_workflow(out)
        require_stage_ready(state, "sig")
        selected = _select_attempt(state, "sig", requested_attempt)
        if not selected.get("valid"):
            raise WorkflowError(
                f"Sig attempt {selected['id']} is invalid and cannot be frozen.",
                code="RCI-ATTEMPT-INVALID",
                action=next_action(state),
            )
        if selected.get("rejected") or selected.get("submitted"):
            raise WorkflowError(
                f"Sig attempt {selected['id']} is no longer eligible for freezing.",
                code="RCI-SIG-FREEZE-NOT-ALLOWED",
                action=next_action(state),
            )
        if selected.get("confirmation_status") != "pending":
            raise WorkflowError(
                f"Sig attempt {selected['id']} has confirmation status "
                f"{selected.get('confirmation_status')!r}.",
                code="RCI-SIG-CONFIRMATION-CONSUMED",
                action=next_action(state),
            )
        attempt_dir = Path(selected["path"])
        if not attempt_dir.is_dir():
            raise FileNotFoundError(f"Sig attempt directory is missing: {attempt_dir}")

        validation_payload = json.loads(
            (attempt_dir / "validation_plan.json").read_text(encoding="utf-8")
        )
        provenance = json.loads(
            (attempt_dir / "provenance.json").read_text(encoding="utf-8")
        )
        plan_source = Path(provenance["plan"]["path"])
        if _source_identity(plan_source) != provenance["plan"]:
            raise WorkflowError(
                "The Sig eval plan changed after the attempt was created; run a new "
                "attempt before confirmation.",
                code="RCI-SIG-PLAN-CHANGED",
                action=next_action(state),
            )
        builder_source = Path(provenance["builder"]["path"])
        if _source_identity(builder_source) != provenance["builder"]:
            raise WorkflowError(
                "The Sig builder changed after the attempt was created; run a new "
                "attempt before confirmation.",
                code="RCI-SIG-BUILDER-CHANGED",
                action=next_action(state),
            )

        plan_module = _load_module(plan_source)
        declaration = _signal_declaration(plan_module)
        validation_plan = _signal_validation_plan(plan_module)
        if validation_plan is None or validation_plan.confirmation is None:
            raise WorkflowError(
                "This Sig plan does not declare a confirmation holdout.",
                code="RCI-SIG-CONFIRMATION-NOT-DECLARED",
                action=next_action(state),
            )
        if validation_plan.as_dict() != validation_payload:
            raise WorkflowError(
                "The declared ValidationPlan changed after the attempt was created.",
                code="RCI-SIG-VALIDATION-PLAN-CHANGED",
                action=next_action(state),
            )

        confirmation = validation_plan.confirmation
        holdout_hash = _path_digest(context_path)
        sealed_hash = _signal_freeze_digest(attempt_dir)
        begin_confirmation(
            out,
            attempt=selected["id"],
            holdout_id=confirmation.holdout_id,
            holdout_hash=holdout_hash,
            sealed_hash=sealed_hash,
        )
        started = True

        context = _load_context(context_path)
        feature_table = pl.read_parquet(out / "feat" / "main.parquet")
        signal_main = pl.read_parquet(attempt_dir / "main.parquet").select(
            *declaration.keys,
            *declaration.signals,
        )
        primary = context.table(confirmation.context_table)
        development_path = (
            attempt_dir / "context" / f"{confirmation.context_table}.parquet"
        )
        development = (
            pl.read_parquet(development_path) if development_path.exists() else None
        )
        scope = _confirmation_scope(
            feature_table,
            primary,
            development,
            declaration,
        )
        holdout_keys = primary.select(*declaration.keys).unique(maintain_order=True)
        confirmation_table = (
            feature_table.select(*declaration.keys, *declaration.features)
            .join(signal_main, on=list(declaration.keys), how="inner")
            .join(holdout_keys, on=list(declaration.keys), how="inner")
        )
        plan = _make_named_plan(
            plan_module,
            confirmation.plan_function,
            confirmation_table,
            "sig",
            context,
        )
        scope_rule = EvalSpec(
            "engine.baseline.signal_confirmation_scope",
            runner=lambda _: EvalObservation.passed_without_output(**scope),
            failure_policy="stop",
            description=(
                "evaluate only final signals on keys disjoint from development"
            ),
            execution_mode="polars_join_group",
        )
        plan = replace(plan, baseline=(scope_rule, *plan.baseline))
        result = run_eval_plan(confirmation_table, plan, context)

        confirmation_dir = attempt_dir / "confirmation"
        confirmation_dir.mkdir(parents=True, exist_ok=False)
        confirmation_context = confirmation_dir / "context"
        confirmation_context.mkdir()
        for name, table in sorted(context.tables.items()):
            _safe_artifact_name(name)
            table.write_parquet(confirmation_context / f"{name}.parquet")
        (confirmation_context / "metadata.json").write_text(
            json.dumps(dict(context.metadata), indent=2, sort_keys=True, default=str)
            + "\n",
            encoding="utf-8",
        )
        evidence_path = confirmation_dir / "eval" / "report.jsonl"
        result.write_report(evidence_path)
        assessment_path = confirmation_dir / "eval" / "assessment.json"
        assessment_path.write_text(
            json.dumps(assess_records(result.records), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status = "revealed_valid" if result.passed else "revealed_invalid"
        holdout_record = {
            "holdout_id": confirmation.holdout_id,
            "holdout_hash": holdout_hash,
            "sealed_hash": sealed_hash,
            "status": status,
            "valid": result.passed,
            "scope": scope,
            "records": len(result.records),
            "plan_function": confirmation.plan_function,
            "context_tables": sorted(context.tables),
            "evidence": str(evidence_path),
            "assessment": str(assessment_path),
        }
        (confirmation_dir / "holdout.json").write_text(
            json.dumps(holdout_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        card, card_identifier = _confirmation_card(
            attempt_dir / "card.md",
            validation_plan.as_dict(),
            holdout_record,
        )
        (attempt_dir / "card.md").write_text(card + "\n", encoding="utf-8")
        manifest_path = attempt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "card_id": card_identifier,
                "confirmation": str(confirmation_dir / "holdout.json"),
                "confirmation_status": status,
                "evidence_scope": (
                    "confirmed_holdout" if result.passed else "confirmation_invalid"
                ),
                "frozen": result.passed,
                "sealed_hash": sealed_hash,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        finish_confirmation(
            out,
            attempt=selected["id"],
            evidence=str(evidence_path),
            valid=result.passed,
            card=card_identifier,
        )
    except Exception as exc:  # noqa: BLE001 - reports external confirmation errors
        if started and selected is not None and attempt_dir is not None:
            _record_failed_confirmation(out, selected, attempt_dir, exc)
        print(f"sig freeze failed: {exc}")
        return 1

    print(f"frozen: sig attempt {selected['id']}")
    print(f"confirmation: {status}")
    print(f"confirmation report: {evidence_path}")
    if result.passed:
        print(
            "next: prepare the frozen result for review with "
            f"`rci sig prepare --out {out} --attempt {selected['id']} "
            "--brief <review-brief.json>`"
        )
        return 0
    print("next: confirmation checks were invalid; use a new attempt and holdout")
    return 1


def _algo_run(
    builder_path: Path,
    plan_path: Path,
    backtester_path: Path,
    out: Path,
    context_path: Path | None = None,
) -> int:
    """Run an external Algo builder and an explicit external backtester."""

    try:
        state = read_workflow(out)
        require_stage_ready(state, "algo")
        attempt, out_algo = _allocate_attempt(out, state, "algo")
        signal_path = out / "sig" / "main.parquet"
        signal_table = pl.read_parquet(signal_path)
        signal_before = signal_table.clone()

        builder = _load_module(builder_path)
        bundle = _build_algo(builder, signal_table)
        if not signal_table.equals(signal_before):
            raise ValueError("algo builder modified the input Sig DataFrame")

        plan_module = _load_module(plan_path)
        declaration = _algo_declaration(plan_module)
        expected_columns = declaration.required
        missing = sorted(set(expected_columns) - set(bundle.main.columns))
        extra = sorted(set(bundle.main.columns) - set(expected_columns))
        if missing or extra:
            raise ValueError(
                "AlgoBundle.main must contain exactly the declared Algo columns; "
                f"missing={missing}, extra={extra}"
            )
        main_table = bundle.main.select(expected_columns)

        backtester = _load_module(backtester_path)
        backtest_result = _run_backtest(backtester, main_table)

        external_context = _load_context(context_path)
        backtest_summary_name = f"{declaration.backtest_table}_summary"
        generated_tables = {
            declaration.backtest_table: backtest_result.timeseries,
            backtest_summary_name: backtest_result.summary,
        }
        if backtest_result.trades is not None:
            generated_tables["trades"] = backtest_result.trades
        collisions = sorted(set(external_context.tables) & set(generated_tables))
        if collisions:
            raise ValueError(
                f"algo backtest sidecars collide with external context: {collisions}"
            )
        context = EvalContext(
            tables={**external_context.tables, **generated_tables},
            metadata={
                **external_context.metadata,
                **bundle.metadata,
                "backtest": dict(backtest_result.metadata),
            },
        )

        plan = _make_plan(plan_module, main_table, "algo", context)
        baseline_rules = (
            EvalSpec(
                "engine.baseline.algo_main_output_shape",
                runner=lambda _: algo_main_output_shape(
                    signal_table,
                    main_table,
                    declaration,
                ),
                failure_policy="stop",
                description="preserve approved Sig inputs and validate positions",
                execution_mode="polars_join_group",
            ),
            EvalSpec(
                "engine.baseline.algo_backtest_output_shape",
                runner=lambda _: algo_backtest_output_shape(
                    signal_table,
                    backtest_result.timeseries,
                    declaration,
                ),
                failure_policy="stop",
                description="validate the external backtest key grid and paths",
                execution_mode="polars_join_group",
            ),
            EvalSpec(
                "engine.baseline.algo_backtest_summary_shape",
                runner=lambda _: algo_backtest_summary_shape(
                    backtest_result.summary,
                    declaration,
                ),
                failure_policy="stop",
                description="require one summary row per strategy",
                execution_mode="polars_select",
            ),
        )
        plan = replace(plan, baseline=(*baseline_rules, *plan.baseline))
        eval_result = run_eval_plan(main_table, plan, context)

        main_table.write_parquet(out_algo / "main.parquet")
        backtest_dir = out_algo / "backtest"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        backtest_result.timeseries.write_parquet(backtest_dir / "timeseries.parquet")
        backtest_result.summary.write_parquet(backtest_dir / "summary.parquet")
        trades_path: Path | None = None
        if backtest_result.trades is not None:
            trades_path = backtest_dir / "trades.parquet"
            backtest_result.trades.write_parquet(trades_path)
        (backtest_dir / "metadata.json").write_text(
            json.dumps(
                dict(backtest_result.metadata),
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        recipe = _builder_recipe(builder, "algo")
        recipe["backtester"] = str(backtester_path)
        (out_algo / "recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        artifact_manifest = _write_stage_artifacts(
            out_algo,
            bundle.artifacts,
            stage="algo",
        )

        eval_result.write_report(out_algo / "eval" / "report.jsonl")
        assessment = assess_records(eval_result.records)
        (out_algo / "eval" / "assessment.json").write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checks_status = _checks_status(eval_result)
        card, card_identifier = algo_card(
            main_table,
            eval_result,
            {**dict(bundle.metadata), "backtest": dict(backtest_result.metadata)},
            backtest_result.summary,
            input_rows=signal_table.height,
            keys=declaration.keys,
            positions=declaration.positions,
            representation=declaration.representation,
            strategies=tuple(declaration.strategies),
            result_status=checks_status,
            workflow_state="ready",
            artifact_role="attempt",
        )
        (out_algo / "card.md").write_text(card + "\n", encoding="utf-8")
        (out_algo / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "algo",
                    "attempt": attempt,
                    "checks": checks_status,
                    "workflow_state": "ready",
                    "card_id": card_identifier,
                    "input_table": str(signal_path),
                    "main_table": str(out_algo / "main.parquet"),
                    "representation": declaration.representation,
                    "recipe": str(out_algo / "recipe.json"),
                    "backtester": str(backtester_path),
                    "backtest": {
                        "timeseries": str(backtest_dir / "timeseries.parquet"),
                        "summary": str(backtest_dir / "summary.parquet"),
                        "trades": str(trades_path) if trades_path else None,
                        "metadata": str(backtest_dir / "metadata.json"),
                    },
                    "eval_report": str(out_algo / "eval" / "report.jsonl"),
                    "assessment": str(out_algo / "eval" / "assessment.json"),
                    "artifacts": artifact_manifest,
                    "rows": main_table.height,
                    "columns": main_table.columns,
                    "keys": declaration.keys,
                    "signals": declaration.signals,
                    "positions": declaration.positions,
                    "strategies": list(declaration.strategies),
                    "context_tables": sorted(context.tables),
                    "records": len(eval_result.records),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record_attempt(
            out,
            stage="algo",
            attempt=attempt,
            path=str(out_algo),
            valid=eval_result.passed,
            card=card_identifier,
            checks=checks_status,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports external workspace errors
        print(f"algo run failed: {exc}")
        return 1
    print(f"algo attempt {attempt}: {checks_status}")
    print(f"attempt report: {out_algo / 'card.md'}")
    if eval_result.passed:
        print(
            "next: inspect the evidence, run another attempt, or prepare this "
            f"result with `rci algo prepare --out {out} --attempt {attempt} "
            "--brief <review-brief.json>`"
        )
    else:
        print("next: fix the reported mandatory checks and run algo again")
    return 0 if eval_result.passed else 1


def _stage_prepare(
    stage: str,
    out: Path,
    requested_attempt: str | None,
    brief_path: Path,
) -> int:
    """Turn internal evidence into the concise Card an external reviewer sees."""

    try:
        state = read_workflow(out)
        require_stage_ready(state, stage)
        selected = _select_attempt(state, stage, requested_attempt)
        if not selected.get("valid"):
            raise WorkflowError(
                f"Attempt {selected['id']} is invalid and cannot be prepared.",
                code="RCI-ATTEMPT-INVALID",
                action=next_action(state),
            )
        if selected.get("stale"):
            raise WorkflowError(
                f"Attempt {selected['id']} is stale and cannot be prepared.",
                code="RCI-ATTEMPT-STALE",
                action=next_action(state),
            )
        if selected.get("confirmation_required") and not selected.get("frozen"):
            raise WorkflowError(
                f"Sig attempt {selected['id']} must finish confirmation first.",
                code="RCI-SIG-CONFIRMATION-REQUIRED",
                action=next_action(state),
            )
        payload = json.loads(brief_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("review brief must be a JSON object")
        brief = ReviewBrief.from_dict(payload)
        attempt_dir = Path(selected["path"])
        root = attempt_dir.resolve()
        for relative in brief.key_plots:
            plot = (attempt_dir / relative).resolve()
            if not plot.is_relative_to(root) or not plot.is_file():
                raise FileNotFoundError(
                    f"review brief key plot is missing from the attempt: {relative}"
                )
        for relative in brief.evidence_refs:
            evidence = (attempt_dir / relative).resolve()
            if not evidence.is_relative_to(root) or not evidence.is_file():
                raise FileNotFoundError(
                    "review brief evidence reference is missing from the attempt: "
                    f"{relative}"
                )
        manifest_path = attempt_dir / "manifest.json"
        assessment_path = attempt_dir / "eval" / "assessment.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
        stage_attempts = state.get("attempts", {}).get(stage, [])
        selected_index = next(
            index
            for index, item in enumerate(stage_attempts)
            if item.get("id") == selected["id"]
        )
        previous = next(
            (
                item["id"]
                for item in reversed(stage_attempts[:selected_index])
                if not item.get("stale")
            ),
            None,
        )
        if previous is not None and not brief.changes:
            raise ValueError(
                "review brief must declare changes from the previous attempt"
            )
        card, card_identifier = review_card(
            stage,
            brief,
            manifest,
            assessment,
            attempt=selected["id"],
            previous_attempt=previous,
        )
        (attempt_dir / "review.json").write_text(
            json.dumps(brief.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (attempt_dir / "card.md").write_text(card + "\n", encoding="utf-8")
        manifest.update(
            {
                "card_id": card_identifier,
                "research_ready": True,
                "review_brief": str(attempt_dir / "review.json"),
                "previous_attempt": previous,
                "evidence_refs": list(brief.evidence_refs),
                "key_plots": list(brief.key_plots),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prepare_attempt(
            out,
            stage=stage,
            attempt=selected["id"],
            card=card_identifier,
            brief=brief.as_dict(),
        )
    except Exception as exc:  # noqa: BLE001 - report external workspace errors
        print(f"{stage} prepare failed: {exc}")
        return 1
    print(f"prepared: {stage} attempt {selected['id']}")
    print(f"review card: {attempt_dir / 'card.md'} ({card_identifier})")
    print(
        f"next: rci {stage} submit --out {out} --attempt {selected['id']} "
        "--recommendation <continue|stop|uncertain>"
    )
    return 0


def _stage_submit(
    stage: str,
    out: Path,
    recommendation: str,
    requested_attempt: str | None,
) -> int:
    try:
        state = read_workflow(out)
        require_stage_ready(state, stage)
        selected = _select_attempt(state, stage, requested_attempt)
        if not selected.get("valid"):
            raise WorkflowError(
                f"Attempt {selected['id']} is invalid and cannot be submitted.",
                code="RCI-ATTEMPT-INVALID",
                action=next_action(state),
            )
        if selected.get("rejected"):
            raise WorkflowError(
                f"Attempt {selected['id']} was rejected; run a new attempt.",
                code="RCI-ATTEMPT-REJECTED",
                action=next_action(state),
            )
        if selected.get("stale"):
            raise WorkflowError(
                f"Attempt {selected['id']} is stale; run a new attempt.",
                code="RCI-ATTEMPT-STALE",
                action=next_action(state),
            )
        if selected.get("submitted"):
            raise WorkflowError(
                f"Attempt {selected['id']} was already submitted; run a new "
                "attempt after reopening the stage.",
                code="RCI-ATTEMPT-SUBMITTED",
                action=next_action(state),
            )
        attempt_dir = Path(selected["path"])
        if not attempt_dir.is_dir():
            raise WorkflowError(
                f"Attempt directory does not exist: {attempt_dir}",
                code="RCI-ATTEMPT-MISSING",
                action=next_action(state),
            )
        if stage == "sig" and selected.get("confirmation_required"):
            confirmation = selected.get("confirmation") or {}
            expected_hash = confirmation.get("sealed_hash")
            if not selected.get("frozen") or not expected_hash:
                raise WorkflowError(
                    f"Sig attempt {selected['id']} must complete confirmation before "
                    "submission.",
                    code="RCI-SIG-CONFIRMATION-REQUIRED",
                    action=next_action(state),
                )
            actual_hash = _signal_freeze_digest(attempt_dir)
            if actual_hash != expected_hash:
                raise WorkflowError(
                    "The frozen Sig recipe or its evidence changed after the holdout "
                    "was revealed; create a new attempt.",
                    code="RCI-SIG-FREEZE-MISMATCH",
                    action=next_action(state),
                )
        if not selected.get("research_ready"):
            raise WorkflowError(
                f"Attempt {selected['id']} passed checks but has not been prepared "
                "for external review.",
                code="RCI-ATTEMPT-NOT-RESEARCH-READY",
                action=next_action(state),
            )
        stage_dir = out / stage
        published_entries = _publish_attempt(attempt_dir, stage_dir)
        card, card_identifier = _submission_card(stage_dir / "card.md", recommendation)
        (stage_dir / "card.md").write_text(card + "\n", encoding="utf-8")
        _publish_manifest(
            stage_dir / "manifest.json",
            attempt_dir,
            stage_dir,
            attempt=selected["id"],
            recommendation=recommendation,
            card=card_identifier,
        )
        (stage_dir / "candidate.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "attempt": selected["id"],
                    "recommendation": recommendation,
                    "card_id": card_identifier,
                    "published_entries": published_entries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        submit_for_review(
            out,
            stage=stage,
            card=card_identifier,
            recommendation=recommendation,
            attempt=selected["id"],
        )
    except Exception as exc:  # noqa: BLE001 - report external workspace errors
        print(f"{stage} submit failed: {exc}")
        return 1
    print(f"submitted: {stage} attempt {selected['id']}")
    print(f"recommendation: {recommendation}")
    print(f"card: {stage_dir / 'card.md'} ({card_identifier})")
    print(
        "next (agent reviewer): rci approve "
        f"--stage {stage} --out {out} --by agent --decision continue "
        f"--card {card_identifier}"
    )
    print(
        "   or: rci approve "
        f"--stage {stage} --out {out} --by agent --decision stop "
        f"--card {card_identifier}"
    )
    print(
        "   or: rci reject "
        f"--stage {stage} --out {out} --by agent --message <reason> "
        f"--card {card_identifier}"
    )
    print("human reviewers may replace `--by agent` with `--by human`")
    return 0


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
        recommendation = state.get("recommendations", {}).get(stage, "-")
        attempts = len(state.get("attempts", {}).get(stage, []))
        current = latest_attempt(state, stage)
        current_summary = (
            f", latest={current['id']}:valid={bool(current.get('valid'))}"
            f":ready={bool(current.get('research_ready'))}"
            f":stale={bool(current.get('stale'))}"
            if current is not None
            else ""
        )
        print(
            f"{stage}: {status} "
            f"(card={card}, recommendation={recommendation}, attempts={attempts}"
            f"{current_summary})"
        )
    approvals = state.get("approvals", [])
    print(f"approvals: {len(approvals)}")
    for item in approvals:
        decision = item.get("decision", "continue")
        print(f"- {item['stage']} by {item['by']}: {decision} ({item['card']})")
    revisions = state.get("revisions", [])
    print(f"revisions: {len(revisions)}")
    action = next_action(state)
    if action["stage"] is None:
        print(f"next: {action.get('outcome', 'complete')}")
    else:
        print(f"next: {action['actor']} -> {action['command']}")
    return 0


def _next(out: Path, as_json: bool = False) -> int:
    try:
        state = read_workflow(out)
    except WorkflowError as exc:
        print(f"next failed: {exc}")
        return 1
    action = next_action(state)
    if as_json:
        print(json.dumps(action, indent=2, sort_keys=True))
        return 0
    print(f"actor: {action.get('actor') or '-'}")
    print(f"stage: {action.get('stage') or '-'}")
    print(f"state: {action.get('state') or '-'}")
    print(f"action: {action['action']}")
    print(f"command: {action.get('command') or '-'}")
    print(f"card: {action.get('card') or '-'}")
    print(f"reason: {action['reason']}")
    if action.get("recommendation"):
        print(f"recommendation: {action['recommendation']}")
    for item in action.get("optional_help", []):
        print(f"optional help: {item}")
    return 0


def _approve(
    stage: str,
    out: Path,
    by: str | None,
    decision: str | None,
    message: str,
    expected_card: str | None,
) -> int:
    interactive = _interactive_terminal()
    needs_prompt = by is None or decision is None or expected_card is None
    if by == "agent" and (decision is None or expected_card is None):
        return _review_arguments_required(stage, out, "approve")
    if needs_prompt and not interactive:
        return _review_arguments_required(stage, out, "approve")
    if by is None:
        by = "human"
    if by == "human" and interactive and needs_prompt:
        try:
            state = read_workflow(out)
            active_card = state.get("cards", {}).get(stage)
            if state.get("stages", {}).get(stage) != "review" or not active_card:
                raise WorkflowError(
                    f"Stage {stage!r} has no submitted review Card.",
                    code="RCI-NO-REVIEW-CARD",
                    action=next_action(state),
                )
            recommendation = state.get("recommendations", {}).get(stage)
            print(
                _terminal_review(
                    out,
                    stage,
                    state,
                    recommendation=recommendation,
                )
            )
            expected_card = expected_card or active_card
            decision = decision or ("stop" if recommendation == "stop" else "continue")
            if not _confirm(f"Approve {stage} with decision {decision!r}?"):
                print("approval cancelled; workflow unchanged")
                return 1
        except (OSError, ValueError, WorkflowError) as exc:
            print(f"approval failed: {exc}")
            return 1
    if decision is None or expected_card is None:
        return _review_arguments_required(stage, out, "approve")
    try:
        state, next_stage = approve_workflow(
            out,
            stage=stage,
            by=by,
            decision=decision,
            message=message,
            expected_card=expected_card,
        )
    except WorkflowError as exc:
        print(f"approval failed: {exc}")
        return 1
    if stage != "contract":
        (out / stage / "approved.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "attempt": state.get("candidates", {}).get(stage),
                    "card_id": state.get("cards", {}).get(stage),
                    "decision": decision,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if decision == "stop":
        print(f"accepted and stopped: {stage} by {by}; workflow completed_no_go")
    elif next_stage is None:
        print(f"approved: {stage} by {by}; workflow completed")
    else:
        print(f"approved: {stage} by {by}; next stage ready: {next_stage}")
    return 0


def _reject(
    stage: str,
    out: Path,
    by: str | None,
    message: str | None,
    expected_card: str | None,
) -> int:
    interactive = _interactive_terminal()
    needs_prompt = by is None or not message or expected_card is None
    if by == "agent" and (not message or expected_card is None):
        return _review_arguments_required(stage, out, "reject")
    if needs_prompt and not interactive:
        return _review_arguments_required(stage, out, "reject")
    if by is None:
        by = "human"
    if by == "human" and interactive and needs_prompt:
        try:
            state = read_workflow(out)
            active_card = state.get("cards", {}).get(stage)
            if state.get("stages", {}).get(stage) != "review" or not active_card:
                raise WorkflowError(
                    f"Stage {stage!r} has no submitted review Card.",
                    code="RCI-NO-REVIEW-CARD",
                    action=next_action(state),
                )
            print(_terminal_review(out, stage, state))
            expected_card = expected_card or active_card
            if not _confirm("Return this Card to the Agent?"):
                print("rejection cancelled; workflow unchanged")
                return 1
            if not message:
                message = input("Feedback: ").strip()
            if not message:
                print("rejection cancelled: feedback is required")
                return 1
        except (OSError, ValueError, WorkflowError, EOFError) as exc:
            print(f"rejection failed: {exc}")
            return 1
    if not message or expected_card is None:
        return _review_arguments_required(stage, out, "reject")
    try:
        reject_workflow(
            out,
            stage=stage,
            by=by,
            message=message,
            expected_card=expected_card,
        )
        candidate_path = out / stage / "candidate.json"
        if candidate_path.exists():
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate["review"] = "rejected"
            candidate["message"] = message
            candidate_path.write_text(
                json.dumps(candidate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:  # noqa: BLE001 - report workflow errors
        print(f"rejection failed: {exc}")
        return 1
    print(f"rejected: {stage}; Agent may revise and run a new attempt")
    return 0


def _interactive_terminal() -> bool:
    """Whether the reviewer can use the human terminal handshake."""

    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _review_arguments_required(stage: str, out: Path, action: str) -> int:
    """Explain the explicit Agent/CI form when a prompt is unavailable."""

    if action == "approve":
        command = (
            f"rci approve --stage {stage} --out {out} --by <human|agent> "
            "--decision <continue|stop> --card <card_id>"
        )
    else:
        command = (
            f"rci reject --stage {stage} --out {out} --by <human|agent> "
            "--message <reason> --card <card_id>"
        )
    print(
        "review failed: approval arguments are required outside an interactive terminal"
    )
    print(f"next: {command}")
    return 1


def _confirm(prompt: str) -> bool:
    """Use the old RCI-style default-no terminal confirmation."""

    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _terminal_review(
    out: Path,
    stage: str,
    state: dict[str, Any],
    *,
    recommendation: str | None = None,
) -> str:
    """Build the compact terminal view from the durable submitted Card."""

    card_path = out / stage / "card.md"
    if not card_path.is_file():
        raise FileNotFoundError(f"review Card does not exist: {card_path}")
    card_identifier = str(state.get("cards", {}).get(stage, ""))
    if not card_identifier:
        raise WorkflowError(
            f"Stage {stage!r} has no active Card.", code="RCI-CARD-MISSING"
        )
    return render_terminal_card(
        stage,
        card_path.read_text(encoding="utf-8"),
        card_identifier=card_identifier,
        recommendation=recommendation,
        dependencies=_review_dependencies(out, stage),
    )


def _review_dependencies(out: Path, stage: str) -> tuple[tuple[str, str], ...]:
    """Hash the small set of files that a reviewer is about to accept."""

    candidates = [
        ("workflow", out / "workflow.json"),
        ("manifest", out / stage / "manifest.json"),
        ("eval", out / stage / "eval"),
        ("main", out / stage / "main.parquet"),
    ]
    result: list[tuple[str, str]] = []
    for name, path in candidates:
        if not path.exists():
            continue
        try:
            result.append((name, _path_digest(path)[:12]))
        except (OSError, ValueError):
            continue
    return tuple(result)


def _reopen(stage: str, out: Path, by: str, message: str) -> int:
    try:
        state = reopen_workflow(out, stage=stage, by=by, message=message)
        start = STAGE_ORDER.index(stage)
        for affected in STAGE_ORDER[start:]:
            if state["stages"].get(affected) != "disabled":
                (out / affected / "approved.json").unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - report workflow errors
        print(f"reopen failed: {exc}")
        return 1
    print(f"reopened: {stage}; downstream stages are locked")
    return 0


def _revise(stage: str, out: Path, message: str) -> int:
    try:
        state = revise_workflow(out, stage=stage, message=message)
        start = STAGE_ORDER.index(stage)
        for affected in STAGE_ORDER[start:]:
            if state["stages"].get(affected) != "disabled":
                (out / affected / "approved.json").unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - report workflow errors
        print(f"revision failed: {exc}")
        return 1
    print(f"revision opened: {stage}; downstream products are stale and locked")
    return 0


def _guide(stage: str, as_json: bool) -> int:
    guide = _STAGE_GUIDES[stage]
    if as_json:
        print(json.dumps(guide, indent=2, sort_keys=True))
        return 0
    print(f"stage: {stage}")
    print(f"objective: {guide['objective']}")
    print(f"formal input: {guide['input']}")
    print(f"formal output: {guide['output']}")
    print("optional construction tools:")
    for item in guide["tools"]:
        print(f"- {item}")
    print("required boundary:")
    for item in guide["required"]:
        print(f"- {item}")
    print("completion checklist:")
    for item in guide["completion_checklist"]:
        print(f"- {item}")
    print("common failure modes:")
    for item in guide["common_failure_modes"]:
        print(f"- {item}")
    print(f"eval menu: rci eval methods --stage {stage} --available-only")
    return 0


def _allocate_attempt(out: Path, state: dict[str, Any], stage: str) -> tuple[str, Path]:
    attempts_dir = out / stage / "attempts"
    used = {
        int(item["id"])
        for item in state.get("attempts", {}).get(stage, [])
        if str(item.get("id", "")).isdigit()
    }
    if attempts_dir.exists():
        used.update(
            int(path.name)
            for path in attempts_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    attempt = f"{max(used, default=0) + 1:04d}"
    path = attempts_dir / attempt
    path.mkdir(parents=True)
    return attempt, path


def _select_attempt(
    state: dict[str, Any], stage: str, requested: str | None
) -> dict[str, Any]:
    if requested is None:
        selected = latest_attempt(state, stage)
        if selected is None:
            raise WorkflowError(
                f"Stage {stage!r} has no attempt to submit.",
                code="RCI-ATTEMPT-MISSING",
                action=next_action(state),
            )
        return selected
    for item in state.get("attempts", {}).get(stage, []):
        if item.get("id") == requested:
            return item
    raise WorkflowError(
        f"Unknown attempt {requested!r} for stage {stage!r}.",
        code="RCI-ATTEMPT-MISSING",
        action=next_action(state),
    )


def _publish_attempt(attempt_dir: Path, stage_dir: Path) -> list[str]:
    """Publish one attempt without touching the attempt history."""

    candidate_path = stage_dir / "candidate.json"
    if candidate_path.exists():
        previous = json.loads(candidate_path.read_text(encoding="utf-8"))
        for name in previous.get("published_entries", []):
            if name in {"attempts", "candidate.json", "approved.json"}:
                continue
            _remove_generated_path(stage_dir / name)

    published: list[str] = []
    for source in sorted(attempt_dir.iterdir(), key=lambda item: item.name):
        destination = stage_dir / source.name
        _remove_generated_path(destination)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        published.append(source.name)
    return published


def _remove_generated_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _submission_card(path: Path, recommendation: str) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8").rstrip()
    marker = "\n\n- card_id: `"
    body = text.rsplit(marker, 1)[0] if marker in text else text
    body = body.replace("- workflow_state: ready", "- workflow_state: review", 1)
    body = body.replace("- artifact_role: attempt", "- artifact_role: candidate", 1)
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line == "- workflow_state: review":
            lines.insert(index + 1, f"- recommendation: {recommendation}")
            break
    return render_card("\n".join(lines))


def _publish_manifest(
    path: Path,
    attempt_dir: Path,
    stage_dir: Path,
    *,
    attempt: str,
    recommendation: str,
    card: str,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = _replace_path_prefix(payload, str(attempt_dir), str(stage_dir))
    payload.update(
        {
            "attempt": attempt,
            "workflow_state": "review",
            "recommendation": recommendation,
            "card_id": card,
        }
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _replace_path_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_path_prefix(item, old, new) for key, item in value.items()
        }
    return value


def _workflow_commands(contract_path: Path, out: Path) -> dict[str, dict[str, str]]:
    lab = contract_path.parent

    def command(*parts: str | Path) -> str:
        return shlex.join(str(part) for part in parts)

    def review(stage: str) -> str:
        return command(
            "rci",
            "approve",
            "--stage",
            stage,
            "--out",
            out,
            "--by",
            "agent",
            "--decision",
            "<continue|stop>",
            "--card",
            "{card}",
        )

    def submit(stage: str) -> str:
        return command(
            "rci",
            stage,
            "submit",
            "--out",
            out,
            "--attempt",
            "{attempt}",
            "--recommendation",
            "<continue|stop|uncertain>",
        )

    def prepare(stage: str) -> str:
        return command(
            "rci",
            stage,
            "prepare",
            "--out",
            out,
            "--attempt",
            "{attempt}",
            "--brief",
            f"<review-brief-{stage}.json>",
        )

    return {
        "contract": {
            "run": command("rci", "contract", "derive", contract_path, "--out", out),
            "approve": review("contract"),
        },
        "data": {
            "run": command(
                "rci",
                "data",
                "run",
                "--contract",
                contract_path,
                "--builder",
                lab / "data.py",
                "--plan",
                lab / "evals" / "data.py",
                "--out",
                out,
            ),
            "prepare": prepare("data"),
            "submit": submit("data"),
            "approve": review("data"),
        },
        "feat": {
            "run": command(
                "rci",
                "feat",
                "run",
                "--builder",
                lab / "feat.py",
                "--plan",
                lab / "evals" / "feat.py",
                "--out",
                out,
            ),
            "prepare": prepare("feat"),
            "submit": submit("feat"),
            "approve": review("feat"),
        },
        "sig": {
            "run": command(
                "rci",
                "sig",
                "run",
                "--builder",
                lab / "sig.py",
                "--plan",
                lab / "evals" / "sig.py",
                "--out",
                out,
            ),
            "prepare": prepare("sig"),
            "freeze": command(
                "rci",
                "sig",
                "freeze",
                "--out",
                out,
                "--attempt",
                "{attempt}",
                "--context",
                "<confirmation-context>",
            ),
            "submit": submit("sig"),
            "approve": review("sig"),
        },
        "algo": {
            "run": command(
                "rci",
                "algo",
                "run",
                "--builder",
                lab / "algo.py",
                "--plan",
                lab / "evals" / "algo.py",
                "--backtester",
                lab / "backtest.py",
                "--out",
                out,
            ),
            "prepare": prepare("algo"),
            "submit": submit("algo"),
            "approve": review("algo"),
        },
    }


def _checks_status(result: Any) -> str:
    return "valid" if result.passed else "invalid"


def _build_table(module: ModuleType) -> pl.DataFrame | DataBundle:
    for name in ("build_data_bundle", "build_main_table", "build"):
        builder = getattr(module, name, None)
        if builder is not None:
            return builder()
    raise AttributeError(
        "data builder must define build_data_bundle(), build_main_table(), or build()"
    )


def _build_features(module: ModuleType, bundle: DataBundle) -> pl.DataFrame:
    bundle_builder = getattr(module, "build_features_from_bundle", None)
    if bundle_builder is not None:
        return bundle_builder(bundle)
    for name in ("build_features", "build"):
        builder = getattr(module, name, None)
        if builder is not None:
            return builder(bundle.main)
    raise AttributeError(
        "feature builder must define build_features_from_bundle(bundle), "
        "build_features(table), or build(table)"
    )


def _build_signals(module: ModuleType, table: pl.DataFrame) -> SignalBundle:
    for name in ("build_signals", "build"):
        builder = getattr(module, name, None)
        if builder is not None:
            result = builder(table)
            if not isinstance(result, SignalBundle):
                raise TypeError(
                    "signal builder must return an research_ci.kit.SignalBundle"
                )
            return result
    raise AttributeError(
        "signal builder must define build_signals(table) or build(table)"
    )


def _build_algo(module: ModuleType, table: pl.DataFrame) -> AlgoBundle:
    for name in ("build_algo", "build"):
        builder = getattr(module, name, None)
        if builder is not None:
            result = builder(table)
            if not isinstance(result, AlgoBundle):
                raise TypeError(
                    "algo builder must return an research_ci.kit.AlgoBundle"
                )
            return result
    raise AttributeError("algo builder must define build_algo(table) or build(table)")


def _run_backtest(module: ModuleType, table: pl.DataFrame) -> BacktestResult:
    runner = getattr(module, "run_backtest", None)
    if runner is None:
        raise AttributeError("backtester must define run_backtest(algo_table)")
    before = table.clone()
    result = runner(table)
    if not isinstance(result, BacktestResult):
        raise TypeError(
            "run_backtest(algo_table) must return research_ci.kit.BacktestResult"
        )
    if not table.equals(before):
        raise ValueError("external backtester modified the Algo main DataFrame")
    return result


def _build_signal_main(
    module: ModuleType,
    data_path: Path,
    signal_table: pl.DataFrame,
) -> pl.DataFrame:
    """Build the case-owned Sig main table consumed directly by Algo."""

    builder = getattr(module, "build_main_table", None)
    if builder is None:
        raise TypeError("signal builder must define build_main_table(data, signals)")
    data_table = pl.read_parquet(data_path)
    data_before = data_table.clone()
    signal_before = signal_table.clone()
    result = builder(data_table, signal_table)
    if not isinstance(result, pl.DataFrame):
        raise TypeError(
            "build_main_table(data, signals) must return a Polars DataFrame"
        )
    if not data_table.equals(data_before) or not signal_table.equals(signal_before):
        raise ValueError("build_main_table modified an input DataFrame")
    return result


def _visualize(stage: str, renderer_path: Path, out: Path) -> int:
    """Run a case renderer only when the explicit visualize command is used."""

    try:
        if not (out / stage / "main.parquet").exists():
            raise FileNotFoundError(
                f"{stage.title()} output is missing; run that stage first"
            )
        module = _load_module(renderer_path)
        renderer = next(
            (
                getattr(module, name)
                for name in ("render_visualizations", "render")
                if getattr(module, name, None) is not None
            ),
            None,
        )
        if renderer is None:
            raise AttributeError(
                "renderer must define render_visualizations(run_dir, stage, output_dir)"
            )
        output_dir = (out / stage / "plots").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered = renderer(out, stage, output_dir)
        if not isinstance(rendered, dict) or not rendered:
            raise TypeError("stage renderer must return a non-empty name-to-path dict")

        root = output_dir.resolve()
        manifest: dict[str, str] = {}
        for name, raw_path in sorted(rendered.items()):
            _safe_artifact_name(name)
            path = Path(raw_path)
            if not path.is_absolute():
                path = output_dir / path
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"visualization {name!r} is outside {output_dir}")
            if not resolved.is_file():
                raise FileNotFoundError(f"visualization {name!r} was not written")
            manifest[name] = str(path)

        plot_manifest = output_dir / "manifest.json"
        plot_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage_manifest_path = out / stage / "manifest.json"
        if stage_manifest_path.exists():
            stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
            stage_manifest["visualizations"] = manifest
            stage_manifest_path.write_text(
                json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:  # noqa: BLE001 - CLI reports external renderer errors
        print(f"{stage} visualization failed: {exc}")
        return 1
    for name, path in manifest.items():
        print(f"{name}: {path}")
    print(f"visualization manifest: {plot_manifest}")
    return 0


def _signal_declaration(module: ModuleType) -> SignalSet:
    for name in ("SIGNAL_SET", "DECLARATION"):
        declaration = getattr(module, name, None)
        if isinstance(declaration, SignalSet):
            return declaration
    raise AttributeError("signal plan must expose SIGNAL_SET or DECLARATION")


def _signal_validation_plan(module: ModuleType) -> ValidationPlan | None:
    value = getattr(module, "VALIDATION_PLAN", None)
    if value is None:
        return None
    if not isinstance(value, ValidationPlan):
        raise TypeError("signal plan VALIDATION_PLAN must be a ValidationPlan")
    return value


def _algo_declaration(module: ModuleType) -> AlgoSet:
    for name in ("ALGO_SET", "DECLARATION"):
        declaration = getattr(module, name, None)
        if isinstance(declaration, AlgoSet):
            return declaration
    raise AttributeError("algo plan must expose ALGO_SET or DECLARATION")


def _label_sidecar_shape(
    feature_table: pl.DataFrame,
    declaration: SignalSet,
    context: EvalContext,
    *,
    key_columns: tuple[str, ...] | None = None,
    allow_subset: bool = False,
) -> EvalObservation:
    try:
        label = context.table(declaration.label_table)
    except KeyError as exc:
        return EvalObservation(
            "failed",
            output_kind="findings",
            details={
                "reason": str(exc),
                "reason_code": "MISSING_CONTEXT",
                "context_table": declaration.label_table,
            },
        )
    keys = tuple(key_columns or declaration.keys)
    required = list(declaration.targets.values())
    if declaration.evaluation_by is not None:
        required.append(declaration.evaluation_by)
    if allow_subset:
        needed = [*keys, *required]
        missing = sorted(set(needed) - set(label.columns))
        label_keys = label.select(*keys) if not missing else None
        duplicate_rows = (
            label_keys.height - label_keys.unique().height
            if label_keys is not None
            else 0
        )
        unknown_rows = (
            label_keys.join(
                feature_table.select(*keys).unique(),
                on=list(keys),
                how="anti",
            ).height
            if label_keys is not None
            else 0
        )
        status = (
            "passed"
            if not missing
            and not duplicate_rows
            and not unknown_rows
            and label.height > 0
            else "failed"
        )
        observation = EvalObservation(
            status,
            output_kind="findings",
            details={
                "mode": "development_subset",
                "rows": label.height,
                "missing_columns": missing,
                "duplicate_key_rows": duplicate_rows,
                "unknown_key_rows": unknown_rows,
            },
        )
    else:
        observation = sidecar_output_shape(
            feature_table,
            label,
            keys,
            required,
        )
    return EvalObservation(
        status=observation.status,
        value=observation.value,
        output_kind=observation.output_kind,
        details={
            **observation.details,
            "context_table": declaration.label_table,
        },
    )


def _confirmation_scope(
    feature_table: pl.DataFrame,
    holdout: pl.DataFrame,
    development: pl.DataFrame | None,
    declaration: SignalSet,
) -> dict[str, Any]:
    keys = list(declaration.keys)
    required = [*keys, *declaration.targets.values()]
    if declaration.evaluation_by is not None:
        required.append(declaration.evaluation_by)
    missing = sorted(set(required) - set(holdout.columns))
    if missing:
        raise ValueError(f"confirmation context is missing columns: {missing}")
    if holdout.is_empty():
        raise ValueError("confirmation context contains no rows")
    holdout_keys = holdout.select(*keys)
    duplicate_rows = holdout_keys.height - holdout_keys.unique().height
    if duplicate_rows:
        raise ValueError(
            f"confirmation context contains {duplicate_rows} duplicate key rows"
        )
    feature_keys = feature_table.select(*keys).unique()
    unknown_rows = holdout_keys.join(feature_keys, on=keys, how="anti").height
    if unknown_rows:
        raise ValueError(
            f"confirmation context contains {unknown_rows} keys outside approved Feat"
        )
    overlap_rows = 0
    if development is not None:
        development_missing = sorted(set(keys) - set(development.columns))
        if development_missing:
            raise ValueError(
                "development context is missing confirmation keys: "
                f"{development_missing}"
            )
        overlap_rows = holdout_keys.join(
            development.select(*keys).unique(), on=keys, how="inner"
        ).height
        if overlap_rows:
            raise ValueError(
                "confirmation keys overlap development evidence; provide an unseen "
                f"context ({overlap_rows} overlapping rows)"
            )
    return {
        "holdout_rows": holdout.height,
        "duplicate_key_rows": duplicate_rows,
        "unknown_key_rows": unknown_rows,
        "development_overlap_rows": overlap_rows,
        "keys": keys,
        "evaluated_columns": list(declaration.signals),
        "candidate_columns_exposed": [],
    }


def _confirmation_card(
    path: Path,
    validation_plan: dict[str, Any],
    holdout: dict[str, Any],
) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8").rstrip()
    marker = "\n\n- card_id: `"
    body = text.rsplit(marker, 1)[0] if marker in text else text
    status = str(holdout["status"])
    evidence_scope = (
        "confirmed_holdout" if holdout.get("valid") else "confirmation_invalid"
    )
    body = _replace_card_field(body, "evidence_scope", evidence_scope)
    body = _replace_card_field(body, "confirmation_status", status)
    body += "\n\n## Frozen confirmation\n\n"
    body += "```json\n"
    body += json.dumps(
        {
            "validation_plan": validation_plan,
            "holdout": holdout,
        },
        indent=2,
        sort_keys=True,
        default=str,
    )
    body += "\n```"
    return render_card(body)


def _replace_card_field(body: str, name: str, value: str) -> str:
    prefix = f"- {name}: "
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{prefix}{value}"
            return "\n".join(lines)
    lines.insert(1, f"{prefix}{value}")
    return "\n".join(lines)


def _record_failed_confirmation(
    out: Path,
    selected: dict[str, Any],
    attempt_dir: Path,
    error: Exception,
) -> None:
    """Keep a failed reveal auditable without hiding the original CLI error."""

    try:
        state = read_workflow(out)
        current = next(
            item
            for item in state.get("attempts", {}).get("sig", [])
            if item.get("id") == selected.get("id")
        )
        if current.get("confirmation_status") != "revealing":
            return
        confirmation = dict(current.get("confirmation") or {})
        confirmation_dir = attempt_dir / "confirmation"
        confirmation_dir.mkdir(parents=True, exist_ok=True)
        error_path = confirmation_dir / "error.json"
        error_path.write_text(
            json.dumps(
                {"error": str(error), "type": type(error).__name__},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        holdout_record = {
            **confirmation,
            "status": "revealed_invalid",
            "valid": False,
            "evidence": str(error_path),
            "error": str(error),
        }
        (confirmation_dir / "holdout.json").write_text(
            json.dumps(holdout_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation_plan = json.loads(
            (attempt_dir / "validation_plan.json").read_text(encoding="utf-8")
        )
        card, card_identifier = _confirmation_card(
            attempt_dir / "card.md", validation_plan, holdout_record
        )
        (attempt_dir / "card.md").write_text(card + "\n", encoding="utf-8")
        manifest_path = attempt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "card_id": card_identifier,
                "confirmation": str(confirmation_dir / "holdout.json"),
                "confirmation_status": "revealed_invalid",
                "evidence_scope": "confirmation_invalid",
                "frozen": False,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        finish_confirmation(
            out,
            attempt=str(selected["id"]),
            evidence=str(error_path),
            valid=False,
            card=card_identifier,
        )
    except Exception:
        # The original error remains the user-facing failure. A partially
        # written "revealing" record still proves that the holdout was used.
        return


def _safe_artifact_name(name: str) -> None:
    if not name.strip() or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"invalid stage artifact name: {name!r}")


def _source_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"source file does not exist: {resolved}")
    return {"path": str(resolved), "sha256": _file_digest(resolved)}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_file():
        files = (resolved,)
        root = resolved.parent
    elif resolved.is_dir():
        files = tuple(sorted(item for item in resolved.rglob("*") if item.is_file()))
        root = resolved
    else:
        raise FileNotFoundError(f"confirmation context does not exist: {resolved}")
    if not files:
        raise ValueError(f"confirmation context is empty: {resolved}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_digest(item)))
    return digest.hexdigest()


def _signal_freeze_digest(attempt_dir: Path) -> str:
    """Hash immutable Sig research products while excluding review envelopes."""

    excluded_files = {"card.md", "manifest.json", "review.json"}
    excluded_directories = {"confirmation", "plots"}
    files = []
    for item in attempt_dir.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(attempt_dir)
        if relative.name in excluded_files:
            continue
        if any(part in excluded_directories for part in relative.parts[:-1]):
            continue
        files.append(item)
    digest = hashlib.sha256()
    for item in sorted(files):
        relative = item.relative_to(attempt_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_digest(item)))
    return digest.hexdigest()


def _write_stage_artifacts(
    stage_dir: Path,
    artifacts: Any,
    *,
    stage: str,
) -> dict[str, str]:
    directory = stage_dir / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for name, value in sorted(dict(artifacts).items()):
        _safe_artifact_name(name)
        if isinstance(value, bytes):
            filename = name if Path(name).suffix else f"{name}.bin"
            (directory / filename).write_bytes(value)
        else:
            filename = name if Path(name).suffix == ".json" else f"{name}.json"
            try:
                payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
            except TypeError as exc:
                raise TypeError(
                    f"{stage} artifact {name!r} must be JSON-serializable or bytes"
                ) from exc
            (directory / filename).write_text(payload, encoding="utf-8")
        manifest[name] = str(directory / filename)
    return manifest


def _write_signal_artifacts(out_sig: Path, artifacts: Any) -> dict[str, str]:
    return _write_stage_artifacts(out_sig, artifacts, stage="signal")


def _builder_recipe(module: ModuleType, stage: str) -> dict[str, Any]:
    """Read optional construction metadata without constraining builder code."""

    info = getattr(module, "BUILD_INFO", None)
    if info is None:
        return {
            "declared": False,
            "stage": stage,
            "steps": [],
            "summary": "",
        }
    if not isinstance(info, BuildInfo):
        raise TypeError("builder BUILD_INFO must be an research_ci.kit.BuildInfo")
    if info.stage != stage:
        raise ValueError(
            f"builder BUILD_INFO stage is {info.stage!r}; expected {stage!r}"
        )
    return {"declared": True, **info.as_dict()}


def _feature_keys(
    module: ModuleType,
    input_table: pl.DataFrame,
    feature_table: pl.DataFrame,
) -> tuple[str, ...]:
    declared = getattr(module, "FEATURE_KEYS", None)
    if declared is not None:
        return tuple(dict.fromkeys(declared))
    for candidate in (
        ("event_time", "inst"),
        ("event_time", "pair"),
        ("event_time",),
        ("t", "inst"),
    ):
        if all(
            column in input_table.columns and column in feature_table.columns
            for column in candidate
        ):
            return candidate
    return ()


def _result_status(result: Any) -> str:
    if not result.passed:
        return "halted"
    if any(record.status in {"failed", "error"} for record in result.records):
        return "review"
    if any(record.status == "not_applicable" for record in result.records):
        return "review"
    return "passed"


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("aci_external_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_plan(
    module: ModuleType,
    table: pl.DataFrame,
    stage: str,
    context: EvalContext,
) -> Any:
    """Call old two-argument plans and new context-aware plans alike."""

    maker = getattr(module, "make_plan")
    parameters = inspect.signature(maker).parameters
    kwargs: dict[str, Any] = {"stage": stage}
    if "context" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs["context"] = context
    return maker(table, **kwargs)


def _make_named_plan(
    module: ModuleType,
    function_name: str,
    table: pl.DataFrame,
    stage: str,
    context: EvalContext,
) -> Any:
    maker = getattr(module, function_name, None)
    if maker is None:
        raise AttributeError(
            f"signal plan must define {function_name}(table, stage, context)"
        )
    parameters = inspect.signature(maker).parameters
    kwargs: dict[str, Any] = {"stage": stage}
    if "context" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs["context"] = context
    return maker(table, **kwargs)


def _load_context(path: Path | None) -> EvalContext:
    """Load explicitly named Parquet sidecars without changing the main table."""

    if path is None:
        return EvalContext()
    if not path.exists():
        raise FileNotFoundError(f"evaluation context does not exist: {path}")
    tables: dict[str, pl.DataFrame] = {}
    metadata: dict[str, Any] = {}
    if path.is_file() and path.suffix == ".parquet":
        tables[path.stem] = pl.read_parquet(path)
    elif path.is_dir():
        for sidecar in sorted(path.glob("*.parquet")):
            tables[sidecar.stem] = pl.read_parquet(sidecar)
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
    elif path.suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            metadata = loaded
    else:
        raise ValueError(
            "context must be a Parquet file, a directory, or a JSON metadata file"
        )
    return EvalContext(tables=tables, metadata=metadata)


def _read_data_bundle(path: Path) -> DataBundle:
    """Read the new bundle layout while accepting an older single-table run."""

    if (path / "catalog.json").exists():
        return DataBundle.read(path)
    return DataBundle.from_main(pl.read_parquet(path / "main.parquet"))


def _with_data_bundle_context(context: EvalContext, bundle: DataBundle) -> EvalContext:
    """Expose native-frequency tables to evals without joining them into main."""

    native = bundle.eval_tables()
    collisions = sorted(set(context.tables) & set(native))
    if collisions:
        raise ValueError(
            f"DataBundle native tables collide with external context: {collisions}"
        )
    return EvalContext(
        tables={**context.tables, **native},
        metadata={
            **context.metadata,
            "data_bundle": {
                "catalog": bundle.as_catalog(),
                "native_tables": sorted(bundle.native),
            },
        },
    )


def _data_bundle_shape(bundle: DataBundle) -> EvalObservation:
    findings: dict[str, Any] = {"tables": {}}
    failed = False
    for name, table in bundle.tables().items():
        spec = bundle.catalog[name]
        keys = list(spec.keys)
        duplicate_rows = table.select(keys).height - table.select(keys).unique().height
        null_key_rows = int(
            table.select(pl.any_horizontal(pl.col(keys).is_null()).sum()).item()
        )
        findings["tables"][name] = {
            "rows": table.height,
            "columns": table.width,
            "keys": keys,
            "duplicate_key_rows": duplicate_rows,
            "null_key_rows": null_key_rows,
            "frequency": spec.frequency,
            "source": spec.source,
        }
        failed = failed or bool(duplicate_rows or null_key_rows or table.is_empty())
    coverage = bundle.coverage()
    coverage_names = (
        coverage.get_column("table").cast(pl.String)
        if "table" in coverage.columns
        else None
    )
    coverage_tables = (
        set(coverage_names.to_list()) if coverage_names is not None else set()
    )
    missing_coverage = sorted(set(bundle.tables()) - coverage_tables)
    unknown_coverage = sorted(coverage_tables - set(bundle.tables()))
    duplicate_coverage = (
        coverage_names.len() - coverage_names.n_unique()
        if coverage_names is not None
        else 0
    )
    findings["missing_coverage_tables"] = missing_coverage
    findings["unknown_coverage_tables"] = unknown_coverage
    findings["duplicate_coverage_rows"] = duplicate_coverage
    failed = failed or bool(missing_coverage or unknown_coverage or duplicate_coverage)
    return EvalObservation(
        "failed" if failed else "passed",
        output_kind="findings",
        details=findings,
    )
