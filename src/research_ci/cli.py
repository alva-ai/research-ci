"""Small CLI for the first protocol layer."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl

from .cards import contract_card, data_card, feat_card, sig_card
from .contract import ContractError, validate_contract
from .eval import (
    DataSet,
    EvalContext,
    EvalObservation,
    EvalSpec,
    FeatureSet,
    LibrarySelection,
    SignalSet,
    assess_features,
    assess_records,
    assess_signal_evidence,
    compile_stage_menu,
)
from .eval.baseline import (
    feature_output_shape,
    sidecar_output_shape,
    signal_algo_input_shape,
    signal_disposition_coverage,
    signal_feature_universe,
    signal_lineage_coverage,
    signal_screening_artifact_coverage,
    signal_screening_plan_coverage,
)
from .eval.library import METHODS as LIBRARY_METHODS
from .eval.runner import run_eval_plan
from .kit import BuildInfo, SignalBundle
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
        data_run.add_argument("--context", type=Path)

    feat = commands.add_parser("feat", help="run an external feature builder")
    feat_commands = feat.add_subparsers(dest="feat_command", required=True)
    feat_run = feat_commands.add_parser("run")
    feat_run.add_argument("--builder", required=True, type=Path)
    feat_run.add_argument("--plan", required=True, type=Path)
    feat_run.add_argument("--out", required=True, type=Path)
    feat_run.add_argument("--context", type=Path)

    sig = commands.add_parser("sig", help="run an external signal builder")
    sig_commands = sig.add_subparsers(dest="sig_command", required=True)
    sig_run = sig_commands.add_parser("run")
    sig_run.add_argument("--builder", required=True, type=Path)
    sig_run.add_argument("--plan", required=True, type=Path)
    sig_run.add_argument("--out", required=True, type=Path)
    sig_run.add_argument("--context", type=Path)

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
        return _data_run(args.contract, args.builder, args.plan, args.out, args.context)
    if args.command == "feat":
        return _feat_run(args.builder, args.plan, args.out, args.context)
    if args.command == "sig":
        return _sig_run(args.builder, args.plan, args.out, args.context)
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
) -> DataSet | FeatureSet | SignalSet:
    module = _load_module(path)
    declarations = [
        value
        for name in ("DECLARATION", "DATA_SET", "FEATURE_SET", "SIGNAL_SET")
        if isinstance(
            (value := getattr(module, name, None)),
            (DataSet, FeatureSet, SignalSet),
        )
    ]
    if stage is not None:
        declarations = [item for item in declarations if item.stage == stage]
    unique = list({id(item): item for item in declarations}.values())
    if not unique:
        raise ValueError(
            "declaration module must expose DECLARATION, DATA_SET, FEATURE_SET, "
            "or SIGNAL_SET"
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
        recipe = _builder_recipe(builder, "data")
        (out_data / "recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        plan_module = _load_module(plan_path)
        context = _load_context(context_path)
        plan = _make_plan(plan_module, table, "data", context)
        result = run_eval_plan(table, plan, context)
        result.write_report(out_data / "eval" / "report.jsonl")
        assessment = assess_records(result.records)
        (out_data / "eval" / "assessment.json").write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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


def _feat_run(
    builder_path: Path,
    plan_path: Path,
    out: Path,
    context_path: Path | None = None,
) -> int:
    try:
        state = read_workflow(out)
        require_stage_ready(state, "feat")
        data_path = out / "data" / "main.parquet"
        data_table = pl.read_parquet(data_path)
        data_before = data_table.clone()

        builder = _load_module(builder_path)
        feature_table = _build_features(builder, data_table)
        if not isinstance(feature_table, pl.DataFrame):
            raise TypeError("feature builder must return a Polars DataFrame")
        if not data_table.equals(data_before):
            raise ValueError("feature builder modified the input DataFrame")

        keys = _feature_keys(builder, data_table, feature_table)
        plan_module = _load_module(plan_path)
        context = _load_context(context_path)
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

        out_feat = out / "feat"
        out_feat.mkdir(parents=True, exist_ok=True)
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
        result_status = _result_status(result)
        card, card_identifier = feat_card(
            feature_table,
            result,
            exploration,
            input_rows=data_table.height,
            keys=keys,
            result_status=result_status,
        )
        (out_feat / "card.md").write_text(card + "\n", encoding="utf-8")
        state = submit_for_review(
            out,
            stage="feat",
            card=card_identifier,
            result_status=result_status,
        )
        feature_columns = [
            column for column in feature_table.columns if column not in keys
        ]
        (out_feat / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "feat",
                    "status": result_status,
                    "workflow_state": state["stages"]["feat"],
                    "card_id": card_identifier,
                    "input_table": str(data_path),
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
    except Exception as exc:  # noqa: BLE001 - CLI reports external workspace errors
        print(f"feat run failed: {exc}")
        return 1
    print(card)
    print(f"feat card: {out_feat / 'card.md'}")
    print(
        "next: rci approve "
        f"--stage feat --out {out} --by human --card {card_identifier}"
    )
    print(
        "   or: rci approve "
        f"--stage feat --out {out} --by agent --card {card_identifier}"
    )
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
        feature_path = out / "feat" / "main.parquet"
        feature_table = pl.read_parquet(feature_path)
        feature_before = feature_table.clone()

        builder = _load_module(builder_path)
        bundle = _build_signals(builder, feature_table)
        if not feature_table.equals(feature_before):
            raise ValueError("signal builder modified the input Feature DataFrame")

        plan_module = _load_module(plan_path)
        declaration = _signal_declaration(plan_module)
        library_selection = getattr(plan_module, "LIBRARY_SELECTION", None)
        if not isinstance(library_selection, LibrarySelection):
            raise TypeError(
                "signal plan must expose a LibrarySelection as LIBRARY_SELECTION"
            )
        keys = declaration.keys
        expected_main = (*keys, *declaration.signals)
        missing_main = sorted(set(expected_main) - set(bundle.main.columns))
        extra_main = sorted(set(bundle.main.columns) - set(expected_main))
        if missing_main or extra_main:
            raise ValueError(
                "signal main table must contain exactly declared keys and signals; "
                f"missing={missing_main}, extra={extra_main}"
            )
        signal_table = bundle.main.select(expected_main)
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
        missing_features = sorted(
            set((*keys, *declaration.features)) - set(feature_table.columns)
        )
        if missing_features:
            raise ValueError(
                f"approved Feat table is missing Sig inputs: {missing_features}"
            )
        algo_input = _build_algo_input(
            builder,
            out / "data" / "main.parquet",
            signal_table,
        )

        external_context = _load_context(context_path)
        duplicate_sidecars = sorted(set(external_context.tables) & set(bundle.sidecars))
        if duplicate_sidecars:
            raise ValueError(
                f"signal sidecars collide with external context: {duplicate_sidecars}"
            )
        context = EvalContext(
            tables={**external_context.tables, **bundle.sidecars},
            metadata={**external_context.metadata, **bundle.metadata},
        )
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
            "engine.baseline.signal_output_shape",
            runner=lambda _: feature_output_shape(
                feature_table,
                signal_table,
                keys,
            ),
            failure_policy="stop",
            description="preserve the approved Feat key grid in the signal output",
            execution_mode="polars_join_group",
        )
        all_outputs = (*declaration.candidates, *declaration.signals)
        baseline_rules = [
            EvalSpec(
                "engine.baseline.signal_feature_universe",
                runner=lambda _: signal_feature_universe(
                    feature_table,
                    keys,
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
                    keys,
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
        if declaration.candidates:
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
                    ),
                    failure_policy="stop",
                    description="preserve keys and declared labels in the sidecar",
                    execution_mode="polars_join_group",
                )
            )
        if algo_input is not None:
            baseline_rules.append(
                EvalSpec(
                    "engine.baseline.signal_algo_input_shape",
                    runner=lambda _: signal_algo_input_shape(
                        signal_table,
                        algo_input,
                        keys,
                        declaration.signals,
                    ),
                    failure_policy="stop",
                    description=(
                        "preserve one wide row per signal key with named instrument "
                        "and price columns"
                    ),
                    execution_mode="polars_join_group",
                )
            )
        plan = replace(plan, baseline=(*baseline_rules, *plan.baseline))
        result = run_eval_plan(evaluation_table, plan, context)

        out_sig = out / "sig"
        out_sig.mkdir(parents=True, exist_ok=True)
        signal_table.write_parquet(out_sig / "main.parquet")
        algo_input_path: Path | None = None
        if algo_input is not None:
            algo_input_path = out_sig / "algo_input.parquet"
            algo_input.write_parquet(algo_input_path)
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
        for name, sidecar in sorted(context.tables.items()):
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
        artifact_manifest = _write_signal_artifacts(out_sig, bundle.artifacts)

        result.write_report(out_sig / "eval" / "report.jsonl")
        assessment = assess_records(result.records)
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
        result_status = _result_status(result)
        card, card_identifier = sig_card(
            signal_table,
            result,
            dict(bundle.metadata),
            input_rows=feature_table.height,
            evaluation_columns=tuple(evaluation_table.columns),
            keys=keys,
            sidecars=tuple(sorted(context.tables)),
            candidates=declaration.candidates,
            dispositions=disposition_table,
            result_status=result_status,
        )
        (out_sig / "card.md").write_text(card + "\n", encoding="utf-8")
        state = submit_for_review(
            out,
            stage="sig",
            card=card_identifier,
            result_status=result_status,
        )
        (out_sig / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "sig",
                    "status": result_status,
                    "workflow_state": state["stages"]["sig"],
                    "card_id": card_identifier,
                    "input_table": str(feature_path),
                    "main_table": str(out_sig / "main.parquet"),
                    "algo_input": (
                        str(algo_input_path) if algo_input_path is not None else None
                    ),
                    "recipe": str(out_sig / "recipe.json"),
                    "selection": str(out_sig / "selection.json"),
                    "eval_report": str(out_sig / "eval" / "report.jsonl"),
                    "assessment": str(out_sig / "eval" / "assessment.json"),
                    "context_tables": sorted(context.tables),
                    "research": research_manifest,
                    "artifacts": artifact_manifest,
                    "rows": signal_table.height,
                    "columns": signal_table.columns,
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
    except Exception as exc:  # noqa: BLE001 - CLI reports external workspace errors
        print(f"sig run failed: {exc}")
        return 1
    print(card)
    print(f"sig card: {out_sig / 'card.md'}")
    print(
        f"next: rci approve --stage sig --out {out} --by human --card {card_identifier}"
    )
    print(
        "   or: rci approve "
        f"--stage sig --out {out} --by agent --card {card_identifier}"
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


def _build_features(module: ModuleType, table: pl.DataFrame) -> pl.DataFrame:
    for name in ("build_features", "build"):
        builder = getattr(module, name, None)
        if builder is not None:
            return builder(table)
    raise AttributeError(
        "feature builder must define build_features(table) or build(table)"
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


def _build_algo_input(
    module: ModuleType,
    data_path: Path,
    signal_table: pl.DataFrame,
) -> pl.DataFrame | None:
    """Run an optional case-owned wide handoff builder for the Algo stage."""

    builder = getattr(module, "build_algo_input", None)
    if builder is None:
        return None
    data_table = pl.read_parquet(data_path)
    data_before = data_table.clone()
    signal_before = signal_table.clone()
    result = builder(data_table, signal_table)
    if not isinstance(result, pl.DataFrame):
        raise TypeError(
            "build_algo_input(data, signals) must return a Polars DataFrame"
        )
    if not data_table.equals(data_before) or not signal_table.equals(signal_before):
        raise ValueError("build_algo_input modified an input DataFrame")
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


def _label_sidecar_shape(
    feature_table: pl.DataFrame,
    declaration: SignalSet,
    context: EvalContext,
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
    required = list(declaration.targets.values())
    if declaration.evaluation_by is not None:
        required.append(declaration.evaluation_by)
    observation = sidecar_output_shape(
        feature_table,
        label,
        declaration.keys,
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


def _safe_artifact_name(name: str) -> None:
    if not name.strip() or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"invalid signal artifact name: {name!r}")


def _write_signal_artifacts(out_sig: Path, artifacts: Any) -> dict[str, str]:
    directory = out_sig / "artifacts"
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
                    f"signal artifact {name!r} must be JSON-serializable or bytes"
                ) from exc
            (directory / filename).write_text(payload, encoding="utf-8")
        manifest[name] = str(directory / filename)
    return manifest


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
    spec = importlib.util.spec_from_file_location("rci_external_module", path)
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
