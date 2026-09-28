import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from research_ci import __version__
from research_ci.api.contract import ContractError
from research_ci.store.ledger import LedgerError
from research_ci.workflow.commands import CONTRACT_FILE, Idea, find_ideas
from research_ci.workflow.machine import WorkflowError
from research_ci.workflow.templates import CONTRACT, TEMPLATES

app = typer.Typer(no_args_is_help=True, add_completion=False)
contract_app = typer.Typer(no_args_is_help=True, help="Work on the contract.")
app.add_typer(contract_app, name="contract")

ERRORS = (WorkflowError, ContractError, LedgerError)
IdeaDir = Annotated[Path, typer.Option("--idea", "-i", help="Idea directory.")]
AsJson = Annotated[bool, typer.Option("--json", help="Machine-readable output.")]


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def init(
    idea: Annotated[Path, typer.Argument(help="Directory of the new idea.")],
    template: Annotated[str, typer.Option("--template", "-t", help=f"One of: {', '.join(TEMPLATES)}.")],
) -> None:
    """Start an idea: write a contract skeleton and open its ledger."""
    with _errors():
        Idea.init(idea, template)
    typer.echo(f"created {idea} ({template}); edit {idea / CONTRACT_FILE}, then run `rci contract derive -i {idea}`")


@app.command()
def status(idea: IdeaDir = Path("."), as_json: AsJson = False) -> None:
    """Where every stage of the idea stands."""
    with _errors():
        current = Idea(idea).status()
    if as_json:
        typer.echo(json.dumps(asdict(current), indent=2))
        return
    typer.echo(f"{current.idea} ({current.template})")
    for s in current.stages:
        trials = "" if s.budget is None else f"{s.trials}/{s.budget}"
        typer.echo(f"  {s.stage:<9} {s.state:<11} v{s.version:<3} {trials:<7} {s.note}".rstrip())


@app.command("next")
def next_(idea: IdeaDir = Path("."), as_json: AsJson = False) -> None:
    """What should happen next, and who should do it."""
    with _errors():
        actions = Idea(idea).next_actions()
    if as_json:
        typer.echo(json.dumps([asdict(a) for a in actions], indent=2))
        return
    for a in actions:
        typer.echo(f"{a.who:<10} {a.command:<36} {a.reason}")
    if not actions:
        typer.echo("nothing to do")


@app.command()
def inbox(
    root: Annotated[Path, typer.Argument(help="Directory holding ideas.")] = Path("."),
    as_json: AsJson = False,
) -> None:
    """Everything waiting for a person, across ideas."""
    items: list[dict[str, Any]] = []
    for path in find_ideas(root):
        where = os.path.relpath(path)
        try:
            items += [{"idea": where, **asdict(a)} for a in Idea(path).next_actions() if a.who == "human"]
        except ERRORS as exc:
            items.append(
                {"idea": where, "who": "human", "stage": "", "kind": "error", "command": "", "reason": str(exc)}
            )
    if as_json:
        typer.echo(json.dumps(items, indent=2))
        return
    for item in items:
        typer.echo(f"{item['idea']:<24} {item['command']:<36} {item['reason']}")
    if not items:
        typer.echo("nothing waiting")


@app.command()
def verify(idea: IdeaDir = Path(".")) -> None:
    """Check the ledger's hash chain."""
    with _errors():
        count = Idea(idea).ledger.verify()
    typer.echo(f"ledger ok: {count} records")


@contract_app.command("derive")
def derive(idea: IdeaDir = Path(".")) -> None:
    """Load contract.py, validate it, and register it for approval."""
    with _errors():
        contract = Idea(idea).derive_contract()
    typer.echo(f"contract registered: {contract.claim}")
    typer.echo("a person freezes it with `rci approve --stage contract`")


@app.command()
def approve(
    stage: Annotated[str, typer.Option("--stage", "-s", help="Stage to freeze.")],
    idea: IdeaDir = Path("."),
    extend: Annotated[int | None, typer.Option(help="Add this many submissions to a paused layer instead.")] = None,
) -> None:
    """Freeze a stage, or extend a paused layer's budget. Only a person at a terminal can do this."""
    if not _interactive():
        typer.echo("error: approve needs a person at an interactive terminal", err=True)
        raise typer.Exit(1)
    with _errors():
        target = Idea(idea)
        if extend is not None:
            _extend(target, stage, extend)
            return
        preview = target.preview_approval(stage)
        typer.echo(_render_preview(target, preview))
        typer.confirm(f"Freeze {stage} v{preview['version']}?", abort=True)
        freeze = target.approve(stage)
    typer.echo(f"frozen {stage} v{preview['version']}: {freeze.hash[:12]}")
    paths = f"{os.path.relpath(target.workspace.root)} {os.path.relpath(target.root / CONTRACT_FILE)}"
    typer.echo(f"to keep it in git: git add {paths} && git commit -m 'rci: freeze {stage} of {target.root.name}'")


def _extend(target: Idea, stage: str, trials: int) -> None:
    current = {s.stage: s for s in target.status().stages}.get(stage)
    if current is None:
        raise WorkflowError(f"{stage!r} is not a stage of this idea")
    typer.echo(f"{stage} is {current.state}; {current.trials} of {current.budget} submissions used")
    typer.confirm(f"Add {trials} submissions to {stage}?", abort=True)
    typer.echo(f"{stage} is {target.extend(stage, trials)} again")


def _render_preview(target: Idea, preview: dict[str, Any]) -> str:
    lines = [f"approve {preview['stage']} v{preview['version']} of {target.root.name}"]
    if preview["stage"] == CONTRACT:
        lines += [f"  {k}: {v}" for k, v in preview["contract"].items()]
    lines.append("  frozen dependencies:")
    lines += [f"    {k:<20} {v[:12] if len(v) == 64 else v}" for k, v in sorted(preview["deps"].items())]
    return "\n".join(lines)


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except ERRORS as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
