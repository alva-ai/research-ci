import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from research_ci import __version__
from research_ci.cli import catalog, common
from research_ci.cli.contract import contract_app
from research_ci.cli.data import data_app
from research_ci.cli.eval import eval_app
from research_ci.cli.feat import feat_app
from research_ci.render.common import card_id
from research_ci.workflow.base import CONTRACT_FILE, Approval, find_ideas
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import WorkflowError
from research_ci.workflow.templates import EXAMPLE_FILE, START_FILE

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Research CI: judge research ideas with fixed code. Start with `rci init <idea>` and read its START.md; after "
    "every step, `rci next -i <idea>` says what to do.",
)
app.add_typer(contract_app, name="contract")
app.add_typer(data_app, name="data")
app.add_typer(feat_app, name="feat")
app.add_typer(eval_app, name="eval")


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def init(
    idea: Annotated[Path, typer.Argument(help="Directory of the new idea.")],
) -> None:
    """Start an idea: write START.md, a contract skeleton, and research/example.py, and open its ledger."""
    with common.errors():
        Idea.init(idea)
    typer.echo(f"created {idea}; read {idea / START_FILE} first: it says how the work goes")
    typer.echo(f"then fill in {idea / CONTRACT_FILE} and run `rci contract derive -i {idea}`")
    typer.echo(f"features start from {idea / EXAMPLE_FILE}; `rci next -i {idea}` says what to do after every step")


app.command()(catalog.plugins)
app.command()(catalog.methods)


@app.command()
def status(idea: common.IdeaDir = Path("."), as_json: common.AsJson = False) -> None:
    """Where every stage of the idea stands."""
    with common.errors():
        current = Idea(idea).status()
    if as_json:
        typer.echo(json.dumps(asdict(current), indent=2))
        return
    typer.echo(current.idea)
    for s in current.stages:
        trials = "" if s.budget is None else f"{s.trials}/{s.budget}"
        typer.echo(f"  {s.stage:<9} {s.state:<7} v{s.version:<3} {trials:<7} {s.note}".rstrip())


@app.command("next")
def next_(idea: common.IdeaDir = Path("."), as_json: common.AsJson = False) -> None:
    """What should happen next, and who should do it."""
    with common.errors():
        actions = Idea(idea).next_actions()
    if as_json:
        typer.echo(json.dumps([asdict(a) for a in actions], indent=2))
        return
    for a in actions:
        typer.echo(f"{a.who:<10} {a.command or '(nothing to run)':<36} {a.reason}")
    if not actions:
        typer.echo("nothing to do")


@app.command()
def inbox(
    root: Annotated[Path, typer.Argument(help="Directory holding ideas.")] = Path("."),
    as_json: common.AsJson = False,
) -> None:
    """Everything waiting for a person, across ideas: results up for review and paused layers."""
    items: list[dict[str, Any]] = []
    for path in find_ideas(root):
        where = os.path.relpath(path)
        try:
            items += [{"idea": where, **asdict(a)} for a in Idea(path).next_actions() if a.who == "human"]
        except common.ERRORS as exc:
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
def verify(idea: common.IdeaDir = Path(".")) -> None:
    """Check the ledger's hash chain."""
    with common.errors():
        count = Idea(idea).ledger.verify()
    typer.echo(f"ledger ok: {count} records")


@app.command()
def approve(
    stage: Annotated[str, typer.Option("--stage", "-s", help="Stage whose result to approve.")],
    idea: common.IdeaDir = Path("."),
    extend: Annotated[int | None, typer.Option(help="Add this many submissions to a paused layer instead.")] = None,
    relayed: Annotated[
        str | None, typer.Option(help="An agent relays a person's approval; quote the person's words.")
    ] = None,
    auto: Annotated[bool, typer.Option(help="Fully automatic research: the agent approves.")] = False,
    card: common.Card = "",
) -> None:
    """Approve the result a stage has up for review, which freezes it, or extend a paused layer's budget.

    A person at a terminal approves by default. In semi-automatic research an agent stops, asks the person, and
    relays the answer with --relayed; in fully automatic research the agent approves with --auto. The ledger records
    which it was.
    """
    if relayed is not None and auto:
        typer.echo("error: choose --relayed or --auto, not both", err=True)
        raise typer.Exit(1)
    by_person = relayed is None and not auto
    if by_person and not common.interactive():
        typer.echo("error: approve needs a person at an interactive terminal (or --relayed / --auto)", err=True)
        raise typer.Exit(1)
    with common.errors():
        target = Idea(idea)
        if extend is not None:
            if not by_person:
                raise WorkflowError("extending a budget needs a person at a terminal")
            _extend(target, stage, extend)
            return
        who = Approval()
        if relayed is not None:
            who = Approval(by="human", channel="relayed", words=relayed, card=card)
        elif auto:
            who = Approval(by="agent", channel="auto")
        preview = target.preview_approval(stage)
        typer.echo(_render_preview(target, preview))
        if by_person:
            typer.confirm(f"Approve {stage} v{preview['version']}?", abort=True)
        freeze = target.approve(stage, approval=who)
        state = target.status().stage(stage).state
    typer.echo(f"{stage} v{preview['version']} {state}: freeze {freeze.hash[:12]} · {who.describe()}")
    paths = f"{os.path.relpath(target.workspace.root)} {os.path.relpath(target.root / CONTRACT_FILE)}"
    typer.echo(f"to keep it in git: git add {paths} && git commit -m 'rci: approve {stage} of {target.root.name}'")


def _extend(target: Idea, stage: str, trials: int) -> None:
    current = {s.stage: s for s in target.status().stages}.get(stage)
    if current is None:
        raise WorkflowError(f"{stage!r} is not a stage of this idea")
    typer.echo(f"{stage} is {current.state}; {current.trials} of {current.budget} submissions used")
    typer.confirm(f"Add {trials} submissions to {stage}?", abort=True)
    typer.echo(f"{stage} is {target.extend(stage, trials)}")


def _render_preview(target: Idea, preview: dict[str, Any]) -> str:
    lines = [f"approve {preview['stage']} v{preview['version']} of {target.root.name}"]
    if "card" in preview:
        lines += [preview["card"], f"card {card_id(preview['card'])}"]
    lines.append("  frozen dependencies:")
    width = max(map(len, preview["deps"]), default=0)
    lines += [f"    {k:<{width}} {v[:12] if len(v) == 64 else v}" for k, v in sorted(preview["deps"].items())]
    return "\n".join(lines)
