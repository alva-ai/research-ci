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
from research_ci.render.card import card_id
from research_ci.store.ledger import LedgerError
from research_ci.workflow.commands import CONTRACT_FILE, DATA_TEXT, Approval, Idea, find_ideas
from research_ci.workflow.machine import WorkflowError
from research_ci.workflow.templates import TEMPLATES

app = typer.Typer(no_args_is_help=True, add_completion=False)
contract_app = typer.Typer(no_args_is_help=True, help="Work on the contract.")
app.add_typer(contract_app, name="contract")
data_app = typer.Typer(no_args_is_help=True, help="Check the data against the contract.")
app.add_typer(data_app, name="data")
source_app = typer.Typer(no_args_is_help=True, help="Admit a data source, once for every idea that uses it.")
app.add_typer(source_app, name="source")

ERRORS = (WorkflowError, ContractError, LedgerError)
IdeaDir = Annotated[Path, typer.Option("--idea", "-i", help="Idea directory.")]
AsJson = Annotated[bool, typer.Option("--json", help="Machine-readable output.")]
Card = Annotated[
    str, typer.Option("--card", help="With --relayed: the id of the card the person was shown, from `card <id>`.")
]


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
        _, card = Idea(idea).derive_contract()
    typer.echo(card)
    typer.echo(f"card {card_id(card)}")
    typer.echo("registered; a person freezes it with `rci approve --stage contract`")


@data_app.command("check")
def data_check(
    idea: IdeaDir = Path("."),
    keep: Annotated[
        list[str] | None, typer.Option("--keep", help="A treatment a person chose not to apply, by its name.")
    ] = None,
) -> None:
    """Admit the data source if needed, check the data against the contract, and print the cards."""
    with _errors():
        target = Idea(idea)
        spec, card = target.check_data(tuple(keep or ()))
        if spec["source"]["status"] != "admitted":
            source = target.source_card()
            typer.echo(source)
            typer.echo(f"card {card_id(source)}")
    typer.echo(card)
    typer.echo(f"card {card_id(card)}")
    status = spec["verdict"]["status"]
    if status == "blocked":
        typer.echo("blocked; see the verdict above")
    else:
        typer.echo("checked; a person freezes it with `rci approve --stage data`")


@data_app.command("show")
def data_show(idea: IdeaDir = Path(".")) -> None:
    """Print the last data card."""
    with _errors():
        target = Idea(idea)
        path = target.workspace.path(DATA_TEXT)
        if not path.exists():
            raise WorkflowError("the data has not been checked; run `rci data check` first")
    text = path.read_text().rstrip()
    typer.echo(text)
    typer.echo(f"card {card_id(text)}")


@data_app.command("submit")
def data_submit(idea: IdeaDir = Path(".")) -> None:
    """Record the frozen data check as passed."""
    with _errors():
        state = Idea(idea).submit_data()
    typer.echo(f"data {state}")


@source_app.command("show")
def source_show(idea: IdeaDir = Path(".")) -> None:
    """Print the admission card of the idea's data source."""
    with _errors():
        text = Idea(idea).source_card()
    typer.echo(text)
    typer.echo(f"card {card_id(text)}")


@source_app.command("approve")
def source_approve(
    idea: IdeaDir = Path("."),
    relayed: Annotated[
        str | None, typer.Option(help="An agent relays a person's approval; quote the person's words.")
    ] = None,
    auto: Annotated[bool, typer.Option(help="Fully automatic research: the agent admits the source.")] = False,
    card: Card = "",
) -> None:
    """Admit the idea's data source, once for this plugin, options and code."""
    who = _who(relayed, auto, card)
    with _errors():
        target = Idea(idea)
        text = target.source_card()
        typer.echo(text)
        typer.echo(f"card {card_id(text)}")
        if who.channel == "terminal":
            typer.confirm("Admit this source?", abort=True)
        record = target.approve_source(who)
    typer.echo(f"admitted {record['stage']}: {record['payload']['approval']['note']}")


def _who(relayed: str | None, auto: bool, card: str = "") -> Approval:
    if relayed is not None and auto:
        typer.echo("error: choose --relayed or --auto, not both", err=True)
        raise typer.Exit(1)
    if relayed is None and not auto and not _interactive():
        typer.echo("error: approving needs a person at an interactive terminal (or --relayed / --auto)", err=True)
        raise typer.Exit(1)
    if relayed is not None:
        return Approval(by="human", channel="relayed", words=relayed, card=card)
    if auto:
        return Approval(by="agent", channel="auto")
    return Approval()


@app.command()
def approve(
    stage: Annotated[str, typer.Option("--stage", "-s", help="Stage to freeze.")],
    idea: IdeaDir = Path("."),
    extend: Annotated[int | None, typer.Option(help="Add this many submissions to a paused layer instead.")] = None,
    relayed: Annotated[
        str | None, typer.Option(help="An agent relays a person's approval; quote the person's words.")
    ] = None,
    auto: Annotated[
        bool, typer.Option(help="Fully automatic research: the agent approves; results are capped.")
    ] = False,
    card: Card = "",
) -> None:
    """Freeze a stage, or extend a paused layer's budget.

    A person at a terminal approves by default. In semi-automatic research an agent stops, asks the person, and
    relays the answer with --relayed; in fully automatic research the agent approves with --auto. The ledger records
    which it was.
    """
    if relayed is not None and auto:
        typer.echo("error: choose --relayed or --auto, not both", err=True)
        raise typer.Exit(1)
    by_person = relayed is None and not auto
    if by_person and not _interactive():
        typer.echo("error: approve needs a person at an interactive terminal (or --relayed / --auto)", err=True)
        raise typer.Exit(1)
    with _errors():
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
            typer.confirm(f"Freeze {stage} v{preview['version']}?", abort=True)
        freeze = target.approve(stage, approval=who)
    typer.echo(f"frozen {stage} v{preview['version']}: {freeze.hash[:12]} · {who.describe()}")
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
    if "card" in preview:
        lines += [preview["card"], f"card {card_id(preview['card'])}"]
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
