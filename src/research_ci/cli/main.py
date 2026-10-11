import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from research_ci import __version__
from research_ci.cli import common
from research_ci.cli.contract import contract_app
from research_ci.cli.data import data_app
from research_ci.cli.feat import feat_app
from research_ci.render.common import card_id
from research_ci.runner import catalog
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


KIND_USE = {
    "data": "Contract(data=Use(name, {options}))",
    "universe": "Universe(name, clusters=<column>, groups=(<column>, ...))",
    "label": "Label(<label name>, Use(name, {options}), horizons=(...))",
    "target": "Effect(on=<label name>, target=Use(name, {options}))",
    "cost": "Contract(costs=Use(name, {options}))",
    "consumer": "Contract(consumer=Use(name, {options})), for its decision times",
}


@app.command()
def plugins(as_json: common.AsJson = False) -> None:
    """Every installed plugin a contract can name: its kind, options, and what it declares."""
    found = catalog.plugins()
    if as_json:
        typer.echo(json.dumps(found, indent=2))
        return
    order = [*KIND_USE, "unknown", "error"]

    def rank(p: dict[str, Any]) -> tuple[int, str]:
        first = p["kind"].split("/")[0]
        return (order.index(first) if first in order else len(order), p["name"])

    shown = ""
    for p in sorted(found, key=rank):
        if p["kind"] != shown:
            shown = p["kind"]
            typer.echo(f"{shown}: {KIND_USE.get(shown.split('/')[0], '')}".rstrip(": "))
        if shown == "error":
            typer.echo(f"  {p['name']}: cannot be loaded: {p['error']}")
            continue
        typer.echo(f"  {p['signature']}")
        for line in _declared(p):
            typer.echo(f"    {line}")
    if not found:
        typer.echo("no plugins installed; a lab registers them under the entry-point group research_ci.plugins")


def _declared(p: dict[str, Any]) -> list[str]:
    """What a plugin says about itself: its docstring's first sentence, and what its kind declares."""
    lines = [p["doc"]] if p["doc"] else []
    if p.get("fields") is not None:
        lines.append(f"fields: {', '.join(p['fields']) or 'not declared'}")
    if p.get("levels"):
        lv = p["levels"]
        lines.append(f"levels: {', '.join(lv['fields'])} · factor {lv['factor']:g} · window {lv['window']}")
        lines[-1] += f" · mask {lv['mask']}"
    if p.get("reference"):
        lines.append("reference source: answers the independent check at admission")
    if "per" in p:
        lines.append(f"per: {p['per'] or 'not declared'} · rolls: {'yes' if p['rolls'] else 'no'}")
    lines.append(f"from the {p['origin']}" if p["origin"] == "registry" else f"from an {p['origin']}")
    return lines


@app.command()
def methods(as_json: common.AsJson = False) -> None:
    """The eval methods a criterion, a data test, or a pit can call, and the worked examples that back each; then how
    a null world is written."""
    found = catalog.methods()
    worlds = catalog.nulls()
    if as_json:
        typer.echo(json.dumps(found + worlds, indent=2))
        return
    formats = catalog.formats()
    shown = ""
    for m in found:
        if m["use"] != shown:
            typer.echo(f"{m['use']}: {formats.get(m['use'], '')}".rstrip(": "))
            shown = m["use"]
        typer.echo(f"  {m['import']}")
        typer.echo(f"    {m['name']}{m['signature']}")
        if m["doc"]:
            typer.echo(f"    {m['doc']}")
        typer.echo(f"    backed by {_count(m['examples'], 'worked example')} · {_count(m['properties'], 'property')}")
    typer.echo(f"null world: {catalog.NULLS_FORMAT}")
    typer.echo(f"  from research_ci.api import {', '.join(w['name'] for w in worlds)}")
    for w in worlds:
        typer.echo(f"    {w['name']}{w['signature']}")
        width = max(len(f["name"]) for f in w["fields"])
        for f in w["fields"]:
            typer.echo(f"      {f['name']:<{width}}  {f['means']}")
    typer.echo(f"  for example: {catalog.NULLS_EXAMPLE}")


def _count(n: int, what: str) -> str:
    many = what[:-1] + "ies" if what.endswith("y") else what + "s"
    return f"{n} {what if n == 1 else many}"


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
