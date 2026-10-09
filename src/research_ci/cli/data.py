from pathlib import Path
from typing import Annotated

import typer

from research_ci.cli import common
from research_ci.render.common import card_id
from research_ci.workflow.data import DATA_TEXT
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import WorkflowError

data_app = typer.Typer(no_args_is_help=True, help="Check the data against the contract.")
source_app = typer.Typer(no_args_is_help=True, help="Admit a data source, once for every idea that uses it.")


@data_app.command("check")
def data_check(
    idea: common.IdeaDir = Path("."),
    keep: Annotated[
        list[str] | None, typer.Option("--keep", help="A treatment a person chose not to apply, by its name.")
    ] = None,
) -> None:
    """Admit the data source if needed, check the data against the contract, and print the cards."""
    with common.errors():
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
def data_show(idea: common.IdeaDir = Path(".")) -> None:
    """Print the last data card."""
    with common.errors():
        target = Idea(idea)
        path = target.workspace.path(DATA_TEXT)
        if not path.exists():
            raise WorkflowError("the data has not been checked; run `rci data check` first")
    text = path.read_text().rstrip()
    typer.echo(text)
    typer.echo(f"card {card_id(text)}")


@data_app.command("submit")
def data_submit(idea: common.IdeaDir = Path(".")) -> None:
    """Record the frozen data check as passed."""
    with common.errors():
        state = Idea(idea).submit_data()
    typer.echo(f"data {state}")


@source_app.command("show")
def source_show(idea: common.IdeaDir = Path(".")) -> None:
    """Print the admission card of the idea's data source."""
    with common.errors():
        text = Idea(idea).source_card()
    typer.echo(text)
    typer.echo(f"card {card_id(text)}")


@source_app.command("approve")
def source_approve(
    idea: common.IdeaDir = Path("."),
    relayed: Annotated[
        str | None, typer.Option(help="An agent relays a person's approval; quote the person's words.")
    ] = None,
    auto: Annotated[bool, typer.Option(help="Fully automatic research: the agent admits the source.")] = False,
    card: common.Card = "",
) -> None:
    """Admit the idea's data source, once for this plugin, options and code."""
    who = common.who(relayed, auto, card)
    with common.errors():
        target = Idea(idea)
        text = target.source_card()
        typer.echo(text)
        typer.echo(f"card {card_id(text)}")
        if who.channel == "terminal":
            typer.confirm("Admit this source?", abort=True)
        record = target.approve_source(who)
    typer.echo(f"admitted {record['stage']}: {record['payload']['approval']['note']}")
