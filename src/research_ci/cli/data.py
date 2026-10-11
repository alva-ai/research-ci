from pathlib import Path
from typing import Annotated

import typer

from research_ci.cli import common
from research_ci.render.common import card_id
from research_ci.workflow.data import DATA_TEXT
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import WorkflowError

data_app = typer.Typer(no_args_is_help=True, help="Check the data against the contract.")


@data_app.command("check")
def data_check(
    idea: common.IdeaDir = Path("."),
    keep: Annotated[
        list[str] | None, typer.Option("--keep", help="A treatment a person chose not to apply, by its name.")
    ] = None,
) -> None:
    """Run the source's admission checks if needed, check the data against the contract, and print the DataCard."""
    with common.errors():
        spec, card = Idea(idea).check_data(tuple(keep or ()))
    typer.echo(card)
    typer.echo(f"card {card_id(card)}")
    if spec["verdict"]["status"] == "blocked":
        typer.echo("blocked; see the verdict above")
    else:
        typer.echo("up for review; a person approves it with `rci approve --stage data`")


@data_app.command("show")
def data_show(idea: common.IdeaDir = Path(".")) -> None:
    """Print the last DataCard."""
    with common.errors():
        target = Idea(idea)
        path = target.workspace.path(DATA_TEXT)
        if not path.exists():
            raise WorkflowError("the data has not been checked; run `rci data check` first")
    text = path.read_text().rstrip()
    typer.echo(text)
    typer.echo(f"card {card_id(text)}")
