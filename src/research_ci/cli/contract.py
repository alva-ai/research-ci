from pathlib import Path

import typer

from research_ci.cli import common
from research_ci.render.common import card_id
from research_ci.workflow.contract import CARD_TEXT
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import State, WorkflowError

contract_app = typer.Typer(no_args_is_help=True, help="Work on the contract.")


@contract_app.command("derive")
def derive(idea: common.IdeaDir = Path(".")) -> None:
    """Load contract.py, derive what the data can decide, and print the ContractCard."""
    with common.errors():
        target = Idea(idea)
        _, card = target.derive_contract()
        state = target.status().stage("contract").state
    typer.echo(card)
    typer.echo(f"card {card_id(card)}")
    if state is State.REVIEW:
        typer.echo("up for review; a person approves it with `rci approve --stage contract`")
    else:
        typer.echo("blocked; see the verdict above")


@contract_app.command("show")
def show(idea: common.IdeaDir = Path(".")) -> None:
    """Print the last ContractCard with its card id."""
    with common.errors():
        path = Idea(idea).workspace.path(CARD_TEXT)
        if not path.exists():
            raise WorkflowError("the contract has not been derived; run `rci contract derive` first")
    text = path.read_text().rstrip()
    typer.echo(text)
    typer.echo(f"card {card_id(text)}")
