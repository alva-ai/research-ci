from pathlib import Path

import typer

from research_ci.cli import common
from research_ci.render.common import card_id
from research_ci.workflow.idea import Idea

contract_app = typer.Typer(no_args_is_help=True, help="Work on the contract.")


@contract_app.command("derive")
def derive(idea: common.IdeaDir = Path(".")) -> None:
    """Load contract.py, validate it, and register it for approval."""
    with common.errors():
        _, card = Idea(idea).derive_contract()
    typer.echo(card)
    typer.echo(f"card {card_id(card)}")
    typer.echo("registered; a person freezes it with `rci approve --stage contract`")
