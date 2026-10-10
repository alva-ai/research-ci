"""What every command shares: the options and how errors are shown."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from research_ci.api.contract import ContractError
from research_ci.store.ledger import LedgerError
from research_ci.workflow.machine import WorkflowError

ERRORS = (WorkflowError, ContractError, LedgerError)
IdeaDir = Annotated[Path, typer.Option("--idea", "-i", help="Idea directory.")]
AsJson = Annotated[bool, typer.Option("--json", help="Machine-readable output.")]
Card = Annotated[
    str, typer.Option("--card", help="With --relayed: the id of the card the person was shown, from `card <id>`.")
]


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


@contextmanager
def errors() -> Iterator[None]:
    try:
        yield
    except ERRORS as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
