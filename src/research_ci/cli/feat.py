from pathlib import Path
from typing import Annotated

import typer

from research_ci.cli import common
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import WorkflowError

feat_app = typer.Typer(no_args_is_help=True, help="Judge features against random scores.")


@feat_app.command("submit")
def feat_submit(
    path: Annotated[Path, typer.Argument(help="A file defining features(fields, candidates).")],
    idea: common.IdeaDir = Path("."),
) -> None:
    """Judge a file of features against the floor world; this uses one submission of the budget."""
    with common.errors():
        _, card, state = Idea(idea).submit_feat(path)
    typer.echo(card)
    typer.echo(f"feat {state}")


@feat_app.command("explore")
def feat_explore(
    path: Annotated[Path, typer.Argument(help="A file defining features(fields, candidates).")],
    idea: common.IdeaDir = Path("."),
) -> None:
    """Run the checks a submission must pass on a file of features; nothing is judged or recorded."""
    with common.errors():
        _, card = Idea(idea).explore_feat(path)
    typer.echo(card)


@feat_app.command("show")
def feat_show(idea: common.IdeaDir = Path(".")) -> None:
    """Print the calibration in use and the ResultCard of every submission."""
    with common.errors():
        results = Idea(idea).results()
        if not results:
            raise WorkflowError("nothing submitted yet; run `rci feat submit <file>` first")
    typer.echo("\n\n".join(results))
