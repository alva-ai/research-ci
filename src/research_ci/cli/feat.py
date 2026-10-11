from pathlib import Path
from typing import Annotated

import typer

from research_ci.cli import common
from research_ci.render.common import card_id
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import State, WorkflowError

feat_app = typer.Typer(no_args_is_help=True, help="Judge features against random scores.")
FEATURES_HELP = (
    "A features file, such as research/<name>.py: features(fields, candidates) gets polars DataFrames and returns t, "
    "entity, score; research/example.py shows one. Row-by-row code gets a warning on the card, which also shows the "
    "time and peak memory the features took."
)


@feat_app.command("submit")
def feat_submit(
    path: Annotated[Path, typer.Argument(help=FEATURES_HELP)],
    idea: common.IdeaDir = Path("."),
) -> None:
    """Judge a file of features against the floor world; this uses one submission of the budget."""
    with common.errors():
        _, card, state = Idea(idea).submit_feat(path)
    typer.echo(card)
    if state is State.REVIEW:
        typer.echo(f"card {card_id(card)}")
        typer.echo("up for review; a person approves it with `rci approve --stage feat`")
    elif state is State.PAUSED:
        typer.echo("feat PAUSED: every submission is used; a person extends the budget or drops the idea")
    else:
        typer.echo("feat OPEN: not up for review; `rci next` says what to do and how many submissions are left")


@feat_app.command("explore")
def feat_explore(
    path: Annotated[Path, typer.Argument(help=FEATURES_HELP)],
    idea: common.IdeaDir = Path("."),
) -> None:
    """Run the checks a submission must pass on a file of features; nothing is judged or recorded."""
    with common.errors():
        _, card = Idea(idea).explore_feat(path)
    typer.echo(card)


@feat_app.command("show")
def feat_show(idea: common.IdeaDir = Path(".")) -> None:
    """Print the calibration in use and the ResultCard of every submission, with the card id of the one up for
    review."""
    with common.errors():
        target = Idea(idea)
        results = target.results()
        if not results:
            raise WorkflowError("nothing submitted yet; run `rci feat submit <file>` first")
        review = target.status().stage("feat").state is State.REVIEW
    typer.echo("\n\n".join(results))
    if review:
        typer.echo(f"card {card_id(results[-1])}")
        typer.echo("up for review; a person approves it with `rci approve --stage feat`")
