from pathlib import Path
from typing import Annotated

import typer

from research_ci.cli import common
from research_ci.render.common import card_id
from research_ci.workflow.feat import FEAT_TEXT
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import WorkflowError

feat_app = typer.Typer(no_args_is_help=True, help="Judge features with criteria and null worlds fixed in advance.")


@feat_app.command("register")
def feat_register(idea: common.IdeaDir = Path(".")) -> None:
    """Load feat.py and nulls/feat.py, calibrate every criterion in every world, and print the method card."""
    with common.errors():
        spec, card = Idea(idea).register_feat()
    typer.echo(card)
    typer.echo(f"card {card_id(card)}")
    if spec["verdict"]["status"] == "blocked":
        typer.echo("blocked; see the verdict above")
    else:
        typer.echo("checked; a person freezes it with `rci approve --stage feat`")


@feat_app.command("show")
def feat_show(idea: common.IdeaDir = Path(".")) -> None:
    """Print the method card and the result card of every submission."""
    with common.errors():
        target = Idea(idea)
        path = target.workspace.path(FEAT_TEXT)
        if not path.exists():
            raise WorkflowError("the feat method has not been registered; run `rci feat register` first")
        results = target.results()
    text = path.read_text().rstrip()
    typer.echo(text)
    typer.echo(f"card {card_id(text)}")
    for result in results:
        typer.echo(result)


@feat_app.command("explore")
def feat_explore(
    path: Annotated[Path, typer.Argument(help="A file defining features(fields, candidates).")],
    idea: common.IdeaDir = Path("."),
) -> None:
    """Run the checks a submission must pass on a file of features; nothing is judged or recorded."""
    with common.errors():
        _, card = Idea(idea).explore_feat(path)
    typer.echo(card)


@feat_app.command("submit")
def feat_submit(
    path: Annotated[Path, typer.Argument(help="A file defining features(fields, candidates).")],
    idea: common.IdeaDir = Path("."),
) -> None:
    """Judge a file of features with the approved method; this uses one submission of the budget."""
    with common.errors():
        _, card, state = Idea(idea).submit_feat(path)
    typer.echo(card)
    typer.echo(f"feat {state}")
