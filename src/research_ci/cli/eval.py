from pathlib import Path
from typing import Annotated

import typer

from research_ci.cli import common
from research_ci.render.common import MARK
from research_ci.store.workspace import STATE_DIR
from research_ci.validation.load import CASE_FORMAT, checked_file

eval_app = typer.Typer(no_args_is_help=True, help="Case evals: the idea's own eval functions, held to their tests.")
PITS = "pits that a data eval claims in covers are planted by `rci data check`, on the idea's data"


@eval_app.command("check")
def eval_check(
    path: Annotated[Path, typer.Argument(help="A module of case evals, such as evals/<name>.py, with its CASES.")],
    idea: Annotated[Path | None, typer.Option("--idea", "-i", help="Idea directory; found from the file.")] = None,
) -> None:
    """Run the tests of every eval in a module: its CASES and PROPERTIES that run it. Free; nothing is recorded."""
    root = idea or _idea_of(path)
    with common.errors():
        found, problem = checked_file(path, root)
    typer.echo(f"{path}")
    width = max((len(name) for name, _ in found), default=0)
    for name, v in found:
        if not v.tests:
            typer.echo(
                f"  {name:<{width}}  no tests: refused if EVALS uses it; write a Case that runs it, or name it _{name}"
            )
            continue
        typer.echo(f"  {name:<{width}}  {v.status} {v.passed}/{len(v.tests)}")
        for t in v.tests:
            typer.echo(f"    {MARK[t.status]:<4} {t.name}: {t.detail}")
        if v.status == "invalid":
            typer.echo(f"    {v.fix().removesuffix(f'; `rci eval check {v.file}` runs them for free')}")
    if problem:
        typer.echo(f"error: {problem}", err=True)
    elif not found:
        typer.echo(f"  no public functions; {CASE_FORMAT}")
    typer.echo(PITS)
    if problem or any(v.status == "invalid" for _, v in found):
        raise typer.Exit(1)


def _idea_of(path: Path) -> Path:
    """The idea a file belongs to: the nearest directory above it with engine state or a contract."""
    here = Path(path).resolve().parent
    for d in [here, *here.parents]:
        if (d / STATE_DIR).is_dir() or (d / "contract.py").is_file():
            return d
    return here
