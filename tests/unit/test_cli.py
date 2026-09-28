from typer.testing import CliRunner

from alpha_ci import __version__
from alpha_ci.cli.main import app


def test_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
