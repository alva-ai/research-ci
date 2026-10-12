"""`rci eval check <file>` runs the tests of every eval in a module any time, for free: it says how each stands and
why, and records nothing."""

from pathlib import Path

from typer.testing import CliRunner

from research_ci.cli.main import app

runner = CliRunner()

GOOD = """
import polars as pl

from research_ci.api import Case
from research_ci.eval.statistics.spread import quantile_spread
from research_ci.eval.synthetic import hand

SIX = hand({"score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "y": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]})


def thirds(table):
    return quantile_spread(table, share=1 / 3)


CASES = [
    Case("the top two of six less the bottom two", thirds, 0.4, given=SIX),
    Case("reversed scores turn it around", thirds, -0.4, given=SIX.with_columns(-pl.col("score"))),
]
"""
BAD = (
    GOOD
    + """

def halves(table):
    return quantile_spread(table, share=0.4)


def untested(table):
    return table


CASES += [Case("the top half less the bottom half", halves, 0.3, given=SIX)]
"""
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "idea" / "evals" / "spread.py"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def test_valid_evals_say_so_with_each_test(tmp_path: Path) -> None:
    path = _write(tmp_path, GOOD)
    result = runner.invoke(app, ["eval", "check", str(path)])
    assert result.exit_code == 0, result.output
    assert (
        "  thirds  valid 2/2\n    ok   case 'the top two of six less the bottom two': as worked by hand"
        in result.output
    )
    assert "pits that a data eval claims in covers are planted by `rci data check`" in result.output
    assert not (tmp_path / "idea" / ".rci").exists(), "nothing is recorded"


def test_an_invalid_eval_says_which_test_failed_what_it_expected_and_got_and_how_to_fix_it(tmp_path: Path) -> None:
    path = _write(tmp_path, BAD)
    (tmp_path / "idea" / "contract.py").write_text("")
    result = runner.invoke(app, ["eval", "check", str(path)])
    assert result.exit_code == 1
    for line in (
        "  halves    invalid 0/1",
        "    FAIL case 'the top half less the bottom half': expected 0.3, got 0.4",
        "    fix the eval or its CASES in evals/spread.py",
        "  thirds    valid 2/2",
        "  untested  no tests: refused if EVALS uses it; write a Case that runs it, or name it _untested",
    ):
        assert line in result.output, result.output


def test_a_module_without_cases_is_an_error_that_says_how_tests_are_written(tmp_path: Path) -> None:
    path = _write(tmp_path, "def lonely(table):\n    return table\n")
    result = runner.invoke(app, ["eval", "check", str(path), "-i", str(tmp_path / "idea")])
    assert result.exit_code == 1
    assert "evals/spread.py defines no CASES; a case eval's module holds its tests next to it" in result.output
    broken = _write(tmp_path / "other", "import nowhere\n")
    result = runner.invoke(app, ["eval", "check", str(broken)])
    assert result.exit_code == 1 and "failed to load: ModuleNotFoundError: No module named 'nowhere'" in result.output
