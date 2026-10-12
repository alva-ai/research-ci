"""Lab evals: the engine finds an eval's own tests in its module, runs only those, keeps the result until the module
or what it imports changes, and refuses an eval without tests or still the example. A library eval is trusted."""

import sys
from pathlib import Path

import pytest

from research_ci.api import Eval
from research_ci.eval.statistics.ic import rank_ic
from research_ci.validation import load as ev

MODULE = """
from functools import partial

import polars as pl

from research_ci.api import Case
from helpers import double


def scaled(frame, k=1.0):
    return frame.select(pl.col("x") * k * double(1) / 2)


def other(frame):
    return frame.height


def _through(frame):
    return scaled(frame, 2.0)


def _closure():
    fn = scaled
    return lambda: fn(pl.DataFrame({"x": [1.0]}))["x"][0]


ONE = pl.DataFrame({"x": [1.0, 2.0]})
CASES = [
    Case("the frame itself", scaled, pl.DataFrame({"x": [1.0, 2.0]}), given=ONE),
    Case("through a helper", lambda: _through(ONE)["x"].to_list(), [2.0, 4.0]),
    Case("through a partial", partial(scaled, k=3.0), pl.DataFrame({"x": [3.0, 6.0]}), given=ONE),
    Case("held in a closure", _closure(), 1.0),
    Case("another eval", other, 2, given=ONE),
]
"""
HELPERS = "def double(x):\n    return 2 * x\n"


def _idea(tmp_path: Path, module: str = MODULE, helpers: str = HELPERS) -> Path:
    root = tmp_path / "idea"
    (root / "evals").mkdir(parents=True, exist_ok=True)
    (root / "evals" / "scale.py").write_text(module)
    (root / "helpers.py").write_text(helpers)
    return root


def _load(root: Path, name: str = "scaled"):
    return getattr(ev.load_module(root / "evals" / "scale.py", root), name)


def test_an_evals_tests_are_the_cases_of_its_module_that_run_it(tmp_path: Path) -> None:
    root = _idea(tmp_path)
    cases, properties = ev.tests_of(_load(root))
    assert [c.name for c in cases] == ["the frame itself", "through a helper", "through a partial", "held in a closure"]
    assert properties == []
    assert [c.name for c in ev.tests_of(_load(root, "other"))[0]] == ["another eval"]
    assert ev.source_of(rank_ic) == "library" and ev.source_of(_load(root)) == "lab"
    assert "helpers" not in sys.modules and "scale" not in sys.modules, "the idea's modules are forgotten"


def test_a_case_eval_without_tests_is_refused_saying_where_they_go(tmp_path: Path) -> None:
    root = _idea(tmp_path, MODULE.replace('    Case("another eval", other, 2, given=ONE),\n', ""))
    with pytest.raises(ev.EvalsError) as refused:
        ev.refuse_untested([Eval(_load(root, "other"))], root, "feat.py", "; no submission was spent")
    said = str(refused.value)
    assert "feat.py: the lab eval 'other' has no tests, so it cannot run: write CASES in evals/scale.py" in said
    assert "`rci eval check evals/scale.py`" in said and said.endswith("no submission was spent")
    ev.refuse_untested([Eval(_load(root)), Eval(rank_ic)], root, "feat.py")


def test_a_validation_is_kept_until_the_module_or_what_it_imports_changes(tmp_path: Path, monkeypatch) -> None:
    root = _idea(tmp_path)
    ran: list[str] = []
    real = ev.run_tests

    def counted(name, cases, properties):
        ran.append(name)
        return real(name, cases, properties)

    monkeypatch.setattr(ev, "run_tests", counted)
    first = ev.validate([Eval(_load(root)), Eval(rank_ic)], root)
    assert first["rank_ic"].status == "trusted" and first["rank_ic"].summary() == "engine library"
    assert first["scaled"].status == "valid" and first["scaled"].summary() == "valid 4/4" and not first["scaled"].cached
    again = ev.validate([Eval(_load(root))], root)
    assert again["scaled"].cached and again["scaled"].summary() == "valid 4/4" and ran == ["scaled"]
    (root / "helpers.py").write_text("def double(x):\n    return 3 * x\n")
    changed = ev.validate([Eval(_load(root))], root)
    assert ran == ["scaled", "scaled"] and changed["scaled"].status == "invalid"
    assert (
        changed["scaled"]
        .summary()
        .startswith(
            "invalid 0/4: case 'the frame itself': 2 of 2 rows differ; at row 0: x: expected 1.0, got 1.5 (and 3 more)"
        )
    )
    assert changed["scaled"].fix() == (
        "fix the eval or its CASES in evals/scale.py; `rci eval check evals/scale.py` runs them for free"
    )
    ev.validate([Eval(_load(root))], root, cache=False)
    assert len(ran) == 3


def test_a_case_eval_module_with_malformed_tests_says_how_they_are_written(tmp_path: Path) -> None:
    root = _idea(tmp_path, MODULE + "\nPROPERTIES = [1]\n")
    with pytest.raises(ev.EvalsError, match=r"PROPERTIES in scale.py must be a list of Size, Power, or Recovers"):
        ev.tests_of(_load(root))


def test_a_lab_eval_from_the_example_module_is_refused_until_it_is_rewritten(tmp_path: Path) -> None:
    """The lab eval skeleton ``rci init`` writes passes its own tests, so only its marker keeps it from being used as
    it is; without the marker, it runs."""
    from research_ci.workflow.templates import EVALS_EXAMPLE

    root = _idea(tmp_path, EVALS_EXAMPLE)
    with pytest.raises(ev.EvalsError, match="comes from evals/scale.py, which is still the example; write your own"):
        ev.refuse_untested([Eval(_load(root, "ranks_label"))], root, "feat.py")
    with pytest.raises(ev.EvalsError, match="evals/scale.py is still the example"):
        ev.checked_file(root / "evals" / "scale.py", root)
    copied = "\n".join(EVALS_EXAMPLE.splitlines()[2:])
    root = _idea(tmp_path / "copy", copied)
    ev.refuse_untested([Eval(_load(root, "ranks_label")), Eval(_load(root, "flags"))], root, "feat.py")
    found, problem = ev.checked_file(root / "evals" / "scale.py", root)
    assert not problem and [(name, v.status) for name, v in found] == [("flags", "valid"), ("ranks_label", "valid")]


def test_a_data_eval_module_lists_its_own_pits(tmp_path: Path) -> None:
    pits = "\nfrom research_ci.api import Pit\nPITS = [Pit('drift', lambda fields, field, rng: fields, 'nothing')]\n"
    root = _idea(tmp_path, MODULE + pits)
    assert [p.category for p in ev.pits_of(_load(root))] == ["drift"]
    root = _idea(tmp_path / "bad", MODULE + "\nPITS = [1]\n")
    with pytest.raises(ev.EvalsError, match=r"PITS in scale.py must be a list of Pit"):
        ev.pits_of(_load(root))
