"""The catalog in docs/catalog.md is read from the code: it must match what `rci methods --markdown` prints, each
stage's table must be its floor list with its default list where the floor marks it, and what it says about each eval
function must hold."""

import inspect
from pathlib import Path

from typer.testing import CliRunner

from research_ci.catalog import collect
from research_ci.catalog.markdown import FILE, REGENERATE
from research_ci.cli.main import app
from research_ci.stages import skeleton
from research_ci.stages.contract import defaults as contract_defaults
from research_ci.stages.contract import floor as contract_floor
from research_ci.stages.data import defaults as data_defaults
from research_ci.stages.data import floor as data_floor
from research_ci.stages.data import source as source_floor
from research_ci.stages.feat import defaults as feat_defaults
from research_ci.stages.feat import floor as feat_floor

ROOT = Path(__file__).parents[2]
LISTS = {
    "contract": (contract_floor, contract_defaults),
    "data": (data_floor, data_defaults),
    "feat": (feat_floor, feat_defaults),
}
runner = CliRunner()


def test_the_catalog_in_docs_is_what_the_code_prints() -> None:
    printed = runner.invoke(app, ["methods", "--markdown"]).stdout
    assert printed == (ROOT / FILE).read_text(), f"{FILE} is out of date; regenerate it: {REGENERATE}"


def test_each_stage_table_is_its_floor_list_with_its_default_list_where_the_floor_marks_it() -> None:
    stages = {s["stage"]: s for s in collect.collect()["stages"]}
    for name, (floor, defaults) in LISTS.items():
        rows = stages[name]["sections"][-1][1]
        listed = skeleton.ordered(floor.STEPS, defaults.CHECKS)
        expected = [(s.name, s.role, "default" if s in defaults.CHECKS else "floor", s.what) for s in listed]
        assert [(r["item"], r["role"], r["list"], r["what"]) for r in rows] == expected, name
    admission = dict(stages["data"]["sections"])["admission, once per source"]
    assert [r["item"] for r in admission] == [s.name for s in source_floor.STEPS]
    every = [r for s in stages.values() for _, rows in s["sections"] for r in rows]
    assert [r["item"] for r in every if r["role"] == "decides"] == ["decisive test"]


def test_the_eval_functions_and_declarations_of_each_step_are_read_from_its_code() -> None:
    stages = {
        s["stage"]: {r["item"]: r for _, rows in s["sections"] for r in rows} for s in collect.collect()["stages"]
    }
    data = stages["data"]
    assert {k: (r["role"], r["needs"]) for k, r in data.items() if r["list"] == "default"} == {
        "data is current": ("warns", ""),
        "level shifts": ("treats", "levels"),
        "level sanity": ("treats", "levels"),
        "identity breaks": ("treats", "levels"),
        "coverage": ("treats", ""),
        "hedge guard": ("treats", "hedge"),
        "influence concentration": ("warns", ""),
    }
    assert "data.sanity.spikes" in data["level sanity"]["methods"]
    assert {"data.reconcile.requests", "data.reconcile.compare"} <= set(data["independent source agrees"]["methods"])
    assert data["field grades"]["methods"] == []
    assert data["decisive test calibration"]["methods"] == ["protocol.target_reads_score"]
    assert data["independent source agrees"]["needs"] == "reference"
    feat = stages["feat"]
    assert [
        feat[name]["methods"] for name in ("rank ic", "rank ic without top {clusters}", "top minus bottom fifth")
    ] == [
        ["statistics.ic.rank_ic"],
        ["statistics.ic.rank_ic", "structure.filters.without_top", "structure.filters.kept_without_top"],
        ["statistics.spread.quantile_spread"],
    ]
    assert stages["contract"]["target reads the score"]["methods"] == ["protocol.target_reads_score"]


def test_every_step_says_what_it_checks_in_a_few_words() -> None:
    for floor, defaults in [*LISTS.values(), (source_floor, None)]:
        listed = skeleton.ordered(floor.STEPS, defaults.CHECKS if defaults else [])
        for s in listed:
            assert len(s.what.split()) <= 10, f"{s.name}: {s.what!r} is long; the rest belongs in the docstring"
            assert inspect.getdoc(s.run), f"{s.name}: its function says the rest in a docstring"


def test_every_eval_function_says_what_it_does_and_where_it_is_used() -> None:
    library = collect.collect()["library"]
    assert [p["package"] for p in library][:7] == list(collect.ORDER)
    functions = {f["key"]: f for p in library for m in p["modules"] for f in m["functions"]}
    assert all(f["doc"] and f["used"] for f in functions.values())
    tests = "\n".join(p.read_text() for p in (ROOT / "tests").rglob("*.py") if p != Path(__file__))
    for key, f in functions.items():
        if f["used"] == ["tests only"]:
            assert f"{f['name']}(" in tests, f"{key} is marked tests only, but no test calls it"
    assert functions["protocol.consumer_pit"]["used"] == ["algo, planned"]
    assert functions["statistics.ic.rank_ic"] == {
        "key": "statistics.ic.rank_ic",
        "name": "rank_ic",
        "doc": "Spearman correlation of score and label at each decision time.",
        "used": ["feat", "research code"],
        "examples": 3,
        "properties": 3,
    }


def test_every_eval_function_says_what_it_does_in_a_few_words() -> None:
    functions = [f for p in collect.collect()["library"] for m in p["modules"] for f in m["functions"]]
    for f in functions:
        assert len(f["doc"].split()) <= 12, f"{f['key']}: {f['doc']!r} is long; the rest belongs in the docstring"


def test_written_descriptions_stand_in_only_for_missing_docstrings() -> None:
    public = {
        collect._key(fn): fn
        for p in collect.packages()
        for m in collect.listing._modules(f"{collect.EVAL}.{p}")
        for fn in collect.public(m)
    }
    for key in collect.UNDOCUMENTED:
        assert key in public, f"{key} is no eval function; take it out of UNDOCUMENTED"
        assert not inspect.getdoc(public[key]), f"{key} has a docstring now; take it out of UNDOCUMENTED"
