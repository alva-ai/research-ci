"""Research code is free: the checker finds row-by-row code in a features file and in the modules of the idea and of
installed labs it imports, for a warning on the card, never a refusal; it stops at plugin modules, never reads
third-party code, and lets loops over small lists through, resolving names by scope. pandas is allowed."""

import importlib
import re
import sys
from pathlib import Path

import pytest

from research_ci.runner import plugins, research
from research_ci.runner.research import ResearchError, findings, warning
from research_ci.workflow.templates import EXAMPLE
from toy import REGISTRY

GOOD = """
import itertools

import pandas as pd
import polars as pl

HORIZONS = ("8h", "24h")
GROUPS = {"fast": ("8h",), "slow": ("72h", "168h")}
LAG = pd.Timedelta("1h")


def features(fields, candidates):
    means = [pl.col("value").rolling_mean_by("known_at", window_size=h).over("inst").alias(h) for h in HORIZONS]
    for name, horizons in GROUPS.items():
        for h in horizons:
            pass
    for i, h in enumerate(HORIZONS):
        pass
    for a, b in itertools.product(HORIZONS, HORIZONS):
        pass
    for c in candidates.collect_schema().names():
        pass
    for k in range(len(HORIZONS)):
        pass
    renamed = {c: c.upper() for c in candidates.columns}
    venues = candidates.to_pandas()["inst"].map({"a:X": 1})
    exprs = [pl.col("value").mean(), pl.col("value").std()]
    exprs = exprs + [pl.col("value").max()]
    exprs += [pl.col("value").min()]
    for e in exprs:
        pass
    windows = ("8h",) if len(HORIZONS) > 1 else ("8h", "24h")
    for w in windows:
        pass
    for group in GROUPS.values():
        for h in group:
            pass
    for w in "8h 24h".split():
        pass
    return candidates.select("t", "inst", pl.lit(1.0).alias("score"))


def helper(cols, times):
    return cols, times


def other(frame):
    cols = ["a", "b"]
    for c in cols:
        pass
    return frame
"""

BAD = {
    "iter_rows": "    for row in df.iter_rows(named=True):\n        pass\n",
    "rows": "    x = df.rows()\n",
    "row": "    x = df.row(0)\n",
    "to_dicts": "    x = df.to_dicts()\n",
    "map_elements": "    x = df['score'].map_elements(lambda v: v + 1)\n",
    "map_rows": "    x = df.map_rows(lambda r: r)\n",
    "iterrows": "    for _, r in df.to_pandas().iterrows():\n        pass\n",
    "itertuples": "    x = list(df.to_pandas().itertuples())\n",
    "apply": "    x = df.to_pandas().apply(sum, axis=1)\n",
    "applymap": "    x = df.to_pandas().applymap(abs)\n",
    "map with a lambda": "    x = df.to_pandas()['score'].map(lambda v: v)\n",
    "map with a function": "    x = df.to_pandas()['score'].map(helper)\n",
    "for over a column": "    for v in df['score']:\n        pass\n",
    "for over a parameter": "    for t in times:\n        pass\n",
    "for over group_by": "    for key, group in df.group_by('inst'):\n        pass\n",
    "for over range of the height": "    for i in range(df.height):\n        pass\n",
    "comprehension over a series": "    x = [v * 2 for v in df['score'].to_list()]\n",
    "while": "    i = 0\n    while i < df.height:\n        i += 1\n",
}


def _file(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _bad(body: str) -> str:
    return (
        "import polars as pl\n\n\ndef helper(v):\n    return v\n\n\n"
        f"def features(fields, candidates, times=()):\n    df = candidates\n{body}    return df\n"
    )


def test_whole_columns_loops_over_literal_lists_and_pandas_pass(tmp_path: Path) -> None:
    assert findings([_file(tmp_path, "good.py", GOOD)], tmp_path) == []


@pytest.mark.parametrize("name", list(BAD))
def test_row_by_row_code_is_a_warning_with_where_it_is(tmp_path: Path, name: str) -> None:
    path = _file(tmp_path, "bad.py", _bad(BAD[name]))
    found = findings([path], tmp_path)
    assert len(found) == 1, found
    said = warning([path], tmp_path)
    assert said == f"row-wise code: bad.py:{found[0].line} {found[0].what} — polars expressions are usually much faster"


def test_the_warning_names_each_file_once_and_counts_what_it_leaves_out(tmp_path: Path) -> None:
    body = "".join(BAD[n] for n in ("iterrows", "for over a column", "while"))
    first = _file(tmp_path, "research/rows.py", _bad(body) + "\nfrom helpers import features as again\n")
    _file(tmp_path, "research/helpers.py", _bad(BAD["to_dicts"]))
    assert warning([first], tmp_path) == (
        "row-wise code: research/rows.py:10 iterrows · :12 for over df['score'] · :15 while loop · "
        "research/helpers.py:10 to_dicts — polars expressions are usually much faster"
    )
    many = _file(tmp_path, "many.py", _bad(BAD["rows"] * 10))
    assert warning([many], tmp_path).endswith(":17 rows · 2 more — polars expressions are usually much faster")


def test_the_example_rci_init_writes_has_no_row_wise_code_even_in_comments(tmp_path: Path) -> None:
    path = _file(tmp_path, "research/example.py", EXAMPLE)
    assert findings([path], tmp_path) == [] and research.still_example(path)
    words = r"iter_rows|iterrows|itertuples|map_elements|map_rows|to_dicts|\.apply\(|for \w+(, \w+)* in "
    assert not re.search(words, EXAMPLE)
    path.write_text("\n".join(line for line in EXAMPLE.splitlines() if research.EXAMPLE_MARKER not in line))
    assert not research.still_example(path)


def test_a_syntax_error_says_where(tmp_path: Path) -> None:
    with pytest.raises(ResearchError, match=r"broken.py:2: .*fix the file"):
        findings([_file(tmp_path, "broken.py", "import polars\ndef features(:\n")], tmp_path)


def test_modules_of_the_idea_are_read_too(tmp_path: Path) -> None:
    _file(tmp_path, "research/helpers.py", _bad(BAD["iter_rows"]))
    _file(tmp_path, "shared/__init__.py", "")
    _file(tmp_path, "shared/loops.py", _bad(BAD["while"]))
    features = _file(tmp_path, "research/main.py", "from helpers import features\nfrom shared import loops\n")
    found = {f.path.name: f.what for f in findings([features], tmp_path)}
    assert set(found) == {"helpers.py", "loops.py"}
    assert "iter_rows" in found["helpers.py"] and "while" in found["loops.py"]


PLUGIN = """
class Universe:
    def candidates(self, fields, times):
        rows = []
        for t in times:
            rows.append(t)
        return rows
"""


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An installed lab package: one plugin module, which may loop, and one module of factors research code imports."""
    root = tmp_path / "site"
    _file(root, "fakelab/__init__.py", "")
    _file(root, "fakelab/plugin.py", PLUGIN)
    _file(root, "fakelab/factors.py", "from fakelab.plugin import Universe\n\n" + _bad(BAD["to_dicts"]))
    monkeypatch.syspath_prepend(str(root))
    for name in ("fakelab", "fakelab.plugin"):
        sys.modules.pop(name, None)
    universe = importlib.import_module("fakelab.plugin").Universe
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "fake.universe": universe})
    return root


@pytest.mark.usefixtures("lab")
def test_lab_modules_research_code_imports_are_read_and_plugin_modules_are_not(tmp_path: Path) -> None:
    idea = tmp_path / "idea"
    features = _file(idea, "research/f.py", "import numpy\nfrom fakelab.factors import features\n")
    found = findings([features], idea)
    assert [(f.path.name, "to_dicts" in f.what) for f in found] == [("factors.py", True)]
    only_plugin = _file(
        idea, "research/g.py", "from fakelab.plugin import Universe\n\ndef features(f, c):\n    return c\n"
    )
    assert findings([only_plugin], idea) == []
