"""Engine code works on whole columns. In ``eval`` and ``stages/defaults``, the methods and default checks that run on
every idea's data, a loop over data-sized things (rows, entities, clusters, decision times) is not allowed, and neither
are the row APIs in ``ROW_APIS``. A structural loop, over horizons, folds, cut points, chunks, iterations of an
algorithm, repetitions, or calls into plugin, research, or criterion code, and a row API on a handful of rows, such as
a worked example's fixture, are allowed only through an entry in ``ALLOWED``: the file, the function, what it is
(``for``, ``while``, or the API), and one line on why, which a reviewer reads. A recursion is written as a compiled
function, not in Python. Comprehensions build small Python lists and are not counted; one over a frame's rows shows up
through the row API it calls."""

import ast
from pathlib import Path

import research_ci

ROOT = Path(research_ci.__file__).parent
SCANNED = ("eval", "stages/defaults")
ROW_APIS = frozenset({"map_groups", "map_elements", "map_rows", "apply", "iter_rows", "iterrows", "itertuples"})
Key = tuple[str, str, str]
ALLOWED: dict[Key, str] = {
    ("eval/calibration/stand_in.py", "_size", "for"): "doubling, then bisection: each step runs the criterion at a "
    "weight the step before chose",
    ("eval/data/replay.py", "replay", "for"): "the data plugin fetches as of each cut point",
    ("eval/data/replay.py", "cut_points", "iter_rows"): "the few cut points become records, one plugin fetch each",
    ("eval/protocol.py", "universe_pit", "for"): "the universe plugin runs on the data cut at each cut point",
    ("eval/protocol.py", "label_cases", "for"): "the label plugin runs on each worked example and on each of its "
    "mutations",
    ("eval/protocol.py", "label_truncation", "for"): "the label plugin runs on the data cut at each cut point",
    ("eval/protocol.py", "consumer_pit", "for"): "the consumer runs on the data cut at each cut point",
    ("eval/protocol.py", "matches_labels", "iter_rows"): "one note per component the consumer books, a handful",
    ("eval/data/baseline.py", "_count", "iter_rows"): "names the three fields with the most flagged rows",
    ("eval/data/baseline.py", "_kinds", "iter_rows"): "worked example: one count per anomaly kind",
    ("eval/data/invariants.py", "_bad_share", "iter_rows"): "worked example: one share per pair of a two-pair market",
    ("eval/data/reconcile.py", "summarize", "iter_rows"): "one grade per field and one count per field or anomaly kind",
    ("eval/data/reconcile.py", "_statuses", "iter_rows"): "worked example: the statuses of each field",
    ("eval/data/treatments.py", "apply_all", "apply"): "Treatment.apply, not pandas: one whole-frame rewrite per "
    "treatment",
    ("eval/data/treatments.py", "apply_candidates", "apply"): "Treatment.apply, not pandas: one whole-frame rewrite "
    "per treatment",
    ("eval/data/treatments.py", "_removed", "apply"): "worked example: Treatment.apply, a whole-frame rewrite",
    ("eval/data/treatments.py", "_split", "apply"): "worked example: Treatment.apply, a whole-frame rewrite",
    ("eval/data/treatments.py", "_split", "iter_rows"): "worked example: rows per entity of a fixed five-entity source",
}
FIX = (
    "fix: work on whole columns: group_by().agg() per group, .over() per entity or cluster, a join or join_asof per "
    "decision time, rolling_*_by for windows; a loop or a row API over rows, entities, clusters, or times does not "
    "belong in eval or stages/defaults"
)
JUSTIFY = (
    "or justify: a structural loop (over horizons, folds, cut points, chunks, iterations of an algorithm, "
    "repetitions, or calls into plugin, research, or criterion code) or a row API on a handful of rows: add its key "
    'to ALLOWED in tests/eval/test_columnar.py with "<one line on why>", which a reviewer reads'
)


def _scan(text: str, rel: str) -> dict[Key, list[int]]:
    """Every loop statement and row API call in one file: (file, function, what) with its lines."""
    found: dict[Key, list[int]] = {}

    def visit(node: ast.AST, where: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = where
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                inner = f"{where}.{child.name}" if where else child.name
            what = ""
            if isinstance(child, ast.For | ast.AsyncFor):
                what = "for"
            elif isinstance(child, ast.While):
                what = "while"
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr in ROW_APIS:
                what = child.func.attr
            if what:
                found.setdefault((rel, where or "<module>", what), []).append(getattr(child, "lineno", 0))
            visit(child, inner)

    visit(ast.parse(text), "")
    return found


def _engine() -> dict[Key, list[int]]:
    found: dict[Key, list[int]] = {}
    for path in sorted(p for root in SCANNED for p in (ROOT / root).rglob("*.py")):
        found |= _scan(path.read_text(), path.relative_to(ROOT).as_posix())
    return found


def test_engine_code_loops_only_where_a_reviewed_reason_allows_it() -> None:
    found = _engine()
    new = [f"  {key[0]}:{found[key][0]}  key {key!r}" for key in sorted(found.keys() - ALLOWED.keys())]
    assert not new, "\n".join(["not allowed in engine code:", *new, FIX, JUSTIFY])
    gone = [f"  {key}" for key in sorted(ALLOWED.keys() - found.keys())]
    assert not gone, "\n".join(["allowed in ALLOWED but no longer in the code; drop them:", *gone])


def test_every_allowed_loop_says_why_in_one_line() -> None:
    for key, why in ALLOWED.items():
        assert why.strip() and "\n" not in why and len(why) <= 120, key


def test_the_rule_sees_loops_and_row_apis_and_where_they_are() -> None:
    text = (
        "def f(df):\n    for r in df.iter_rows():\n        pass\n    while True:\n        break\n\n\n"
        "class A:\n    def g(self, df):\n        return df.group_by('e').map_groups(len)\n\n\n"
        "X = [h for h in ('8h', '24h')]\n"
    )
    assert _scan(text, "eval/x.py") == {
        ("eval/x.py", "f", "for"): [2],
        ("eval/x.py", "f", "iter_rows"): [2],
        ("eval/x.py", "f", "while"): [4],
        ("eval/x.py", "A.g", "map_groups"): [10],
    }
