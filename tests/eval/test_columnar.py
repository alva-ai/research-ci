"""Methods and default checks work on whole columns: no ``for`` or ``while`` statement in ``eval`` or
``stages/defaults`` except the few that call plugin or research code on different inputs, listed here with why."""

import ast
from pathlib import Path

import research_ci

ROOT = Path(research_ci.__file__).parent
SCANNED = ("eval", "stages/defaults")
ALLOWED = {
    ("eval/calibration/stand_in.py", "_size"): "doubling, then bisection: each step runs the criterion at a weight "
    "the step before chose",
    ("eval/data/replay.py", "replay"): "the data plugin fetches as of each cut point",
    ("eval/protocol.py", "universe_pit"): "the universe plugin runs on the data cut at each cut point",
    ("eval/protocol.py", "label_cases"): "the label plugin runs on each worked example and on each of its mutations",
    ("eval/protocol.py", "label_truncation"): "the label plugin runs on the data cut at each cut point",
    ("eval/protocol.py", "consumer_pit"): "the consumer runs on the data cut at each cut point",
}


def _loops() -> set[tuple[str, str]]:
    found = set()
    for path in sorted(p for root in SCANNED for p in (ROOT / root).rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()

        def visit(node: ast.AST, where: str, rel: str = rel) -> None:
            for child in ast.iter_child_nodes(node):
                inner = where
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    inner = f"{where}.{child.name}" if where else child.name
                if isinstance(child, ast.For | ast.AsyncFor | ast.While):
                    found.add((rel, inner or "<module>"))
                visit(child, inner)

        visit(ast.parse(path.read_text()), "")
    return found


def test_methods_and_default_checks_loop_only_over_calls() -> None:
    loops = _loops()
    assert not loops - ALLOWED.keys(), "use column operations, or allow the loop here with why"
    assert not ALLOWED.keys() - loops, "an allowed loop is gone; drop it from ALLOWED"
