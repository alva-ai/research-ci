import shutil
from pathlib import Path

import pytest

import research_ci
from research_ci.store.environment import judging_deps
from research_ci.store.freeze import changed, make_freeze, read_freeze, write_freeze
from research_ci.store.ledger import Ledger, LedgerError
from research_ci.store.workspace import Workspace, hash_file, hash_tree


def _ledger(tmp_path: Path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("approve", "feat", {"freeze": "abc"}, at_ns=1)
    ledger.append("submit", "feat", {"verdict": "FAIL", "z": 1.25}, at_ns=2)
    ledger.append("submit", "feat", {"verdict": "PASS", "z": 3.5}, at_ns=3)
    ledger.append("submit", "signal", {"verdict": "FAIL"}, at_ns=4)
    return ledger


def test_ledger_verifies_and_counts_trials(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.verify()
    assert ledger.n_trials("feat") == 2
    assert ledger.n_trials("signal") == 1
    assert ledger.n_trials("algo") == 0


def test_one_byte_edit_breaks_verification(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    text = ledger.path.read_text()
    ledger.path.write_text(text.replace('"z":1.25', '"z":1.35', 1))
    with pytest.raises(LedgerError, match="line 2"):
        ledger.verify()


def test_deleting_a_middle_line_breaks_verification(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    lines = ledger.path.read_text().splitlines(keepends=True)
    ledger.path.write_text("".join(lines[:1] + lines[2:]))
    with pytest.raises(LedgerError):
        ledger.verify()


def test_reordering_lines_breaks_verification(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    lines = ledger.path.read_text().splitlines(keepends=True)
    ledger.path.write_text("".join([lines[1], lines[0], *lines[2:]]))
    with pytest.raises(LedgerError):
        ledger.verify()


def test_head_hash_changes_with_every_append(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    head = ledger.head()
    ledger.append("submit", "feat", {"verdict": "FAIL"}, at_ns=5)
    assert ledger.head() != head


def test_freeze_hash_is_stable_and_sees_any_dependency_change() -> None:
    deps = {"contract": "c1", "evaluators": "e1", "kernel": "k1", "uv.lock": "u1"}
    first = make_freeze("feat", deps, at_ns=1)
    again = make_freeze("feat", dict(reversed(deps.items())), at_ns=99)
    assert first.hash == again.hash
    for key in deps:
        assert make_freeze("feat", {**deps, key: "other"}).hash != first.hash
    assert changed(first, {**deps, "kernel": "k2", "plugins": "p1"}) == ["kernel", "plugins"]


def test_freeze_round_trips_through_the_workspace(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "idea")
    freeze = make_freeze("feat", {"contract": "c1"}, at_ns=7)
    write_freeze(workspace, freeze)
    assert read_freeze(workspace, "feat") == freeze
    assert read_freeze(workspace, "signal") is None


def test_workspace_writes_atomically_and_reads_back(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "idea")
    workspace.write_json("state.json", {"feat": "FROZEN"})
    assert workspace.read_json("state.json") == {"feat": "FROZEN"}
    assert workspace.read_json("missing.json") is None
    assert not list(workspace.root.glob("*.tmp"))


def test_tree_hash_sees_content_and_names_but_not_listing_order(tmp_path: Path) -> None:
    root = tmp_path / "study"
    (root / "sub").mkdir(parents=True)
    (root / "a.py").write_text("x = 1\n")
    (root / "sub" / "b.py").write_text("y = 2\n")
    first = hash_tree(root)
    assert hash_tree(root) == first
    (root / "a.py").write_text("x = 2\n")
    assert hash_tree(root) != first
    (root / "a.py").write_text("x = 1\n")
    (root / "sub" / "b.py").rename(root / "sub" / "c.py")
    assert hash_tree(root) != first
    assert hash_file(root / "a.py") == hash_file(root / "a.py")


@pytest.mark.parametrize(
    "left_out",
    [
        "render/contract.py",
        "cli/main.py",
        "catalog/markdown.py",
        "stages/contract/defaults.py",
        "stages/feat/defaults.py",
        "stages/data/defaults.py",
    ],
)
def test_rewording_a_card_or_a_command_or_changing_a_default_list_keeps_the_engine_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, left_out: str
) -> None:
    package = tmp_path / "research_ci"
    shutil.copytree(Path(research_ci.__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(research_ci, "__file__", str(package / "__init__.py"))
    first = judging_deps()["engine"]
    with (package / left_out).open("a") as handle:
        handle.write("\n# reworded\n")
    assert judging_deps()["engine"] == first
    for judged in (
        "stages/data/run.py",
        "stages/feat/run.py",
        "stages/data/floor.py",
        "stages/feat/floor.py",
        "stages/data/treatments.py",
        "stages/skeleton.py",
    ):
        before = judging_deps()["engine"]
        with (package / judged).open("a") as handle:
            handle.write("\n# changed\n")
        assert judging_deps()["engine"] != before, judged


def _anchored(tmp_path: Path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl", anchors=tmp_path / "heads")
    for i in range(4):
        ledger.append("submit", "feat", {"verdict": "FAIL", "i": i}, at_ns=i)
    return ledger


def test_an_anchor_catches_records_removed_from_the_end(tmp_path: Path) -> None:
    ledger = _anchored(tmp_path)
    lines = ledger.path.read_text().splitlines(keepends=True)
    ledger.path.write_text("".join(lines[:-1]))
    Ledger(ledger.path).verify()
    with pytest.raises(LedgerError, match="removed or rewritten"):
        ledger.verify()


def test_an_anchor_catches_a_consistent_rewrite(tmp_path: Path) -> None:
    ledger = _anchored(tmp_path)
    other = Ledger(tmp_path / "other.jsonl")
    for i in range(4):
        other.append("submit", "feat", {"verdict": "PASS" if i == 3 else "FAIL", "i": i}, at_ns=i)
    other.verify()
    ledger.path.write_bytes(other.path.read_bytes())
    with pytest.raises(LedgerError, match="line 4"):
        ledger.verify()


def test_an_anchor_follows_a_ledger_that_grew_elsewhere(tmp_path: Path) -> None:
    ledger = _anchored(tmp_path)
    Ledger(ledger.path).append("submit", "feat", {"verdict": "PASS"}, at_ns=9)
    assert ledger.verify() == 5
    lines = ledger.path.read_text().splitlines(keepends=True)
    ledger.path.write_text("".join(lines[:-1]))
    with pytest.raises(LedgerError):
        ledger.verify()


def test_an_anchor_is_keyed_by_the_ledger_not_its_path(tmp_path: Path) -> None:
    ledger = _anchored(tmp_path)
    moved = tmp_path / "moved" / "ledger.jsonl"
    moved.parent.mkdir()
    moved.write_text("".join(ledger.path.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(LedgerError):
        Ledger(moved, anchors=tmp_path / "heads").verify()
