from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_rci_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "rci-home"
    monkeypatch.setenv("RCI_HOME", str(home))
    return home
