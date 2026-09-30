from pathlib import Path

import pytest

from research_ci.runner import plugins
from toy import REGISTRY


@pytest.fixture(autouse=True)
def isolated_rci_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "rci-home"
    monkeypatch.setenv("RCI_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def toy_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", dict(REGISTRY))
