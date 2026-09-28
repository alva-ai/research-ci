from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_aci_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "aci-home"
    monkeypatch.setenv("ACI_HOME", str(home))
    return home
