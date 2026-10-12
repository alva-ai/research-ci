from functools import partial
from pathlib import Path

import pytest

from research_ci.runner import plugins
from research_ci.stages.feat import run as feat
from toy import REGISTRY


@pytest.fixture(autouse=True)
def isolated_rci_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "rci-home"
    monkeypatch.setenv("RCI_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def toy_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", dict(REGISTRY))


@pytest.fixture(autouse=True)
def quick_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine's own tests calibrate the decisive test, at every data check, with fewer repetitions than research
    does; a test that passes its own settings keeps them."""
    quick = partial(feat.FeatSettings, calibration_reps=40, calibration_nulls=20, floor_reps=60, cuts=2)
    monkeypatch.setattr(feat, "FeatSettings", quick)
