"""What a call into research code costs: its wall time, and the process's resident memory at its peak during it,
read with the standard library alone."""

import sys
import threading
import time
from dataclasses import asdict

import numpy as np
import pytest

from research_ci.render.feat import cost_line
from research_ci.runner import meter
from research_ci.runner.meter import HIGH_WATER, SAMPLED, UNAVAILABLE, measured

MEASURED = sys.platform == "darwin" or sys.platform.startswith("linux")


def _held(nbytes: int) -> int:
    block = np.ones(nbytes // 8)
    time.sleep(0.05)
    return int(block.size)


@pytest.mark.skipif(not MEASURED, reason="the resident size is read on Linux and macOS")
def test_the_peak_counts_memory_freed_before_the_call_returns() -> None:
    out, cost = measured(lambda: _held(400 * 10**6))
    assert out == 50 * 10**6 and cost.memory == SAMPLED and cost.seconds >= 0.05
    assert cost.peak_bytes is not None and cost.start_bytes is not None
    assert cost.peak_bytes - cost.start_bytes > 300 * 10**6


def test_without_memory_only_the_time_is_measured() -> None:
    out, cost = measured(lambda: time.sleep(0.02) or 7, memory=False)
    assert out == 7 and cost.seconds >= 0.02 and cost.memory == UNAVAILABLE and cost.peak_bytes is None


def test_a_failing_call_raises_and_stops_the_sampler() -> None:
    def fails() -> None:
        raise ValueError("no")

    with pytest.raises(ValueError, match="no"):
        measured(fails)
    assert not any(t.name == "rci-meter" for t in threading.enumerate())


def test_where_the_size_cannot_be_read_the_line_says_it_is_the_high_water_mark(monkeypatch) -> None:
    monkeypatch.setattr(meter, "_resident_reader", lambda: None)
    _, cost = measured(lambda: None)
    assert cost.memory in (HIGH_WATER, UNAVAILABLE) and cost.start_bytes is None
    line = cost_line(asdict(cost) | {"cut_seconds": [0.0004, 0.25, 1.25]})
    assert line is not None and line.startswith("full call ") and "on cut data 0.4 ms, 250 ms, 1.2 s" in line
    if cost.memory == HIGH_WATER:
        assert "process high-water mark, not the call's own" in line
