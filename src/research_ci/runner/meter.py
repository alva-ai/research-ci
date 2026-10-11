"""What a call into research code costs: its wall time and, for a call that runs alone, the memory the process holds at
its peak during it. Cards show it and records keep it; nothing judges it.

Memory is the process's resident set size, read with the standard library alone: ``/proc/self/statm`` on Linux and
``proc_pid_rusage`` through ``ctypes`` on macOS, sampled on a background thread every ``EVERY`` seconds and at both
ends of the call. When the call raises the process's high-water mark (``getrusage``), that mark is its exact peak.
Where the current size cannot be read, only the high-water mark is known, which may predate the call, and ``memory``
says so.
"""

import ctypes
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")
EVERY = 0.01
SAMPLED = "sampled"
HIGH_WATER = "high-water mark"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Cost:
    """``seconds`` of wall time; ``peak_bytes`` and ``start_bytes`` resident, ``None`` when not measured."""

    seconds: float
    peak_bytes: int | None = None
    start_bytes: int | None = None
    memory: str = UNAVAILABLE


def measured(call: Callable[[], T], memory: bool = True) -> tuple[T, Cost]:
    """``call()`` and what it cost; with ``memory``, the process's resident memory at its peak during the call, which
    is the call's own only when nothing else runs meanwhile."""
    read = _resident_reader() if memory else None
    if read is None:
        started = time.perf_counter()
        out = call()
        seconds = round(time.perf_counter() - started, 4)
        mark = _high_water() if memory else None
        return out, Cost(seconds, mark, None, HIGH_WATER if mark is not None else UNAVAILABLE)
    before = _high_water()
    start = read()
    peak = [start]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(EVERY):
            peak[0] = max(peak[0], read())

    sampler = threading.Thread(target=sample, name="rci-meter", daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        out = call()
    finally:
        seconds = round(time.perf_counter() - started, 4)
        stop.set()
        sampler.join()
    top = max(peak[0], read())
    after = _high_water()
    if before is not None and after is not None and after > before:
        top = max(top, after)
    return out, Cost(seconds, top, start, SAMPLED)


def _resident_reader() -> Callable[[], int] | None:
    if sys.platform.startswith("linux") and os.path.exists("/proc/self/statm"):
        page = os.sysconf("SC_PAGE_SIZE")

        def statm() -> int:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * page

        return statm
    if sys.platform == "darwin":
        return _darwin_reader()
    return None


class _RusageInfoV0(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
        )
    ]


def _darwin_reader() -> Callable[[], int] | None:
    try:
        rusage = ctypes.CDLL(None).proc_pid_rusage
    except (OSError, AttributeError):
        return None
    rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    rusage.restype = ctypes.c_int
    pid, info = os.getpid(), _RusageInfoV0()
    if rusage(pid, 0, ctypes.byref(info)) != 0:
        return None

    def read() -> int:
        rusage(pid, 0, ctypes.byref(info))
        return int(info.resident_size)

    return read


def _high_water() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    mark = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(mark) if sys.platform == "darwin" else int(mark) * 1024
