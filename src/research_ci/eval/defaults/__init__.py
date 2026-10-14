"""Stage-specific proposals from the Engine's general eval library.

These names are a menu, not an automatic verdict. The caller still supplies
the stage's columns, target, horizon, and other research context.
"""

from __future__ import annotations

from typing import Literal

from . import algo, data, feat, sig

Stage = Literal["data", "feat", "sig", "algo"]

DEFAULT_METHODS: dict[Stage, tuple[str, ...]] = {
    "data": data.METHOD_NAMES,
    "feat": feat.METHOD_NAMES,
    "sig": sig.METHOD_NAMES,
    "algo": algo.METHOD_NAMES,
}


def default_method_names(stage: str) -> tuple[str, ...]:
    """Return general-method proposals for a stage."""

    try:
        return DEFAULT_METHODS[stage]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"unknown stage: {stage}") from exc


__all__ = ["DEFAULT_METHODS", "default_method_names"]
