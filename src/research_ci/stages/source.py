"""Admitting a data source: does the plugin give, for any past time, what was known then, and does an independent
source agree with what it gives? Run once per plugin, options and code; every idea using the source reuses it, and
the first approval of an idea's data that uses it admits it. The steps are the admission's floor list
(``stages/floor/source.py``), run in order."""

import time
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl

from research_ci.api.contract import Use
from research_ci.kernel.tables import field_table
from research_ci.kernel.time.clock import from_ns
from research_ci.runner.plugins import code_hash
from research_ci.stages import steps
from research_ci.stages.floor import source as floor
from research_ci.stages.floor.source import SourceSettings, can_replay


def admit(use: Use, plugin: Any, fields: pl.DataFrame, settings: SourceSettings | None = None) -> dict[str, Any]:
    s = settings or SourceSettings()
    started = time.monotonic()
    names = sorted(fields["field"].unique().to_list())
    spec: dict[str, Any] = {
        "plugin": use.name,
        "options": dict(use.options),
        "code": code_hash(plugin),
        "rows": fields.height,
        "entities": fields["entity"].n_unique(),
        "fields": names,
        "settings": asdict(s),
    }

    def fetch(at: int, entities: list[str]) -> pl.DataFrame:
        return field_table(plugin.fetch(as_of=from_ns(at), entities=entities))

    seen = floor.Seen(plugin, fields, names, s, np.random.default_rng(s.seed), fetch if can_replay(plugin) else None)
    found = steps.combined(step.run(seen) for step in floor.STEPS)
    spec |= found.shown
    spec["checks"] = [asdict(c) for c in found.checks]
    failed = [c.name for c in found.checks if c.status == "fail"]
    status = "not admitted" if failed else "admissible"
    notes = [f"failed: {', '.join(failed)}"] if failed else ["admitted once, by approving the data of an idea"]
    spec["verdict"] = {"status": status, "notes": notes}
    spec["seconds"] = round(time.monotonic() - started, 1)
    return spec
