"""Replaying a data plugin as of past times. A plugin that builds, repairs or filters its rows with information from
later gives a different answer at ``T`` than its full fetch cut at ``T``. The engine picks where to look: inside every
hole of the full output, just before entities first appear, and at random."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from research_ci.api.methods import Case, Source
from research_ci.eval.data.baseline import events
from research_ci.eval.synthetic import HOUR_NS, START_NS, source
from research_ci.kernel.time.clock import from_ns, lit_time
from research_ci.kernel.verdict.check import Check

Fetch = Callable[[int, list[str]], pl.DataFrame]
KEY = ["entity", "field", "event_time", "vintage"]
DAY_NS = 86_400 * 10**9


@dataclass(frozen=True)
class Cut:
    at: int
    why: str
    entities: tuple[str, ...]


def cut_points(
    full: pl.DataFrame,
    rng: np.random.Generator,
    holes: int = 200,
    late: int = 40,
    randoms: int = 5,
    random_entities: int = 20,
    hole_factor: float = 1.5,
) -> list[Cut]:
    """Inside every hole (an event missing where the usual spacing expects one), at its first missing slot; a day
    before each week's newly appearing entities, so rows a plugin dropped from before an entity's start show up; and
    at random times for random entities."""
    e = events(full)
    first = int(full.select(pl.col("available_at").min().dt.epoch("ns")).item())
    last = int(full.select(pl.col("available_at").max().dt.epoch("ns")).item())
    found = e.filter((pl.col("n") >= 3) & (pl.col("gap") > hole_factor * pl.col("spacing")))
    found = found.select(
        "entity", "field", (pl.col("prev").dt.epoch("ns") + pl.col("spacing").cast(pl.Int64)).alias("at")
    ).unique(["entity", "at"])
    cuts = [Cut(int(at), f"hole in {f}", (ent,)) for ent, f, at in _sample(found, holes, rng).iter_rows()]
    starts = full.group_by("entity").agg(pl.col("event_time").min().dt.epoch("ns").alias("t"))
    starts = starts.filter(pl.col("t") > first + DAY_NS).with_columns(week=(pl.col("t") - first) // (7 * DAY_NS))
    weeks = starts.group_by("week").agg(pl.col("entity").sort(), pl.col("t").min()).sort("week")
    for ents, t in _sample(weeks, late, rng).select("entity", "t").iter_rows():
        cuts.append(Cut(max(first, int(t) - DAY_NS), f"before {len(ents)} entities start", tuple(ents)))
    names = full["entity"].unique().sort().to_list()
    for _ in range(randoms):
        at = int(rng.integers(first + (last - first) // 10, last - (last - first) // 10))
        picked = rng.choice(names, size=min(random_entities, len(names)), replace=False)
        cuts.append(Cut(at, "random", tuple(sorted(str(x) for x in picked))))
    return sorted(cuts, key=lambda c: (c.at, c.why))


def replay(fetch: Fetch, full: pl.DataFrame, cuts: Sequence[Cut]) -> tuple[Check, list[dict[str, Any]]]:
    """Each cut: the plugin's fetch as of ``at`` for the cut's entities must equal the full fetch cut at ``at``."""
    results = []
    for cut in cuts:
        try:
            got = fetch(cut.at, list(cut.entities))
        except Exception as exc:  # noqa: BLE001
            results.append(_result(cut, error=f"{type(exc).__name__}: {exc}"[:200]))
            continue
        after = got.filter(pl.col("available_at") > lit_time(cut.at)).height
        got = got.filter(pl.col("available_at") <= lit_time(cut.at))
        want = full.filter(pl.col("entity").is_in(list(cut.entities)) & (pl.col("available_at") <= lit_time(cut.at)))
        results.append(_result(cut, *_compare(got, want), after=after))
    bad = [r for r in results if r["status"] == "fail"]
    if not results:
        return Check("replay as of past times", "warn", "no cut points"), results
    if bad:
        where = sorted({e for r in bad for e in r["differ"]})
        detail = f"{len(bad)} of {len(results)} cut points differ from the full fetch cut there: {', '.join(where[:8])}"
        return Check("replay as of past times", "fail", detail), results
    return Check("replay as of past times", "pass", f"{len(results)} cut points agree with the full fetch"), results


def _compare(got: pl.DataFrame, want: pl.DataFrame) -> tuple[int, int, int, list[str]]:
    a = got.select(*KEY, pl.col("value").alias("a"))
    b = want.select(*KEY, pl.col("value").alias("b"))
    only_replay = a.join(b, on=KEY, how="anti")
    only_full = b.join(a, on=KEY, how="anti")
    both = a.join(b, on=KEY)
    x, y = both["a"].to_numpy(), both["b"].to_numpy()
    same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-9, atol=1e-12)
    changed = both.filter(pl.Series(~same))
    differ = sorted(set(only_replay["entity"]) | set(only_full["entity"]) | set(changed["entity"]))
    return only_replay.height, only_full.height, changed.height, differ


def _result(
    cut: Cut,
    only_replay: int = 0,
    only_full: int = 0,
    changed: int = 0,
    differ: list[str] | None = None,
    after: int = 0,
    error: str = "",
) -> dict[str, Any]:
    failed = bool(error or only_replay or only_full or changed or after)
    return {
        "at": from_ns(cut.at).strftime("%Y-%m-%dT%H:%MZ"),
        "why": cut.why,
        "entities": len(cut.entities),
        "only_replay": only_replay,
        "only_full": only_full,
        "changed": changed,
        "after_cut": after,
        "differ": differ or ([] if not error else list(cut.entities[:3])),
        "error": error,
        "status": "fail" if failed else "pass",
    }


def _sample(frame: pl.DataFrame, k: int, rng: np.random.Generator) -> pl.DataFrame:
    if frame.height <= k:
        return frame
    return frame.sample(k, seed=int(rng.integers(0, 2**31)))


def _cut(rows: pl.DataFrame, at: int, entities: list[str]) -> pl.DataFrame:
    return rows.filter(pl.col("entity").is_in(entities) & (pl.col("available_at") <= lit_time(at)))


def _honest(seed: int) -> str:
    full = source(Source(lag=2, revised=0.3), seed)
    cuts = cut_points(full, np.random.default_rng(seed))
    return replay(lambda at, ents: _cut(full, at, ents), full, cuts)[0].status


def _hindsight(seed: int) -> str:
    """Rows of e0's price in hours 100 to 109 are deleted once hour 130 shows they preceded a jump; as of earlier
    times the source still had them."""
    original = source(Source(), seed)
    t = pl.col("event_time").dt.epoch("ns")
    window = (pl.col("entity") == "e0") & (pl.col("field") == "price")
    window = window & t.is_between(START_NS + 100 * HOUR_NS, START_NS + 109 * HOUR_NS)
    full = original.filter(~window)
    learned = START_NS + 130 * HOUR_NS

    def fetch(at: int, ents: list[str]) -> pl.DataFrame:
        return _cut(original if at < learned else full, at, ents)

    return replay(fetch, full, cut_points(full, np.random.default_rng(seed)))[0].status


def _ahead(seed: int) -> str:
    full = source(Source(lag=2), seed)
    cuts = cut_points(full, np.random.default_rng(seed))
    return replay(lambda at, ents: full.filter(pl.col("entity").is_in(ents)), full, cuts)[0].status


CASES = [
    Case(
        "a source that replays honestly passes, with lags and revisions",
        lambda: {_honest(s) for s in range(3)},
        {"pass"},
    ),
    Case("a source that deleted rows with hindsight fails", lambda: {_hindsight(s) for s in range(3)}, {"fail"}),
    Case("a source that answers with rows not yet known fails", lambda: {_ahead(s) for s in range(3)}, {"fail"}),
]
