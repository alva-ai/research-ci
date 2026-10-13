"""Replaying a data plugin as of past times. A plugin that builds, repairs or filters its rows with information from
later gives a different answer at ``T`` than its full fetch cut at ``T``. The engine picks where to look: inside every
hole of a source's rows of an inst, just before insts first appear, and at random."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from research_ci.api.evals import Case, Source
from research_ci.eval import synthetic
from research_ci.eval.data.baseline import events, rows
from research_ci.eval.synthetic import HOUR_NS, START_NS
from research_ci.kernel.tables import values
from research_ci.kernel.time.clock import from_ns, lit_time
from research_ci.kernel.verdict.check import Check

Sources = Mapping[str, pl.DataFrame]
Fetch = Callable[[int, list[str]], Sources]
KEY = ["inst", "event_time", "vintage"]
DAY_NS = 86_400 * 10**9


@dataclass(frozen=True)
class Cut:
    at: int
    why: str
    insts: tuple[str, ...]


def cut_points(
    full: Sources,
    rng: np.random.Generator,
    holes: int = 200,
    late: int = 40,
    randoms: int = 5,
    random_insts: int = 20,
    hole_factor: float = 1.5,
) -> list[Cut]:
    """Cut points: holes, before new insts, and random times. Inside every hole (an event of a source missing where
    the usual spacing of the inst's rows expects one), at its first missing slot; a day before each week's newly
    appearing insts, so rows a plugin dropped from before an inst's start show up; and at random times for random
    insts."""
    r = rows(full)
    e = events(r)
    first = int(r.select(pl.col("available_at").min().dt.epoch("ns")).item())
    last = int(r.select(pl.col("available_at").max().dt.epoch("ns")).item())
    found = e.filter((pl.col("n") >= 3) & (pl.col("gap") > hole_factor * pl.col("spacing")))
    found = found.select(
        "source", "inst", (pl.col("prev").dt.epoch("ns") + pl.col("spacing").cast(pl.Int64)).alias("at")
    ).unique(["inst", "at"])
    holed = _sample(found, holes, rng).select(
        "at", pl.format("hole in {}", "source").alias("why"), pl.concat_list("inst").alias("inst")
    )
    starts = r.group_by("inst").agg(pl.col("event_time").min().dt.epoch("ns").alias("t"))
    starts = starts.filter(pl.col("t") > first + DAY_NS).with_columns(week=(pl.col("t") - first) // (7 * DAY_NS))
    weeks = starts.group_by("week").agg(pl.col("inst").sort(), pl.col("t").min()).sort("week")
    before = _sample(weeks, late, rng).select(
        pl.max_horizontal(pl.lit(first), pl.col("t") - DAY_NS).alias("at"),
        pl.format("before {} insts start", pl.col("inst").list.len()).alias("why"),
        "inst",
    )
    names = r["inst"].unique().sort()
    k = min(random_insts, names.len())
    ats = rng.integers(first + (last - first) // 10, last - (last - first) // 10, size=randoms)
    picked = np.sort(rng.permuted(np.tile(np.arange(names.len()), (randoms, 1)), axis=1)[:, :k], axis=1)
    drawn = pl.DataFrame({"_c": np.repeat(np.arange(randoms), k), "_i": picked.ravel()})
    drawn = drawn.with_columns(inst=pl.lit(names).gather("_i")).group_by("_c", maintain_order=True).agg("inst")
    random = drawn.select(pl.Series("at", ats), pl.lit("random").alias("why"), "inst")
    found_cuts = pl.concat([f.cast({"at": pl.Int64}) for f in (holed, before, random)]).sort("at", "why")
    return [Cut(int(at), why, tuple(ents)) for at, why, ents in found_cuts.iter_rows()]


def replay(fetch: Fetch, full: Sources, cuts: Sequence[Cut]) -> tuple[Check, list[dict[str, Any]]]:
    """Each cut's fetch must equal the full fetch cut there. For each cut, the plugin's fetch as of ``at`` for the cut's
    insts must give every source as the full fetch cut at ``at`` gives it."""
    results = []
    for cut in cuts:
        try:
            got = fetch(cut.at, list(cut.insts))
        except Exception as exc:  # noqa: BLE001
            results.append(_result(cut, error=f"{type(exc).__name__}: {exc}"[:200]))
            continue
        later = pl.col("available_at") > lit_time(cut.at)
        after = sum(table.filter(later).height for table in got.values())
        mine = {name: table.filter(~later) for name, table in got.items()}
        mask = pl.col("inst").is_in(list(cut.insts)) & ~later
        want = {name: table.filter(mask) for name, table in full.items()}
        results.append(_result(cut, *_compare(mine, want), after=after))
    bad = [r for r in results if r["status"] == "fail"]
    if not results:
        return Check("replay as of past times", "warn", "no cut points"), results
    if bad:
        where = sorted({e for r in bad for e in r["differ"]})
        detail = f"{len(bad)} of {len(results)} cut points differ from the full fetch cut there: {', '.join(where[:8])}"
        return Check("replay as of past times", "fail", detail), results
    return Check("replay as of past times", "pass", f"{len(results)} cut points agree with the full fetch"), results


def _compare(got: Sources, want: Sources) -> tuple[int, int, int, list[str]]:
    """Rows only the replay has, rows only the full fetch has, rows both have with another value, and the insts of all
    of them, over every source either has."""
    found = [_one(got.get(name), want.get(name)) for name in sorted(set(got) | set(want))]
    differ = sorted({e for *_, insts in found for e in insts})
    return sum(f[0] for f in found), sum(f[1] for f in found), sum(f[2] for f in found), differ


def _one(got: pl.DataFrame | None, want: pl.DataFrame | None) -> tuple[int, int, int, list[str]]:
    columns = sorted({c for t in (got, want) if t is not None for c in values(t)})
    a, b = (_aligned(t, columns) for t in (got, want))
    only_replay = a.join(b, on=KEY, how="anti")
    only_full = b.join(a, on=KEY, how="anti")
    both = a.join(b, on=KEY, suffix="_b")
    changed = both.filter(pl.any_horizontal(_differs(c) for c in columns)) if columns else both.head(0)
    differ = sorted(set(only_replay["inst"]) | set(only_full["inst"]) | set(changed["inst"]))
    return only_replay.height, only_full.height, changed.height, differ


def _aligned(table: pl.DataFrame | None, columns: list[str]) -> pl.DataFrame:
    """The keys and ``columns`` of a source's rows; none when the source is missing, nulls for a missing column."""
    if table is None:
        table = pl.DataFrame(schema={"inst": pl.String, "event_time": pl.Datetime("ns", "UTC"), "vintage": pl.Int64})
    own = [pl.col(c) if c in table.columns else pl.lit(None, dtype=pl.Float64).alias(c) for c in columns]
    return table.select(*KEY, *own)


def _differs(c: str) -> pl.Expr:
    """Whether the two sides of a column differ: both null or both NaN agree, numbers within ``np.isclose``."""
    x, y = pl.col(c), pl.col(f"{c}_b")
    close = (x - y).abs() <= 1e-12 + 1e-9 * y.abs()
    same = (x.is_null() & y.is_null()) | (x.is_nan() & y.is_nan()).fill_null(False) | close.fill_null(False)
    return ~same


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
        "insts": len(cut.insts),
        "only_replay": only_replay,
        "only_full": only_full,
        "changed": changed,
        "after_cut": after,
        "differ": differ or ([] if not error else list(cut.insts[:3])),
        "error": error,
        "status": "fail" if failed else "pass",
    }


def _sample(frame: pl.DataFrame, k: int, rng: np.random.Generator) -> pl.DataFrame:
    if frame.height <= k:
        return frame
    return frame.sample(k, seed=int(rng.integers(0, 2**31)))


def _cut(tables: Sources, at: int, insts: list[str]) -> dict[str, pl.DataFrame]:
    keep = pl.col("inst").is_in(insts) & (pl.col("available_at") <= lit_time(at))
    return {name: t.filter(keep) for name, t in tables.items()}


def _honest(seed: int) -> str:
    full = synthetic.sources(Source(lag=2, revised=0.3), seed)
    cuts = cut_points(full, np.random.default_rng(seed))
    return replay(lambda at, ents: _cut(full, at, ents), full, cuts)[0].status


def _hindsight(seed: int) -> str:
    """Rows of e0's price in hours 100 to 109 are deleted once hour 130 shows they preceded a jump; as of earlier
    times the source still had them."""
    original = synthetic.sources(Source(), seed)
    t = pl.col("event_time").dt.epoch("ns")
    window = (pl.col("inst") == "e0") & t.is_between(START_NS + 100 * HOUR_NS, START_NS + 109 * HOUR_NS)
    full = {**original, "prices": original["prices"].filter(~window)}
    learned = START_NS + 130 * HOUR_NS

    def fetch(at: int, ents: list[str]) -> dict[str, pl.DataFrame]:
        return _cut(original if at < learned else full, at, ents)

    return replay(fetch, full, cut_points(full, np.random.default_rng(seed)))[0].status


def _ahead(seed: int) -> str:
    full = synthetic.sources(Source(lag=2), seed)
    cuts = cut_points(full, np.random.default_rng(seed))
    check, _ = replay(lambda at, ents: {n: t.filter(pl.col("inst").is_in(ents)) for n, t in full.items()}, full, cuts)
    return check.status


def _revalued(seed: int) -> str:
    """A replay whose flows are a tenth larger than the full fetch's, as of every time."""
    full = synthetic.sources(Source(), seed)

    def fetch(at: int, ents: list[str]) -> dict[str, pl.DataFrame]:
        cut = _cut(full, at, ents)
        return {**cut, "flows": cut["flows"].with_columns(pl.col("flow") * 1.1)}

    return replay(fetch, full, cut_points(full, np.random.default_rng(seed)))[0].status


CASES = [
    Case(
        "a source that replays honestly passes, with lags and revisions",
        lambda: {_honest(s) for s in range(3)},
        {"pass"},
    ),
    Case("a source that deleted rows with hindsight fails", lambda: {_hindsight(s) for s in range(3)}, {"fail"}),
    Case("a source that answers with rows not yet known fails", lambda: {_ahead(s) for s in range(3)}, {"fail"}),
    Case("a source that answers with other values fails", lambda: {_revalued(s) for s in range(3)}, {"fail"}),
]
