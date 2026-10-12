"""Judging an eval: its value per decision time, averaged by blocks of the horizon, against the same average in each
null world.

``judge_all`` judges many evals at once. A lane is one evaluation table shuffled one way: its layout is computed once,
and each seed's shuffle once, for every eval and world that reads the lane. A ``Statistic`` with a batched form is
computed on many shuffles at once by that form, on a lane whose shuffles move scores only within a decision time; any
other eval on many shuffles at once by stacking (``Stacked``): one copy of the table per shuffle, each later in time
than the one before, in one call. Either is used only after it gives the eval's own numbers on the table as it is;
otherwise the eval runs shuffle by shuffle on the table with the shuffled scores. A request takes the seeds
``0 .. reps - 1``, or fewer when a lane has fewer distinct shuffles; a sequential one takes them a batch at a time and
stops once its decision cannot change (``sequential.py``).
"""

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import polars as pl

from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict import sequential
from research_ci.kernel.verdict.power import block_means

Evaluate = Callable[[pl.DataFrame], pl.DataFrame]
Index = Callable[[int], np.ndarray]
CHUNK = 1 << 22
STACKED_ROWS = 1 << 20
CHECKED_COPIES = 8
WEEK_NS = 7 * 86_400 * 10**9


class EvalError(Exception):
    pass


class Prepared(Protocol):
    """A batched statistic ready for one table: its decision ``times`` (ns, ascending), and for a matrix of shuffles
    (one per row) the value at each time, NaN where there is none."""

    @property
    def times(self) -> np.ndarray: ...

    def __call__(self, index: np.ndarray) -> np.ndarray: ...


Batched = Callable[[pl.DataFrame], Prepared]


@dataclass(frozen=True)
class Statistic:
    """An eval, with the same computed on many shuffles at once when ``batched`` is given
    (``eval/statistics/shuffled.py``)."""

    evaluate: Evaluate
    batched: Batched | None = None

    def __call__(self, table: pl.DataFrame) -> pl.DataFrame:
        return self.evaluate(table)


@dataclass(frozen=True)
class Lane:
    """An evaluation table, its horizon in ns, and one way of shuffling its scores, built by ``shuffle()``; lanes with
    the same ``key`` share every shuffle. ``within``: its shuffles move scores only between the rows of one decision
    time, as the batched forms assume. ``distinct``: how many distinct shuffles it has, when that may be fewer than a
    request asks for."""

    table: pl.DataFrame
    h_ns: int
    key: Hashable
    shuffle: Callable[[], Index]
    within: bool = True
    distinct: int | None = None


@dataclass(frozen=True)
class Sequential:
    """Shuffles ``batch`` at a time, up to the request's ``reps``, until the decision against ``need`` is settled:
    the family's, or with ``horizons`` every counted horizon's."""

    batch: int
    need: float
    horizons: bool = False


@dataclass(frozen=True)
class Request:
    """One eval against one world: ``lanes`` maps each horizon to its lane, ``evaluate`` may differ by horizon.
    Horizons with fewer than ``min_blocks`` blocks are shown but not counted. With ``combine="max"`` the family's z
    compares the best standardized horizon with the best of each shuffle; with ``"each"`` it is the weakest
    horizon's z."""

    evaluate: Evaluate | Mapping[str, Evaluate]
    lanes: Mapping[str, Lane]
    reps: int
    min_blocks: int
    combine: str
    sequential: Sequential | None = None


def per_t(evaluate: Evaluate, table: pl.DataFrame) -> pl.DataFrame:
    out = evaluate(table)
    if not isinstance(out, pl.DataFrame) or not {"t", "value"} <= set(out.columns):
        raise EvalError("an eval must return a polars DataFrame with columns t and value")
    out = out.select(pl.col("t").cast(TIME), pl.col("value").cast(pl.Float64))
    return out.filter(pl.col("value").is_finite()).sort("t")


def blocks(values: pl.DataFrame, h_ns: int) -> tuple[float, float, int]:
    """Mean of the block means, its standard error, and the number of blocks."""
    means = block_means(values, h_ns)["value"].drop_nulls().to_numpy()
    n = len(means)
    if n == 0:
        return math.nan, math.nan, 0
    se = float(np.std(means, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return float(means.mean()), se, n


def with_scores(table: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    return table.with_columns(pl.Series("score", scores, dtype=pl.Float64))


def block_mean_rows(values: np.ndarray, times: np.ndarray, h_ns: int) -> np.ndarray:
    """``blocks(...)[0]`` of each row of per-time values (NaN where a time has none)."""
    out = np.full(values.shape[0], np.nan)
    finite = np.isfinite(values)
    has = finite.any(axis=1)
    if not has.any():
        return out
    t0 = times[np.argmax(finite, axis=1)]
    block = (times[None, :] - t0[:, None]) // h_ns
    width = int(block[finite].max()) + 1
    flat = (np.arange(values.shape[0])[:, None] * width + block)[finite]
    size = values.shape[0] * width
    sums = np.bincount(flat, weights=values[finite], minlength=size).reshape(-1, width)
    counts = np.bincount(flat, minlength=size).reshape(-1, width)
    filled = counts > 0
    means = np.where(filled, sums / np.maximum(counts, 1), 0.0)
    out[has] = (means.sum(axis=1) / np.maximum(filled.sum(axis=1), 1))[has]
    return out


def null_values(prepared: Prepared, index: np.ndarray, h_ns: int) -> np.ndarray:
    """``blocks(...)[0]`` of a batched statistic for each shuffle (row) of ``index``, a few shuffles at a time."""
    step = max(1, CHUNK // max(index.shape[1], 1))
    parts = [prepared(index[i : i + step]) for i in range(0, index.shape[0], step)]
    return np.concatenate([block_mean_rows(v, prepared.times, h_ns) for v in parts]) if parts else np.zeros(0)


def judge(
    evaluate: Evaluate | Mapping[str, Evaluate],
    tables: dict[str, tuple[pl.DataFrame, int]],
    index: Callable[[str], Index],
    reps: int,
    min_blocks: int,
    combine: str,
    within: bool = True,
) -> dict[str, Any]:
    """One eval against one null world over its horizons: ``tables`` maps a horizon to its evaluation table and
    length, and ``index(h)(seed)`` gives the world's shuffle for that horizon; ``within`` as in ``Lane``."""
    lanes = {h: Lane(table, h_ns, object(), _bound(index, h), within) for h, (table, h_ns) in tables.items()}
    result = judge_all([Request(evaluate, lanes, reps, min_blocks, combine)])[0]
    if isinstance(result, Exception):
        raise result
    return result


def judge_all(requests: Sequence[Request], workers: int = 1, batched: bool = True) -> list[dict[str, Any] | Exception]:
    """Every request judged, each lane shuffled once per seed for all the requests that read it; a request that fails
    gives its exception. ``batched=False`` evaluates every eval shuffle by shuffle."""
    unique = list({id(r): r for r in requests}.values())
    lanes: dict[Hashable, _Lane] = {}
    taken: list[_Taken] = []
    for r in unique:
        state = _Taken(r)
        taken.append(state)
        for h, lane in r.lanes.items():
            f = r.evaluate[h] if isinstance(r.evaluate, Mapping) else r.evaluate
            consumer = _Consumer(state, h, f, batched)
            state.consumers.append(consumer)
            lanes.setdefault(lane.key, _Lane(lane)).consumers.append(consumer)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while True:
            busy = [lane for lane in lanes.values() if lane.pending()]
            if not busy:
                break
            for job in [pool.submit(lane.run) for lane in busy]:
                job.result()
            for state in taken:
                state.advance()
            lanes = {k: lane for k, lane in lanes.items() if lane.waiting()}
    results = {id(state.request): state.result() for state in taken}
    return [results[id(r)] for r in requests]


def _bound(index: Callable[[str], Index], h: str) -> Callable[[], Index]:
    return lambda: index(h)


@dataclass(eq=False)
class _Taken:
    """A request's progress: the seeds drawn, the next target, the most it draws (its ``reps``, or fewer when a lane
    has fewer distinct shuffles), and its consumers (one per horizon)."""

    request: Request
    consumers: list["_Consumer"] = field(default_factory=list)
    drawn: int = 0
    target: int = 0
    most: int = 0
    error: Exception | None = None
    done: bool = False

    def __post_init__(self) -> None:
        r = self.request
        self.most = min([r.reps, *(lane.distinct for lane in r.lanes.values() if lane.distinct)])
        self.target = self.most if r.sequential is None else min(self.most, r.sequential.batch)

    def advance(self) -> None:
        if self.done:
            return
        self.drawn = self.target
        errors = [c.error for c in self.consumers if c.error is not None]
        if errors:
            self.error = errors[0]
            self.finish()
            return
        r = self.request
        if r.sequential is None or self.drawn >= self.most:
            self.finish()
            return
        reals, nulls = self.values()
        rule = r.sequential
        if settled(summary(reals, nulls, r.min_blocks, r.combine), reals, nulls, r.combine, rule.need, rule.horizons):
            self.finish()
        else:
            self.target = min(self.most, self.drawn + r.sequential.batch)

    def finish(self) -> None:
        self.done = True
        for c in self.consumers:
            c.prepared = None

    def values(self) -> tuple[dict[str, tuple[float, float, int]], dict[str, np.ndarray]]:
        return {c.h: c.real for c in self.consumers}, {c.h: c.null[: self.drawn] for c in self.consumers}

    def result(self) -> dict[str, Any] | Exception:
        """The judgement; ``distinct`` says it drew every distinct shuffle there is, fewer than its ``reps``."""
        if self.error is not None:
            return self.error
        r = self.request
        out = summary(*self.values(), r.min_blocks, r.combine)
        return out | ({"distinct": self.most} if self.most < r.reps and self.drawn >= self.most else {})


def summary(
    reals: Mapping[str, tuple[float, float, int]], nulls: Mapping[str, np.ndarray], min_blocks: int, combine: str
) -> dict[str, Any]:
    """The judgement from each horizon's real value (``blocks``) and null values: z per horizon, the family's, and
    each counted horizon's z against the null the family's uses (``adjusted``, ``sequential.horizon_z``)."""
    rows, standard = [], []
    for h, (real, se, n) in reals.items():
        null = nulls[h]
        mu, sd = float(np.nanmean(null)), float(np.nanstd(null, ddof=1))
        z = (real - mu) / sd if sd > 0 else math.nan
        counted = n >= min_blocks and math.isfinite(z)
        rows.append({"h": h, "real": real, "se": se, "blocks": n, "null_mean": mu, "null_sd": sd, "z": z})
        rows[-1]["counted"] = counted
        if counted:
            standard.append((null - mu) / sd)
    shuffles = min((len(v) for v in nulls.values()), default=0)
    return _combined(rows, standard, min_blocks, combine) | {"shuffles": shuffles}


def settled(
    result: dict[str, Any],
    reals: Mapping[str, tuple[float, float, int]],
    nulls: Mapping[str, np.ndarray],
    combine: str,
    need: float,
    horizons: bool = False,
) -> bool:
    """Whether more shuffles could still change ``result``'s decision against ``need``, or with ``horizons`` that of
    any counted horizon (``sequential.py``)."""
    counted = [r for r in result["horizons"] if r["counted"]]
    if not counted:
        return sequential.settled(result["z"], math.nan, need)
    real = np.array([reals[r["h"]][0] for r in counted])
    shuffled = np.column_stack([nulls[r["h"]] for r in counted])
    if not horizons:
        return sequential.settled(result["z"], sequential.standard_error(real, shuffled, combine), need)
    ses = sequential.standard_errors(real, shuffled, combine)
    return all(sequential.settled(r["adjusted"], se, need) for r, se in zip(counted, ses, strict=True))


def _combined(rows: list[dict[str, Any]], nulls: list[np.ndarray], min_blocks: int, combine: str) -> dict[str, Any]:
    counted = [r for r in rows if r["counted"]]
    for r in rows:
        r["adjusted"] = r["z"] if r["counted"] else math.nan
    if not counted:
        return {"z": math.nan, "best": None, "horizons": rows, "note": f"fewer than {min_blocks} blocks everywhere"}
    if combine == "each":
        worst = min(counted, key=lambda r: r["z"])
        return {"z": worst["z"], "best": worst["h"], "horizons": rows, "note": "weakest horizon"}
    best = max(counted, key=lambda r: r["z"])
    if len(counted) == 1:
        return {"z": best["z"], "best": best["h"], "horizons": rows, "note": "one horizon"}
    top = np.nanmax(np.vstack(nulls), axis=0)
    sd, mean = float(np.nanstd(top, ddof=1)), float(np.nanmean(top))
    for r in counted:
        r["adjusted"] = (r["z"] - mean) / sd if sd > 0 else math.nan
    return {"z": best["adjusted"], "best": best["h"], "horizons": rows, "note": f"best of {len(counted)} horizons"}


@dataclass(eq=False)
class _Consumer:
    """One request at one horizon: its real value, and its null value per seed as the lane draws them."""

    taken: _Taken
    h: str
    evaluate: Evaluate
    batched: bool
    real: tuple[float, float, int] = (math.nan, math.nan, 0)
    null: np.ndarray = field(default_factory=lambda: np.zeros(0))
    prepared: Prepared | None = None
    started: bool = False
    error: Exception | None = None

    def wants(self) -> tuple[int, int]:
        t = self.taken
        return (t.drawn, t.target) if not t.done and self.error is None else (0, 0)

    def start(self, lane: Lane, other: Callable[[], np.ndarray | None]) -> None:
        """The real value, and the fastest form that gives the eval's numbers: a batched one on a lane that shuffles
        within each decision time, else stacked, checked with ``other()``, another shuffle of the lane."""
        if self.started:
            return
        self.started = True
        self.null = np.full(self.taken.request.reps, np.nan)
        values = per_t(self.evaluate, lane.table)
        self.real = blocks(values, lane.h_ns)
        if not self.batched:
            return
        form = getattr(self.evaluate, "batched", None) if lane.within else None
        found = checked(form(lane.table), self.evaluate, lane.table, values) if form is not None else None
        self.prepared = found if found is not None else stacked(self.evaluate, lane.table, values, other())


def checked(prepared: Prepared, evaluate: Evaluate, table: pl.DataFrame, values: pl.DataFrame) -> Prepared | None:
    """A batched form, if it gives what ``evaluate`` gives (``values``, its ``per_t``) on the table as a shuffle
    leaves it (``with_scores``) with every score in place; otherwise nothing."""
    score = table["score"]
    if score.null_count():
        values = per_t(evaluate, with_scores(table, score.to_numpy()))
    found = prepared(np.arange(table.height)[None, :])[0]
    return prepared if _agrees(found, prepared.times, values) else None


class Stacked:
    """``evaluate`` on many shuffles at once, for any eval: one copy of the table per shuffle, holding that shuffle's
    scores, each copy later in time than the one before by whole weeks (so an hour, a weekday, and a gap between
    times stay what they are), stacked and evaluated in one call; each copy's values come back at their own times.
    ``with_scores`` gives the same on the table with other scores."""

    def __init__(self, evaluate: Evaluate, table: pl.DataFrame) -> None:
        t = table["t"].dt.epoch("ns").to_numpy()
        self.evaluate, self.base = evaluate, table.drop("score")
        self.score = table["score"].cast(pl.Float64).to_numpy()
        self.times = np.unique(t)
        self.dtype = table.schema["t"]
        self.start = int(t.min()) if t.size else 0
        self.step = ((int(t.max()) - self.start) // WEEK_NS + 2) * WEEK_NS if t.size else WEEK_NS
        room = (np.iinfo(np.int64).max - (int(t.max()) if t.size else 0)) // self.step
        self.per_call = int(max(1, min(STACKED_ROWS // max(table.height, 1), room)))

    def __call__(self, index: np.ndarray) -> np.ndarray:
        out = np.full((index.shape[0], self.times.size), np.nan)
        for first in range(0, index.shape[0], self.per_call):
            part = index[first : first + self.per_call]
            values = per_t(self.evaluate, self._stack(part))
            at = values["t"].dt.epoch("ns").to_numpy() - self.start
            copy, own = at // self.step, at % self.step + self.start
            pos = np.minimum(np.searchsorted(self.times, own), self.times.size - 1)
            kept = (copy >= 0) & (copy < part.shape[0]) & (self.times[pos] == own)
            out[first + copy[kept], pos[kept]] = values["value"].to_numpy()[kept]
        return out

    def with_scores(self, scores: np.ndarray) -> "Stacked":
        return Stacked(self.evaluate, with_scores(self.base, scores))

    def _stack(self, index: np.ndarray) -> pl.DataFrame:
        k, n = index.shape
        t = self.base["t"].dt.epoch("ns").to_numpy()
        moved = (t[None, :] + (np.arange(k, dtype=np.int64) * self.step)[:, None]).ravel()
        when = pl.Series("t", moved).cast(pl.Datetime("ns"))
        tz = getattr(self.dtype, "time_zone", None)
        when = when.dt.replace_time_zone(tz) if tz else when
        frame = pl.concat([self.base] * k) if k > 1 else self.base
        return frame.with_columns(when.cast(self.dtype), pl.Series("score", self.score[index.ravel()]))


def stacked(
    evaluate: Evaluate, table: pl.DataFrame, values: pl.DataFrame, other: np.ndarray | None = None
) -> Stacked | None:
    """``Stacked`` for ``evaluate``, if it gives what ``evaluate`` gives alone (``values``) on the table as it is, in a
    copy between two copies holding other scores (``other``, a shuffle; by default each decision time's rows reversed)
    and in the last copy of a stack, the farthest in time; otherwise nothing, and the eval runs shuffle by shuffle. An
    eval that reads across decision times, across the rows of other copies, or the calendar beyond the hour and the
    weekday gives other numbers there."""
    form = Stacked(evaluate, table)
    copies = min(form.per_call, CHECKED_COPIES)
    if table.height == 0 or copies < 2:
        return None
    other, same = _reversed(table) if other is None else other, np.arange(table.height)
    index = np.stack([other, same, *([other] * (copies - 3)), same][:copies])
    score = table["score"].cast(pl.Float64)
    alone = per_t(evaluate, with_scores(table, score.to_numpy())) if score.null_count() else values
    try:
        found = form(index)
    except Exception:  # noqa: BLE001
        return None
    return form if all(_agrees(found[i], form.times, alone) for i in {1, copies - 1}) else None


def _reversed(table: pl.DataFrame) -> np.ndarray:
    """A shuffle that reverses the rows of each decision time: row ``i`` takes the score of ``index[i]``."""
    t = table["t"].dt.epoch("ns").to_numpy()
    order = np.argsort(t, kind="stable")
    ts = t[order]
    first = np.concatenate([[True], ts[1:] != ts[:-1]])
    starts = np.flatnonzero(first)
    ends = np.append(starts[1:], ts.size)
    seg = np.cumsum(first) - 1
    partner = starts[seg] + ends[seg] - 1 - np.arange(ts.size)
    index = np.empty(ts.size, dtype=np.int64)
    index[order] = order[partner]
    return index


def _agrees(found: np.ndarray, times: np.ndarray, values: pl.DataFrame) -> bool:
    at = np.searchsorted(times, values["t"].dt.epoch("ns").to_numpy())
    if at.size and (at.max() >= times.size or not np.array_equal(times[at], values["t"].dt.epoch("ns"))):
        return False
    expect = np.full(found.shape, np.nan)
    expect[at] = values["value"].to_numpy()
    return bool(np.allclose(np.where(np.isfinite(found), found, np.nan), expect, rtol=1e-9, atol=1e-12, equal_nan=True))


@dataclass(eq=False)
class _Lane:
    """A lane and the consumers that read it; its shuffle is built once and kept while a consumer may want more."""

    lane: Lane
    consumers: list[_Consumer] = field(default_factory=list)
    index: Index | None = None

    def pending(self) -> bool:
        return any(lo < hi for lo, hi in (c.wants() for c in self.consumers))

    def waiting(self) -> bool:
        return any(not c.taken.done and c.error is None for c in self.consumers)

    def run(self) -> None:
        """Draws the seeds the lane's consumers want this round, a few at a time, and gives each its null values; a
        consumer that fails keeps its exception, and a shuffle that fails gives it to every consumer."""
        work = [(c, lo, hi) for c in self.consumers for lo, hi in [c.wants()] if lo < hi]
        for c, _, _ in work:
            try:
                c.start(self.lane, self._other)
            except Exception as exc:  # noqa: BLE001
                c.error = exc
        work = [w for w in work if w[0].error is None]
        try:
            self._draw(work)
        except Exception as exc:  # noqa: BLE001
            for c, _, _ in work:
                c.error = c.error or exc
        for c, _, _ in work:
            if c.taken.request.sequential is None:
                c.prepared = None
        if all(c.taken.request.sequential is None for c in self.consumers):
            self.index = None

    def _other(self) -> np.ndarray | None:
        """The shuffle the stacked check sets beside the table: by default each time's rows reversed, which a lane that
        moves scores across decision times would never draw, so there its first shuffle."""
        if self.lane.within:
            return None
        if self.index is None:
            self.index = self.lane.shuffle()
        return self.index(0)

    def _draw(self, work: list[tuple[_Consumer, int, int]]) -> None:
        if not work:
            return
        if self.index is None:
            self.index = self.lane.shuffle()
        table, h_ns = self.lane.table, self.lane.h_ns
        score = table["score"].to_numpy()
        first, last = min(lo for _, lo, _ in work), max(hi for _, _, hi in work)
        step = max(1, CHUNK // max(table.height, 1))
        for start in range(first, last, step):
            seeds = range(start, min(start + step, last))
            index = np.stack([self.index(s) for s in seeds])
            for c, lo, hi in work:
                a, b = max(lo, seeds.start), min(hi, seeds.stop)
                if c.error is not None or a >= b:
                    continue
                try:
                    c.null[a:b] = self._values(c, index[a - seeds.start : b - seeds.start], table, score, h_ns)
                except Exception as exc:  # noqa: BLE001
                    c.error = exc

    @staticmethod
    def _values(c: _Consumer, index: np.ndarray, table: pl.DataFrame, score: np.ndarray, h_ns: int) -> np.ndarray:
        if c.prepared is not None:
            return null_values(c.prepared, index, h_ns)
        return np.array([blocks(per_t(c.evaluate, with_scores(table, score[i])), h_ns)[0] for i in index])
