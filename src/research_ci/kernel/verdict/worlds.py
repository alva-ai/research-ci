"""Null worlds as the engine runs them: how a world shuffles the scores of each horizon's evaluation table, and the
floor world every eval is judged in, which the universe's shape sets."""

from dataclasses import dataclass
from functools import partial

import polars as pl

from research_ci.api.contract import FLOOR_ALTERNATIVES, span
from research_ci.kernel.time.clock import span_ns
from research_ci.kernel.verdict.judge import Index, Lane
from research_ci.kernel.verdict.permute import Layout, Permutation
from research_ci.kernel.verdict.shift import Shift


@dataclass(frozen=True)
class World:
    """A null world as the engine runs it: scores shuffled within ``bucket`` (``None``: each decision time), whole
    ``clusters`` at a time, the same way within ``block``; or, with ``shift``, each inst's scores shifted in time by
    at least that much (``ShiftInTime``). It claims to keep what lives in ``keeps``. It judges the evals named in
    ``evals``, or every eval when empty."""

    name: str
    explains: tuple[str, ...]
    bucket: str | None
    clusters: str | None
    block: str | None
    keeps: str | None
    evals: tuple[str, ...]
    reps: int
    by: str
    note: str = ""
    shift: str | None = None

    def block_ns(self, h_ns: int) -> int:
        if self.block is None:
            return 1
        return h_ns if self.block == "horizon" else span_ns(span(self.block))

    def shift_ns(self, h_ns: int) -> int:
        assert self.shift is not None
        return h_ns if self.shift == "horizon" else span_ns(span(self.shift))

    def describe(self) -> str:
        if self.shift is not None:
            return f"ShiftInTime(min_shift={self.shift})"
        return f"PermuteWithin({self.bucket or 'time'}, clusters={self.clusters}, block={self.block})"


def lanes(tables: dict[str, tuple[pl.DataFrame, int]], w: World) -> dict[str, Lane]:
    """The world's shuffle of each horizon's table; two evals or worlds on the same table and layout share it. A
    shift in time moves scores across decision times and has as many shuffles as offsets."""
    out = {}
    for h, (table, h_ns) in tables.items():
        if w.shift is not None:
            shift = Shift(table, w.shift_ns(h_ns))
            key = (id(table), "shift", w.shift_ns(h_ns))
            out[h] = Lane(table, h_ns, key, partial(_index, shift), within=False, distinct=shift.distinct)
            continue
        layout = Layout(w.bucket, w.clusters, w.block_ns(h_ns))
        out[h] = Lane(table, h_ns, (id(table), layout), partial(_shuffle, table, layout))
    return out


def shuffler(table: pl.DataFrame, w: World, h_ns: int) -> tuple[Index, int | None]:
    """The world's shuffle of one table, built now, and how many distinct shuffles it has (``None``: no fewer than
    any request draws)."""
    if w.shift is not None:
        shift = Shift(table, w.shift_ns(h_ns))
        return shift.index, shift.distinct
    return Permutation(table, w.bucket, w.clusters, w.block_ns(h_ns)).index, None


def floor_world(clusters: str, reps: int, shape: str = "cross") -> World:
    """The engine's world for the universe's shape: scores shuffled across clusters at each decision time
    (``cross``), or each inst's scores shifted in time by at least the horizon (``series`` and ``events``)."""
    if shape != "cross":
        return World(
            name="floor",
            explains=FLOOR_ALTERNATIVES[:1],
            bucket=None,
            clusters=None,
            block=None,
            keeps="inst",
            evals=(),
            reps=reps,
            by="engine",
            note="each inst's scores shifted in time by at least the horizon, the same offset for every inst",
            shift="horizon",
        )
    return World(
        name="floor",
        explains=FLOOR_ALTERNATIVES,
        bucket=None,
        clusters=clusters,
        block="horizon",
        keeps=None,
        evals=(),
        reps=reps,
        by="engine",
        note="scores shuffled across clusters at each decision time, the same way within a horizon",
    )


def _shuffle(table: pl.DataFrame, layout: Layout) -> Index:
    return Permutation(table, layout.bucket, layout.clusters, layout.block_ns).index


def _index(shift: Shift) -> Index:
    return shift.index
