"""Null worlds as the engine runs them: how a world shuffles the scores of each horizon's evaluation table, and the
floor world every eval is judged in."""

from dataclasses import dataclass
from functools import partial

import polars as pl

from research_ci.api.contract import FLOOR_ALTERNATIVES, span
from research_ci.kernel.time.clock import span_ns
from research_ci.kernel.verdict.judge import Index, Lane
from research_ci.kernel.verdict.permute import Layout, Permutation


@dataclass(frozen=True)
class World:
    """A null world as the engine runs it: scores shuffled within ``bucket`` (``None``: each decision time), whole
    ``clusters`` at a time, the same way within ``block``; it claims to keep what lives in ``keeps``. It judges the
    evals named in ``evals``, or every eval when empty."""

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

    def block_ns(self, h_ns: int) -> int:
        if self.block is None:
            return 1
        return h_ns if self.block == "horizon" else span_ns(span(self.block))

    def describe(self) -> str:
        return f"PermuteWithin({self.bucket or 'time'}, clusters={self.clusters}, block={self.block})"


def lanes(tables: dict[str, tuple[pl.DataFrame, int]], w: World) -> dict[str, Lane]:
    """The world's shuffle of each horizon's table; two evals or worlds on the same table and layout share it."""
    out = {}
    for h, (table, h_ns) in tables.items():
        layout = Layout(w.bucket, w.clusters, w.block_ns(h_ns))
        out[h] = Lane(table, h_ns, (id(table), layout), partial(_shuffle, table, layout))
    return out


def floor_world(clusters: str, reps: int) -> World:
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
