"""Null worlds on the evaluation table. A world says, for every row, which row's score it takes; the labels stay.

``PermuteWithin`` shuffles scores across the rows of each decision time and bucket. Whole clusters move: a cluster's
rows take the scores of another cluster with as many rows at that time, in random order, so what a cluster's scores
share survives while their link to its members' labels does not. Both shuffles are uniformly random and the same at
every decision time of a block, so overlapping label windows see one shuffle.
"""

import numpy as np
import polars as pl


class Permutation:
    """``PermuteWithin`` on one table: the layout is computed once, and each seed gives one shuffle."""

    def __init__(self, table: pl.DataFrame, bucket: str | None, cluster: str | None, block_ns: int) -> None:
        if block_ns <= 0:
            raise ValueError("a block must be positive")
        n = table.height
        frame = table.select(
            pl.int_range(n, dtype=pl.Int64).alias("row"),
            pl.col("t").dt.epoch("ns").alias("t"),
            _key(bucket).alias("b"),
            (_key(cluster) if cluster else pl.col("entity").cast(pl.String)).alias("c"),
            pl.col("entity").cast(pl.String),
        ).sort("t", "b", "c", "entity")
        new = (pl.col("t") != pl.col("t").shift(1)) | (pl.col("b") != pl.col("b").shift(1))
        new = new | (pl.col("c") != pl.col("c").shift(1))
        frame = frame.with_columns(inst=new.fill_null(True).cum_sum() - 1)
        inst = frame.group_by("inst", maintain_order=True).agg(
            pl.col("t").first(), pl.col("b").first(), pl.col("c").first(), pl.len().alias("size")
        )
        t0 = int(inst["t"].min()) if inst.height else 0  # type: ignore[arg-type]
        block = (pl.col("t") - t0) // block_ns
        self.members = frame.select(block.alias("block"), "b", "c", "entity")
        self.rows = frame["row"].to_numpy()
        self.inst_of_pos = frame["inst"].to_numpy()
        size = inst["size"].to_numpy()
        self.start = np.concatenate([[0], np.cumsum(size)[:-1]]).astype(np.int64)
        group = inst.select(pl.struct("t", "b", "size").rank("dense").alias("g"))["g"].to_numpy()
        cluster_rank = inst.select(pl.col("c").rank("dense").alias("r"))["r"].to_numpy()
        self.group = group
        self.order = np.lexsort((cluster_rank, group))
        self.keys = inst.select(block.alias("block"), "b", "c")
        self.n = n

    def index(self, seed: int) -> np.ndarray:
        """Row ``i`` takes the score of row ``index[i]``."""
        if self.n == 0:
            return np.zeros(0, dtype=np.int64)
        key = self.keys.select(pl.struct("block", "b", "c").hash(seed).alias("k"))["k"].to_numpy()
        shuffled = np.lexsort((key, self.group))
        source = np.empty_like(self.order)
        source[self.order] = shuffled
        inst = self.inst_of_pos
        inner = self.members.select(pl.struct("block", "b", "c", "entity").hash(seed + 1).alias("k"))["k"].to_numpy()
        by = np.lexsort((inner, inst))
        offset = np.empty(self.n, dtype=np.int64)
        offset[by] = np.arange(self.n, dtype=np.int64) - self.start[inst[by]]
        out = np.empty(self.n, dtype=np.int64)
        out[self.rows] = self.rows[self.start[source[inst]] + offset]
        return out


def _key(column: str | None) -> pl.Expr:
    if column is None or column == "time":
        return pl.lit("")
    return pl.col(column).cast(pl.String).fill_null("")
