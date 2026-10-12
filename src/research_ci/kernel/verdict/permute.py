"""Null worlds on the evaluation table. A world says, for every row, which row's score it takes; the labels stay.

``PermuteWithin`` shuffles scores across the rows of each decision time and bucket. Whole clusters move: a cluster's
rows take the scores of another cluster with as many rows at that time, in random order, so what a cluster's scores
share survives while their link to its members' labels does not. Both shuffles are uniformly random and the same at
every decision time of a block, so overlapping label windows see one shuffle.

The sort layout is computed once per table: which clusters can trade places (a group: the same time, bucket and size),
which rows make up each cluster, and the distinct keys a seed draws for. A seed then hashes those keys, orders each
group and each cluster by them, and gathers.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Layout:
    """How a world shuffles one table: within ``bucket`` (``None``: each decision time), whole ``clusters`` at a
    time (``None``: each inst alone), the same way within blocks of ``block_ns``."""

    bucket: str | None
    clusters: str | None
    block_ns: int


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
            (_key(cluster) if cluster else pl.col("inst").cast(pl.String)).alias("c"),
            pl.col("inst").cast(pl.String),
        ).sort("t", "b", "c", "inst")
        new = (pl.col("t") != pl.col("t").shift(1)) | (pl.col("b") != pl.col("b").shift(1))
        new = new | (pl.col("c") != pl.col("c").shift(1))
        frame = frame.with_columns(seg=new.fill_null(True).cum_sum() - 1)
        seg = frame.group_by("seg", maintain_order=True).agg(
            pl.col("t").first(), pl.col("b").first(), pl.col("c").first(), pl.len().alias("size")
        )
        t0 = int(seg["t"].min()) if seg.height else 0  # type: ignore[arg-type]
        block = ((pl.col("t") - t0) // block_ns).alias("block")
        self.n = n
        self.rows = frame["row"].to_numpy()
        self.seg_of_pos = frame["seg"].to_numpy()
        size = seg["size"].to_numpy()
        self.start = np.concatenate([[0], np.cumsum(size)[:-1]]).astype(np.int64)
        codes = seg.select(
            block,
            pl.col("b").rank("dense").cast(pl.Int64).alias("b"),
            pl.col("c").rank("dense").cast(pl.Int64).alias("c"),
            pl.struct("t", "b", "size").rank("dense").cast(pl.Int64).alias("g"),
        )
        order = np.lexsort((codes["c"].to_numpy(), codes["g"].to_numpy()))
        keys = seg.select(block, "b", "c")
        trading = _classes(codes["g"].to_numpy()[order], order)
        self.groups, self.key_frame, self.key_ids = _drawn(trading, keys, codes.select("block", "b", "c"))
        members = self.seg_of_pos
        sized = _classes(members, np.arange(n, dtype=np.int64))
        member_codes = frame.select(
            block, pl.col("b").rank("dense"), pl.col("c").rank("dense"), pl.col("inst").rank("dense")
        ).cast(pl.Int64)
        member_keys = frame.select(block, "b", "c", "inst")
        self.clusters, self.member_frame, self.member_ids = _drawn(sized, member_keys, member_codes)

    def index(self, seed: int) -> np.ndarray:
        """Row ``i`` takes the score of row ``index[i]``."""
        if self.n == 0:
            return np.zeros(0, dtype=np.int64)
        source = np.arange(self.start.size, dtype=np.int64)
        drawn = _hashed(self.key_frame, seed)
        for members, ids in zip(self.groups, self.key_ids, strict=True):
            order = np.argsort(drawn[ids], axis=1, kind="stable")
            source[members] = np.take_along_axis(members, order, axis=1)
        offset = np.zeros(self.n, dtype=np.int64)
        drawn = _hashed(self.member_frame, seed + 1)
        for positions, ids in zip(self.clusters, self.member_ids, strict=True):
            order = np.argsort(drawn[ids], axis=1, kind="stable")
            ranks = np.broadcast_to(np.arange(positions.shape[1], dtype=np.int64), positions.shape)
            offset[np.take_along_axis(positions, order, axis=1)] = ranks
        out = np.empty(self.n, dtype=np.int64)
        out[self.rows] = self.rows[self.start[source[self.seg_of_pos]] + offset]
        return out


def _key(column: str | None) -> pl.Expr:
    if column is None or column == "time":
        return pl.lit("")
    return pl.col(column).cast(pl.String).fill_null("")


def _classes(labels: np.ndarray, items: np.ndarray) -> list[np.ndarray]:
    """``items`` (laid out so that equal ``labels`` are contiguous) as one matrix per run length of two or more: each
    row is one run."""
    if not labels.size:
        return []
    starts = np.flatnonzero(np.concatenate([[True], labels[1:] != labels[:-1]]))
    lengths = np.diff(np.concatenate([starts, [labels.size]]))
    out = []
    for length in np.unique(lengths[lengths >= 2]).tolist():
        first = starts[lengths == length]
        out.append(items[first[:, None] + np.arange(length)])
    return out


def _drawn(
    classes: list[np.ndarray], keys: pl.DataFrame, codes: pl.DataFrame
) -> tuple[list[np.ndarray], pl.DataFrame, list[np.ndarray]]:
    """The distinct ``keys`` of the items in ``classes`` (a seed hashes only these), and each item's key in them."""
    if not classes:
        return [], keys.clear(), []
    involved = np.unique(np.concatenate([c.ravel() for c in classes]))
    found = codes[involved].to_numpy()
    _, first, inverse = np.unique(found, axis=0, return_index=True, return_inverse=True)
    lookup = np.full(keys.height, -1, dtype=np.int64)
    lookup[involved] = inverse.ravel()
    return classes, keys[involved[first]], [lookup[c] for c in classes]


def _hashed(keys: pl.DataFrame, seed: int) -> np.ndarray:
    return keys.select(pl.struct(keys.columns).hash(seed).alias("k"))["k"].to_numpy()
