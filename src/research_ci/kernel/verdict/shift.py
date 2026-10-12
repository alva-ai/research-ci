"""``ShiftInTime`` on the evaluation table: each inst's scores move along its own decision times, circularly, by one
offset per seed, the same for every inst; the outcomes stay.

An offset counts decision times: the ``k``-th of an inst's rows, in time order, takes the score of its row ``k``
places earlier, the first ones those of its last rows. Positions ``m`` or more apart are at least ``m`` of the table's
shortest spacing between decision times apart, so an offset of at least ``m = ceil(min_shift / spacing)`` and at most
the shortest inst's row count less ``m`` keeps every score at least ``min_shift`` away from its own row, either way
round. A seed picks one of those offsets from a fixed random order of them, so the seeds below their count draw each
once.
"""

import math

import numpy as np
import polars as pl

ORDER_SEED = 0


class Shift:
    """``ShiftInTime`` on one table: the layout is computed once, and each seed gives one shuffle."""

    def __init__(self, table: pl.DataFrame, min_ns: int) -> None:
        if min_ns <= 0:
            raise ValueError("a shift must be positive")
        frame = table.select(
            pl.int_range(table.height, dtype=pl.Int64).alias("row"),
            pl.col("t").dt.epoch("ns").alias("t"),
            pl.col("inst").cast(pl.String),
        ).sort("inst", "t")
        frame = frame.with_columns(
            pos=pl.int_range(pl.len(), dtype=pl.Int64).over("inst"), size=pl.len().over("inst").cast(pl.Int64)
        )
        times = np.unique(frame["t"].to_numpy())
        gaps = np.diff(times)
        spacing = int(gaps.min()) if gaps.size else 0
        self.min_steps = max(1, math.ceil(min_ns / spacing)) if spacing else 1
        self.length = int(frame["size"].min()) if frame.height else 0  # type: ignore[arg-type]
        self.rows = frame["row"].to_numpy()
        self.pos = frame["pos"].to_numpy()
        self.size = frame["size"].to_numpy()
        self.first = np.arange(frame.height, dtype=np.int64) - self.pos
        allowed = np.arange(self.min_steps, self.length - self.min_steps + 1, dtype=np.int64)
        self.offsets = np.random.default_rng(ORDER_SEED).permutation(allowed)

    @property
    def distinct(self) -> int:
        """How many distinct shuffles there are: one per allowed offset."""
        return int(self.offsets.size)

    def offset(self, seed: int) -> int:
        if not self.distinct:
            raise ValueError(
                f"no shift in time fits: an offset of at least {self.min_steps} decision times each way needs every "
                f"inst to have {2 * self.min_steps} or more, and the shortest has {self.length}"
            )
        return int(self.offsets[seed % self.distinct])

    def index(self, seed: int) -> np.ndarray:
        """Row ``i`` takes the score of row ``index[i]``."""
        out = np.empty(self.rows.size, dtype=np.int64)
        if not self.rows.size:
            return out
        k = self.offset(seed)
        source = self.first + (self.pos - k) % self.size
        out[self.rows] = self.rows[source]
        return out
