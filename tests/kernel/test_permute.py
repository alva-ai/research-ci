from collections import Counter

import numpy as np
import polars as pl

from research_ci.kernel.time.clock import TIME
from research_ci.kernel.verdict.permute import Permutation

HOUR = 3_600 * 10**9


def _table() -> pl.DataFrame:
    """Four decision times; symbols A, B, C with three pairs each and D, E with one; pairs in venue buckets x or y."""
    rows = []
    for h in range(4):
        for sym, pairs in (("A", 3), ("B", 3), ("C", 3), ("D", 1), ("E", 1)):
            for k in range(pairs):
                rows.append((h * HOUR, f"{sym}:{k}", sym, "x" if k == 0 else "y"))
    frame = pl.DataFrame(rows, schema=["t", "inst", "symbol", "venues"], orient="row")
    return frame.with_columns(pl.col("t").cast(TIME))


def test_a_shuffle_is_a_bijection_that_keeps_time_and_bucket() -> None:
    table = _table()
    idx = Permutation(table, "venues", "symbol", 2 * HOUR).index(7)
    assert sorted(idx.tolist()) == list(range(table.height))
    for col in ("t", "venues"):
        assert (table[col].to_numpy() == table[col].to_numpy()[idx]).all()


def test_whole_clusters_move_to_clusters_of_the_same_size() -> None:
    table = _table()
    for seed in range(20):
        idx = Permutation(table, None, "symbol", HOUR).index(seed)
        moved = table.with_columns(source=table["symbol"].to_numpy()[idx])
        per = moved.group_by("t", "symbol").agg(pl.col("source").n_unique().alias("k"), pl.col("source").first())
        assert per["k"].to_list() == [1] * per.height
        big = per.filter(pl.col("symbol").is_in(["A", "B", "C"]))["source"]
        small = per.filter(pl.col("symbol").is_in(["D", "E"]))["source"]
        assert set(big) <= {"A", "B", "C"} and set(small) <= {"D", "E"}


def test_one_shuffle_holds_for_every_decision_time_of_a_block_and_changes_between_blocks() -> None:
    table = _table()
    perm = Permutation(table, None, "symbol", 2 * HOUR)
    changes = 0
    for seed in range(30):
        moved = table.with_columns(source=table["symbol"].to_numpy()[perm.index(seed)])
        maps = moved.group_by("t", "symbol").agg(pl.col("source").first()).sort("t", "symbol")
        by_t = {t: tuple(g["source"]) for (t,), g in maps.group_by("t", maintain_order=True)}
        times = sorted(by_t)
        assert by_t[times[0]] == by_t[times[1]] and by_t[times[2]] == by_t[times[3]]
        changes += by_t[times[0]] != by_t[times[2]]
    assert changes > 15


def test_shuffles_are_uniform_over_clusters() -> None:
    table = _table()
    perm = Permutation(table, None, "symbol", HOUR)
    first = table.filter(pl.col("t") == pl.col("t").min()).with_row_index("row")
    rows = first.filter(pl.col("inst") == "A:0")["row"].to_list()
    counts: Counter[str] = Counter()
    for seed in range(600):
        idx = perm.index(seed)
        counts[table["symbol"][int(idx[rows[0]])]] += 1
    assert set(counts) == {"A", "B", "C"}
    assert all(abs(n / 600 - 1 / 3) < 0.06 for n in counts.values())
    assert np.isclose(sum(counts.values()), 600)


def test_rows_inside_a_moved_cluster_are_shuffled_too_and_held_within_a_block() -> None:
    """Clusters laid out alike (pair k of every symbol) must not keep their position: that would keep what the
    position tells about the label."""
    table = _table()
    perm = Permutation(table, None, "symbol", 2 * HOUR)
    same, total = 0, 0
    for seed in range(200):
        idx = perm.index(seed)
        source = table["inst"].to_numpy()[idx]
        for i, inst in enumerate(table["inst"].to_list()):
            if inst.startswith(("A", "B", "C")):
                same += inst.split(":")[1] == source[i].split(":")[1]
                total += 1
        moved = table.with_columns(source=pl.Series(source))
        first = moved.filter(pl.col("t") == pl.col("t").min()).sort("inst")["source"].to_list()
        second = moved.filter(pl.col("t") == pl.col("t").min() + pl.duration(hours=1)).sort("inst")["source"]
        assert first == second.to_list()
    assert abs(same / total - 1 / 3) < 0.05


def _sorted_by_hash(
    table: pl.DataFrame, bucket: str | None, cluster: str | None, block_ns: int, seed: int
) -> np.ndarray:
    """The shuffle as first written: every instance and every row sorted by its hash, with ``np.lexsort``."""
    key = lambda c: pl.lit("") if c is None else pl.col(c).cast(pl.String).fill_null("")  # noqa: E731
    frame = table.select(
        pl.int_range(table.height, dtype=pl.Int64).alias("row"),
        pl.col("t").dt.epoch("ns").alias("t"),
        key(bucket).alias("b"),
        (key(cluster) if cluster else pl.col("inst").cast(pl.String)).alias("c"),
        pl.col("inst").cast(pl.String),
    ).sort("t", "b", "c", "inst")
    new = (pl.col("t") != pl.col("t").shift(1)) | (pl.col("b") != pl.col("b").shift(1))
    frame = frame.with_columns(inst=(new | (pl.col("c") != pl.col("c").shift(1))).fill_null(True).cum_sum() - 1)
    inst = frame.group_by("inst", maintain_order=True).agg(
        pl.col("t").first(), pl.col("b").first(), pl.col("c").first(), pl.len().alias("size")
    )
    block = ((pl.col("t") - inst["t"].min()) // block_ns).alias("block")
    rows, of = frame["row"].to_numpy(), frame["inst"].to_numpy()
    start = np.concatenate([[0], np.cumsum(inst["size"].to_numpy())[:-1]]).astype(np.int64)
    group = inst.select(pl.struct("t", "b", "size").rank("dense"))["t"].to_numpy()
    order = np.lexsort((inst.select(pl.col("c").rank("dense"))["c"].to_numpy(), group))
    hashed = inst.select(block, "b", "c").select(pl.struct("block", "b", "c").hash(seed))["block"].to_numpy()
    source = np.empty_like(order)
    source[order] = np.lexsort((hashed, group))
    inner = frame.select(block, "b", "c", "inst").select(pl.struct(pl.all()).hash(seed + 1))["block"].to_numpy()
    by = np.lexsort((inner, of))
    offset = np.empty(table.height, dtype=np.int64)
    offset[by] = np.arange(table.height) - start[of[by]]
    out = np.empty(table.height, dtype=np.int64)
    out[rows] = rows[start[source[of]] + offset]
    return out


def test_a_shuffle_is_the_one_sorting_every_row_by_its_hash_gives() -> None:
    """The layout computed once gives, seed by seed, exactly the shuffle of sorting every instance and row by hash."""
    rng = np.random.default_rng(0)
    for trial in range(8):
        sizes = rng.integers(1, 5, int(rng.integers(2, 30)))
        rows = [
            (h * HOUR, f"s{c}:{k}", f"s{c}", f"v{rng.integers(0, 3)}")
            for h in range(int(rng.integers(3, 25)))
            for c, n in enumerate(sizes)
            for k in range(n)
            if rng.random() > 0.2
        ]
        frame = pl.DataFrame(rows, schema=["t", "inst", "symbol", "venues"], orient="row", strict=False)
        table = frame.with_columns(pl.col("t").cast(TIME)).sample(fraction=1.0, shuffle=True, seed=trial)
        for bucket, cluster, block in ((None, "symbol", HOUR), ("venues", "symbol", 4 * HOUR), (None, None, HOUR)):
            perm = Permutation(table, bucket, cluster, block)
            for seed in range(4):
                expected = _sorted_by_hash(table, bucket, cluster, block, seed)
                assert np.array_equal(perm.index(seed), expected), (trial, bucket, cluster, block, seed)
