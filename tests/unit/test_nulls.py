from collections import Counter

import numpy as np
import polars as pl

from research_ci.kernel.judge.nulls import Permutation
from research_ci.kernel.time.clock import TIME

HOUR = 3_600 * 10**9


def _table() -> pl.DataFrame:
    """Four decision times; symbols A, B, C with three pairs each and D, E with one; pairs in venue buckets x or y."""
    rows = []
    for h in range(4):
        for sym, pairs in (("A", 3), ("B", 3), ("C", 3), ("D", 1), ("E", 1)):
            for k in range(pairs):
                rows.append((h * HOUR, f"{sym}:{k}", sym, "x" if k == 0 else "y"))
    frame = pl.DataFrame(rows, schema=["t", "entity", "symbol", "venues"], orient="row")
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
    rows = first.filter(pl.col("entity") == "A:0")["row"].to_list()
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
        source = table["entity"].to_numpy()[idx]
        for i, entity in enumerate(table["entity"].to_list()):
            if entity.startswith(("A", "B", "C")):
                same += entity.split(":")[1] == source[i].split(":")[1]
                total += 1
        moved = table.with_columns(source=pl.Series(source))
        first = moved.filter(pl.col("t") == pl.col("t").min()).sort("entity")["source"].to_list()
        second = moved.filter(pl.col("t") == pl.col("t").min() + pl.duration(hours=1)).sort("entity")["source"]
        assert first == second.to_list()
    assert abs(same / total - 1 / 3) < 0.05
