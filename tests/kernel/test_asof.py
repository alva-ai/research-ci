import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from research_ci.kernel.tables import SchemaError, source_table, source_tables
from research_ci.kernel.time.asof import asof, known, wide, wide_schema
from research_ci.kernel.time.clock import from_ns, to_ns

T0 = pd.Timestamp("2026-01-01", tz="UTC")
H = pd.Timedelta(hours=1)
COLUMNS = ["inst", "event_time", "available_at", "vintage", "x"]


def _table(rows: list[tuple], columns: list[str] = COLUMNS) -> pl.DataFrame:
    return source_table("s", pd.DataFrame(rows, columns=columns))


def _cell(frame: pl.DataFrame, at: pd.Timestamp, inst: str, column: str) -> float:
    row = frame.filter((pl.col("event_time").dt.epoch("ns") == to_ns(at)) & (pl.col("inst") == inst))
    return row[column].item()


def test_publication_lag_hides_the_row_until_available() -> None:
    table = _table([("a", T0, T0 + 30 * H, 0, 7.0)])
    assert asof(table, T0 + 29 * H).is_empty()
    assert _cell(asof(table, T0 + 30 * H), T0, "a", "x") == 7.0


def test_revision_is_used_only_after_it_is_published() -> None:
    table = _table([("a", T0, T0, 0, 7.0), ("a", T0, T0 + 10 * H, 1, 9.0)])
    assert _cell(asof(table, T0 + 9 * H), T0, "a", "x") == 7.0
    assert _cell(asof(table, T0 + 10 * H), T0, "a", "x") == 9.0


def test_a_source_keeps_its_columns_and_casts_them_to_floats() -> None:
    table = _table([("a", T0, T0, 0, 1, 2)], [*COLUMNS, "y"])
    assert table.columns == COLUMNS + ["y"] and table.schema["x"] == pl.Float64
    assert asof(table, T0).columns == ["inst", "event_time", "x", "y"]


def test_missing_column_is_rejected() -> None:
    with pytest.raises(SchemaError, match="missing columns"):
        source_table("s", pd.DataFrame({"inst": ["a"], "event_time": [T0], "available_at": [T0], "x": [1.0]}))


def test_a_source_needs_value_columns_that_are_numbers() -> None:
    with pytest.raises(SchemaError, match="no value columns"):
        _table([("a", T0, T0, 0)], COLUMNS[:4])
    with pytest.raises(SchemaError, match="x is not"):
        _table([("a", T0, T0, 0, "high")])


def test_names_wide_gives_are_reserved() -> None:
    with pytest.raises(SchemaError, match="rename age_s"):
        _table([("a", T0, T0, 0, 1.0)], [*COLUMNS[:4], "age_s"])
    with pytest.raises(SchemaError, match="a source's name"):
        source_table("my source", _table([("a", T0, T0, 0, 1.0)]))


def test_a_value_column_is_held_by_one_source() -> None:
    table = _table([("a", T0, T0, 0, 1.0)])
    with pytest.raises(SchemaError, match="x is in more than one"):
        source_tables({"one": table, "two": table})
    assert list(source_tables({"two": table, "one": table.rename({"x": "y"})})) == ["one", "two"]


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(SchemaError):
        _table([("a", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01"), 0, 1.0)])


def test_duplicate_key_is_rejected() -> None:
    with pytest.raises(SchemaError):
        _table([("a", T0, T0, 0, 1.0), ("a", T0, T0 + H, 0, 2.0)])


def test_revision_published_before_the_original_is_rejected() -> None:
    with pytest.raises(SchemaError):
        _table([("a", T0, T0 + 5 * H, 0, 1.0), ("a", T0, T0 + H, 1, 2.0)])


def test_non_utc_input_is_normalized_to_utc() -> None:
    local = pd.Timestamp("2026-01-01 08:00", tz="Asia/Shanghai")
    table = _table([("a", local, local, 0, 1.0)])
    assert table.schema["event_time"] == pl.Datetime("ns", "UTC")
    assert table["event_time"].dt.epoch("ns").item() == to_ns(T0)


def _hand() -> dict[str, pl.DataFrame]:
    """Prices of a every hour, known on the hour; a's 1h price revised at 5h; b's first price at 3h. Flows of a
    settled at 2h and 6h, each known two hours later."""
    prices = [("a", T0 + h * H, T0 + h * H, 0, 100.0 + h) for h in range(8)]
    prices += [("a", T0 + H, T0 + 5 * H, 1, 500.0), ("b", T0 + 3 * H, T0 + 3 * H, 0, 50.0)]
    flows = [("a", T0 + 2 * H, T0 + 4 * H, 0, 0.01), ("a", T0 + 6 * H, T0 + 8 * H, 0, 0.02)]
    return source_tables(
        {
            "prices": pd.DataFrame(prices, columns=[*COLUMNS[:4], "price"]),
            "flows": pd.DataFrame(flows, columns=[*COLUMNS[:4], "flow"]),
        }
    )


def _at(hours: list[int]) -> list[int]:
    return [to_ns(T0 + h * H) for h in hours]


def test_wide_takes_the_latest_event_known_and_how_old_it_is() -> None:
    w = wide(_hand(), _at([2, 3, 4, 5, 8]))
    assert w.columns == ["t", "inst", "flow", "price", "age_flows", "age_prices"]
    got = w.select(
        pl.col("t").dt.hour(), "inst", "flow", "price", pl.col("age_flows").dt.total_hours(), "age_prices"
    ).rows()
    zero = pd.Timedelta(0).to_pytimedelta()
    assert got == [
        (2, "a", None, 102.0, None, zero),
        (3, "a", None, 103.0, None, zero),
        (3, "b", None, 50.0, None, zero),
        (4, "a", 0.01, 104.0, 0, zero),
        (4, "b", None, 50.0, None, H.to_pytimedelta()),
        (5, "a", 0.01, 105.0, 1, zero),
        (5, "b", None, 50.0, None, 2 * H.to_pytimedelta()),
        (8, "a", 0.02, 107.0, 0, H.to_pytimedelta()),
        (8, "b", None, 50.0, None, 5 * H.to_pytimedelta()),
    ]


def test_a_late_revision_of_an_older_event_is_not_the_latest_event() -> None:
    """At 5h the revision of the 1h price becomes known: a join on when rows became known would take it, but the
    latest event known then is the 5h price."""
    w = wide(_hand(), _at([5]))
    assert w.filter(pl.col("inst") == "a")["price"].item() == 105.0
    by_known = _hand()["prices"].sort("available_at", "vintage").filter(pl.col("available_at") <= T0 + 5 * H)
    assert by_known.filter(pl.col("inst") == "a")["price"][-1] == 500.0


def test_wide_on_one_time_is_the_last_event_of_the_as_of_snapshot() -> None:
    sources = _hand()
    for h in range(9):
        t = to_ns(T0 + h * H)
        row = wide(sources, [t])
        for name, table in sources.items():
            column = "price" if name == "prices" else "flow"
            snapshot = asof(table, t).group_by("inst").agg(pl.col(column).sort_by("event_time").last())
            got = row.select("inst", column).drop_nulls(column)
            assert dict(got.rows()) == dict(snapshot.rows()), (h, name)


def test_wide_has_its_columns_with_nothing_known() -> None:
    assert dict(wide(_hand(), _at([-1])).schema) == wide_schema(_hand())
    assert wide(_hand(), []).is_empty()


def test_known_cuts_every_source_at_a_time() -> None:
    cut = known(_hand(), to_ns(T0 + 3 * H))
    assert {name: t.height for name, t in cut.items()} == {"flows": 0, "prices": 5}


rows_strategy = st.lists(
    st.tuples(
        st.sampled_from(["a", "b"]),
        st.integers(0, 20),
        st.integers(0, 10),
        st.floats(-1e6, 1e6, allow_nan=False),
    ),
    min_size=1,
    max_size=40,
)


def _random_table(rows: list[tuple]) -> tuple[list[tuple], pl.DataFrame]:
    table_rows = []
    seen: dict[tuple, int] = {}
    for inst, event, lag, value in rows:
        vintage = seen.get((inst, event), -1) + 1
        seen[(inst, event)] = vintage
        table_rows.append((inst, T0 + event * H, T0 + (event + lag + 11 * vintage) * H, vintage, value))
    return table_rows, _table(table_rows)


@settings(max_examples=200, deadline=None)
@given(rows=rows_strategy, t=st.integers(0, 40))
def test_asof_matches_brute_force(rows: list[tuple], t: int) -> None:
    table_rows, table = _random_table(rows)
    cutoff = T0 + t * H
    expected: dict[tuple, float] = {}
    best: dict[tuple, int] = {}
    for inst, event_time, available_at, vintage, value in table_rows:
        if available_at <= cutoff and vintage >= best.get((inst, event_time), -1):
            best[(inst, event_time)] = vintage
            expected[(to_ns(event_time), inst)] = value
    found = asof(table, cutoff)
    actual = {
        (ns, inst): value
        for ns, inst, value in zip(found["event_time"].dt.epoch("ns").to_list(), found["inst"], found["x"], strict=True)
    }
    assert actual == expected


@settings(max_examples=200, deadline=None)
@given(rows=rows_strategy, ts=st.lists(st.integers(0, 40), min_size=1, max_size=5))
def test_wide_matches_brute_force(rows: list[tuple], ts: list[int]) -> None:
    """At each time, an inst's row is its latest event known by then, at the latest vintage known by then, and its
    age is the time since that vintage became known."""
    table_rows, table = _random_table(rows)
    expected = {}
    for t in sorted(set(ts)):
        at = T0 + t * H
        for inst in ("a", "b"):
            mine = [r for r in table_rows if r[0] == inst and r[2] <= at]
            if not mine:
                continue
            event = max(r[1] for r in mine)
            row = max((r for r in mine if r[1] == event), key=lambda r: r[3])
            expected[(to_ns(at), inst)] = (row[4], to_ns(at) - to_ns(row[2]))
    found = wide({"s": table}, [to_ns(T0 + t * H) for t in ts])
    actual = {
        (int(t), inst): (x, int(age))
        for t, inst, x, age in found.select(
            pl.col("t").dt.epoch("ns"), "inst", "x", pl.col("age_s").dt.total_nanoseconds()
        ).rows()
    }
    assert actual == expected


@settings(max_examples=100, deadline=None)
@given(rows=rows_strategy, t=st.integers(0, 40), seed=st.integers(0, 2**16))
def test_asof_does_not_depend_on_row_order(rows: list[tuple], t: int, seed: int) -> None:
    _, table = _random_table(rows)
    shuffled = table.sample(fraction=1.0, shuffle=True, seed=seed)
    assert asof(shuffled, T0 + t * H).equals(asof(table, T0 + t * H))
    assert wide({"s": shuffled}, [to_ns(T0 + t * H)]).equals(wide({"s": table}, [to_ns(T0 + t * H)]))


@settings(max_examples=200, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(["a", "b", "c"]),
            st.integers(0, 4),
            st.integers(0, 2),
            st.integers(0, 6),
        ),
        min_size=1,
        max_size=30,
    )
)
def test_duplicates_and_early_revisions_are_found_as_a_check_by_key_finds_them(rows: list[tuple]) -> None:
    """The key checks run on one sort of coded keys; brute force over the rows says what they must find."""
    frame = pl.DataFrame(
        [(e, T0 + event * H, T0 + at * H, v, 1.0) for e, event, v, at in rows],
        schema=COLUMNS,
        orient="row",
    )
    keys = [(e, event, v) for e, event, v, _ in rows]
    duplicated = sum(keys.count(k) > 1 for k in keys)
    early = any(a[:2] == b[:2] and a[2] < b[2] and a[3] > b[3] for a in rows for b in rows if keys.count(a[:3]) == 1)
    if duplicated:
        with pytest.raises(SchemaError, match=f"{duplicated} duplicate rows"):
            source_table("s", frame)
    elif early:
        with pytest.raises(SchemaError, match="before the vintage it revises"):
            source_table("s", frame)
    else:
        assert source_table("s", frame).height == len(rows)


def test_from_ns_round_trips() -> None:
    assert from_ns(to_ns(T0 + H)) == T0 + H
