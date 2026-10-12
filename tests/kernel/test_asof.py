import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from research_ci.kernel.tables import SchemaError, field_table
from research_ci.kernel.time.asof import asof
from research_ci.kernel.time.clock import from_ns, to_ns

T0 = pd.Timestamp("2026-01-01", tz="UTC")
H = pd.Timedelta(hours=1)
COLUMNS = ["inst", "field", "event_time", "available_at", "value", "vintage"]


def _table(rows: list[tuple]) -> pl.DataFrame:
    return field_table(pd.DataFrame(rows, columns=COLUMNS))


def _cell(frame: pl.DataFrame, at: pd.Timestamp, inst: str, column: str) -> float:
    row = frame.filter((pl.col("time").dt.epoch("ns") == to_ns(at)) & (pl.col("inst") == inst))
    return row[column].item()


def test_publication_lag_hides_the_row_until_available() -> None:
    fields = _table([("a", "inv", T0, T0 + 30 * H, 7.0, 0)])
    assert asof(fields, T0 + 29 * H).is_empty()
    assert _cell(asof(fields, T0 + 30 * H), T0, "a", "inv") == 7.0


def test_revision_is_used_only_after_it_is_published() -> None:
    fields = _table([("a", "inv", T0, T0, 7.0, 0), ("a", "inv", T0, T0 + 10 * H, 9.0, 1)])
    assert _cell(asof(fields, T0 + 9 * H), T0, "a", "inv") == 7.0
    assert _cell(asof(fields, T0 + 10 * H), T0, "a", "inv") == 9.0


def test_lookback_keeps_only_recent_event_times() -> None:
    fields = _table([("a", "x", T0 + i * H, T0 + i * H, float(i), 0) for i in range(10)])
    panel = asof(fields, T0 + 9 * H, lookback=3 * H)
    assert [from_ns(ns) for ns in panel["time"].dt.epoch("ns").to_list()] == [T0 + 7 * H, T0 + 8 * H, T0 + 9 * H]


def test_panel_keeps_every_field_as_a_column() -> None:
    fields = _table([("a", "x", T0, T0, 1.0, 0), ("a", "y", T0 + 5 * H, T0 + 5 * H, 2.0, 0)])
    assert asof(fields, T0).columns == ["time", "inst", "x", "y"]
    assert asof(fields, T0 - H).columns == ["time", "inst", "x", "y"]


def test_missing_column_is_rejected() -> None:
    with pytest.raises(SchemaError):
        field_table(pd.DataFrame({"inst": ["a"], "field": ["x"], "event_time": [T0], "value": [1.0]}))


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(SchemaError):
        _table([("a", "x", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01"), 1.0, 0)])


def test_duplicate_key_is_rejected() -> None:
    with pytest.raises(SchemaError):
        _table([("a", "x", T0, T0, 1.0, 0), ("a", "x", T0, T0 + H, 2.0, 0)])


def test_revision_published_before_the_original_is_rejected() -> None:
    with pytest.raises(SchemaError):
        _table([("a", "x", T0, T0 + 5 * H, 1.0, 0), ("a", "x", T0, T0 + H, 2.0, 1)])


def test_non_utc_input_is_normalized_to_utc() -> None:
    local = pd.Timestamp("2026-01-01 08:00", tz="Asia/Shanghai")
    fields = _table([("a", "x", local, local, 1.0, 0)])
    assert fields.schema["event_time"] == pl.Datetime("ns", "UTC")
    assert fields["event_time"].dt.epoch("ns").item() == to_ns(T0)


rows_strategy = st.lists(
    st.tuples(
        st.sampled_from(["a", "b"]),
        st.sampled_from(["x", "y"]),
        st.integers(0, 20),
        st.integers(0, 10),
        st.floats(-1e6, 1e6, allow_nan=False),
    ),
    min_size=1,
    max_size=40,
)


@settings(max_examples=200, deadline=None)
@given(rows=rows_strategy, t=st.integers(0, 40))
def test_asof_matches_brute_force(rows: list[tuple], t: int) -> None:
    table_rows = []
    seen: dict[tuple, int] = {}
    for inst, field, event, lag, value in rows:
        key = (inst, field, event)
        vintage = seen.get(key, -1) + 1
        seen[key] = vintage
        table_rows.append((inst, field, T0 + event * H, T0 + (event + lag + 11 * vintage) * H, value, vintage))
    fields = _table(table_rows)
    cutoff = T0 + t * H

    expected: dict[tuple, float] = {}
    best: dict[tuple, int] = {}
    for inst, field, event_time, available_at, value, vintage in table_rows:
        if available_at <= cutoff and vintage >= best.get((inst, field, event_time), -1):
            best[(inst, field, event_time)] = vintage
            expected[(to_ns(event_time), inst, field)] = value

    long = asof(fields, cutoff).unpivot(index=["time", "inst"]).drop_nulls("value")
    actual = {
        (ns, inst, field): value
        for ns, inst, field, value in zip(
            long["time"].dt.epoch("ns").to_list(), long["inst"], long["variable"], long["value"], strict=True
        )
    }
    assert actual == expected


@settings(max_examples=100, deadline=None)
@given(rows=rows_strategy, t=st.integers(0, 40), seed=st.integers(0, 2**16))
def test_asof_does_not_depend_on_row_order(rows: list[tuple], t: int, seed: int) -> None:
    table_rows, seen = [], {}
    for inst, field, event, lag, value in rows:
        vintage = seen.get((inst, field, event), -1) + 1
        seen[(inst, field, event)] = vintage
        table_rows.append((inst, field, T0 + event * H, T0 + (event + lag + 11 * vintage) * H, value, vintage))
    fields = _table(table_rows)
    shuffled = fields.sample(fraction=1.0, shuffle=True, seed=seed)
    assert not shuffled["available_at"].flags["SORTED_ASC"] or shuffled.equals(fields)
    assert asof(shuffled, T0 + t * H).equals(asof(fields, T0 + t * H))


@settings(max_examples=200, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(["a", "b", "c"]),
            st.sampled_from(["x", "y"]),
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
        [(e, f, T0 + event * H, T0 + at * H, 1.0, v, "") for e, f, event, v, at in rows],
        schema=["inst", "field", "event_time", "available_at", "value", "vintage", "source"],
        orient="row",
    )
    keys = [(e, f, event, v) for e, f, event, v, _ in rows]
    duplicated = sum(keys.count(k) > 1 for k in keys)
    early = any(a[:3] == b[:3] and a[3] < b[3] and a[4] > b[4] for a in rows for b in rows if keys.count(a[:4]) == 1)
    if duplicated:
        with pytest.raises(SchemaError, match=f"^{duplicated} duplicate rows"):
            field_table(frame)
    elif early:
        with pytest.raises(SchemaError, match="before the vintage it revises"):
            field_table(frame)
    else:
        assert field_table(frame).height == len(rows)
