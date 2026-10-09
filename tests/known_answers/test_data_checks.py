"""Every data check must pass on a correct source and fail on one broken in the way it claims to catch; every
built-in pit must change the data where it lands."""

import numpy as np
import polars as pl
import pytest

from research_ci.api.contract import Use
from research_ci.kernel.checks import invariants as inv
from research_ci.kernel.checks.baseline import baseline
from research_ci.kernel.checks.pits import BUILTIN
from research_ci.kernel.tables import field_table
from research_ci.stages.source import admit
from toy import ToyCorrupt, ToyData, ToyHindsight, ToyNoAsOf, ToyPairs, ToySwapped, toy_fields


def _admit(plugin) -> dict:
    return admit(Use("toy.data"), plugin, field_table(plugin.fetch()))


def _checks(spec: dict) -> dict[str, str]:
    return {c["name"]: c["status"] for c in spec["checks"]}


def test_a_correct_source_is_admissible_and_its_canaries_bite() -> None:
    spec = _admit(ToyData())
    checks = _checks(spec)
    assert checks["replay as of past times"] == "pass"
    assert checks["independent source agrees"] == "pass"
    assert checks["canary: replay catches a hindsight deletion"] == "pass"
    assert checks["canary: reconciliation catches altered values"] == "pass"
    assert spec["grades"] == {"close": "verified", "funding_rate": "verified", "funding_interval_hours": "unverified"}
    assert spec["verdict"]["status"] == "admissible"


@pytest.mark.parametrize(
    ("plugin", "failed"),
    [
        (ToyHindsight(), {"replay as of past times", "independent source agrees"}),
        (ToyNoAsOf(), {"replay as of past times"}),
        (ToyCorrupt(), {"independent source agrees"}),
    ],
)
def test_a_broken_source_is_not_admitted(plugin, failed: set[str]) -> None:
    spec = _admit(plugin)
    assert {name for name, status in _checks(spec).items() if status == "fail"} == failed
    assert spec["verdict"]["status"] == "not admitted"


def test_a_hole_made_with_hindsight_is_found_by_the_cut_points_alone() -> None:
    spec = _admit(ToyHindsight())
    holes = [r for r in spec["replay"] if r["why"].startswith("hole")]
    assert holes and all(r["status"] == "fail" and r["differ"] == ["b:Y"] for r in holes)


def test_hedged_legs_flag_an_asset_listed_under_another_ticker() -> None:
    for plugin, expect_bad in ((ToyData(), False), (ToySwapped(), True)):
        fields = field_table(plugin.fetch())
        times = fields.filter(pl.col("field") == "close")["available_at"].unique().sort().to_list()[::6]
        cands = ToyPairs().candidates(fields, times)
        h = ToyPairs.hedge
        legs = inv.hedged_legs(cands, fields, h.price, h.max_ratio, h.min_corr, 24, 12, h.min_move, keep=("symbol",))
        bad = legs.filter(~pl.col("ok"))
        assert (bad.height > 0) is expect_bad
        if expect_bad:
            assert set(bad["symbol"]) == {"Z"}
            assert set(bad["pair"].str.split("|").list.last()) | set(bad["pair"].str.split("|").list.first()) >= {"c:Z"}


@pytest.mark.parametrize("name", sorted(BUILTIN))
def test_every_builtin_pit_changes_the_field_it_targets(name: str) -> None:
    fields = toy_fields(hours=24 * 6)
    planted = field_table(BUILTIN[name](fields, "close", np.random.default_rng(0)))
    key = ["entity", "event_time", "value"]
    a = fields.filter(pl.col("field") == "close").select(key).sort(key)
    b = planted.filter(pl.col("field") == "close").select(key).sort(key)
    assert not a.equals(b)


def test_baseline_flags_gaps_near_duplicates_and_level_shifts() -> None:
    fields = toy_fields(hours=24 * 6)
    rng = np.random.default_rng(1)
    broken = field_table(BUILTIN["gap"](BUILTIN["duplicate"](fields, "close", rng), "close", rng))
    later = pl.col("event_time") > fields["event_time"].min() + pl.duration(hours=72, time_unit="ns")
    hit = (pl.col("entity") == "a:X") & (pl.col("field") == "close") & later
    shifted = field_table(fields.with_columns(value=pl.when(hit).then(pl.col("value") * 10).otherwise("value")))
    levels = (("close",), 5.0, 24)
    clean_checks, clean = baseline(fields, *levels)
    _, found = baseline(broken, *levels)
    _, moved = baseline(shifted, *levels)
    assert clean.is_empty() and all(c.status == "pass" for c in clean_checks)
    assert {"gap", "near"} <= set(found["kind"])
    assert moved.filter(pl.col("kind") == "shift")["entity"].to_list() == ["a:X"]
    undeclared_checks, undeclared = baseline(shifted)
    assert undeclared.filter(pl.col("kind") == "shift").is_empty()
    assert {c.name: c.status for c in undeclared_checks}["no level shifts"] == "skip"
