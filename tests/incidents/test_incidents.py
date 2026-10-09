"""Real incidents, frozen as small slices of the data they happened on. Each must keep being caught.

The slices come from a perpetual-futures source (four venues, 2026-04-08 to 09-14); the folders hold only what each
check reads.
"""

from pathlib import Path

import polars as pl

from research_ci.api.plugins import Reference
from research_ci.eval.data import reconcile as rc
from research_ci.eval.data.baseline import baseline
from research_ci.kernel.time.clock import TIME

HERE = Path(__file__).parent
TOL = 60 * 10**9


def _at(text: str) -> pl.Expr:
    return pl.lit(text).str.to_datetime(time_zone="UTC").cast(TIME)


def _requests(ours: pl.DataFrame, venue: pl.DataFrame) -> dict[str, set[str]]:
    _, anomalies = baseline(ours, ("close",), 5.0, 24)
    reqs = rc.requests(ours, anomalies, 10, 10, TOL, 0)
    ref = Reference(venue.select("entity", "field", "event_time", "value"), reqs.select("rid"))
    out: dict[str, set[str]] = {}
    for r in rc.compare(reqs, ours, ref, TOL):
        out.setdefault(r["kind"], set()).add(r["status"])
    return out


def test_candles_dropped_with_hindsight_around_a_jump_are_a_gap_the_venue_does_not_have() -> None:
    """A 2026-09-15 audit listed a 700-million-fold close jump on one venue on 07-06 between 00:00 and 06:00, and the
    first version of the source dropped those candles. As of 03:00 nobody knew where the jump would end. The venue
    still has the candles: the gap we made must not pass as faithful, while the venue's own gap before the jump, and
    the jump itself, are kept."""
    venue = pl.read_parquet(HERE / "hindsight-repair" / "venue.parquet")
    window = pl.col("event_time").is_between(_at("2026-07-06T00:00:00Z"), _at("2026-07-06T07:00:00Z"))
    ours = venue.filter(~window)
    assert _requests(ours, venue) == {"sample": {"match"}, "gap": {"mismatch"}, "shift": {"match"}}
    assert _requests(venue, venue) == {"sample": {"match"}, "gap": {"match"}, "shift": {"match"}}


def test_a_ticker_reused_for_a_new_contract_leaves_a_gap_until_it_is_split() -> None:
    """One venue relaunched a delisted symbol's ticker for a different asset on 2026-07-14; its funding history from
    April is the old contract's. Under one name the series has a three-month hole; split at the relaunch, neither
    contract has one."""
    funding = pl.read_parquet(HERE / "ticker-reuse" / "funding.parquet")
    _, merged = baseline(funding)
    gaps = merged.filter(pl.col("kind") == "gap")
    assert gaps.height == 1 and gaps["start"][0].month == 4 and gaps["end"][0].month == 7
    later = pl.col("event_time") >= _at("2026-07-14T06:00:00Z")
    split = funding.with_columns(entity=pl.when(later).then(pl.col("entity") + "#2").otherwise("entity"))
    assert baseline(split)[1].filter(pl.col("kind") == "gap").is_empty()
