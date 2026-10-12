"""The checks the data layer runs on every idea's data by default, like feat's diagnostics: one list, to adjust from
experience.

Admitting the source (replay as of past times, the independent source, the canaries, the field grades) is not here:
it checks fidelity, that the rows are what the venue reported, and the data layer runs it first. The checks in
``CHECKS`` come after it, in order, and check fitness, whether faithful rows can be used for research: a ticker reused
for another asset, a price at zero or stuck, an inst too short for the horizons; the last only shows how much of
the noise rests on one cluster. They read only the FieldTable, what the plugins declare (``levels``, ``hedge``) and
the contract; none needs reference data or knows an asset class. Each is a step (``stages/skeleton.py``) that reads the
data check's context (``Seen`` in ``stages/data/floor.py``) and returns lines for the DataCard, warnings, treatments,
and items a person must decide. The data layer applies the treatments by default and weighs each by its influence on
the decisive number; a check sees the data with the treatments before it applied (``Seen.prior``). This list is slot 2
of the data layer, with the library evals an idea lists; the idea's lab evals (``EVALS`` in ``data.py``, slot 3) run
after these, once validated; the built-in pits of the categories in ``PITS``, with those of an eval's own module
(``PITS`` next to its ``CASES``), test the ones that claim to cover them. Tools that need knowledge of a case, such as
adjusting stock prices for splits or rolling futures, are not defaults: they are evals an idea lists on purpose, from
the engine's library or its own lab evals.

The engine hash leaves ``stages/data/defaults.py`` out: the data freeze pins the treated data by content, so a changed
list cannot change a frozen result; it shows at the next ``rci data check``.
"""

from collections.abc import Sequence
from typing import Any

import polars as pl

from research_ci.api.contract import span
from research_ci.api.data import Hedge
from research_ci.eval.data import concentration as cc
from research_ci.eval.data import coverage as cv
from research_ci.eval.data import identity as idn
from research_ci.eval.data import invariants as inv
from research_ci.eval.data import sanity as sn
from research_ci.eval.data.baseline import level_shift_anomalies
from research_ci.kernel.time.clock import from_ns, span_ns
from research_ci.kernel.verdict.check import Check
from research_ci.kernel.verdict.power import DAY_NS
from research_ci.stages.data import treatments as tr
from research_ci.stages.data.floor import Seen
from research_ci.stages.data.source import declared_levels
from research_ci.stages.skeleton import Found, Step

STALE_AFTER = "2 days"
IDENTITY_GAP = 3.0
STALE_RUN = 24.0
MIN_CANDIDATES = 5
CONCENTRATION = 0.5
CONCENTRATION_SHOWN = 3
PITS = ("unit", "level_break", "gap", "duplicate", "shift", "leak", "spike", "swap")
KEYS = ["inst", "field", "event_time"]


def fresh(seen: Seen) -> Found:
    """Data whose newest row is older than ``STALE_AFTER`` is a warning: collect live data first."""
    return freshness(seen.latest, seen.now)


def freshness(latest: int, now: int) -> Found:
    """Data whose newest row is older than ``STALE_AFTER`` is a warning: collect live data first. Derive warns too."""
    age = (now - latest) / DAY_NS
    stale = age > span(STALE_AFTER).total_seconds() / 86_400
    found = Found([Check("data is current", "warn" if stale else "pass", f"latest data is {age:.1f} days old")])
    if stale:
        found.warnings.append(f"the data source has nothing newer than {_iso(latest)}; collect live data first")
    return found


def level_shifts(seen: Seen) -> Found:
    """Lasting jumps in the fields the source declares as levels (``Levels``): those the independent source did not
    confirm at admission are masked for the source's ``mask`` on each side; confirmed ones are kept."""
    found = Found()
    shifts = level_shift_anomalies(seen.fields, *declared_levels(seen.source))
    unconfirmed = shifts.join(_confirmed(seen.admission), on=["inst", "field", "end"], how="anti")
    if unconfirmed.height:
        around = span_ns(span(seen.source.levels.mask))
        reason = f"{unconfirmed.height} level shifts the independent source did not confirm"
        mask = tr.mask_windows("mask unconfirmed level shifts", reason, unconfirmed, around, seen.fields)
        found.treatments.append(mask)
    if shifts.height > unconfirmed.height:
        kept = f"{shifts.height - unconfirmed.height} confirmed by the source; kept"
        found.checks.append(Check("level shifts", "pass", kept))
    return found


def level_sanity(seen: Seen) -> Found:
    """In the declared levels: values at zero or below; the repeats in a run of one value longer than ``STALE_RUN``
    usual spacings; and moves of the declared factor or more that do not last, which the source did not confirm. Each
    kind is a mask of the rows it flags; rows a treatment before it already removes are left to that one."""
    names, factor, window = declared_levels(seen.source)
    if not names:
        return Found([Check("level sanity", "skip", "the source declares no level fields")])
    kinds = [
        ("at zero or below", "mask levels at zero or below"),
        (f"repeated in a run longer than {STALE_RUN:g} usual spacings", "mask stale level runs"),
        (f"{factor:g}x away from the level before for a moment", "mask level spikes"),
    ]
    flagged = pl.concat(
        [
            sn.nonpositive(seen.fields, names).with_columns(kind=pl.lit(0)),
            sn.stale(seen.fields, names, STALE_RUN).with_columns(kind=pl.lit(1)),
            _unconfirmed(sn.spikes(seen.fields, names, factor, window), seen.admission).with_columns(kind=pl.lit(2)),
        ]
    )
    near = seen.fields.join(flagged.select(KEYS).unique(), on=KEYS, how="semi")
    flagged = _still_there(flagged, near, seen.prior)
    per_kind = flagged.partition_by("kind", as_dict=True, include_key=False)
    found = Found()
    found.treatments += [
        tr.mask_rows(name, f"{per_kind[(k,)].height:,} events {what}", per_kind[(k,)], near)
        for k, (what, name) in enumerate(kinds)
        if (k,) in per_kind
    ]
    counts = [f"{per_kind[(k,)].height:,} {what}" for k, (what, _) in enumerate(kinds) if (k,) in per_kind]
    if counts:
        found.checks.append(Check("level sanity", "warn", f"masked: {' · '.join(counts)}"))
    else:
        found.checks.append(Check("level sanity", "pass", "no declared level at zero or below, stale, or spiking"))
    return found


def identity_breaks(seen: Seen) -> Found:
    """A declared level that comes back after a gap longer than ``IDENTITY_GAP`` usual spacings at the declared factor
    or more away from where it was is another asset under the same name: the inst is split there, and its rows from
    the break on become ``<inst>#2`` (``#3`` after a second break)."""
    names, factor, window = declared_levels(seen.source)
    name = "identity breaks"
    if not names:
        return Found([Check(name, "skip", "the source declares no level fields")])
    found = idn.breaks(seen.fields, names, factor, window, IDENTITY_GAP)
    if found.is_empty():
        return Found([Check(name, "pass", f"no declared level comes back {factor:g}x away after a long gap")])
    splits = idn.renames(found, seen.fields["inst"].unique())
    shown = found.head(6).select("inst", _ns("at"), "ratio").rows()
    where = ", ".join(f"{e} {_iso(int(at))} (x{r:.3g})" for e, at, r in shown)
    more = f" and {found.height - 6} more" if found.height > 6 else ""
    insts = found["inst"].n_unique()
    detail = f"{_count(insts, 'inst comes', 'insts come')} back at another level after a long gap: "
    detail += where + more
    reason = f"{_count(found.height, 'break', 'breaks')}; the rows from each are a new inst"
    split = tr.split_insts("split insts at identity breaks", reason, splits, seen.fields)
    return Found([Check(name, "warn", detail + "; split")], treatments=[split])


def coverage(seen: Seen) -> Found:
    """Insts whose whole history is shorter than the contract's longest horizon are dropped; decision times with
    fewer than ``MIN_CANDIDATES`` candidates are only counted, for a cross shape, which ranks them."""
    horizons = [span(h) for label in seen.contract.labels for h in label.horizons]
    if not horizons:
        return Found([Check("coverage", "skip", "the contract has no labels")])
    longest = max(horizons)
    treated = tr.apply_all(seen.prior, seen.fields.select("inst", "field", "event_time"))
    short = cv.short_lived(treated, span_ns(longest))
    found, notes = Found(), []
    if short.height:
        found.treatments.append(
            tr.drop_spans(
                "drop insts too short for the horizon",
                f"{_count(short.height, 'inst', 'insts')} with less than {_days(longest)} of data",
                _origins(short, seen.fields),
                seen.fields,
            )
        )
        names = short["inst"].to_list()
        more = f" and {len(names) - 6} more" if len(names) > 6 else ""
        notes.append(
            f"{_count(len(names), 'inst', 'insts')} shorter than {_days(longest)}, dropped: "
            f"{', '.join(names[:6])}{more}"
        )
    u = seen.contract.universe
    if seen.candidates is not None and (u is None or u.shape == "cross"):
        thin = cv.thin(seen.candidates, MIN_CANDIDATES)
        if thin.height:
            times = seen.candidates["t"].n_unique()
            notes.append(f"{thin.height:,} of {times:,} decision times have fewer than {MIN_CANDIDATES} candidates")
    if notes:
        found.checks.append(Check("coverage", "warn", "; ".join(notes)))
    else:
        found.checks.append(Check("coverage", "pass", f"every inst covers {_days(longest)}"))
    return found


def hedge_guard(seen: Seen) -> Found:
    """Legs the universe declares as one asset on two venues (``Hedge``): at each decision time, the candidate rows
    whose legs do not trade at about the same level or do not move together are dropped."""
    hedge = getattr(seen.universe, "hedge", None)
    if hedge is not None and not isinstance(hedge, Hedge):
        raise TypeError("a universe plugin's hedge must be an research_ci.api.Hedge")
    cands, u = seen.candidates, seen.contract.universe
    if cands is None or u is None or hedge is None or not {"long", "short"} <= set(cands.columns):
        return Found()
    keep = (u.clusters,) if u.clusters in cands.columns else ()
    h = hedge
    legs = inv.hedged_legs(cands, seen.fields, h.price, h.max_ratio, h.min_corr, h.window, h.min_obs, h.min_move, keep)
    found = Found([inv.hedge_check(legs, h.price, h.max_ratio, h.min_corr, keep[0] if keep else "pair")])
    bad = legs.filter(~pl.col("ok"))
    if bad.height:
        reason = f"{bad.height:,} candidate rows whose legs do not trade as one asset"
        found.treatments.append(tr.drop_candidates("hedge guard at each decision time", reason, bad))
    return found


def concentration(seen: Seen) -> Found:
    """How much of the noise the decisive number's MDE rests on comes from each cluster of the universe, on the data
    as treated so far: the clusters that carry the most are shown, with a warning when one of several carries more than
    ``CONCENTRATION``. It is information, never a treatment or an item to decide: broken data is what the identity and
    sanity checks catch, and an effect can be concentrated too."""
    d, u = seen.decisive, seen.contract.universe
    if d is None or u is None:
        return Found()
    found = cc.shares(d.rows(seen.prior, [u.clusters, "value"]), u.clusters, "h_ns", by=["key"])
    top = found.group_by("key", maintain_order=True).head(CONCENTRATION_SHOWN)
    shown = [[str(k), [list(x) for x in zip(c, v, strict=True)]] for k, c, v in _listed(top).rows()]
    out = Found(shown={"concentration": {"clusters": u.clusters, "shares": shown, "threshold": CONCENTRATION}})
    name = "influence concentration"
    where = pl.format("{}% at {}", (pl.col("share") * 100).round(0).cast(pl.Int64), pl.col("key"))
    heavy = found.filter((pl.col("share") > CONCENTRATION) & (pl.len().over("key") > 1))
    heavy = heavy.group_by("cluster", maintain_order=True)
    heavy = heavy.agg(where.str.join(" and ").alias("where"))
    if heavy.is_empty():
        out.checks.append(Check(name, "pass", f"no {u.clusters} carries more than {CONCENTRATION:.0%} of the noise"))
        return out
    listed = heavy.head(6).select(pl.format("{} {}", "cluster", "where")).to_series().to_list()
    more = f" and {heavy.height - 6} more" if heavy.height > 6 else ""
    detail = f"more than {CONCENTRATION:.0%} of the noise from one {u.clusters}: {', '.join(listed)}{more}"
    out.checks.append(Check(name, "warn", detail + "; shown, not treated"))
    return out


CHECKS = [
    Step("data is current", "warns", fresh, "Warns when the newest row is too old"),
    Step("level shifts", "treats", level_shifts, "Masks level jumps the source did not confirm", "levels"),
    Step("level sanity", "treats", level_sanity, "Masks levels at zero, stuck, or spiking", "levels"),
    Step("identity breaks", "treats", identity_breaks, "Splits an inst that returns at another level", "levels"),
    Step("coverage", "treats", coverage, "Drops insts shorter than the longest horizon"),
    Step("hedge guard", "treats", hedge_guard, "Drops candidates whose legs are not one asset", "hedge"),
    Step("influence concentration", "warns", concentration, "Shows the clusters most of the noise rests on"),
]


def _confirmed(admission: dict[str, Any]) -> pl.DataFrame:
    """Level shifts the independent source confirmed at admission: ``inst``, ``field``, ``end``. Admission records
    the end of the window it asked about, the reconciliation tolerance after the shift."""
    rows = admission.get("confirmed_shifts", [])
    tol = span_ns(span(admission.get("settings", {}).get("tolerance", "0s")))
    frame = pl.DataFrame(rows, schema={"inst": pl.String, "field": pl.String, "end": pl.String}, orient="row")
    end = pl.col("end").str.to_datetime(time_zone="UTC").cast(pl.Datetime("ns", "UTC"))
    return frame.with_columns(end - pl.duration(nanoseconds=tol, time_unit="ns"))


def _unconfirmed(flagged: pl.DataFrame, admission: dict[str, Any]) -> pl.DataFrame:
    confirmed = _confirmed(admission).rename({"end": "event_time"})
    return flagged.join(confirmed, on=KEYS, how="anti")


def _still_there(flagged: pl.DataFrame, near: pl.DataFrame, prior: Sequence[tr.Treatment]) -> pl.DataFrame:
    """The flagged rows the treatments before this check leave in the data; ``near`` holds the rows of the data the
    flags name, and a treatment decides about each row by itself."""
    if flagged.is_empty() or not any(t.on == "fields" for t in prior):
        return flagged
    left = tr.apply_all(prior, near).select(KEYS).unique()
    return flagged.join(left, on=KEYS, how="semi", maintain_order="left")


def _origins(short: pl.DataFrame, fields: pl.DataFrame) -> pl.DataFrame:
    """Spans to drop for short insts; an inst a split made also covers the rows it came from, so the drop does the
    same with or without the split."""
    raw = fields.select("inst").unique()
    base = pl.col("inst").str.replace(r"#\d+$", "").alias("_base")
    made = short.join(raw, on="inst", how="anti").with_columns(base)
    came = made.join(raw.with_columns(base), on="_base", suffix="_raw").select(
        pl.col("inst_raw").alias("inst"), "start", "end"
    )
    return pl.concat([short, came.select(short.columns)]).cast(short.schema)  # type: ignore[arg-type]


def _listed(found: pl.DataFrame) -> pl.DataFrame:
    return found.group_by("key", maintain_order=True).agg("cluster", "share")


def _ns(column: str) -> pl.Expr:
    return pl.col(column).dt.epoch("ns")


def _iso(ns: int) -> str:
    return from_ns(ns).strftime("%Y-%m-%dT%H:%MZ")


def _count(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def _days(delta: Any) -> str:
    hours = delta.total_seconds() / 3600
    return f"{hours / 24:g} days" if hours % 24 == 0 and hours >= 48 else f"{hours:g}h"
