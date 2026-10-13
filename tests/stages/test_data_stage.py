from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from research_ci.api.contract import Use, span
from research_ci.api.evals import Source
from research_ci.eval import synthetic
from research_ci.eval.data import coverage as cv
from research_ci.eval.synthetic import HOUR_NS, START_NS
from research_ci.kernel.time.clock import span_ns, to_ns
from research_ci.kernel.verdict import power as pw
from research_ci.render.common import card_id
from research_ci.runner import plugins
from research_ci.runner.research import ResearchError
from research_ci.stages.contract.run import Settings
from research_ci.stages.data import defaults as data_defaults
from research_ci.stages.data import run as data_stage
from research_ci.stages.data import treatments as tr
from research_ci.stages.data.floor import Seen
from research_ci.stages.data.run import DataSettings
from research_ci.stages.skeleton import Found, Step
from research_ci.store import vault
from research_ci.workflow.base import Approval
from research_ci.workflow.idea import Idea
from research_ci.workflow.machine import State, WorkflowError
from toy import (
    DATA_EVALS,
    REGISTRY,
    START,
    ToyData,
    ToyHindsight,
    ToyPairs,
    ToyRelaunched,
    ToyRenamed,
    ToyShort,
    ToySwapped,
    ToyZeros,
    carry_contract,
    lab,
)

FAST = Settings(workers=2, universe_cuts=2, outcome_cuts=1)
RELAYED = Approval(by="human", channel="relayed", words="ok")

LEGS = '''
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from research_ci.api import Case, Pit


def drifting_legs(sources):
    """Venues of one symbol whose hourly returns do not move with the symbol's other venues."""
    close = sources["candles"].sort("inst", "event_time")
    r = close.with_columns(r=(pl.col("close") / pl.col("close").shift(1).over("inst")).log())
    r = r.with_columns(symbol=pl.col("inst").str.split(":").list.last())
    mean = r.group_by("symbol", "event_time").agg(pl.col("r").mean().alias("m"))
    r = r.join(mean, on=["symbol", "event_time"])
    corr = r.group_by("inst").agg(pl.corr("r", "m").alias("c"))
    bad = corr.filter(pl.col("c") < 0.8)["inst"]
    flagged = close.filter(pl.col("inst").is_in(bad.implode()))
    return flagged.select(pl.lit("candles").alias("source"), "inst", "event_time")


def nothing(sources):
    return sources["candles"].head(0).select(pl.lit("candles").alias("source"), "inst", "event_time")


def _venues(apart):
    walks = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, (2, 48)), axis=1))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = pl.datetime_range(start, start + timedelta(hours=47), "1h", eager=True)
    closes = {"a:X": walks[0], "b:X": walks[0], "c:X": walks[1] if apart else walks[0]}
    return {"candles": pl.concat([pl.DataFrame({"inst": e, "event_time": t, "close": v}) for e, v in closes.items()])}


def _wander(sources, column, rng):
    """One venue's closes walking off on their own: the same moves, in another order."""
    candles = sources["candles"]
    inst = sorted(candles["inst"].unique().to_list())[0]
    hit = pl.col("inst") == inst
    found = candles.filter(hit).sort("event_time")
    moves = rng.permutation(np.diff(np.log(found[column].to_numpy())))
    walk = found[column][0] * np.exp(np.concatenate([[0.0], np.cumsum(moves)]))
    return {**sources, "candles": pl.concat([candles.filter(~hit), found.with_columns(pl.Series(column, walk))])}


CASES = [
    Case("a venue moving apart is flagged", lambda: drifting_legs(_venues(True))["inst"].unique().to_list(), ["c:X"]),
    Case("venues moving together are not", drifting_legs, 0, given=_venues(False)),
    Case("nothing flags nothing", nothing, 0, given=_venues(True)),
]
PITS = [Pit("wander", _wander, "one venue walks off on its own")]
'''
SWAP_TEST = """
from research_ci.api import Eval
from evals.legs import drifting_legs, nothing

EVALS = [
    Eval(drifting_legs, name="drifting legs", h0="every venue of a symbol moves with the others",
         covers=[("swap", "close")], on_fail="drop_insts"),
    {lazy}
]
"""
LAZY = 'Eval(nothing, name="lazy", h0="nothing is wrong", covers=[("swap", "close")]),'


def _evals(idea: "Idea", data: str, legs: str = LEGS) -> None:
    (idea.root / "evals").mkdir(exist_ok=True)
    (idea.root / "evals" / "legs.py").write_text(legs)
    (idea.root / "data.py").write_text(data)


def _relayed(idea: Idea, stage: str) -> Approval:
    return Approval(by="human", channel="relayed", words="ok", card=card_id(idea.card(stage)))


def _idea(tmp_path: Path, settings: DataSettings | None = None) -> Idea:
    idea = Idea.init(tmp_path / "idea", settings=FAST, data_settings=settings)
    (idea.root / "contract.py").write_text(carry_contract())
    idea.derive_contract()
    idea.approve("contract")
    lab(idea.root)
    return idea


def _checked(tmp_path: Path, monkeypatch, plugin: type, keep: tuple[str, ...] = ()) -> tuple[Idea, dict, str]:
    """The data check of the toy contract on ``plugin``."""
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": plugin})
    idea = _idea(tmp_path)
    spec, card = idea.check_data(keep=keep)
    return idea, spec, card


def _items(spec: dict) -> list[str]:
    return [d.split(":")[0] for d in spec["decide"]]


def _treatment(spec: dict, name: str) -> dict:
    return next(t for t in spec["treatments"] if t["name"] == name)


def test_the_data_layer_runs_from_admission_to_a_passed_verdict(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    spec, card = idea.check_data()
    assert spec["source"]["status"] == "waiting" and spec["verdict"]["status"] == "ready" and spec["decide"] == []
    assert "not yet admitted: approving this card admits it for every idea [!]" in card
    assert "ok   replay as of past times" in card and "reference: " in card
    assert "approving also admits toy.data for every idea" in card
    checks = {c["name"]: c for c in spec["checks"]}
    assert checks["hedged legs are one asset"]["status"] == "pass"
    assert checks["influence concentration"]["status"] == "warn", "Z is a third of the toy, so it often carries half"
    assert "READY" in card and "noise from one symbol: Z 55% at ref.mean 24h" in card
    assert idea.status().stage("data").state is State.REVIEW
    assert [(a.who, a.command) for a in idea.next_actions()] == [("human", "rci approve --stage data")]
    idea.approve("data", approval=_relayed(idea, "data"))
    assert idea.status().stage("data").state is State.PASSED
    assert idea.status().stage("data").budget is None and idea.ledger.n_trials("data") == 0
    assert idea.next_actions()[0].stage == "feat"
    again, card = idea.check_data()
    assert again["source"]["status"] == "admitted" and "not yet admitted" not in card
    assert idea.status().stage("data").version == 2


def test_the_data_freeze_stores_its_products_as_parquet_and_pins_them(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    spec, card = idea.check_data()
    idea.approve("data", approval=_relayed(idea, "data"))
    pinned = list(idea.ledger.records())[-1]["payload"]["deps"]
    assert tuple(sorted(spec["products"])) == tuple(sorted(data_stage.PRODUCTS))
    for kind, p in spec["products"].items():
        assert pinned[kind] == p["hash"] and vault.has(f"toy.data.{kind}", p["hash"])
        assert list(Path(p["path"]).glob("*.parquet")) and p["path"] in card
    cands = vault.get("toy.data.candidates", pinned["candidates"])["candidates"]
    outcomes = vault.get("toy.data.outcomes", pinned["outcomes"])["outcomes"]
    assert cands.height == spec["products"]["candidates"]["rows"] > 0
    assert set(outcomes["h"]) == {"8h", "24h"} and {"t", "inst", "y", "y_end"} <= set(outcomes.columns)
    assert outcomes.join(cands, on=["t", "inst"], how="anti").is_empty()


EVERY_THIRD = (
    DATA_EVALS
    + """
import polars as pl


def grid(sources):
    hours = sources["candles"].select("inst", pl.col("available_at").alias("t"))
    return hours.filter(pl.col("t").dt.hour() % 3 == 0)
"""
)
UNEVEN = (
    DATA_EVALS
    + """
import polars as pl


def grid(sources, wide):
    return wide.filter(pl.col("t").dt.hour().is_in([0, 1, 9, 12])).select("t", "inst")
"""
)
PEEK = (
    DATA_EVALS
    + """
import polars as pl


def grid(sources):
    closes = sources["candles"].sort("inst", "event_time")
    rises = pl.col("close").shift(-1).over("inst") > pl.col("close")
    return closes.filter(rises).select("inst", pl.col("available_at").alias("t"))
"""
)
SPREAD = """
import polars as pl


def features(wide, candidates):
    return candidates.select("t", "inst", pl.col("spread_h").alias("score"))
"""


def _grid_checked(tmp_path: Path, data: str) -> tuple[Idea, dict, str]:
    idea = _idea(tmp_path)
    (idea.root / "data.py").write_text(data)
    spec, card = idea.check_data()
    return idea, spec, card


def test_without_a_sampler_the_decisions_follow_the_contracts_clock(tmp_path: Path) -> None:
    _, spec, card = _grid_checked(tmp_path, DATA_EVALS)
    assert spec["grid"]["by"] == "the contract's clock, every 1h" and spec["grid"]["even"]
    assert spec["grid"]["longest_gap"] == "1h" and spec["grid"]["rows"] == spec["products"]["wide"]["rows"]
    assert "grid point-in-time" not in {c["name"] for c in spec["checks"]}
    assert "grid       the contract's clock, every 1h · " in card and "longest gap 1h · even" in card


def test_a_sampler_sets_the_decision_grid_that_wide_candidates_and_outcomes_follow(tmp_path: Path) -> None:
    idea, spec, card = _grid_checked(tmp_path, EVERY_THIRD)
    grid = vault.get("toy.data.grid", spec["products"]["grid"]["hash"])["grid"]
    table = vault.get("toy.data.wide", spec["products"]["wide"]["hash"])["wide"]
    cands = vault.get("toy.data.candidates", spec["products"]["candidates"]["hash"])["candidates"]
    outcomes = vault.get("toy.data.outcomes", spec["products"]["outcomes"]["hash"])["outcomes"]
    assert set((grid["t"].dt.hour() % 3).unique()) == {0} and grid.height > 0
    assert table.select("t", "inst").equals(grid.sort("t", "inst"))
    assert cands.height > 0 and set(cands["t"]) <= set(grid["t"]) and set(outcomes["t"]) <= set(grid["t"])
    assert spec["grid"] | {"per_inst": None} == {
        "by": "grid in data.py",
        "times": grid["t"].n_unique(),
        "rows": grid.height,
        "insts": 9,
        "per_inst": None,
        "longest_gap": "3h",
        "even": True,
    }
    check = {c["name"]: c for c in spec["checks"]}["grid point-in-time"]
    assert check["status"] == "pass" and spec["verdict"]["status"] != "blocked"
    assert "grid       grid in data.py · " in card and "longest gap 3h · even" in card
    assert "uneven" not in card
    idea.approve("data", approval=_relayed(idea, "data"))
    research = idea.root / "research"
    research.mkdir(exist_ok=True)
    (research / "spread.py").write_text(SPREAD)
    explored, _ = idea.explore_feat(research / "spread.py")
    assert [f["status"] for f in explored["floors"]] == ["pass", "pass"], "the cut check builds wide on the grid"


def test_an_uneven_grid_is_shown_with_a_warning(tmp_path: Path) -> None:
    _, spec, card = _grid_checked(tmp_path, UNEVEN)
    assert not spec["grid"]["even"] and spec["grid"]["longest_gap"] == "12h"
    assert "longest gap 12h · uneven [!]" in card
    assert "[!] the decision grid is uneven: the shuffles, blocks and the MDE assume an even one" in card


def test_a_sampler_that_reads_ahead_blocks_the_data(tmp_path: Path) -> None:
    _, spec, card = _grid_checked(tmp_path, PEEK)
    check = {c["name"]: c for c in spec["checks"]}["grid point-in-time"]
    assert check["status"] == "fail" and spec["verdict"]["status"] == "blocked"
    assert spec["verdict"]["notes"][0].startswith(
        "grid point-in-time (floor) failed: the sources cut at a time change the decision times: up to "
    )
    assert "(cut data vs all data); a decision time may depend only on rows known by then" in card
    assert "FAIL grid point-in-time" in card


def test_a_source_that_fails_admission_blocks_the_data_and_cannot_be_admitted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyHindsight})
    idea = _idea(tmp_path)
    spec, card = idea.check_data()
    assert spec["source"]["status"] == "not admitted" and "NOT ADMITTED" in card
    assert set(spec["source"]["failed"]) == {"replay as of past times", "independent source agrees"}
    data = idea.status().stage("data")
    assert data.state is State.OPEN and data.note.startswith("blocked: source toy.data not admitted")
    assert [(a.who, a.command) for a in idea.next_actions()] == [("agent", "rci data check")]
    with pytest.raises(WorkflowError, match="cannot approve"):
        idea.approve("data", approval=RELAYED)


def test_changing_a_plugin_after_the_check_blocks_approval(tmp_path: Path, monkeypatch) -> None:
    idea = _idea(tmp_path)
    idea.check_data()
    with monkeypatch.context() as m:
        m.setattr(data_stage, "code_hash", lambda plugin: "changed")
        with pytest.raises(WorkflowError, match="changed since the data check"):
            idea.approve("data", approval=Approval())


def test_a_swapped_leg_is_filtered_and_its_influence_measured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToySwapped})
    idea = _idea(tmp_path)
    spec, card = idea.check_data()
    guard = next(t for t in spec["treatments"] if t["name"] == "hedge guard at each decision time")
    assert guard["rows"] > 0 and set(guard["influence"]) == {"ref.mean 8h", "ref.mean 24h"}
    assert all(v["mde"] >= 0 for v in guard["influence"].values())
    assert all(p["mde"] < p["mde_before"] for p in spec["power"])
    assert "hedge guard" in card


def test_a_treatment_that_matters_needs_a_person(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToySwapped})
    idea = _idea(tmp_path, DataSettings(materiality=1e-6))
    spec, _ = idea.check_data()
    assert spec["verdict"]["status"] == "decide" and spec["decide"]
    with pytest.raises(WorkflowError, match="asks a person"):
        idea.approve("data", approval=Approval(by="agent", channel="auto"))
    with pytest.raises(WorkflowError, match="must quote the card"):
        idea.approve("data", approval=RELAYED)
    with pytest.raises(WorkflowError, match="must quote the card"):
        idea.approve("data", approval=Approval(by="human", channel="relayed", words="ok", card="00000000"))
    idea.approve("data", approval=_relayed(idea, "data"))


def test_a_person_can_keep_a_treatment_from_being_applied(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToySwapped})
    idea = _idea(tmp_path)
    applied, _ = idea.check_data()
    kept, card = idea.check_data(keep=("hedge guard at each decision time",))
    assert kept["kept"] == ["hedge guard at each decision time"]
    assert next(t for t in kept["treatments"] if t["name"].startswith("hedge"))["kept"] is True
    assert kept["products"]["exclusions"]["hash"] != applied["products"]["exclusions"]["hash"]
    assert all(p["mde"] == p["mde_before"] for p in kept["power"])
    assert "kept, not applied" in card
    with pytest.raises(data_stage.DataError, match="no treatment named"):
        idea.check_data(keep=("nothing like this",))


def test_treated_data_of_a_superseded_check_leaves_the_vault_unless_frozen(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToySwapped})
    idea = _idea(tmp_path)
    first, _ = idea.check_data()
    idea.approve("data", approval=_relayed(idea, "data"))
    second, _ = idea.check_data(keep=("hedge guard at each decision time",))
    third, _ = idea.check_data()
    assert vault.has("toy.data.exclusions", first["products"]["exclusions"]["hash"])
    assert not vault.has("toy.data.exclusions", second["products"]["exclusions"]["hash"])
    assert third["products"]["exclusions"]["hash"] == first["products"]["exclusions"]["hash"]


@pytest.mark.parametrize("lazy", [False, True])
def test_an_idea_eval_must_catch_the_pits_it_claims(tmp_path: Path, monkeypatch, lazy: bool) -> None:
    idea = _idea(tmp_path, DataSettings(materiality=0.05, seeds=1))
    _evals(idea, SWAP_TEST.replace("{lazy}", LAZY if lazy else ""))
    spec, card = idea.check_data()
    evals = {e["name"]: e for e in spec["evals"]}
    legs = evals["drifting legs"]["validation"]
    assert legs["status"] == "valid" and legs["file"] == "evals/legs.py"
    assert [t["name"] for t in legs["tests"]][-1] == "built-in swap in close, seed 0"
    assert [t["status"] for t in legs["tests"]] == ["pass"] * 3, "two cases run it, and the pit"
    if lazy:
        lazy_v = evals["lazy"]["validation"]
        assert lazy_v["status"] == "invalid" and evals["lazy"]["flagged"] is None
        assert spec["verdict"]["status"] == "blocked"
        assert "FAIL lazy (evals/legs.py) · H0: nothing is wrong" in card
        assert "on_fail mask: invalid 1/2: built-in swap in close, seed 0: expected it flagged, got nothing" in card
        assert (
            "FAIL lazy (evals/legs.py): invalid 1/2: built-in swap in close, seed 0" in card
            and "0/1 pits caught" in card
        )
        assert "lab eval 'lazy' (slot 4, tests of lab evals) is invalid" in card
        assert "make it catch the pits it claims in covers" in card
        assert not [t for t in spec["treatments"] if t["name"].startswith("eval lazy")]
    else:
        assert spec["verdict"]["status"] == "ready"
        assert "ok   drifting legs (evals/legs.py): valid 3/3 · 1/1 pits caught" in card


def test_the_default_list_decides_which_checks_run_and_which_pits_calibrate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(data_defaults, "CHECKS", data_defaults.CHECKS[:1])
    monkeypatch.setattr(data_defaults, "PITS", ())
    idea = _idea(tmp_path, DataSettings(materiality=0.05, seeds=1))
    _evals(idea, SWAP_TEST.replace("{lazy}", ""))
    spec, card = idea.check_data()
    assert [c["name"] for c in spec["checks"]] == ["data is current", "drifting legs", "column grades"]
    tests = spec["evals"][0]["validation"]["tests"]
    assert spec["evals"][0]["validation"]["status"] == "invalid" and tests[-1]["name"] == "pits: swap in close"
    assert (
        "no pit plants swap, so the claim cannot be checked" in card
        and "add a Pit('swap', plant) to PITS in legs.py" in card
    )


SPLIT = "split insts at identity breaks"


def test_a_ticker_relaunched_for_another_asset_is_split_and_a_person_decides(tmp_path: Path, monkeypatch) -> None:
    idea, spec, card = _checked(tmp_path, monkeypatch, ToyRelaunched)
    checks = {c["name"]: c for c in spec["checks"]}
    assert checks["level shifts"]["status"] == "pass", "the venue confirmed the relaunch, so it is not masked"
    assert checks["identity breaks"]["status"] == "warn"
    assert "c:Z 2026-01-07T18:00Z" in checks["identity breaks"]["detail"]
    split = _treatment(spec, SPLIT)
    assert split["rows"] > 0 and split["largest"] >= spec["materiality"]
    assert spec["verdict"]["status"] == "decide" and any(d.startswith(SPLIT) for d in spec["decide"])
    candles = vault.get("toy.data.treated", spec["products"]["treated"]["hash"])["candles"]
    z = candles.filter(pl.col("inst").str.starts_with("c:Z")).group_by("inst").agg(pl.col("event_time").min())
    assert dict(z.iter_rows())["c:Z#2"] == pd.Timestamp("2026-01-07T18:00Z")
    assert set(z["inst"]) == {"c:Z", "c:Z#2"}
    assert all(p["mde"] < p["mde_before"] / 100 for p in spec["power"] if p["horizon"] == "24h")
    assert SPLIT in card
    with pytest.raises(WorkflowError, match="asks a person"):
        idea.approve("data", approval=Approval(by="agent", channel="auto"))


def test_one_cluster_carrying_the_noise_is_shown_and_only_warned(tmp_path: Path, monkeypatch) -> None:
    _, applied, card = _checked(tmp_path, monkeypatch, ToyRelaunched)
    shares = dict(applied["concentration"]["shares"])
    assert set(shares) == {"ref.mean 8h", "ref.mean 24h"} and all(s == 0 for _, s in shares["ref.mean 8h"])
    assert "clusters   share of the noise that goes away without each symbol" in card
    kept, card = _checked(tmp_path / "kept", monkeypatch, ToyRelaunched, keep=(SPLIT,))[1:]
    top, share = dict(kept["concentration"]["shares"])["ref.mean 24h"][0]
    assert top == "Z" and share > 0.99 > dict(shares["ref.mean 24h"])["Z"], "the split takes the relaunch's noise away"
    check = {c["name"]: c for c in kept["checks"]}["influence concentration"]
    assert check["status"] == "warn" and "one symbol: Z " in check["detail"]
    assert not [d for d in kept["decide"] if "symbol Z" in d], "concentration asks no person"
    assert kept["products"]["treated"]["rows"] == kept["data"]["rows"], "concentration never drops anything"


MASK = "mask b:Z before the relaunch"


def _mask_b_z(seen: Seen) -> Found:
    """Masks b:Z's closes over the hours before c:Z is delisted, when the outcomes of c:Z's pairs read the relaunch."""
    hours = pl.col("event_time").is_between(START + pd.Timedelta(hours=140), START + pd.Timedelta(hours=146))
    found = seen.sources["candles"].filter((pl.col("inst") == "b:Z") & hours)
    cells = found.select(pl.lit("candles").alias("source"), "inst", "event_time", pl.lit("close").alias("column"))
    return Found(treatments=[tr.mask_cells(MASK, f"{cells.height} cells", cells, seen.sources)])


def test_other_treatments_are_weighed_on_top_of_the_split(tmp_path: Path, monkeypatch) -> None:
    """A mask next to a relaunch moves the number as much as where the venue named the relaunch apart itself: the
    relaunch that the split renames is not the mask's doing. The split is weighed against the data as it came."""
    monkeypatch.setattr(data_defaults, "CHECKS", [*data_defaults.CHECKS, Step(MASK, "treats", _mask_b_z, MASK)])
    split = _checked(tmp_path / "split", monkeypatch, ToyRelaunched)[1]
    apart = _checked(tmp_path / "apart", monkeypatch, ToyRenamed)[1]
    assert SPLIT not in {t["name"] for t in apart["treatments"]}
    ours, theirs = _treatment(split, MASK)["influence"], _treatment(apart, MASK)["influence"]
    assert ours["ref.mean 24h"]["bp"] > 0
    assert {k: v["bp"] for k, v in ours.items()} == pytest.approx({k: v["bp"] for k, v in theirs.items()}, rel=1e-9)
    assert _treatment(split, SPLIT)["largest"] >= split["materiality"] > _treatment(split, MASK)["largest"]


def _from_scratch(spec: dict) -> dict[str, float]:
    """The MDE per horizon from the frozen sources, wide and exclusions alone, computed the way feat reads them."""
    treated = vault.get("toy.data.treated", spec["products"]["treated"]["hash"])
    table = vault.get("toy.data.wide", spec["products"]["wide"]["hash"])["wide"]
    excluded = vault.get("toy.data.exclusions", spec["products"]["exclusions"]["hash"])["exclusions"]
    start = min(int(t.select(pl.col("available_at").min().dt.epoch("ns")).item()) for t in treated.values())
    end = to_ns(pd.Timestamp("2026-01-15T00:00:00Z"))
    cands = ToyPairs().candidates(table, treated).join(excluded, on=["t", "inst"], how="anti")
    outcome = plugins.load(Use("ref.carry", {"flow": "funding_rate"}))
    target = plugins.load(Use("ref.mean", {"cluster": "symbol", "top": 0.34}))
    costs = plugins.load(Use("ref.flat_costs", {"commission_bps": 1, "slippage_bps": 1, "legs": 2}))
    out = {}
    for h in ("8h", "24h"):
        found = outcome.values(treated, cands, span(h)).join(cands, on=["t", "inst"], suffix="_candidate")
        per_t = target.reduce(found, span(h), costs)
        out[h] = pw.power(pw.block_se(per_t, span_ns(span(h))), end - start, span_ns(span(h))).mde
    return out


def test_the_power_after_is_that_of_the_treated_data_computed_again(tmp_path: Path, monkeypatch) -> None:
    """Weighing a mask leaves out the outcome rows that read what it masks; the power after does not stand in like
    that: the outcomes are computed again on the treated data, as feat will read it."""
    _, spec, _ = _checked(tmp_path, monkeypatch, ToyZeros)
    assert {p["horizon"]: p["mde"] for p in spec["power"]} == pytest.approx(_from_scratch(spec), rel=1e-9)


def test_prices_at_zero_are_masked(tmp_path: Path, monkeypatch) -> None:
    _, spec, card = _checked(tmp_path, monkeypatch, ToyZeros)
    zero = _treatment(spec, "mask levels at zero or below")
    assert zero["rows"] == 3
    assert spec["verdict"]["status"] == "decide", "a number that is not finite before the treatment is for a person"
    candles = vault.get("toy.data.treated", spec["products"]["treated"]["hash"])["candles"]
    assert candles.filter(pl.col("close") <= 0).is_empty() and candles["close"].null_count() == 3
    assert "mask levels at zero or below" in card and "inf" in card


def test_an_inst_shorter_than_the_longest_horizon_is_dropped(tmp_path: Path, monkeypatch) -> None:
    _, spec, _ = _checked(tmp_path, monkeypatch, ToyShort)
    short = _treatment(spec, "drop insts too short for the horizon")
    assert short["rows"] == 22, "twenty candles and two funding rows"
    assert "c:Y" in {c["name"]: c for c in spec["checks"]}["coverage"]["detail"]
    treated = vault.get("toy.data.treated", spec["products"]["treated"]["hash"])
    assert all("c:Y" not in set(t["inst"]) for t in treated.values())


def test_dropping_a_short_series_a_split_made_does_the_same_with_or_without_the_split() -> None:
    sources = synthetic.sources(Source(insts=3), 0)
    at = pl.lit(START_NS + 330 * HOUR_NS).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")
    splits = pl.DataFrame({"inst": ["e0"], "to": ["e0#2"]}).with_columns(at=at)
    split = tr.split_insts("split", "", splits, sources)
    short = cv.short_lived(split.apply(sources), 24 * HOUR_NS)
    assert short["inst"].to_list() == ["e0#2"]
    drop = tr.drop_spans("drop", "", data_defaults._origins(short, sources), sources)
    alone, both = drop.apply(sources), drop.apply(split.apply(sources))
    gone = sum(t.height for t in sources.values()) - sum(t.height for t in alone.values())
    assert drop.rows == gone > 0
    assert all(alone[name].equals(both[name]) for name in sources)


ROW_TEST = """
import polars as pl
from research_ci.api import Case, Eval


def zeros(sources):
    flagged = [(e, t) for e, t, v in sources["candles"].select("inst", "event_time", "close").iter_rows() if v == 0]
    return pl.DataFrame(flagged, schema=["inst", "event_time"], orient="row").with_columns(source=pl.lit("candles"))


CASES = [Case("no zeros, no flags", zeros, 0, given={"candles": pl.DataFrame({"inst": ["a"], "event_time": [0],
                                                                              "close": [1.0]})})]
EVALS = [Eval(zeros, h0="no value is zero", covers=[("spike", "close")])]
"""


def test_a_wrong_eval_file_says_how_to_fix_it_and_a_row_wise_one_runs_with_a_warning(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    (idea.root / "data.py").write_text("EVALS = [\n")
    with pytest.raises(ResearchError, match=r"data.py:1: .*fix the file"):
        idea.check_data()
    (idea.root / "data.py").write_text("import polars as pl\n")
    with pytest.raises(
        data_stage.DataError, match=r"data.py defines no EVALS; data.py defines EVALS = \[Eval\(fn, h0="
    ):
        idea.check_data()
    (idea.root / "data.py").write_text("EVALS = [1]\n")
    with pytest.raises(data_stage.DataError, match=r"EVALS\[0\] is a int: an entry is a function or Eval.*methods"):
        idea.check_data()
    (idea.root / "data.py").write_text("from research_ci.api import Eval\nEVALS = [Eval(print, covers=[('x',)])]\n")
    with pytest.raises(data_stage.DataError, match=r"covers is a list of \(category, column\) pairs"):
        idea.check_data()
    assert not [r for r in idea.ledger.records() if r["kind"] == "check"]
    (idea.root / "data.py").write_text(ROW_TEST)
    spec, card = idea.check_data()
    warning = "row-wise code: data.py:7 iter_rows — polars expressions are usually much faster"
    assert spec["warnings"][0] == warning and f"warnings   [!] {warning}" in card
    assert [e["name"] for e in spec["evals"]] == ["zeros"]


LIBRARY = """
from functools import partial

from research_ci.api import Eval
from research_ci.eval.data.sanity import nonpositive
from evals.toy import closes_at_zero

EVALS = [
    Eval(partial(nonpositive, levels=("close",)), name="closes at zero or below"),
    Eval(closes_at_zero, h0="every close is above zero", on_fail="ask"),
]
"""


def test_a_library_eval_is_trusted_and_runs_with_its_options(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyZeros})
    idea = _idea(tmp_path)
    (idea.root / "data.py").write_text(LIBRARY)
    spec, card = idea.check_data()
    row = spec["evals"][0]
    assert (row["by"], row["validation"]["status"], row["flagged"]) == ("library", "trusted", 3)
    assert "library  [!]  closes at zero or below: engine library · flagged 3 rows; masked" in card
    kept = [p.read_text() for p in (idea.root / ".rci" / "evals").glob("*.json")]
    assert len(kept) == 1 and '"closes_at_zero"' in kept[0], "only the lab eval is validated"


BLOCKS = DATA_EVALS.replace('zero")', 'zero", on_fail="block")')


def test_a_lab_eval_that_blocks_vetoes_what_the_treatments_before_it_leave_with_its_h0_and_the_rows(
    tmp_path: Path, monkeypatch
) -> None:
    """Without level sanity in the default list, nothing before the eval masks the zeros: the eval flags them, and the
    data is BLOCKED, naming the eval, its H0, how many rows, and a few of them."""
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyZeros})
    monkeypatch.setattr(data_defaults, "CHECKS", [c for c in data_defaults.CHECKS if c.name != "level sanity"])
    idea = _idea(tmp_path)
    (idea.root / "data.py").write_text(BLOCKS)
    (idea.root / "nulls" / "data.py").write_text("raise RuntimeError('nulls/data.py is not read')\n")
    spec, card = idea.check_data()
    assert spec["verdict"]["status"] == "blocked" and "calibration" not in spec
    (note,) = spec["verdict"]["notes"]
    assert note.startswith(
        "lab eval 'closes_at_zero' (slot 3) vetoes the data: H0: every close is above zero; it flagged 3 rows that the "
        "treatments before it left, such as candles.close b:X 2026-01-05T04:00Z, candles.close b:X 2026-01-05T05:00Z, "
        "candles.close b:X 2026-01-05T06:00Z; with on_fail='block'"
    )
    assert "FAIL closes_at_zero (evals/toy.py) · H0: every close is above zero" in card and note in card
    assert "on_fail block: valid 1/1 · flagged 3 rows of the data as it came; on_fail=block: a veto" in card
    assert not [t for t in spec["treatments"] if t["name"].startswith("eval closes_at_zero")], "a veto treats nothing"
    assert idea.status().stage("data").state is State.OPEN
    assert idea.next_actions()[0].reason.startswith("blocked: lab eval 'closes_at_zero' (slot 3) vetoes the data")


def test_a_lab_eval_that_blocks_does_not_veto_what_the_treatments_before_it_took_away(
    tmp_path: Path, monkeypatch
) -> None:
    """With the default list, level sanity masks the zeros before the lab evals run: the blocking eval judges the
    data after that mask, finds nothing, and the check goes on; a masking eval still reads the data as it came."""
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyZeros})
    idea = _idea(tmp_path)
    (idea.root / "data.py").write_text(BLOCKS)
    spec, card = idea.check_data()
    assert spec["verdict"]["status"] != "blocked" and "calibration" in spec, spec["verdict"]
    (row,) = [e for e in spec["evals"] if e["name"] == "closes_at_zero"]
    assert row["flagged"] == 0 and "examples" not in row
    assert (
        "on_fail block: valid 1/1 · flagged 0 rows of the data after the treatments before it (mask levels at zero or "
        "below)"
    ) in card
    (idea.root / "data.py").write_text(DATA_EVALS)
    spec, card = idea.check_data()
    (row,) = [e for e in spec["evals"] if e["name"] == "closes_at_zero"]
    assert row["flagged"] == 3 and "on_fail mask: valid 1/1 · flagged 3 rows; masked" in card


def test_a_data_py_without_a_valid_lab_eval_is_refused_before_the_data_is_fetched(tmp_path: Path, monkeypatch) -> None:
    idea = _idea(tmp_path)

    def never(self: object, *args: object, **kwargs: object) -> None:
        raise AssertionError("the data was fetched")

    monkeypatch.setattr(ToyData, "fetch", never)
    (idea.root / "data.py").unlink()
    with pytest.raises(data_stage.DataError, match="slot 3, lab evals: there is no data.py, and the data layer needs"):
        idea.check_data()
    (idea.root / "data.py").write_text(
        LIBRARY.replace('    Eval(closes_at_zero, h0="every close is above zero", on_fail="ask"),\n', "")
    )
    with pytest.raises(data_stage.DataError, match="slot 3, lab evals: data.py lists no lab eval") as refused:
        idea.check_data()
    assert "Eval(fn, h0='<the H0 it rejects>', on_fail=...," in str(refused.value)
    (idea.root / "data.py").write_text(DATA_EVALS)
    toy = idea.root / "evals" / "toy.py"
    toy.write_text(toy.read_text().replace("closes_at_zero, 1, given", "closes_at_zero, 2, given"))
    with pytest.raises(data_stage.DataError) as refused:
        idea.check_data()
    assert str(refused.value) == (
        "slot 4, tests of lab evals: the lab eval 'closes_at_zero' (evals/toy.py) failed its tests, so it cannot "
        "judge: invalid 0/1: case 'a close at zero is flagged': expected 2, got 1; fix the eval or its CASES in "
        "evals/toy.py; `rci eval check evals/toy.py` runs them for free; nothing was recorded"
    )
    (idea.root / "data.py").write_text(DATA_EVALS.replace(', h0="every close is above zero"', ""))
    with pytest.raises(data_stage.DataError, match="the lab eval 'closes_at_zero' names no h0"):
        idea.check_data()
    assert not [r for r in idea.ledger.records() if r["kind"] == "check"]


def test_a_data_eval_is_tested_on_the_pits_of_its_own_module(tmp_path: Path) -> None:
    idea = _idea(tmp_path, DataSettings(materiality=0.05, seeds=1))
    _evals(idea, SWAP_TEST.replace("{lazy}", "").replace('[("swap", "close")]', '[("wander", "close")]'))
    spec, card = idea.check_data()
    tests = spec["evals"][0]["validation"]["tests"]
    assert tests[-1] == {
        "name": "pit wander: one venue walks off on its own in close, seed 0",
        "status": "pass",
        "detail": "caught",
        "by": "",
    }
    assert spec["verdict"]["status"] == "ready" and "1/1 pits caught" in card


STALE = '''
from datetime import datetime, timedelta, timezone

import polars as pl

from research_ci.api import Case

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def stale_closes(wide):
    """Rows of wide whose close became known more than two hours before the decision."""
    return wide.filter(pl.col("age_candles") > pl.duration(hours=2)).select("t", "inst")


AGES = pl.DataFrame({"t": [T, T], "inst": ["a", "b"], "age_candles": [timedelta(hours=1), timedelta(hours=3)]})
CASES = [Case("a close three hours old is stale", stale_closes, 1, given=AGES)]
'''
STALE_DATA = """
from research_ci.api import Eval
from evals.stale import stale_closes

EVALS = [Eval(stale_closes, h0="every close a decision reads is fresh")]
"""


def test_a_data_eval_on_wide_masks_the_candidate_rows_that_read_what_it_flags(tmp_path: Path, monkeypatch) -> None:
    """After venue c stops listing Y, wide carries its last close forward, older every hour: an eval on wide flags
    those rows, and masking them drops the candidate rows whose legs read them, and nothing else."""
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyShort})
    idea = _idea(tmp_path)
    _evals(idea, STALE_DATA)
    (idea.root / "evals" / "stale.py").write_text(STALE)
    spec, card = idea.check_data()
    mask = _treatment(spec, "eval stale_closes: mask")
    assert mask["on"] == "candidates" and mask["rows"] > 0
    excluded = vault.get("toy.data.exclusions", spec["products"]["exclusions"]["hash"])["exclusions"]
    legs = {tuple(pair.split(":")[1].split(">")) for pair in excluded["inst"]}
    assert excluded.height > 0 and all(pair.startswith("Y:") for pair in excluded["inst"])
    assert all("c" in pair for pair in legs), "only the pairs with a leg on venue c"
    assert "on_fail mask: valid 1/1 · flagged" in card and "masked" in card
