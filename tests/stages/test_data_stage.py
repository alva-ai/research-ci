from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from research_ci.api.contract import Use, span
from research_ci.api.evals import Source
from research_ci.eval.data import coverage as cv
from research_ci.eval.synthetic import HOUR_NS, START_NS, source
from research_ci.kernel.time.clock import from_ns, span_ns, to_ns
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
    REGISTRY,
    START,
    ToyHindsight,
    ToyPairs,
    ToyRelaunched,
    ToyRenamed,
    ToyShort,
    ToySwapped,
    ToyZeros,
    label_contract,
)

FAST = Settings(workers=2, universe_cuts=2, label_cuts=1)
RELAYED = Approval(by="human", channel="relayed", words="ok")

LEGS = '''
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from research_ci.api import Case


def drifting_legs(fields):
    """Venues of one symbol whose hourly returns do not move with the symbol's other venues."""
    close = fields.filter(pl.col("field") == "close").sort("entity", "event_time")
    r = close.with_columns(r=(pl.col("value") / pl.col("value").shift(1).over("entity")).log())
    r = r.with_columns(symbol=pl.col("entity").str.split(":").list.last())
    mean = r.group_by("symbol", "event_time").agg(pl.col("r").mean().alias("m"))
    r = r.join(mean, on=["symbol", "event_time"])
    corr = r.group_by("entity").agg(pl.corr("r", "m").alias("c"))
    bad = corr.filter(pl.col("c") < 0.8)["entity"]
    return close.filter(pl.col("entity").is_in(bad.implode())).select("entity", "event_time")


def nothing(fields):
    return fields.head(0).select("entity", "event_time")


def _venues(apart):
    walks = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, (2, 48)), axis=1))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = pl.datetime_range(start, start + timedelta(hours=47), "1h", eager=True)
    closes = {"a:X": walks[0], "b:X": walks[0], "c:X": walks[1] if apart else walks[0]}
    return pl.concat(
        [pl.DataFrame({"entity": e, "field": "close", "event_time": t, "value": v}) for e, v in closes.items()]
    )


CASES = [
    Case("a venue moving apart is flagged", lambda: drifting_legs(_venues(True))["entity"].unique().to_list(), ["c:X"]),
    Case("venues moving together are not", drifting_legs, 0, given=_venues(False)),
    Case("nothing flags nothing", nothing, 0, given=_venues(True)),
]
'''
SWAP_TEST = """
from research_ci.api import Eval
from evals.legs import drifting_legs, nothing

EVALS = [
    Eval(drifting_legs, name="drifting legs", covers=[("swap", "close")], on_fail="drop_entities"),
    {lazy}
]
"""
LAZY = 'Eval(nothing, name="lazy", covers=[("swap", "close")]),'


def _evals(idea: "Idea", data: str, legs: str = LEGS) -> None:
    (idea.root / "evals").mkdir(exist_ok=True)
    (idea.root / "evals" / "legs.py").write_text(legs)
    (idea.root / "data.py").write_text(data)


def _relayed(idea: Idea, stage: str) -> Approval:
    return Approval(by="human", channel="relayed", words="ok", card=card_id(idea.card(stage)))


def _idea(tmp_path: Path, settings: DataSettings | None = None) -> Idea:
    idea = Idea.init(tmp_path / "idea", settings=FAST, data_settings=settings)
    (idea.root / "contract.py").write_text(label_contract())
    idea.derive_contract()
    idea.approve("contract")
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
    assert "READY" in card and "noise from one symbol: Z 55% at carry 24h" in card
    assert idea.status().stage("data").state is State.REVIEW
    assert [(a.who, a.command) for a in idea.next_actions()] == [("human", "rci approve --stage data")]
    idea.approve("data", approval=_relayed(idea, "data"))
    assert idea.status().stage("data").state is State.PASSED
    assert idea.status().stage("data").budget is None and idea.ledger.n_trials("data") == 0
    assert idea.next_actions()[0].stage == "feat"
    again, card = idea.check_data()
    assert again["source"]["status"] == "admitted" and "not yet admitted" not in card
    assert idea.status().stage("data").version == 2


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
    assert guard["rows"] > 0 and set(guard["influence"]) == {"carry 8h", "carry 24h"}
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
    assert kept["panel"]["exclusions"] != applied["panel"]["exclusions"]
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
    assert vault.has("toy.data.exclusions", first["panel"]["exclusions"])
    assert not vault.has("toy.data.exclusions", second["panel"]["exclusions"])
    assert third["panel"]["exclusions"] == first["panel"]["exclusions"]


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
        assert (
            "case     FAIL lazy: invalid 1/2: built-in swap in close, seed 0: expected it flagged, got nothing" in card
        )
        assert "lazy (case eval) is invalid" in card and "make it catch the pits it claims in covers" in card
        assert not [t for t in spec["treatments"] if t["name"].startswith("eval lazy")]
    else:
        assert spec["verdict"]["status"] == "ready"


def test_the_default_list_decides_which_checks_run_and_which_pits_calibrate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(data_defaults, "CHECKS", data_defaults.CHECKS[:1])
    monkeypatch.setattr(data_defaults, "PITS", ())
    idea = _idea(tmp_path, DataSettings(materiality=0.05, seeds=1))
    _evals(idea, SWAP_TEST.replace("{lazy}", ""))
    spec, card = idea.check_data()
    assert [c["name"] for c in spec["checks"]] == ["data is current", "drifting legs", "field grades"]
    tests = spec["evals"][0]["validation"]["tests"]
    assert spec["evals"][0]["validation"]["status"] == "invalid" and tests[-1]["name"] == "pits: swap in close"
    assert "no pit plants swap, so the claim cannot be checked" in card and "nulls/data.py" in card


SPLIT = "split entities at identity breaks"


def test_a_ticker_relaunched_for_another_asset_is_split_and_a_person_decides(tmp_path: Path, monkeypatch) -> None:
    idea, spec, card = _checked(tmp_path, monkeypatch, ToyRelaunched)
    checks = {c["name"]: c for c in spec["checks"]}
    assert checks["level shifts"]["status"] == "pass", "the venue confirmed the relaunch, so it is not masked"
    assert checks["identity breaks"]["status"] == "warn"
    assert "c:Z 2026-01-07T18:00Z" in checks["identity breaks"]["detail"]
    split = _treatment(spec, SPLIT)
    assert split["rows"] > 0 and split["largest"] >= spec["materiality"]
    assert spec["verdict"]["status"] == "decide" and any(d.startswith(SPLIT) for d in spec["decide"])
    panel = vault.get("toy.data.panel", spec["panel"]["hash"])
    z = panel.filter(pl.col("entity").str.starts_with("c:Z")).group_by("entity").agg(pl.col("event_time").min())
    assert dict(z.iter_rows())["c:Z#2"] == pd.Timestamp("2026-01-07T18:00Z")
    assert set(z["entity"]) == {"c:Z", "c:Z#2"}
    assert all(p["mde"] < p["mde_before"] / 100 for p in spec["power"] if p["horizon"] == "24h")
    assert SPLIT in card
    with pytest.raises(WorkflowError, match="asks a person"):
        idea.approve("data", approval=Approval(by="agent", channel="auto"))


def test_one_cluster_carrying_the_noise_is_shown_and_only_warned(tmp_path: Path, monkeypatch) -> None:
    _, applied, card = _checked(tmp_path, monkeypatch, ToyRelaunched)
    shares = dict(applied["concentration"]["shares"])
    assert set(shares) == {"carry 8h", "carry 24h"} and all(s == 0 for _, s in shares["carry 8h"])
    assert "clusters   share of the noise that goes away without each symbol" in card
    kept, card = _checked(tmp_path / "kept", monkeypatch, ToyRelaunched, keep=(SPLIT,))[1:]
    top, share = dict(kept["concentration"]["shares"])["carry 24h"][0]
    assert top == "Z" and share > 0.99 > dict(shares["carry 24h"])["Z"], "the split takes the relaunch's noise away"
    check = {c["name"]: c for c in kept["checks"]}["influence concentration"]
    assert check["status"] == "warn" and "one symbol: Z " in check["detail"]
    assert not [d for d in kept["decide"] if "symbol Z" in d], "concentration asks no person"
    assert kept["panel"]["rows"] == kept["data"]["rows"], "concentration never drops anything"


MASK = "mask b:Z before the relaunch"


def _mask_b_z(seen: Seen) -> Found:
    """Masks b:Z's closes over the hours before c:Z is delisted, when the labels of c:Z's pairs read the relaunch."""
    hours = pl.col("event_time").is_between(START + pd.Timedelta(hours=140), START + pd.Timedelta(hours=146))
    rows = seen.fields.filter((pl.col("entity") == "b:Z") & (pl.col("field") == "close") & hours)
    return Found(treatments=[tr.mask_rows(MASK, f"{rows.height} rows", rows, seen.fields)])


def test_other_treatments_are_weighed_on_top_of_the_split(tmp_path: Path, monkeypatch) -> None:
    """A mask next to a relaunch moves the number as much as where the venue named the relaunch apart itself: the
    relaunch that the split renames is not the mask's doing. The split is weighed against the data as it came."""
    monkeypatch.setattr(data_defaults, "CHECKS", [*data_defaults.CHECKS, Step(MASK, "treats", _mask_b_z, MASK)])
    split = _checked(tmp_path / "split", monkeypatch, ToyRelaunched)[1]
    apart = _checked(tmp_path / "apart", monkeypatch, ToyRenamed)[1]
    assert SPLIT not in {t["name"] for t in apart["treatments"]}
    ours, theirs = _treatment(split, MASK)["influence"], _treatment(apart, MASK)["influence"]
    assert ours["carry 24h"]["bp"] > 0
    assert {k: v["bp"] for k, v in ours.items()} == pytest.approx({k: v["bp"] for k, v in theirs.items()}, rel=1e-9)
    assert _treatment(split, SPLIT)["largest"] >= split["materiality"] > _treatment(split, MASK)["largest"]


def _from_scratch(spec: dict) -> dict[str, float]:
    """The MDE per horizon from the frozen panel and exclusions alone, computed the way feat reads them."""
    panel = vault.get("toy.data.panel", spec["panel"]["hash"])
    excluded = vault.get("toy.data.exclusions", spec["panel"]["exclusions"])
    start = int(panel.select(pl.col("available_at").min().dt.epoch("ns")).item())
    end = to_ns(pd.Timestamp("2026-01-15T00:00:00Z"))
    times = [from_ns(t) for t in pw.decision_grid(HOUR_NS, 0, start, end)]
    cands = ToyPairs().candidates(panel, times).join(excluded, on=["t", "entity"], how="anti")
    label = plugins.load(Use("ref.carry", {"flow": "funding_rate"}))
    target = plugins.load(Use("ref.mean", {"cluster": "symbol", "top": 0.34}))
    costs = plugins.load(Use("ref.flat_costs", {"commission_bps": 1, "slippage_bps": 1, "legs": 2}))
    out = {}
    for h in ("8h", "24h"):
        rows = label.values(panel, cands, span(h)).join(cands, on=["t", "entity"], suffix="_candidate")
        per_t = target.reduce(rows, span(h), costs)
        out[h] = pw.power(pw.block_se(per_t, span_ns(span(h))), end - start, span_ns(span(h))).mde
    return out


def test_the_power_after_is_that_of_the_treated_data_computed_again(tmp_path: Path, monkeypatch) -> None:
    """Weighing a mask leaves out the label rows that read what it masks; the power after does not stand in like
    that: the labels are computed again on the treated data, as feat will read it."""
    _, spec, _ = _checked(tmp_path, monkeypatch, ToyZeros)
    assert {p["horizon"]: p["mde"] for p in spec["power"]} == pytest.approx(_from_scratch(spec), rel=1e-9)


def test_prices_at_zero_are_masked(tmp_path: Path, monkeypatch) -> None:
    _, spec, card = _checked(tmp_path, monkeypatch, ToyZeros)
    zero = _treatment(spec, "mask levels at zero or below")
    assert zero["rows"] == 3
    assert spec["verdict"]["status"] == "decide", "a number that is not finite before the treatment is for a person"
    panel = vault.get("toy.data.panel", spec["panel"]["hash"])
    assert panel.filter((pl.col("field") == "close") & (pl.col("value") <= 0)).is_empty()
    assert "mask levels at zero or below" in card and "inf" in card


def test_an_entity_shorter_than_the_longest_horizon_is_dropped(tmp_path: Path, monkeypatch) -> None:
    _, spec, _ = _checked(tmp_path, monkeypatch, ToyShort)
    short = _treatment(spec, "drop entities too short for the horizon")
    assert short["rows"] == 24
    assert "c:Y" in {c["name"]: c for c in spec["checks"]}["coverage"]["detail"]
    panel = vault.get("toy.data.panel", spec["panel"]["hash"])
    assert "c:Y" not in set(panel["entity"])


def test_dropping_a_short_series_a_split_made_does_the_same_with_or_without_the_split() -> None:
    fields = source(Source(entities=3), 0)
    at = pl.lit(START_NS + 330 * HOUR_NS).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")
    splits = pl.DataFrame({"entity": ["e0"], "to": ["e0#2"]}).with_columns(at=at)
    split = tr.split_entities("split", "", splits, fields)
    short = cv.short_lived(split.apply(fields), 24 * HOUR_NS)
    assert short["entity"].to_list() == ["e0#2"]
    drop = tr.drop_spans("drop", "", data_defaults._origins(short, fields), fields)
    alone, both = drop.apply(fields), drop.apply(split.apply(fields))
    assert drop.rows == fields.height - alone.height > 0
    assert alone.equals(both)


ROW_TEST = """
import polars as pl
from research_ci.api import Case, Eval


def zeros(fields):
    flagged = [(e, t) for e, f, t, v in fields.select("entity", "field", "event_time", "value").iter_rows() if v == 0]
    return pl.DataFrame(flagged, schema=["entity", "event_time"], orient="row")


CASES = [Case("no zeros, no flags", zeros, 0, given=pl.DataFrame({"entity": ["a"], "field": ["close"],
                                                                   "event_time": [0], "value": [1.0]}))]
EVALS = [Eval(zeros, covers=[("spike", "close")])]
"""


def test_a_wrong_eval_file_says_how_to_fix_it_and_a_row_wise_one_runs_with_a_warning(tmp_path: Path) -> None:
    idea = _idea(tmp_path)
    (idea.root / "data.py").write_text("EVALS = [\n")
    with pytest.raises(ResearchError, match=r"data.py:1: .*fix the file"):
        idea.check_data()
    (idea.root / "data.py").write_text("import polars as pl\n")
    with pytest.raises(data_stage.DataError, match=r"data.py defines no EVALS; data.py defines EVALS = \[fn, Eval"):
        idea.check_data()
    (idea.root / "data.py").write_text("EVALS = [1]\n")
    with pytest.raises(data_stage.DataError, match=r"EVALS\[0\] is a int: an entry is a function or Eval.*methods"):
        idea.check_data()
    (idea.root / "data.py").write_text("from research_ci.api import Eval\nEVALS = [Eval(print, covers=[('x',)])]\n")
    with pytest.raises(data_stage.DataError, match=r"covers is a list of \(category, field\) pairs"):
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

EVALS = [Eval(partial(nonpositive, levels=("close",)), name="closes at zero or below")]
"""


def test_a_library_eval_is_trusted_and_runs_with_its_options(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plugins, "EXTRA", {**REGISTRY, "toy.data": ToyZeros})
    idea = _idea(tmp_path)
    (idea.root / "data.py").write_text(LIBRARY)
    spec, card = idea.check_data()
    row = spec["evals"][0]
    assert (row["by"], row["validation"]["status"], row["flagged"]) == ("library", "trusted", 3)
    assert "library  [!]  closes at zero or below: engine library · flagged 3 rows; masked" in card
    assert not (idea.root / ".rci" / "evals").exists(), "nothing to validate"
