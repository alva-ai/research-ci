"""What `rci plugins` and `rci methods` read: every installed plugin with its kind, options, and declarations, and every
method research code may call with the worked examples and properties that back it."""

from dataclasses import fields

from research_ci.catalog import listing


def test_every_plugin_with_its_kind_options_and_declarations() -> None:
    found = {p["name"]: p for p in listing.plugins()}
    assert {"toy.data", "toy.pairs", "toy.carry", "ref.mean", "ref.carry", "ref.flat_costs"} <= set(found)
    data = found["toy.data"]
    assert (data["kind"], data["origin"]) == ("data", "registry")
    assert data["fields"] == ["close", "funding_rate", "funding_interval_hours"] and data["reference"] is True
    assert data["levels"] == {"fields": ("close",), "factor": 5.0, "window": 24, "mask": "24h"}
    assert data["doc"].startswith("Replays as of any time")
    assert (found["toy.pairs"]["kind"], found["toy.carry"]["kind"]) == ("universe", "consumer")
    assert [o["name"] for o in found["toy.carry"]["options"]] == ["k", "commission_bps", "slippage_bps"]
    mean = found["ref.mean"]
    assert (mean["kind"], mean["per"], mean["rolls"]) == ("target", "entry", True)
    assert mean["origin"] == "entry point of research-ci"
    carry = found["ref.carry"]
    assert carry["kind"] == "label" and carry["signature"] == "ref.carry(price: str = 'close', flow: str | None = None)"
    assert found["ref.flat_costs"]["options"][0] == {
        "name": "commission_bps",
        "type": "float",
        "required": True,
        "default": None,
    }


def test_every_method_with_its_import_signature_and_backing() -> None:
    found = {(m["module"].rsplit(".", 1)[-1], m["name"]): m for m in listing.methods()}
    ic = found[("ic", "rank_ic")]
    assert ic["use"] == "feat eval" and ic["import"] == "from research_ci.eval.statistics.ic import rank_ic"
    assert ic["signature"] == "(table: pl.DataFrame, label: str = 'y') -> pl.DataFrame"
    assert (ic["examples"], ic["properties"]) == (3, 3)
    assert (found[("spread", "quantile_spread")]["examples"], found[("spread", "quantile_spread")]["properties"]) == (
        3,
        2,
    )
    assert found[("sanity", "spikes")]["use"] == "data eval" and found[("sanity", "spikes")]["examples"] == 5
    assert found[("pits", "spike")]["use"] == "pit"
    assert all(m["doc"] and m["examples"] >= 1 for m in found.values()), "every listed method says what it does"
    assert set(listing.formats()) == {"feat eval", "data eval", "pit"}


def test_null_worlds_list_every_field_of_null_and_permute_within() -> None:
    for cls, listed in listing.NULL_FIELDS.items():
        assert [f.name for f in fields(cls)] == list(listed), cls
    null, permute = listing.nulls()
    assert null["import"] == "from research_ci.api import Null"
    assert null["signature"].startswith("(explains: str, world: ")
    assert permute["signature"] == "(bucket_by: str, clusters: str | None = None, block: str | None = None)"
