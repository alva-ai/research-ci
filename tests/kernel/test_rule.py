"""The verdict, horizon by horizon: the decisive test passes a horizon, a lab eval must cover it, and every lab eval
and null world that covers it must reach z 2; lab evals and worlds only take horizons away, and a FAILED verdict says
which slot rejected the claim, with each veto's H0, z against the bar, and whether it could have seen the effect."""

import math

from research_ci.kernel.verdict import power as pw
from research_ci.kernel.verdict.rule import VETO_Z, combine, passing, votes

NEED = 2.6


def _decisive(adjusted: dict[str, float], z: float | None = None) -> dict:
    rows = [{"h": h, "counted": True, "adjusted": a} for h, a in adjusted.items()]
    best = max(adjusted, key=lambda h: adjusted[h])
    family = max(adjusted.values()) if z is None else z
    return {"name": "ref.mean", "z": family, "best": best, "effect": 1e-4, "se": 2e-5, "per": "entry", "horizons": rows}


def _horizon(h: str, z: float, sd: float = 0.01, counted: bool = True) -> dict:
    return {"h": h, "z": z, "counted": counted, "gain": z * sd, "mde": pw.mde(sd, VETO_Z)}


def _row(name: str, z: dict[str, float], combine: str = "max", world: str = "floor", by: str = "lab") -> dict:
    family = max(z.values()) if combine == "max" else min(z.values())
    return {
        "name": name,
        "by": by,
        "h0": f"the scores do not rank {name}",
        "world": world,
        "explains": ["chance", "timing"] if world == "floor" else ["tail"],
        "alternative": "" if world == "floor" else "a few symbols carry the result",
        "combine": combine,
        "z": family,
        "horizons": [_horizon(h, x) for h, x in z.items()],
    }


def test_a_horizon_passes_where_the_decisive_test_passes_and_every_lab_eval_that_covers_it_reaches_two() -> None:
    d = _decisive({"8h": 3.0, "24h": 2.0})
    assert passing(d, NEED) == ["8h"]
    v = combine([d], [_row("funding", {"8h": 5.0, "24h": 5.0})], NEED)
    assert v["status"] == "PASSED" and v["horizons"] == {"ref.mean": ["8h"]} and v["slots"] == []
    assert v["notes"][0].startswith("ref.mean: passes at 8h: the decisive test beats random there, z 3 ≥ 2.60 (slot 1)")


def test_a_lab_eval_votes_once_for_all_its_horizons_with_max_and_horizon_by_horizon_with_each() -> None:
    assert votes(_row("x", {"8h": 1.0, "24h": 2.5})) == {"8h": True, "24h": True}
    assert votes(_row("x", {"8h": 1.0, "24h": 2.5}, "each")) == {"8h": False, "24h": True}
    uncounted = _row("x", {"8h": 3.0}, "each")
    uncounted["horizons"][0]["counted"] = False
    assert votes(uncounted) == {"8h": False}
    d = _decisive({"8h": 3.0, "24h": 3.0})
    v = combine([d], [_row("x", {"8h": 1.0, "24h": 2.5}, "each")], NEED)
    assert v["status"] == "PASSED" and v["horizons"] == {"ref.mean": ["24h"]}
    assert any(n.startswith("ref.mean 8h: beats random, but slot 3 rejects it: lab eval 'x'") for n in v["notes"])


def test_a_horizon_no_lab_eval_covers_cannot_pass() -> None:
    d = _decisive({"8h": 3.0, "24h": 3.0})
    only = _row("x", {"24h": 4.0})
    v = combine([d], [only], NEED)
    assert v["status"] == "PASSED" and v["horizons"] == {"ref.mean": ["24h"]}
    assert "ref.mean 8h: beats random, but slot 3 rejects it: no lab eval covers 8h" in v["notes"]
    world = _row("rank ic", {"8h": 4.0, "24h": 4.0}, world="null 1", by="default")
    v = combine([d], [world], NEED)
    assert v["status"] == "FAILED" and v["slots"] == [3], "a world is no lab eval"


def test_lab_evals_never_turn_a_failed_decisive_test_into_a_pass() -> None:
    d = _decisive({"8h": 2.0, "24h": 1.0})
    v = combine([d], [_row("funding", {"8h": 9.0, "24h": 9.0})], NEED)
    assert v["status"] == "FAILED" and v["slots"] == [1] and v["horizons"] == {"ref.mean": []}
    assert v["notes"] == [
        "ref.mean: does not beat random, z 2 < 2.60 (decisive test, slot 1): gain over random 1 bp per entry ± 0.2 "
        "at 8h"
    ]


def test_a_veto_fails_the_claim_in_slot_3_and_says_its_h0_its_z_and_whether_it_could_see_the_effect() -> None:
    d = _decisive({"8h": 3.0, "24h": 3.0})
    weak = _row("funding", {"8h": 1.2, "24h": 0.5}, "each")
    world = _row("decisive", {"8h": 1.5, "24h": 3.0}, "each", world="null 1", by="decisive")
    world["name"], world["h0"], world["per"] = "ref.mean", "", "entry"
    v = combine([d], [weak, world], NEED)
    assert v["status"] == "FAILED" and v["slots"] == [3] and v["horizons"] == {"ref.mean": []}
    (note,) = v["notes"]
    assert note.startswith("ref.mean: beats random at 8h, 24h, z 3 ≥ 2.60 (decisive test, slot 1), but slot 3 rejects")
    assert (
        "8h: lab eval 'funding' (H0: the scores do not rank funding) vs the floor world: z 1.2 < 2.00 (at 8h); not "
        "rejected: its gain over the null at 8h, 0.012, is below what it can detect there (MDE 0.02842)"
    ) in note
    assert (
        "null 1 (tail: a few symbols carry the result) judging the decisive effect: z 1.5 < 2.00 (at 8h); not "
        "rejected: its gain over the null at 8h, 150 bp, is below what it can detect there (MDE 284 bp)"
    ) in note
    assert "24h: lab eval 'funding'" in note and "improve the features and submit again" in note


def test_a_veto_that_could_have_seen_the_effect_says_so() -> None:
    d = _decisive({"8h": 3.0, "24h": 3.0})
    row = _row("funding", {"8h": 4.0, "24h": 1.0})
    row["z"] = 1.5
    v = combine([d], [row], NEED)
    assert v["status"] == "FAILED" and v["slots"] == [3]
    assert (
        "lab eval 'funding' (H0: the scores do not rank funding) vs the floor world: z 1.5 < 2.00 (its family z, one "
        "vote for all its horizons); not rejected although it could detect effects of 0.02842 at 8h"
    ) in v["notes"][0]


def test_a_veto_whose_gain_is_not_above_the_null_says_it_points_the_other_way() -> None:
    d = _decisive({"8h": 3.0, "24h": 3.0})
    row = _row("funding", {"8h": -2.0, "24h": 0.0}, "each")
    v = combine([d], [row], NEED)
    (note,) = v["notes"]
    assert "z -2 < 2.00 (at 8h); not rejected: it points the other way (gain over the null -0.02 at 8h)" in note
    assert "z 0 < 2.00 (at 24h); not rejected: it points the other way (gain over the null 0 at 24h)" in note
    assert "below what it can detect" not in note and "although it could detect" not in note


def test_the_mde_of_a_veto_is_the_effect_that_reaches_two_four_times_in_five() -> None:
    assert math.isclose(pw.mde(1.0, VETO_Z), VETO_Z + 0.8416, rel_tol=1e-4)
    assert math.isclose(pw.mde(0.5, 3.0), 0.5 * (3.0 + pw.Z_BETA))
