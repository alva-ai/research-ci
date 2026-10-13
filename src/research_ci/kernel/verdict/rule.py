"""A feat submission's verdict, horizon by horizon.

The decisive test (slot 1) passes at a horizon whose z against the floor world, set against the null the family's z
uses (``adjusted``), reaches the bar ``z_need``. There, at least one lab eval must cover the horizon, and every lab
eval and null world that covers it (slot 3) must reach ``VETO_Z``, a bar that does not rise with the submissions: false
passes are held off by the decisive test's bar alone, so lab evals and worlds can only take horizons away, never add
one. A decisive effect passes at the horizons that pass; a submission is PASSED when every decisive effect passes at
one or more, and the verdict names them; otherwise it is FAILED, with the slot that rejected it and why.
"""

import math
from typing import Any

VETO_Z = 2.0
FIX = (
    "a lab eval or a null world below z 2.00 at a horizon fails the claim there: improve the features and submit "
    "again; where an eval's MDE is above its gain it cannot see an effect that small on this data, and where no lab "
    "eval covers a horizon, list one in feat.py whose horizons include it"
)


def passing(d: dict[str, Any], need: float) -> list[str]:
    """The horizons where the decisive test passes: counted, with a z against the family's null of ``need`` or
    more."""
    return [h["h"] for h in d["horizons"] if h.get("counted") and _reaches(h.get("adjusted"), need)]


def votes(row: dict[str, Any]) -> dict[str, bool]:
    """Whether a lab eval's or a null world's judgement reaches ``VETO_Z`` at each horizon it covers: one vote for all
    of them with its family's z (``combine="max"``), or each counted horizon's own z (``each``)."""
    if row["combine"] == "max":
        return dict.fromkeys((h["h"] for h in row["horizons"]), _reaches(row["z"], VETO_Z))
    return {h["h"]: bool(h["counted"]) and _reaches(h["z"], VETO_Z) for h in row["horizons"]}


def combine(decisive: list[dict[str, Any]], vetoes: list[dict[str, Any]], need: float) -> dict[str, Any]:
    """The verdict from the decisive effects (slot 1) and what slot 3 judged; ``horizons`` names each effect's
    passing horizons, and ``slots`` the slots that rejected a FAILED one."""
    covered = {h["h"] for r in vetoes if r["by"] == "lab" and r["world"] == "floor" for h in r["horizons"]}
    lines: list[str] = []
    passed: dict[str, list[str]] = {}
    slots: set[int] = set()
    for d in decisive:
        beats = passing(d, need)
        rejected = {h: _rejections(h, vetoes, covered) for h in beats}
        ok = [h for h in beats if not rejected[h]]
        passed[d["name"]] = ok
        z = f"z {_num(d['z'])}"
        if ok:
            them = "it" if len(ok) == 1 else "them"
            lines.append(
                f"{d['name']}: passes at {', '.join(ok)}: the decisive test beats random there, {z} ≥ {need:.2f} "
                f"(slot 1), and every lab eval and null world that covers {them} reaches z ≥ {VETO_Z:.2f} (slot 3)"
            )
            lines += [
                f"{d['name']} {h}: beats random, but slot 3 rejects it: {'; '.join(rejected[h])}"
                for h in beats
                if rejected[h]
            ]
        elif beats:
            slots.add(3)
            why = " · ".join(f"{h}: {'; '.join(rejected[h])}" for h in beats)
            lines.append(
                f"{d['name']}: beats random at {', '.join(beats)}, {z} ≥ {need:.2f} (decisive test, slot 1), but slot "
                f"3 rejects every one of them: {why}; {FIX}"
            )
        else:
            slots.add(1)
            lines.append(f"{d['name']}: does not beat random, {z} < {need:.2f} (decisive test, slot 1): {_effect(d)}")
    status = "PASSED" if all(passed.values()) else "FAILED"
    return verdict(status, lines) | {"horizons": passed, "slots": sorted(slots)}


def _rejections(h: str, vetoes: list[dict[str, Any]], covered: set[str]) -> list[str]:
    """Why slot 3 rejects horizon ``h``: no lab eval covers it, or what covers it and does not reach ``VETO_Z``."""
    out = [] if h in covered else [f"no lab eval covers {h}"]
    for row in vetoes:
        at = next((x for x in row["horizons"] if x["h"] == h), None)
        if at is not None and not votes(row)[h]:
            out.append(_vetoed(row, at))
    return out


def _vetoed(row: dict[str, Any], at: dict[str, Any]) -> str:
    """One veto in words: who, its H0, its z against the bar, and whether it could have seen the effect."""
    judged = f"{row['by']} eval {row['name']!r}" if row["by"] != "decisive" else "the decisive effect"
    judged += f" (H0: {row['h0']})" if row["by"] == "lab" else ""
    who = f"{judged} vs the floor world"
    if row["world"] != "floor":
        who = f"{row['world']} ({', '.join(row['explains'])}: {row['alternative']}) judging {judged}"
    scope = "its family z, one vote for all its horizons" if row["combine"] == "max" else f"at {at['h']}"
    z = row["z"] if row["combine"] == "max" else at["z"]
    return f"{who}: z {_num(z)} < {VETO_Z:.2f} ({scope}); {_power(row, at)}"


def _power(row: dict[str, Any], at: dict[str, Any]) -> str:
    """Why a veto's H0 stands: the evidence points the other way, the eval could not see an effect this small, or it
    could and saw none."""
    gain, mde = at.get("gain", math.nan), at.get("mde", math.nan)
    if math.isfinite(gain) and gain <= 0:
        return f"not rejected: it points the other way (gain over the null {_amount(gain, row)} at {at['h']})"
    if math.isfinite(gain) and math.isfinite(mde) and gain < mde:
        return (
            f"not rejected: its gain over the null at {at['h']}, {_amount(gain, row)}, is below what it can detect "
            f"there (MDE {_amount(mde, row)})"
        )
    return f"not rejected although it could detect effects of {_amount(mde, row)} at {at['h']}"


def _effect(d: dict[str, Any]) -> str:
    """What the decisive effect came to, for a FAILED verdict: the gain over random and its error at the best horizon,
    or why no z could be computed."""
    if d.get("best") is None:
        return "no horizon has enough blocks for a z; the evaluation tables are too short for this contract"
    per = f" per {d['per']}" if d.get("per") else ""
    scale, unit = (1e4, " bp") if d.get("per") else (1.0, "")
    gain, se = d["effect"] * scale, d["se"] * scale
    return f"gain over random {gain:.3g}{unit}{per} ± {se:.3g} at {d['best']}"


def verdict(status: str, notes: list[str]) -> dict[str, Any]:
    return {"status": status, "notes": notes}


def _reaches(z: Any, bar: float) -> bool:
    return z is not None and math.isfinite(z) and z >= bar


def _amount(x: float, row: dict[str, Any]) -> str:
    return f"{x * 1e4:.3g} bp" if row.get("per") and math.isfinite(x) else _num(x)


def _num(x: Any) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.4g}"
