"""The DataCard."""

import math
from typing import Any

from research_ci.render.common import adder, bp, check_line
from research_ci.render.feat import calibration_lines


def data_card(spec: dict[str, Any], idea: str, version: int) -> str:
    out = [f"DataCard · {idea} · v{version}"]
    add = adder(out)
    data = spec["data"]
    add(
        "data",
        f"{data['source']} {data['hash'][:12]} · {data['rows']:,} rows · {data['entities']} entities · "
        f"{data['first']} → {data['explore_end']} · {data['age_days']} days old",
    )
    add("source", *_source(spec["source"]))
    add("checks", *[check_line(c) for c in spec["checks"]])
    rows = spec["treatments"]
    if rows:
        add("treatments", f"{'':<40}{'rows':>10}{'|Δ| bp':>10}{'MDE':>9}  at", *[_treatment(t) for t in rows])
        add("", f"below {spec.get('materiality', 0.5):g} MDE: applied and logged · sizes only, never direction")
        if spec.get("kept"):
            add("", f"kept by a person: {', '.join(spec['kept'])}")
    else:
        add("treatments", "none needed")
    for t in spec["tests"]:
        mark = {"holds": "ok  ", "returned": "FAIL", "not calibrated": "[!] "}.get(t.get("status", ""), "--  ")
        add("tests", f"{mark} {t['name']}: flagged {t['flagged']} rows · on fail {t['on_fail']}")
    if spec["power"]:
        add("power", f"{'H':<12}{'MDE before':>11}{'after':>8}", *[_power_row(p) for p in spec["power"]])
    cal = spec.get("calibration")
    if cal is not None:
        add("calibrate", *(calibration_lines(cal) or [f"FAIL {n}" for n in cal["notes"]]))
    conc = spec.get("concentration")
    if conc and any(v for _, v in conc["shares"]):
        head = f"share of the noise that goes away without each {conc['clusters']}, largest first"
        add("clusters", head, *[_shares(k, v) for k, v in conc["shares"] if v])
    add("grades", _grades(spec["grades"]))
    add("decide", *(spec["decide"] or ["nothing"]))
    v = spec["verdict"]
    add("verdict", v["status"].upper(), *v["notes"])
    if spec["warnings"]:
        add("warnings", *[f"[!] {w}" for w in spec["warnings"]])
    out.append("-" * 100)
    return "\n".join(out)


def _source(src: dict[str, Any]) -> list[str]:
    """One line for a source someone admitted; its admission checks while no one has, since approving this card
    admits it for every idea."""
    head = f"{src['plugin']} · admission {src['admission'][:12]}"
    if src["status"] == "admitted":
        return [f"{head} · admitted"]
    status = "not yet admitted: approving this card admits it for every idea [!]"
    status = status if src["status"] == "waiting" else "NOT ADMITTED: it failed its admission checks [!]"
    lines = [f"{head} · {status}", f"code {src['code'][:12]} · {src['entities']} entities · {src['rows']:,} rows"]
    for c in src["checks"]:
        lines.append(check_line(c))
        if c["name"] == "independent source agrees" and src["reference"]:
            lines.append(f"     reference: {src['reference']}")
    return lines


def _grades(grades: dict[str, str]) -> str:
    by: dict[str, list[str]] = {}
    for f, g in sorted(grades.items()):
        by.setdefault(g, []).append(f)
    order = ("verified", "unverified", "not sampled", "failed")
    return " · ".join(f"{g}: {' '.join(by[g])}" for g in order if g in by)


def _treatment(t: dict[str, Any]) -> str:
    rows = f"{t['rows']:,}" if t["rows"] else ""
    if t.get("kept"):
        return _treatment({**t, "kept": False}) + "  kept, not applied"
    if not t.get("influence"):
        return f"{t['name'][:39]:<40}{rows:>10}{'n/a':>10}{'n/a':>9}"
    key, top = max(t["influence"].items(), key=lambda kv: _rank(kv[1]))
    if top.get("unbounded"):
        return f"{t['name'][:39]:<40}{rows:>10}{'inf':>10}{'inf':>9}  {key}"
    return f"{t['name'][:39]:<40}{rows:>10}{_size(top['bp']):>10}{_size(top['mde']):>9}  {key}"


def _rank(influence: dict[str, Any]) -> float:
    """An unbounded influence (a value that is not finite before the treatment) ranks first; one not measured last."""
    if influence.get("unbounded"):
        return math.inf
    x = influence["mde"]
    return x if x is not None and math.isfinite(x) else -1.0


def _size(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.2f}" if abs(x) < 1e4 else f"{x:.3g}"


def _shares(key: str, shares: list[list[Any]]) -> str:
    return f"{key:<12}" + " · ".join(f"{c} {x:.0%}" for c, x in shares)


def _power_row(p: dict[str, Any]) -> str:
    return f"{p['on'] + ' ' + p['horizon']:<12}{bp(p['mde_before']):>11}{bp(p['mde']):>8}"
