"""The SourceCard and the DataCard."""

import math
from typing import Any

from research_ci.render.common import adder, bp, check_line, days


def source_card(spec: dict[str, Any]) -> str:
    out = [f"SourceCard · {spec['plugin']} · code {spec['code'][:12]}"]
    add = adder(out)
    add("fields", f"{' '.join(spec['fields'])} · {spec['entities']} entities · {spec['rows']:,} rows")
    checks = {c["name"]: c for c in spec["checks"]}
    first = [c for c in spec["checks"] if not c["name"].startswith(("replay", "independent", "canary"))]
    add("baseline", *[check_line(c) for c in first])
    if "replay as of past times" in checks:
        bad = [r for r in spec.get("replay", []) if r["status"] == "fail"]
        add("replay", check_line(checks["replay as of past times"]), *[f"     {_cut(r)}" for r in bad[:8]])
    if "independent source agrees" in checks:
        ref = spec.get("reference", {})
        counts = [f"{k}: {_counts(v)}" for k, v in ref.get("counts", {}).items()]
        note = f"     source: {ref['note']}" if ref.get("note") else None
        add("reference", check_line(checks["independent source agrees"]), note, *[f"     {c}" for c in counts])
    add("canaries", *[check_line(c) for c in spec["checks"] if c["name"].startswith("canary")])
    add("grades", _grades(spec["grades"]))
    v = spec["verdict"]
    add("verdict", v["status"].upper(), *v["notes"])
    out.append("-" * 100)
    return "\n".join(out)


def data_card(spec: dict[str, Any], idea: str, version: int) -> str:
    out = [f"DataCard · {idea} · v{version}"]
    add = adder(out)
    data = spec["data"]
    add(
        "data",
        f"{data['source']} {data['hash'][:12]} · {data['rows']:,} rows · {data['entities']} entities · "
        f"{data['first']} → {data['explore_end']} · {data['age_days']} days old",
    )
    src = spec["source"]
    status = {"admitted": "admitted", "waiting": "WAITING for a person [!]", "not admitted": "NOT ADMITTED [!]"}
    detail = f"{src['plugin']} {status[src['status']]} · admission {src['admission'][:12]}"
    failed = f"failed: {', '.join(src['failed'])}" if src["failed"] else None
    add("source", detail, failed)
    add("checks", *[check_line(c) for c in spec["checks"]])
    rows = spec["treatments"]
    if rows:
        add("treatments", f"{'':<40}{'rows':>10}{'|Δ| bp':>9}{'MDE':>7}  at", *[_treatment(t) for t in rows])
        add("", f"below {spec.get('materiality', 0.5):g} MDE: applied and logged · sizes only, never direction")
        if spec.get("kept"):
            add("", f"kept by a person: {', '.join(spec['kept'])}")
    else:
        add("treatments", "none needed")
    for t in spec["tests"]:
        mark = {"holds": "ok  ", "returned": "FAIL", "not calibrated": "[!] "}.get(t.get("status", ""), "--  ")
        add("tests", f"{mark} {t['name']}: flagged {t['flagged']} rows · on fail {t['on_fail']}")
    if spec["power"]:
        add(
            "power", f"{'H':<12}{'MDE before':>11}{'after':>8}{'new data':>11}", *[_power_row(p) for p in spec["power"]]
        )
    add("grades", _grades(spec["grades"]))
    add("decide", *(spec["decide"] or ["nothing"]))
    v = spec["verdict"]
    add("verdict", v["status"].upper(), *v["notes"])
    if spec["warnings"]:
        add("warnings", *[f"[!] {w}" for w in spec["warnings"]])
    out.append("-" * 100)
    return "\n".join(out)


def _cut(r: dict[str, Any]) -> str:
    what = r["error"] or (
        f"{r['only_replay']} rows only as of then, {r['only_full']} only in the full fetch, {r['changed']} changed"
    )
    return f"{r['at']}  {r['why']}: {what} · {', '.join(r['differ'][:3])}"


def _counts(c: dict[str, int]) -> str:
    return " · ".join(f"{k} {v}" for k, v in sorted(c.items()))


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
        return f"{t['name'][:39]:<40}{rows:>10}{'n/a':>9}{'n/a':>7}"
    key, top = max(
        t["influence"].items(), key=lambda kv: kv[1]["mde"] if math.isfinite(kv[1]["mde"] or math.nan) else -1
    )
    mde = top["mde"] if top["mde"] is not None else math.nan
    bp = top["bp"] if top["bp"] is not None else math.nan
    return f"{t['name'][:39]:<40}{rows:>10}{bp:>9.2f}{mde:>7.2f}  {key}"


def _power_row(p: dict[str, Any]) -> str:
    return f"{p['on'] + ' ' + p['horizon']:<12}{bp(p['mde_before']):>11}{bp(p['mde']):>8}{days(p['forward_days']):>11}"
