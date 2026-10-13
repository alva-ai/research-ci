"""The DataCard, slot by slot."""

import math
from typing import Any

from research_ci.render.common import MARK, adder, bp, check_line, eval_line, failed_tests
from research_ci.render.contract import data_line
from research_ci.render.feat import calibration_lines

SLOT_HEADS = {
    1: "1 floor · the source's admission, and what the engine weighs on every idea's data",
    2: "2 engine lists · the default checks and the library evals data.py lists; what they flag is treated by default",
    3: (
        "3 lab evals · tests of this idea's data: what each flags is treated as its on_fail says; with on_fail=block "
        "any row it flags in the data after the treatments before it vetoes the data"
    ),
    4: "4 tests of lab evals · their CASES and PROPERTIES before the data was fetched, then the pits each covers",
}


def data_card(spec: dict[str, Any], idea: str, version: int) -> str:
    out = [f"DataCard · {idea} · v{version}"]
    add = adder(out)
    add("data", data_line(spec["data"]))
    out.append(SLOT_HEADS[1])
    add("source", *_source(spec["source"]))
    add("checks", *[check_line(c) for c in spec["checks"] if c.get("by") in ("floor", "")])
    add("grades", _grades(spec["grades"]))
    rows = spec["treatments"]
    if rows:
        add("treatments", f"{'':<40}{'rows':>10}{'|Δ| bp':>10}{'MDE':>9}  at", *[_treatment(t) for t in rows])
        add("", f"below {spec.get('materiality', 0.5):g} MDE: applied and logged · sizes only, never direction")
        if spec.get("kept"):
            add("", f"kept by a person: {', '.join(spec['kept'])}")
    else:
        add("treatments", "none needed")
    if spec["power"]:
        add("power", f"{'H':<16}{'MDE before':>11}{'after':>8}", *[_power_row(p) for p in spec["power"]])
    cal = spec.get("calibration")
    if cal is not None:
        add("calibrate", *(calibration_lines(cal) or [f"FAIL {n}" for n in cal["notes"]]))
    out.append(SLOT_HEADS[2])
    add("checks", *_evals(spec, ("default", "library")))
    conc = spec.get("concentration")
    if conc and any(v for _, v in conc["shares"]):
        head = f"share of the noise that goes away without each {conc['clusters']}, largest first"
        add("clusters", head, *[_shares(k, v) for k, v in conc["shares"] if v])
    lab = [r for r in spec.get("evals", []) if r["by"] == "lab"]
    out.append(SLOT_HEADS[3])
    add("lab evals", *[line for r in lab for line in _lab(r, spec)] or ["none"])
    out.append(SLOT_HEADS[4])
    add("tests", *[_tested(r) for r in lab] or ["none"])
    add("decide", *(spec["decide"] or ["nothing"]))
    v = spec["verdict"]
    add("verdict", v["status"].upper(), *v["notes"])
    if spec["warnings"]:
        add("warnings", *[f"[!] {w}" for w in spec["warnings"]])
    out.append("-" * 100)
    return "\n".join(out)


def _evals(spec: dict[str, Any], by: tuple[str, ...]) -> list[str]:
    """The checks of slot 2 that ran, in order, with whose they are: the default list, then the library evals, each
    with its validation; under one invalid by its pits, every other pit it missed."""
    rows = {r["name"]: r for r in spec.get("evals", [])}
    out = []
    for c in spec["checks"]:
        if c.get("by") not in by:
            continue
        out.append(eval_line(c))
        row = rows.get(c["name"]) if c.get("by") == "library" else None
        out += failed_tests(row["validation"]) if row else []
    return out or ["nothing"]


def _lab(r: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """A lab eval with its H0, what it flagged and what that did; a veto with a few of the rows it flagged."""
    check = next((c for c in spec["checks"] if c["name"] == r["name"] and c.get("by") == "lab"), None)
    file = r["validation"].get("file", "")
    mark = MARK[check["status"]] if check is not None else MARK["skip"]
    lines = [f"{mark:<4} {r['name']}{f' ({file})' if file else ''} · H0: {r['h0']}"]
    if check is not None:
        lines.append(f"     on_fail {r['on_fail']}: {check['detail']}")
    if r.get("examples"):
        lines.append(f"     such as {', '.join(r['examples'])}")
    return lines + failed_tests(r["validation"])


def _tested(r: dict[str, Any]) -> str:
    v = r["validation"]
    pits = [t for t in v["tests"] if t["name"].startswith(("built-in ", "pit ", "pits: "))]
    caught = f" · {sum(t['status'] == 'pass' for t in pits)}/{len(pits)} pits caught" if pits else ""
    status = "FAIL" if v["status"] == "invalid" else "ok  "
    file = f" ({v['file']})" if v.get("file") else ""
    return f"{status} {r['name']}{file}: {v['summary']}{caught}"


def _source(src: dict[str, Any]) -> list[str]:
    """One line for a source someone admitted; its admission checks while no one has, since approving this card
    admits it for every idea."""
    head = f"{src['plugin']} · admission {src['admission'][:12]}"
    if src["status"] == "admitted":
        return [f"{head} · admitted"]
    status = "not yet admitted: approving this card admits it for every idea [!]"
    status = status if src["status"] == "waiting" else "NOT ADMITTED: it failed its admission checks [!]"
    shown = " · ".join(f"{name} ({', '.join(columns)})" for name, columns in src["sources"].items())
    lines = [f"{head} · {status}", f"code {src['code'][:12]} · {src['insts']} insts · {src['rows']:,} rows · {shown}"]
    for c in src["checks"]:
        lines.append(check_line(c))
        if c["name"] == "independent source agrees" and src["reference"]:
            lines.append(f"     reference: {src['reference']}")
    return lines


def _grades(grades: dict[str, str]) -> str:
    """Each column of the sources, ``<source>.<column>``, under its grade."""
    by: dict[str, list[str]] = {}
    for c, g in sorted(grades.items()):
        by.setdefault(g, []).append(c)
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
    return f"{key:<16}" + " · ".join(f"{c} {x:.0%}" for c, x in shares)


def _power_row(p: dict[str, Any]) -> str:
    return f"{p['name'] + ' ' + p['horizon']:<16}{bp(p['mde_before']):>11}{bp(p['mde']):>8}"
