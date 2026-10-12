"""The ContractCard a person reads before freezing: rendered from the derived spec by fixed code, no model."""

from typing import Any

from research_ci.api.contract import SHAPES
from research_ci.render.common import MARK, adder, figure, rolled


def contract_card(spec: dict[str, Any], idea: str, version: int) -> str:
    out = [f"ContractCard · {idea} · v{version}"]
    add = adder(out)
    add("claim", spec["claim"])
    for d in spec["decisive"]:
        add("decisive", _decisive_line(d))
    dec = spec["decisions"]
    add("decisions", f"every {dec['every']} from {dec['from']} · {dec['count']} decision times", dec.get("note"))
    data = spec["data"]
    add(
        "data",
        f"{data['source']} {data['hash'][:12]} · {data['rows']:,} rows · {data['insts']} insts · "
        f"{data['first']} → {data['explore_end']} · {data['age_days']} days old",
    )
    if "universe" in spec:
        u = spec["universe"]
        add(
            "universe",
            f"{u['plugin']} · median {u['candidates_median']:.0f} candidates in {u['clusters_median']:.0f} "
            f"{u['clusters']} clusters per decision · membership {u['membership']}",
        )
        shape = u.get("shape", "cross")
        world = f" · floor world {u['world']}" if u.get("world") else ""
        add("shape", f"{shape}: {SHAPES[shape]}{world}")
    c = spec["costs"]
    options = ", ".join(f"{k}={v}" for k, v in c["options"].items())
    round_trip = "" if c["round_trip_median"] is None else f" · median round trip {c['round_trip_median'] * 1e4:.1f} bp"
    add("costs", f"{c['plugin']}({options}){round_trip}")
    if spec.get("labels"):
        add("labels", *[f"{lab['name']}: {lab['plugin']} · H {' '.join(lab['horizons'])}" for lab in spec["labels"]])
    add("checks", *[f"{MARK[ch['status']]:<4} {ch['name']}: {ch['detail']}" for ch in spec["checks"]])
    for d in spec["decisive"]:
        add("power", f"{d['on']}: the smallest gain over random the data can detect (MDE)", *_label_table(d))
    alts = spec["alternatives"]
    add(
        "alts",
        " · ".join(a["id"] for a in alts["idea"]) + f"  (engine floor: {', '.join(alts['floor'])})",
        "each needs a null world in nulls/feat.py, or feat refuses a submission before it is counted; a world can veto",
    )
    used = [
        f"{u['start'][:10]} → {u['end'][:10]}: {u['what']} (affects {', '.join(u['affects'])})" for u in spec["used"]
    ]
    add("used", *(used or ["nothing declared"]))
    history = spec["history"]
    add("history", "only new data can confirm" if history == "explore-only" else "history can confirm")
    fwd = spec.get("forward")
    if fwd:
        add("forward", f"{fwd['source']} · {'installed' if fwd['installed'] else 'NOT INSTALLED [!]'}")
    budget = spec["budget"]
    budget_text = " · ".join(f"{k} {v}" for k, v in budget.items()) if isinstance(budget, dict) else str(budget)
    add("budget", budget_text)
    v = spec["verdict"]
    add("verdict", v["status"].upper(), *v["notes"])
    if spec["warnings"]:
        add("warnings", *[f"[!] {w}" for w in spec["warnings"]])
    out.append("-" * 100)
    return "\n".join(out)


def _decisive_line(d: dict[str, Any]) -> str:
    options = ", ".join(f"{k}={v}" for k, v in d.get("options", {}).items())
    return f"{d['on']}: beats random significantly · target {d['target']}({options})"


def _label_table(d: dict[str, Any]) -> list[str]:
    rolls = d.get("rolls")
    mde = f"MDE bp/{d['per']}" if d.get("per") else "MDE"
    rows = [f"{'H':<8}{'decisions':>10}{'blocks':>8}{mde:>14}" + ("  if rolled" if rolls else "")]
    for r in d["horizons"]:
        line = f"{r['horizon']:<8}{r['decisions']:>10}{r['blocks']:>8.1f}{figure(r['mde'], d):>14}"
        line += f"  {rolled(r['mde'], r['horizon'], d):<10}" if rolls else ""
        flag = "  [!] few blocks" if r["blocks"] < 5 else ""
        rows.append((line + flag).rstrip())
    return rows
