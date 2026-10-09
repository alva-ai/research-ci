"""The ContractCard a person reads before freezing: rendered from the derived spec by a fixed template, no model."""

from typing import Any

from research_ci.render.common import MARK, adder, bp, days


def contract_card(spec: dict[str, Any], idea: str, version: int) -> str:
    skip = f" (skips {', '.join(spec['skip'])})" if spec.get("skip") else ""
    out = [f"ContractCard · {idea} · v{version} · {spec['template']}{skip}"]
    add = adder(out)
    add("claim", spec["claim"])
    for d in spec["decisive"]:
        add("decisive", _decisive_line(d))
    dec = spec["decisions"]
    add("decisions", f"every {dec['every']} from {dec['from']} · {dec['count']} decision times", dec.get("note"))
    data = spec["data"]
    add(
        "data",
        f"{data['source']} {data['hash'][:12]} · {data['rows']:,} rows · {data['entities']} entities · "
        f"{data['first']} → {data['explore_end']} · {data['age_days']} days old",
    )
    if "universe" in spec:
        u = spec["universe"]
        add(
            "universe",
            f"{u['plugin']} · median {u['candidates_median']:.0f} candidates in {u['clusters_median']:.0f} "
            f"{u['clusters']} clusters per decision · membership {u['membership']}",
        )
    c = spec["costs"]
    options = ", ".join(f"{k}={v}" for k, v in c["options"].items())
    round_trip = "" if c["round_trip_median"] is None else f" · median round trip {c['round_trip_median'] * 1e4:.1f} bp"
    add("costs", f"{c['plugin']}({options}){round_trip}")
    if spec.get("labels"):
        add("labels", *[f"{lab['name']}: {lab['plugin']} · H {' '.join(lab['horizons'])}" for lab in spec["labels"]])
    if "baseline" in spec:
        add("baseline", *_baseline_lines(spec["baseline"]))
    for d in spec["decisive"]:
        if d["kind"] == "consumer":
            add("backtest", *_run_lines(d["baseline_run"]))
    add("checks", *[f"{MARK[ch['status']]:<4} {ch['name']}: {ch['detail']}" for ch in spec["checks"]])
    for d in spec["decisive"]:
        add("power", f"{d['on']}:", *(_label_table(d) if d["kind"] == "label" else _consumer_lines(d)))
    alts = spec["alternatives"]
    add("alts", " · ".join(a["id"] for a in alts["idea"]) + f"  (engine floor: {', '.join(alts['floor'])})")
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
    add("budget", f"{budget_text}    deadline {spec['deadline_days']:.0f} days")
    v = spec["verdict"]
    add("verdict", v["status"].upper(), *v["notes"])
    if spec["warnings"]:
        add("warnings", *[f"[!] {w}" for w in spec["warnings"]])
    out.append("-" * 100)
    return "\n".join(out)


def _decisive_line(d: dict[str, Any]) -> str:
    if d["kind"] == "label":
        return f"{d['on']} ≥ {d['at_least'] * 1e4:.1f} bp {d['unit']} · target {d['target']}"
    return f"consumer ≥ {d['at_least']:.1%}/yr {d['unit']}"


def _baseline_lines(b: dict[str, Any]) -> list[str]:
    files = " · ".join(f"{k} {v['hash'][:8]}" for k, v in b["files"].items())
    lines = [f"score {b['score']} · {files}"]
    live = b.get("live_config")
    if live:
        differ = ", ".join(live["differ"]) or "none"
        lines.append(f"live config: {live['common']} shared keys, differing: {differ}; {live['only_live']} only live")
    return lines


def _run_lines(r: dict[str, Any]) -> list[str]:
    parts = " · ".join(f"{k} {v:+.2%}" for k, v in r["components"].items())
    lines = [
        f"baseline {r['total']:+.2%} ({r['per_year']:+.1%}/yr) · max drawdown {r['max_drawdown']:.2%} · "
        f"turnover {r['turnover']:.1f}× · {r['trades']} trades",
        parts,
    ]
    lines += [f"[!] {k}: {v}" for k, v in r["policies"].items()]
    return lines


def _label_table(d: dict[str, Any]) -> list[str]:
    rows = [f"{'H':<8}{'decisions':>10}{'blocks':>8}{'MDE bp':>9}{'FAIL if <':>11}{'new data':>11}"]
    for r in d["horizons"]:
        flag = "  [!] few blocks" if r["blocks"] < 5 else ""
        fail_below = bp(None if r["fail_margin"] is None else d["at_least"] - r["fail_margin"])
        rows.append(
            f"{r['horizon']:<8}{r['decisions']:>10}{r['blocks']:>8.1f}{bp(r['mde']):>9}{fail_below:>11}"
            f"{days(r['forward_days']):>11}{flag}"
        )
    return rows


def _consumer_lines(d: dict[str, Any]) -> list[str]:
    pilot = d.get("pilot")
    if not pilot:
        return ["no pilot: the consumer has no baseline scores to perturb"]
    p = d["power"]
    change = f"{pilot['kind']} {pilot['share']:.0%} of the baseline's candidates at random"
    wait = "never at this noise" if p["forward_months"] is None else f"about {p['forward_months']:.1f} months"
    if pilot["design"] == "branch":
        return [
            f"design: branch every {pilot['every']}, {change} for {pilot['horizon']}, count {pilot['tail']} after",
            f"  {pilot['branches']} branches · sd {pilot['sd']:.3%} · mean change {pilot['mean_shift']:+.3%} · "
            f"positions still shared {pilot['overlap']:.0%}",
            f"history detects ≥ {p['history_mde']:.1%}/yr · new data: {wait}",
        ]
    lines = [f"design: full path · pilot: {change}, per {pilot['block']}"]
    for w in pilot["windows"]:
        lines.append(
            f"  {w['days']:.0f} days × {w['parts']}: {w['runs']} runs · sd {w['sd']:.2%} · "
            f"mean change {w['mean_shift']:+.2%}"
        )
    beta = pilot["beta"]
    lines.append(f"noise grows as days^{beta:.2f} · history detects ≥ {p['history_mde']:.1%}/yr · new data: {wait}")
    return lines
