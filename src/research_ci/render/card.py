"""The contract card a person reads before freezing: rendered from the derived spec by a fixed template, no model."""

import math
from typing import Any

from research_ci.kernel.hashing import sha256_hex

MARK = {"pass": "ok", "fail": "FAIL", "warn": "[!]", "skip": "--"}


def card_id(text: str) -> str:
    """A short fingerprint of a card's text. An agent relaying a person's approval quotes it, so the approval is tied
    to the card the person was shown."""
    return sha256_hex(text.encode())[:8]


def contract_card(spec: dict[str, Any], idea: str, version: int) -> str:
    skip = f" (skips {', '.join(spec['skip'])})" if spec.get("skip") else ""
    out = [f"ContractCard · {idea} · v{version} · {spec['template']}{skip}"]
    add = _adder(out)
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


def _adder(out: list[str]) -> Any:
    def add(key: str, *lines: str | None) -> None:
        body = [line for line in lines if line]
        for i, line in enumerate(body):
            out.append(f"{key if i == 0 else '':<11}{line}")

    return add


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
        fail_below = _bp(None if r["fail_margin"] is None else d["at_least"] - r["fail_margin"])
        rows.append(
            f"{r['horizon']:<8}{r['decisions']:>10}{r['blocks']:>8.1f}{_bp(r['mde']):>9}{fail_below:>11}"
            f"{_days(r['forward_days']):>11}{flag}"
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


def _bp(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x * 1e4:.1f}"


def _days(days: float | None) -> str:
    if days is None or not math.isfinite(days):
        return "n/a"
    return f"{days:.0f} d" if days < 60 else f"{days / 30.4:.0f} mo"


def source_card(spec: dict[str, Any]) -> str:
    out = [f"SourceCard · {spec['plugin']} · code {spec['code'][:12]}"]
    add = _adder(out)
    add("fields", f"{' '.join(spec['fields'])} · {spec['entities']} entities · {spec['rows']:,} rows")
    checks = {c["name"]: c for c in spec["checks"]}
    first = [c for c in spec["checks"] if not c["name"].startswith(("replay", "independent", "canary"))]
    add("baseline", *[_mark(c) for c in first])
    if "replay as of past times" in checks:
        bad = [r for r in spec.get("replay", []) if r["status"] == "fail"]
        add("replay", _mark(checks["replay as of past times"]), *[f"     {_cut(r)}" for r in bad[:8]])
    if "independent source agrees" in checks:
        ref = spec.get("reference", {})
        counts = [f"{k}: {_counts(v)}" for k, v in ref.get("counts", {}).items()]
        note = f"     source: {ref['note']}" if ref.get("note") else None
        add("reference", _mark(checks["independent source agrees"]), note, *[f"     {c}" for c in counts])
    add("canaries", *[_mark(c) for c in spec["checks"] if c["name"].startswith("canary")])
    add("grades", _grades(spec["grades"]))
    v = spec["verdict"]
    add("verdict", v["status"].upper(), *v["notes"])
    out.append("-" * 100)
    return "\n".join(out)


def data_card(spec: dict[str, Any], idea: str, version: int) -> str:
    out = [f"DataCard · {idea} · v{version}"]
    add = _adder(out)
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
    add("checks", *[_mark(c) for c in spec["checks"]])
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


def _mark(c: dict[str, Any]) -> str:
    return f"{MARK[c['status']]:<4} {c['name']}: {c['detail']}"


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
    return (
        f"{p['on'] + ' ' + p['horizon']:<12}{_bp(p['mde_before']):>11}{_bp(p['mde']):>8}{_days(p['forward_days']):>11}"
    )
