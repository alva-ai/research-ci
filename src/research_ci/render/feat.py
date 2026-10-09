"""The MethodCard, the ResultCard and the ExploreCard."""

from typing import Any

from research_ci.render.common import adder, bp, check_line, num, pct


def method_card(spec: dict[str, Any], idea: str, version: int, budget: int | None = None) -> str:
    out = [f"MethodCard · {idea} · feat v{version}"]
    add = adder(out)
    combine = {"max": "best of its horizons", "each": "every horizon"}
    for c in spec["criteria"]:
        role = "required" if c["required"] else "shown only"
        note = f" · {c['note']}" if c["note"] else ""
        add(
            "criteria",
            f"{c['name']}: {c['label']} · H {' '.join(c['horizons'])} · {combine[c['combine']]} · {role}{note}",
        )
    for w in spec["worlds"]:
        world = f"PermuteWithin({w['bucket'] or 'time'}, clusters={w['clusters']}, block={w['block']})"
        add(
            "worlds",
            f"{w['name']} ({w['by']}): {world} · keeps {w['keeps'] or 'time'} · explains {', '.join(w['explains'])} · "
            f"{w['reps']} reps · {', '.join(w['criteria'])}",
            f"  {w['note']}" if w["note"] else None,
        )
    add("alts", *[f"{a['id']}: {a['text']}" for a in spec["alternatives"]])
    head = f"{'':<24}{'world':<10}{'H':<7}{'dies':>6}{'lives':>7}{'kept':>6}{'MDE':>10}  dies needs"
    add("calibrate", head, *[_calibration_row(r) for r in spec["calibration"]])
    add("", "a stand-in carries only what the world keeps: it may live only as often as chance allows;")
    add("", "a planted signal of 1.5 MDE must live ≥ 80%; kept: share of rows whose kept level survives the shuffle")
    for d in spec["decisive"]:
        add("decisive", *_decisive_method_lines(d))
    tables = [
        f"{label}: " + " · ".join(f"{h} {v['times']} times" for h, v in by_h.items())
        for label, by_h in spec["tables"].items()
    ]
    stride = f"every {spec['stride']} decision times" if spec["stride"] > 1 else "every decision time"
    add("tables", *tables, stride)
    z = spec["settings"]["z_base"]
    add(
        "judging",
        f"a criterion passes when it beats every world that judges it: z ≥ {z:g}·√(1+ln n) on the n-th submission",
        f"decisive: passed when z ≥ that and effect − {z:g}·se ≥ at_least; failed when effect + {z:g}·se < at_least;",
        "otherwise INSUFFICIENT · a leak in the scores TAINTS the submission",
    )
    if budget is not None:
        add("budget", f"{budget} submissions")
    add("checks", *[check_line(c) for c in spec["checks"]])
    v = spec["verdict"]
    add("verdict", v["status"].upper(), *v["notes"])
    out.append("-" * 100)
    return "\n".join(out)


def _calibration_row(r: dict[str, Any]) -> str:
    if "error" in r:
        return f"{r['criterion'][:23]:<24}{r['world']:<10}FAIL {r['error']}"
    kept = "--" if r["kept"] is None else f"{r['kept']:.0%}"
    mark = "" if r["ok"] else "  FAIL"
    return (
        f"{r['criterion'][:23]:<24}{r['world']:<10}{r['horizon']:<7}{r['stand_in_dies']:>6.0%}"
        f"{r['planted_lives']:>7.0%}{kept:>6}{num(r['mde']):>10}  ≥ {r['dies_at_least']:.0%}{mark}"
    )


def _decisive_method_lines(d: dict[str, Any]) -> list[str]:
    if d["kind"] == "label":
        return [
            f"{d['on']} ≥ {d['at_least'] * 1e4:.1f} bp {d['unit']} · target {d['target']} with the scores",
            f"  H {' '.join(d['horizons'])}: z of the best against the floor world; passed when one horizon passes,",
            "  failed when every horizon fails",
        ]
    p = d["pilot"]
    design = d["design"]
    how = (
        f"branch every {design['every']} for {design['horizon']}, counted {design['tail']} after"
        if design["kind"] == "branch"
        else "full path"
    )
    return [
        f"consumer ≥ {d['at_least']:.1%}/yr {d['unit']}",
        f"  over {d['reference']} · change: {p['kind']} {p['share']:.0%} per {p['block']} · {how}",
    ]


def result_card(spec: dict[str, Any], idea: str, version: int, budget: int | None = None) -> str:
    of = f" of {budget}" if budget else ""
    out = [f"ResultCard · {idea} · feat v{version} · submission {spec['n']}{of}"]
    add = adder(out)
    f = spec["features"]
    add("features", f"{f['path']} · {f['hash'][:12]}")
    sc = spec["scores"]
    add("scores", f"{sc['rows']:,} rows · {sc['share']:.0%} of candidate rows · {sc['hash'][:12]}")
    add("floors", *[check_line(c) for c in spec["floors"]])
    add("need", f"z ≥ {spec['need']:.2f} on submission {spec['n']}")
    for r in spec.get("criteria", []):
        mark = "ok  " if r["passed"] else "FAIL"
        best = f" at {r['best']}" if r.get("best") else ""
        explains = ", ".join(r["explains"])
        add("criteria", f"{mark} {r['criterion']} vs {r['world']} ({explains}): z {num(r['z'])}{best} · {r['note']}")
        for h in r["horizons"]:
            counted = "" if h["counted"] else "  not counted"
            add(
                "",
                f"       {h['h']:<7}value {num(h['real']):>10} ± {num(h['se']):<10} null {num(h['null_mean']):>10} "
                f"sd {num(h['null_sd']):<10} {h['blocks']:>4} blocks  z {num(h['z'])}{counted}",
            )
    for d in spec.get("decisive", []):
        add("decisive", *_decisive_result_lines(d))
    v = spec["verdict"]
    add("verdict", v["status"], *v["notes"])
    out.append("-" * 100)
    return "\n".join(out)


def _decisive_result_lines(d: dict[str, Any]) -> list[str]:
    if d["kind"] == "label":
        return [
            f"{d['on']}: {bp(d['effect'])} bp ± {bp(d['se'])} at {d['best']} · z {num(d['z'])} · "
            f"needs ≥ {d['at_least'] * 1e4:.1f} bp · {d['outcome'].upper()}"
        ]
    lines = [
        f"consumer: {pct(d['effect'])}/yr ± {pct(d['se'])} over {d['over']} · z {num(d['z'])} · "
        f"needs ≥ {d['at_least']:.1%}/yr · {d['outcome'].upper()}",
    ]
    if d["design"] == "branch":
        lines.append(f"  {d['branches']} branches · noise {pct(d['null_sd'])}/yr per branch")
    else:
        r = d["run"]
        lines.append(
            f"  {d['runs']} runs with the change · noise {pct(d['null_sd'])}/yr · this run {pct(r['per_year'])}/yr · "
            f"max drawdown {r['max_drawdown']:.2%} · turnover {r['turnover']:.1f}× · {r['trades']} trades"
        )
    return lines


def explore_card(spec: dict[str, Any], idea: str) -> str:
    out = [f"ExploreCard · {idea} · feat (not a submission; nothing is judged)"]
    add = adder(out)
    f = spec["features"]
    add("features", f"{f['path']} · {f['hash'][:12]}")
    sc = spec["scores"]
    add("scores", f"{sc['rows']:,} rows · {sc['share']:.0%} of candidate rows · {sc['hash'][:12]}")
    add("floors", *[check_line(c) for c in spec["floors"]])
    out.append("-" * 100)
    return "\n".join(out)
