"""The ResultCard and the ExploreCard."""

from typing import Any

from research_ci.render.common import MARK, adder, amount, check_line, figure, num, rolled


def calibration_lines(cal: dict[str, Any]) -> list[str]:
    """The decisive effect's calibration in the floor world: a failed check, then one line per effect."""
    lines = [check_line(c) for c in cal["checks"] if c["status"] == "fail"]
    for r in cal["rows"]:
        if "error" in r:
            lines.append(f"FAIL {r['effect']} vs {r['world']}: {r['error']}")
            continue
        lines.append(
            f"{MARK['pass' if r['ok'] else 'fail']:<4} {r['effect']} at {r['horizon']} vs {r['world']}: stand-ins die "
            f"{r['stand_in_dies']:.0%} (needs ≥ {r['dies_at_least']:.0%}) · a signal planted at 1.5 MDE "
            f"({amount(1.5 * r['mde'], r)}) lives {r['planted_lives']:.0%} (needs ≥ 80%)"
        )
    return lines


def result_card(spec: dict[str, Any], idea: str, version: int) -> str:
    """A submission's card, rendered from its ledger record alone. The calibration of the decisive effect shows on the
    submission that ran it; later ones reuse it."""
    of = f" of {spec['budget']}" if spec.get("budget") else ""
    out = [f"ResultCard · {idea} · feat v{version} · submission {spec['n']}{of}"]
    add = adder(out)
    f = spec["features"]
    add("features", f"{f['path']} · {f['hash'][:12]}")
    sc = spec["scores"]
    add("scores", f"{sc['rows']:,} rows · {sc['share']:.0%} of candidate rows · {sc['hash'][:12]}")
    add("floors", *[check_line(c) for c in spec["floors"]])
    add("cost", cost_line(spec.get("cost")))
    cal = spec["calibration"]
    if cal["ran"]:
        add("calibrate", *calibration_lines(cal))
    add("need", f"z ≥ {spec['need']:.2f} on submission {spec['n']}")
    for d in spec.get("decisive", []):
        add("decisive", _decisive_line(d))
    if spec.get("criteria"):
        add("shown", "diagnostics, not deciding: the engine's defaults and the idea's criteria")
    for r in spec.get("criteria", []):
        mark = "ok  " if r["passed"] else "[!] "
        best = f" at {r['best']}" if r.get("best") else ""
        explains = ", ".join(r["explains"])
        head = f"{mark} {r['criterion']} ({r['by']}) vs {r['world']} ({explains}): z {num(r['z'])}{best} · {r['note']}"
        add("", head)
        for h in r["horizons"]:
            counted = "" if h["counted"] else "  not counted"
            add(
                "",
                f"       {h['h']:<7}value {num(h['real']):>10} ± {num(h['se']):<10} null {num(h['null_mean']):>10} "
                f"sd {num(h['null_sd']):<10} {h['blocks']:>4} blocks  z {num(h['z'])}{counted}",
            )
    add("worlds", check_line(spec["coverage"]))
    v = spec["verdict"]
    add("verdict", v["status"], *v["notes"])
    add("warnings", *[f"[!] {w}" for w in spec.get("warnings", [])])
    out.append("-" * 100)
    return "\n".join(out)


def _decisive_line(d: dict[str, Any]) -> str:
    at = f" at {d['best']}" if d["best"] else ""
    year = rolled(d["effect"], d["best"], d) if d["best"] else ""
    year = f" ({year} if rolled)" if year else ""
    return (
        f"{d['on']}: gain over random {amount(d['effect'], d)} ± {figure(d['se'], d)}{at}{year} · "
        f"z {num(d['z'])} · {d['outcome'].upper()}"
    )


def explore_card(spec: dict[str, Any], idea: str) -> str:
    out = [f"ExploreCard · {idea} · feat (not a submission; nothing is judged)"]
    add = adder(out)
    f = spec["features"]
    add("features", f"{f['path']} · {f['hash'][:12]}")
    sc = spec["scores"]
    add("scores", f"{sc['rows']:,} rows · {sc['share']:.0%} of candidate rows · {sc['hash'][:12]}")
    add("floors", *[check_line(c) for c in spec["floors"]])
    add("cost", cost_line(spec.get("cost")))
    add("warnings", *[f"[!] {w}" for w in spec.get("warnings", [])])
    out.append("-" * 100)
    return "\n".join(out)


def cost_line(cost: dict[str, Any] | None) -> str | None:
    """What the features cost: the wall time of the full call and of the calls on cut data, and the peak memory of the
    full call. Shown, never judged."""
    if not cost:
        return None
    cuts = ", ".join(_seconds(c) for c in cost["cut_seconds"]) or "none"
    return f"full call {_seconds(cost['seconds'])}, peak memory {_memory(cost)} · on cut data {cuts} · not judged"


def _memory(cost: dict[str, Any]) -> str:
    peak, start, how = cost["peak_bytes"], cost["start_bytes"], cost["memory"]
    if peak is None:
        return "n/a"
    if how == "high-water mark":
        return f"{_bytes(peak)} (process high-water mark, not the call's own)"
    return f"+{_bytes(peak - start)} (process {_bytes(peak)})"


def _seconds(x: float) -> str:
    if x >= 1:
        return f"{x:.1f} s"
    return f"{x * 1000:.1f} ms" if x < 0.01 else f"{x * 1000:.0f} ms"


def _bytes(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    return f"{n / 1e6:.1f} MB" if n < 1e7 else f"{n / 1e6:.0f} MB"
