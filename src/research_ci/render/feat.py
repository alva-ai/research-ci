"""The ResultCard, slot by slot, and the ExploreCard."""

from typing import Any

from research_ci.kernel.verdict.rule import VETO_Z
from research_ci.render.common import MARK, adder, amount, check_line, eval_line, figure, num, rolled


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
    """A submission's card, rendered from its ledger record alone, slot by slot: the floor's premises and decisive
    test, the engine's lists, the lab's evals and null worlds, and the lab evals' tests; then the verdict. The
    calibration of the decisive effect shows on the submission that ran it; later ones reuse it."""
    of = f" of {spec['budget']}" if spec.get("budget") else ""
    again = f" · judged again under v{version}, no new trial" if spec.get("again") else ""
    out = [f"ResultCard · {idea} · feat v{version} · submission {spec['n']}{of}{again}"]
    add = adder(out)
    f = spec["features"]
    add("features", f"{f['path']} · {f['hash'][:12]}")
    sc = spec["scores"]
    add("scores", f"{sc['rows']:,} rows · {sc['share']:.0%} of candidate rows · {sc['hash'][:12]}")
    add("cost", cost_line(spec.get("cost")))
    cal = spec["calibration"]
    if cal["ran"]:
        add("calibrate", *calibration_lines(cal))
    out.append(SLOT_HEADS[1])
    add("premises", *[eval_line(c) for c in spec["floors"]])
    add("need", f"z ≥ {spec['need']:.2f} on submission {spec['n']}")
    for d in spec.get("decisive", []):
        add("decisive", _decisive_line(d, spec["need"]), *_decisive_horizons(d))
    if "vetoes" in spec:
        out.append(SLOT_HEADS[2])
        add("shown", *[line for r in spec.get("shown", []) for line in _shown_rows(r, spec["need"])] or ["nothing"])
        out.append(SLOT_HEADS[3].format(bar=VETO_Z))
        vetoes = spec["vetoes"]
        lab = [line for name in _names(vetoes) for line in _lab_rows(name, vetoes)]
        add("lab evals", *lab)
        add("worlds", *[line for r in vetoes if r["by"] != "lab" for line in _world_rows(r)], _coverage(spec))
    if spec.get("tests"):
        out.append(SLOT_HEADS[4])
        add("tests", *[_test_line(t) for t in spec["tests"]])
    v = spec["verdict"]
    passing = [f"{on} {', '.join(hs)}" for on, hs in v.get("horizons", {}).items() if hs]
    status = f"{v['status']} at {' · '.join(passing)}" if v["status"] == "PASSED" and passing else v["status"]
    add("verdict", status, *v["notes"])
    add("warnings", *[f"[!] {w}" for w in spec.get("warnings", [])])
    out.append("-" * 100)
    return "\n".join(out)


SLOT_HEADS = {
    1: "1 floor · the premises and the decisive test, fixed before any result; only the decisive test can pass a claim",
    2: "2 engine lists · the default diagnostics and the library evals feat.py lists, vs the floor world; shown only",
    3: (
        "3 lab evals · tests of the claim: each lab eval, and each null world, must reach z ≥ {bar:.2f} at every "
        "horizon it covers, or the claim fails there"
    ),
    4: "4 tests of lab evals · run before the submission was counted",
}


def _decisive_horizons(d: dict[str, Any]) -> list[str]:
    """The decisive test at each horizon: the gain over random, its MDE at the bar, the horizon's own z, and its z
    against the family's null, which decides whether it passes there."""
    lines = []
    for h in d["horizons"]:
        mark = "✓" if h.get("passed") else "✗"
        counted = "" if h["counted"] else "  not counted"
        lines.append(
            f"  {h['h']:<7}gain {figure(h['gain'], d):>9} ± {figure(h['se'], d):<9} MDE {figure(h.get('mde'), d):>8}"
            f" {h['blocks']:>5} blocks  z {num(h['z']):<8} vs the family's null {num(h.get('adjusted'))} "
            f"{mark}{counted}"
        )
    return lines


def _shown_rows(r: dict[str, Any], need: float) -> list[str]:
    """A diagnostic of slot 2 against the floor world, with its horizons."""
    mark = "ok  " if r["passed"] else "[!] "
    best = f" at {r['best']}" if r.get("best") else ""
    z = f"z {num(r['z'])}" if r["passed"] else f"z {num(r['z'])} < {need:.2f}"
    head = f"{mark} {r['name']} ({r['by']}) vs {r['world']} ({', '.join(r['explains'])}): {z}{best} · {r['note']}"
    return [head + _shuffles(r), *[_horizon(h, r, veto=False) for h in r["horizons"]]]


def _names(vetoes: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(r["name"] for r in vetoes if r["by"] == "lab"))


def _lab_rows(name: str, vetoes: list[dict[str, Any]]) -> list[str]:
    """A lab eval with its H0, then its judgement in the floor world and in every null world that names it."""
    rows = [r for r in vetoes if r["by"] == "lab" and r["name"] == name]
    file = rows[0].get("validation", {}).get("file", "")
    mark = "✓" if all(r["passed"] for r in rows) else "✗"
    out = [f"{mark} {name}{f' ({file})' if file else ''} · H0: {rows[0]['h0']}"]
    for r in rows:
        out += [f"    {_judgement(r)}", *[f"  {_horizon(h, r, veto=True)}" for h in r["horizons"]]]
    return out


def _world_rows(r: dict[str, Any]) -> list[str]:
    """A null world of the idea and what it judges: a diagnostic of slot 2 it names, or the decisive effect."""
    mark = "✓" if r["passed"] else "✗"
    what = "the decisive effect" if r["by"] == "decisive" else f"{r['name']} ({r['by']})"
    head = f"{mark} {r['world']} ({', '.join(r['explains'])}: {r['alternative']}) judges {what}"
    return [head, f"    {_judgement(r)}", *[f"  {_horizon(h, r, veto=True)}" for h in r["horizons"]]]


def _judgement(r: dict[str, Any]) -> str:
    """A slot-3 judgement in one world: its z against the bar, and how its horizons vote."""
    where = f"{r['world']} ({', '.join(r['explains'])})"
    sign = "≥" if r["passed"] else "<"
    vote = "one vote for all its horizons" if r["combine"] == "max" else "each horizon votes"
    z = f"z {num(r['z'])} {sign} {VETO_Z:.2f}" if r["combine"] == "max" else f"weakest z {num(r['z'])}"
    return f"vs {where}: {z} · {vote} · {r['note']}{_shuffles(r)}"


def _horizon(h: dict[str, Any], r: dict[str, Any], veto: bool) -> str:
    """One horizon of a judgement: the value and its error, the null's mean and spread, the blocks, and its z; for a
    veto, its MDE at the bar and its vote."""
    counted = "" if h["counted"] else "  not counted"
    line = (
        f"{'':<5}{h['h']:<7}value {figure(h['real'], r):>10} ± {figure(h['se'], r):<10} null "
        f"{figure(h['null_mean'], r):>10} sd {figure(h['null_sd'], r):<10} {h['blocks']:>4} blocks"
    )
    if not veto:
        return f"{line}  z {num(h['z'])}{counted}"
    mark = "✓" if r.get("votes", {}).get(h["h"]) else "✗"
    return f"{line}  MDE {figure(h.get('mde'), r):<10} z {num(h['z'])} {mark}{counted}"


def _decisive_line(d: dict[str, Any], need: float) -> str:
    at = f" at {d['best']}" if d["best"] else ""
    year = rolled(d["effect"], d["best"], d) if d["best"] else ""
    year = f" ({year} if rolled)" if year else ""
    z = f"z {num(d['z'])} {'≥' if d['outcome'] == 'passed' else '<'} {need:.2f}"
    passing = d.get("passing", [])
    where = f" at {', '.join(passing)}" if passing and d["outcome"] == "passed" else ""
    return (
        f"{d['on']}: gain over random {amount(d['effect'], d)} ± {figure(d['se'], d)}{at}{year} · "
        f"{z}{_shuffles(d)} · {d['outcome'].upper()}{where}"
    )


def _coverage(spec: dict[str, Any]) -> str:
    c = spec["coverage"]
    return f"{'ok' if c['status'] == 'pass' else MARK[c['status']]:<4} {c['name']}: {c['detail']}"


def _test_line(t: dict[str, Any]) -> str:
    """A lab eval's own tests, as they stood before the submission was counted."""
    v = t["validation"]
    status = "ok  " if v["status"] != "invalid" else "FAIL"
    pits = [x for x in v["tests"] if x["name"].startswith(("built-in ", "pit "))]
    caught = f" · {sum(x['status'] == 'pass' for x in pits)}/{len(pits)} pits caught" if pits else ""
    return f"{status} {t['name']} ({t['file']}): {v['summary']}{caught}"


def _shuffles(r: dict[str, Any]) -> str:
    """How many shuffles of the world the z rests on, and when they are every one it has; records from before it was
    kept show nothing."""
    every = ", every shift there is" if r.get("distinct") else ""
    return f" · {r['shuffles']} shuffles{every}" if r.get("shuffles") else ""


def explore_card(spec: dict[str, Any], idea: str) -> str:
    out = [f"ExploreCard · {idea} · feat (not a submission; nothing is judged)"]
    add = adder(out)
    f = spec["features"]
    add("features", f"{f['path']} · {f['hash'][:12]}")
    sc = spec["scores"]
    add("scores", f"{sc['rows']:,} rows · {sc['share']:.0%} of candidate rows · {sc['hash'][:12]}")
    add("floors", *[eval_line(c) for c in spec["floors"]])
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
