"""The catalog as Markdown, like a table of contents: what each stage runs, how an idea's case evals plug in, and what
the eval library holds."""

from typing import Any

FILE = "docs/catalog.md"
REGENERATE = f"uv run rci methods --markdown > {FILE}"
LEGEND = (
    "**List**: floor steps run on every idea, are part of the engine hash, and no idea can change them; default steps "
    "are adjusted from experience, and the engine hash leaves them out. *(levels)*, *(hedge)*, *(reference)*: only "
    "when a plugin declares it; *(data.py)* and the other files: only when the idea adds the file. **Used by**: "
    "contract, data, feat run it; research code may call it (`rci methods`); inside eval: other eval functions call "
    "it; worked examples: other modules' examples build on it; algo, planned: waits for the algo layer; tests only: "
    "nothing else calls it. **Backing**: ex worked examples, prop properties on synthetic markets, — none declared "
    "(the protocol tests are held to broken toy plugins in `tests/eval/test_protocol.py`)."
)


def markdown(catalog: dict[str, Any]) -> str:
    out = [
        "# Research CI catalog",
        "",
        f"Generated from the code by `rci methods --markdown`; a test fails when it is out of date: `{REGENERATE}`.",
        "",
        "## Stages",
    ]
    for s in catalog["stages"]:
        out += ["", f"### {s['stage']}: `{s['command']}`", ""]
        out += [f"Lists: {', '.join(f'`{f}`' for f in s['files'])}.", ""]
        out += [_line("Step", "Checks or computes", "Role", "List", "Eval method"), _line(*["---"] * 5)]
        for group, rows in s["sections"]:
            if group:
                out.append(_line(f"**{group}**", "", "", "", ""))
            for r in rows:
                item = r["item"] + (f" *({r['needs']})*" if r["needs"] else "")
                out.append(_line(item, r["what"], r["role"], r["list"], _methods(r["methods"])))
    out += _cases(catalog["cases"])
    out += ["", "## Eval library"]
    for p in catalog["library"]:
        out += ["", f"### {p['package']}", ""]
        out += [_line("Module", "Function", "Does", "Used by", "Backing"), _line(*["---"] * 5)]
        for m in p["modules"]:
            for i, f in enumerate(m["functions"]):
                module = f"`{m['module']}`" if i == 0 else ""
                out.append(_line(module, f"`{f['name']}`", f["doc"], ", ".join(f["used"]), _backing(f)))
    roles = " · ".join(f"`{role}`: {means}" for role, means in catalog["roles"].items())
    out += ["", "## Legend", "", f"**Role**: {roles}. {LEGEND}"]
    return "\n".join(out)


def _cases(cases: dict[str, Any]) -> list[str]:
    """How an idea's evals plug into the data and feat layers: what each layer hands an eval and reads back, what a
    case eval's module declares, and what each outcome of its validation does."""
    out = ["", "## Case evals", "", f"A case eval is {cases['about']}.", ""]
    out += [_line("Layer", "Listed in", "An eval gets", "It returns", "Eval(fn, ...) adds", "Validated by")]
    out.append(_line(*["---"] * 6))
    for r in cases["layers"]:
        out.append(_line(r["layer"], f"`{r['file']}`", r["gets"], r["returns"], r["wraps"], r["validated"]))
    out += ["", _line("Declaration", "Means"), _line("---", "---")]
    out += [_line(f"`{d['name']}{d['signature']}`", d["doc"]) for d in cases["declarations"]]
    out += ["", f"For example: `{cases['example']}`.", "", _line("Outcome", "What happens"), _line("---", "---")]
    out += [_line(k, v) for k, v in cases["outcomes"].items()]
    return out


def _line(*cells: str) -> str:
    escaped = (c.replace("|", "\\|") for c in cells)
    return "|" + "|".join(f" {c} " if c else " " for c in escaped) + "|"


def _methods(keys: list[str]) -> str:
    """``module.function``, with the functions of one module in braces."""
    by_module: dict[str, list[str]] = {}
    for key in keys:
        module, name = key.rsplit(".", 1)
        by_module.setdefault(module, []).append(name)
    shown = [f"`{m}.{n[0]}`" if len(n) == 1 else f"`{m}.{{{', '.join(n)}}}`" for m, n in by_module.items()]
    return " · ".join(shown) or "—"


def _backing(f: dict[str, Any]) -> str:
    parts = [f"{f['examples']} ex" if f["examples"] else "", f"{f['properties']} prop" if f["properties"] else ""]
    return " · ".join(p for p in parts if p) or "—"
