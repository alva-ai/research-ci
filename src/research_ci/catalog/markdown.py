"""The catalog as Markdown, like a table of contents: what each stage runs, slot by slot, how an idea's lab evals plug
in and what holds them to their tests, and what the eval library holds."""

from typing import Any

FILE = "docs/catalog.md"
REGENERATE = f"uv run rci methods --markdown > {FILE}"
SLOTS = (
    "Every layer is the same four slots. **1 floor**: the premises and the decisive test, which alone can pass a claim "
    "(z ≥ z_need, rising with the submissions). **2 engine lists**: the default list and the library evals an idea "
    "lists; they test generic H0s and are shown, or treat the data. **3 lab evals**: the evals the idea writes for its "
    "own case, and its null worlds; they can veto, never pass. **4 tests of lab evals**: what holds a lab eval to its "
    "own tests before it judges anything."
)
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
        SLOTS,
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
    out += _lab(catalog["lab"])
    out += _tests(catalog["tests"])
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


def _lab(lab: dict[str, Any]) -> list[str]:
    """Slot 3: how an idea's lab evals plug into the data and feat layers, what each layer hands an eval and reads
    back, when it vetoes, and how its module and a null world are written."""
    out = ["", "## Slot 3: lab evals", "", f"A lab eval is {lab['about']}.", ""]
    head = ("Layer", "Listed in", "An eval gets", "It returns", "Eval(fn, ...) adds", "It vetoes when")
    out += [_line(*head), _line(*["---"] * len(head))]
    for r in lab["layers"]:
        out.append(_line(r["layer"], f"`{r['file']}`", r["gets"], r["returns"], r["wraps"], r["vetoes"]))
    out += ["", _line("Declaration", "Means"), _line("---", "---")]
    out += [_line(f"`{d['name']}{d['signature']}`", d["doc"]) for d in lab["declarations"]]
    out += ["", f"For example: `{lab['example']}`.", ""]
    out += ["A null world, in `nulls/feat.py`, one for every alternative of the contract:", ""]
    out += [_line("Field", "Means"), _line("---", "---")]
    out += [_line(f"`{w['name']}.{f['name']}`", f["means"]) for w in lab["worlds"] for f in w["fields"]]
    return out


def _tests(tests: dict[str, Any]) -> list[str]:
    """Slot 4: what holds a lab eval to its own tests, what each outcome does, and the engine internals that carry
    worked examples of their own."""
    out = ["", "## Slot 4: tests of lab evals", "", _line("Test", "Where", "Does"), _line(*["---"] * 3)]
    out += [_line(t["test"], f"`{t['file']}`: `{t['function']}`", t["does"]) for t in tests["checks"]]
    out += ["", _line("Outcome", "What happens"), _line("---", "---")]
    out += [_line(k, v) for k, v in tests["outcomes"].items()]
    out += ["", "### Engine internals", "", INTERNALS, "", _line("Module", "Function", "Does", "Backing")]
    out.append(_line(*["---"] * 4))
    for m in tests["internals"]:
        for i, f in enumerate(m["functions"]):
            module = f"`{m['module']}`" if i == 0 else ""
            out.append(_line(module, f"`{f['name']}`", f["doc"], f"{f['examples']} ex" if f["examples"] else "—"))
    return out


INTERNALS = (
    "What the engine runs on its own, with worked examples like the library's, that no idea lists: the tests of evals "
    "(`validation/`) and the data layer's machinery, its treatments, their influence, and the decisive numbers "
    "(`stages/data/treatments.py`)."
)


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
