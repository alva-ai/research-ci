import json
from typing import Annotated, Any

import typer

from research_ci.catalog import listing
from research_ci.catalog.collect import collect
from research_ci.catalog.markdown import FILE, markdown
from research_ci.cli import common

KIND_USE = {
    "data": "Contract(data=Use(name, {options}))",
    "universe": "Universe(name, clusters=<column>, groups=(<column>, ...))",
    "label": "Label(<label name>, Use(name, {options}), horizons=(...))",
    "target": "Effect(on=<label name>, target=Use(name, {options}))",
    "cost": "Contract(costs=Use(name, {options}))",
    "consumer": "Contract(consumer=Use(name, {options})), for its decision times",
}


def plugins(as_json: common.AsJson = False) -> None:
    """Every installed plugin a contract can name: its kind, options, and what it declares."""
    found = listing.plugins()
    if as_json:
        typer.echo(json.dumps(found, indent=2))
        return
    order = [*KIND_USE, "unknown", "error"]

    def rank(p: dict[str, Any]) -> tuple[int, str]:
        first = p["kind"].split("/")[0]
        return (order.index(first) if first in order else len(order), p["name"])

    shown = ""
    for p in sorted(found, key=rank):
        if p["kind"] != shown:
            shown = p["kind"]
            typer.echo(f"{shown}: {KIND_USE.get(shown.split('/')[0], '')}".rstrip(": "))
        if shown == "error":
            typer.echo(f"  {p['name']}: cannot be loaded: {p['error']}")
            continue
        typer.echo(f"  {p['signature']}")
        for line in _declared(p):
            typer.echo(f"    {line}")
    if not found:
        typer.echo("no plugins installed; a lab registers them under the entry-point group research_ci.plugins")


def _declared(p: dict[str, Any]) -> list[str]:
    """What a plugin says about itself: its docstring's first sentence, and what its kind declares."""
    lines = [p["doc"]] if p["doc"] else []
    if p.get("fields") is not None:
        lines.append(f"fields: {', '.join(p['fields']) or 'not declared'}")
    if p.get("levels"):
        lv = p["levels"]
        lines.append(f"levels: {', '.join(lv['fields'])} · factor {lv['factor']:g} · window {lv['window']}")
        lines[-1] += f" · mask {lv['mask']}"
    if p.get("reference"):
        lines.append("reference source: answers the independent check at admission")
    if "per" in p:
        lines.append(f"per: {p['per'] or 'not declared'} · rolls: {'yes' if p['rolls'] else 'no'}")
    lines.append(f"from the {p['origin']}" if p["origin"] == "registry" else f"from an {p['origin']}")
    return lines


def methods(
    as_json: common.AsJson = False,
    as_markdown: Annotated[
        bool, typer.Option("--markdown", help=f"The catalog of every stage and the whole eval library, as {FILE}.")
    ] = False,
) -> None:
    """The general evals of the engine's library an idea's EVALS can list or its lab evals can call, with the worked
    examples that back each; how a lab eval and its tests are written; then how a null world is written."""
    if as_markdown:
        typer.echo(markdown(collect()))
        return
    found = listing.methods()
    cases = listing.lab_evals()
    worlds = listing.nulls()
    if as_json:
        typer.echo(json.dumps(found + cases + worlds, indent=2))
        return
    formats = listing.formats()
    shown = ""
    for m in found:
        if m["use"] != shown:
            typer.echo(f"{m['use']}: {formats.get(m['use'], '')}".rstrip(": "))
            shown = m["use"]
        typer.echo(f"  {m['import']}")
        typer.echo(f"    {m['name']}{m['signature']}")
        if m["doc"]:
            typer.echo(f"    {m['doc']}")
        typer.echo(f"    backed by {_count(m['examples'], 'worked example')} · {_count(m['properties'], 'property')}")
    typer.echo(f"lab eval: {listing.LAB_EVAL}")
    typer.echo(f"  from research_ci.api import {', '.join(c['name'] for c in cases)}")
    for c in cases:
        typer.echo(f"    {c['name']}{c['signature']}")
        typer.echo(f"      {c['doc']}")
    typer.echo(f"  for example: {listing.CASE_EXAMPLE}")
    typer.echo(f"null world: {listing.NULLS_FORMAT}")
    typer.echo(f"  from research_ci.api import {', '.join(w['name'] for w in worlds)}")
    for w in worlds:
        typer.echo(f"    {w['name']}{w['signature']}")
        width = max(len(f["name"]) for f in w["fields"])
        for f in w["fields"]:
            typer.echo(f"      {f['name']:<{width}}  {f['means']}")
    typer.echo(f"  for example: {listing.NULLS_EXAMPLE}")


def _count(n: int, what: str) -> str:
    many = what[:-1] + "ies" if what.endswith("y") else what + "s"
    return f"{n} {what if n == 1 else many}"
