"""Snapshots of named tables, such as an idea's sources, kept under the vault by content: one folder per hash, one
parquet file per table."""

import shutil
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from research_ci.kernel.hashing import hash_tables
from research_ci.store.workspace import rci_home


def vault_root() -> Path:
    return rci_home() / "data"


def put(kind: str, tables: Mapping[str, pl.DataFrame]) -> tuple[str, Path]:
    """Keep a snapshot of the tables under the vault, named by content; return its hash and folder."""
    digest = hash_tables(tables)
    folder = vault_root() / kind / digest
    if not folder.exists():
        tmp = folder.with_suffix(".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name, table in tables.items():
            table.write_parquet(tmp / f"{name}.parquet")
        tmp.replace(folder)
    return digest, folder


def get(kind: str, digest: str) -> dict[str, pl.DataFrame]:
    folder = vault_root() / kind / digest
    return {path.stem: pl.read_parquet(path) for path in sorted(folder.glob("*.parquet"))}


def has(kind: str, digest: str) -> bool:
    return (vault_root() / kind / digest).is_dir()


def drop(kind: str, digest: str) -> None:
    shutil.rmtree(vault_root() / kind / digest, ignore_errors=True)
