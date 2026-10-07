from pathlib import Path

import polars as pl

from research_ci.kernel.hashing import hash_frame
from research_ci.store.workspace import rci_home


def vault_root() -> Path:
    return rci_home() / "data"


def put(source: str, fields: pl.DataFrame) -> tuple[str, Path]:
    """Keep a snapshot of normalized fields under the vault, named by content; return its hash and path."""
    digest = hash_frame(fields)
    path = vault_root() / source / f"{digest}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fields.write_parquet(tmp)
        tmp.replace(path)
    return digest, path


def get(source: str, digest: str) -> pl.DataFrame:
    return pl.read_parquet(vault_root() / source / f"{digest}.parquet")


def has(source: str, digest: str) -> bool:
    return (vault_root() / source / f"{digest}.parquet").exists()


def drop(source: str, digest: str) -> None:
    (vault_root() / source / f"{digest}.parquet").unlink(missing_ok=True)
