"""Typed stage artifacts shared between the Engine and external Labs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import polars as pl


@dataclass(frozen=True)
class DataTableSpec:
    """Describe the grain and source semantics of one formal Data table."""

    keys: tuple[str, ...]
    frequency: str
    source: str
    event_time: str = "event_time"
    available_at: str | None = None
    units: Mapping[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(dict.fromkeys(self.keys)))
        object.__setattr__(self, "units", dict(self.units))
        if not self.keys:
            raise ValueError("DataTableSpec.keys must not be empty")
        if not self.frequency.strip():
            raise ValueError("DataTableSpec.frequency must not be empty")
        if not self.source.strip():
            raise ValueError("DataTableSpec.source must not be empty")

    def validate(self, name: str, table: pl.DataFrame) -> None:
        required = {*self.keys, self.event_time, *self.units}
        if self.available_at is not None:
            required.add(self.available_at)
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(
                f"Data table {name!r} is missing catalog columns: {missing}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "keys": list(self.keys),
            "frequency": self.frequency,
            "source": self.source,
            "event_time": self.event_time,
            "available_at": self.available_at,
            "units": dict(self.units),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataTableSpec:
        return cls(
            keys=tuple(value["keys"]),
            frequency=str(value["frequency"]),
            source=str(value["source"]),
            event_time=str(value.get("event_time", "event_time")),
            available_at=(
                str(value["available_at"])
                if value.get("available_at") is not None
                else None
            ),
            units=dict(value.get("units", {})),
            description=str(value.get("description", "")),
        )


@dataclass(frozen=True)
class DataBundle:
    """Formal Data result: one aligned table plus native-frequency sources."""

    main: pl.DataFrame
    native: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    catalog: Mapping[str, DataTableSpec] = field(default_factory=dict)
    source_coverage: pl.DataFrame | None = None
    request_log: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_files: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.main, pl.DataFrame):
            raise TypeError("DataBundle.main must be a Polars DataFrame")
        native = dict(self.native)
        catalog = dict(self.catalog)
        raw_files = {name: Path(path) for name, path in self.raw_files.items()}
        object.__setattr__(self, "native", native)
        object.__setattr__(self, "catalog", catalog)
        object.__setattr__(
            self, "request_log", tuple(dict(row) for row in self.request_log)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "raw_files", raw_files)

        for name, table in native.items():
            _validate_artifact_name(name, "native table")
            if not isinstance(table, pl.DataFrame):
                raise TypeError(
                    f"DataBundle.native[{name!r}] must be a Polars DataFrame"
                )
        expected = {"main", *native}
        if set(catalog) != expected:
            raise ValueError(
                "DataBundle.catalog must describe main and every native table; "
                f"expected={sorted(expected)}, actual={sorted(catalog)}"
            )
        for name, spec in catalog.items():
            if not isinstance(spec, DataTableSpec):
                raise TypeError(f"DataBundle.catalog[{name!r}] must be DataTableSpec")
            spec.validate(name, self.main if name == "main" else native[name])
        if self.source_coverage is not None and not isinstance(
            self.source_coverage, pl.DataFrame
        ):
            raise TypeError("DataBundle.source_coverage must be a Polars DataFrame")
        if (
            self.source_coverage is not None
            and "table" not in self.source_coverage.columns
        ):
            raise ValueError("DataBundle.source_coverage must contain a 'table' column")
        for name, path in raw_files.items():
            _validate_relative_path(name, "raw file")
            if not path.is_file():
                raise FileNotFoundError(f"DataBundle raw file does not exist: {path}")

    @classmethod
    def from_main(cls, table: pl.DataFrame) -> DataBundle:
        """Wrap a legacy single-table builder without inventing source claims."""

        if not table.columns:
            raise ValueError("legacy Data builder returned a table without columns")
        event_time = "event_time" if "event_time" in table.columns else table.columns[0]
        keys = (event_time,)
        for candidate in ("inst", "pair"):
            if candidate in table.columns:
                keys = (event_time, candidate)
                break
        return cls(
            main=table,
            catalog={
                "main": DataTableSpec(
                    keys=keys,
                    frequency="unspecified",
                    source="legacy_builder",
                    event_time=event_time,
                    description="legacy single-table Data builder",
                )
            },
            metadata={"legacy_single_table": True},
        )

    def coverage(self) -> pl.DataFrame:
        """Return declared coverage or a deterministic table-level inventory."""

        if self.source_coverage is not None:
            return self.source_coverage
        rows: list[dict[str, Any]] = []
        for name, table in self.tables().items():
            spec = self.catalog[name]
            series = table.get_column(spec.event_time)
            rows.append(
                {
                    "table": name,
                    "source": spec.source,
                    "frequency": spec.frequency,
                    "rows": table.height,
                    "columns": table.width,
                    "event_time_min": str(series.min()),
                    "event_time_max": str(series.max()),
                }
            )
        return pl.DataFrame(rows)

    def tables(self) -> dict[str, pl.DataFrame]:
        return {"main": self.main, **dict(self.native)}

    def eval_tables(self) -> dict[str, pl.DataFrame]:
        """Expose native tables to evals under an explicit reserved namespace."""

        return {f"data.native.{name}": table for name, table in self.native.items()}

    def as_catalog(self) -> dict[str, Any]:
        return {name: spec.as_dict() for name, spec in sorted(self.catalog.items())}

    def write(self, root: Path) -> dict[str, Any]:
        """Materialize the bundle into one immutable Data attempt directory."""

        root.mkdir(parents=True, exist_ok=True)
        self.main.write_parquet(root / "main.parquet")
        native_dir = root / "native"
        native_dir.mkdir(exist_ok=True)
        for name, table in sorted(self.native.items()):
            table.write_parquet(native_dir / f"{name}.parquet")
        (root / "catalog.json").write_text(
            json.dumps(self.as_catalog(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        coverage = self.coverage()
        coverage.write_parquet(root / "source_coverage.parquet")
        (root / "metadata.json").write_text(
            json.dumps(dict(self.metadata), indent=2, sort_keys=True, default=str)
            + "\n",
            encoding="utf-8",
        )
        with (root / "request_log.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.request_log:
                handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
        raw_dir = root / "raw"
        if self.raw_files:
            raw_dir.mkdir(exist_ok=True)
        for name, source in sorted(self.raw_files.items()):
            destination = raw_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        main_summary = _table_summary(self.main, self.catalog["main"])
        main_summary.update(
            {"path": "main.parquet", "sha256": _file_digest(root / "main.parquet")}
        )
        native_summary: dict[str, dict[str, Any]] = {}
        for name, table in sorted(self.native.items()):
            summary = _table_summary(table, self.catalog[name])
            relative = f"native/{name}.parquet"
            summary.update({"path": relative, "sha256": _file_digest(root / relative)})
            native_summary[name] = summary
        return {
            "main": main_summary,
            "native": dict(sorted(native_summary.items())),
            "catalog": "catalog.json",
            "source_coverage": "source_coverage.parquet",
            "request_log": "request_log.jsonl",
            "metadata": "metadata.json",
            "raw_files": sorted(self.raw_files),
        }

    @classmethod
    def read(cls, root: Path) -> DataBundle:
        """Read an approved bundle; raw snapshots are intentionally excluded."""

        catalog_payload = json.loads(
            (root / "catalog.json").read_text(encoding="utf-8")
        )
        catalog = {
            name: DataTableSpec.from_dict(spec)
            for name, spec in catalog_payload.items()
        }
        native_dir = root / "native"
        native = {
            path.stem: pl.read_parquet(path)
            for path in sorted(native_dir.glob("*.parquet"))
        }
        metadata_path = root / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        coverage_path = root / "source_coverage.parquet"
        coverage = pl.read_parquet(coverage_path) if coverage_path.exists() else None
        return cls(
            main=pl.read_parquet(root / "main.parquet"),
            native=native,
            catalog=catalog,
            source_coverage=coverage,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ReviewBrief:
    """Agent-authored result summary used to prepare an attempt for review."""

    question: str
    conclusion: str
    evidence: tuple[str, ...]
    counterevidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    changes: tuple[str, ...] = ()
    key_plots: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("ReviewBrief.question must not be empty")
        if not self.conclusion.strip():
            raise ValueError("ReviewBrief.conclusion must not be empty")
        if not self.evidence:
            raise ValueError("ReviewBrief.evidence must contain at least one item")
        for field_name in (
            "evidence",
            "counterevidence",
            "limitations",
            "changes",
            "key_plots",
            "evidence_refs",
        ):
            values = tuple(str(value).strip() for value in getattr(self, field_name))
            if any(not value for value in values):
                raise ValueError(f"ReviewBrief.{field_name} contains an empty item")
            object.__setattr__(self, field_name, values)
        for path in self.key_plots:
            _validate_relative_path(path, "key plot")
        for path in self.evidence_refs:
            _validate_relative_path(path, "evidence reference")

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "conclusion": self.conclusion,
            "evidence": list(self.evidence),
            "counterevidence": list(self.counterevidence),
            "limitations": list(self.limitations),
            "changes": list(self.changes),
            "key_plots": list(self.key_plots),
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReviewBrief:
        return cls(
            question=str(value.get("question", "")),
            conclusion=str(value.get("conclusion", "")),
            evidence=tuple(value.get("evidence", ())),
            counterevidence=tuple(value.get("counterevidence", ())),
            limitations=tuple(value.get("limitations", ())),
            changes=tuple(value.get("changes", ())),
            key_plots=tuple(value.get("key_plots", ())),
            evidence_refs=tuple(value.get("evidence_refs", ())),
        )


def _table_summary(table: pl.DataFrame, spec: DataTableSpec) -> dict[str, Any]:
    event_time = table.get_column(spec.event_time)
    return {
        "rows": table.height,
        "columns": table.width,
        "keys": list(spec.keys),
        "frequency": spec.frequency,
        "source": spec.source,
        "event_time": {
            "column": spec.event_time,
            "min": str(event_time.min()),
            "max": str(event_time.max()),
        },
        "available_at": spec.available_at,
    }


def _validate_artifact_name(name: str, kind: str) -> None:
    path = Path(name)
    if not name or path.name != name or name in {".", ".."}:
        raise ValueError(f"Invalid {kind} name: {name!r}")


def _validate_relative_path(value: str, kind: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid {kind} path: {value!r}")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
