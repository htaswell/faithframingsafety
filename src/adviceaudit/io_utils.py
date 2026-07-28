"""Table and configuration I/O shared by both pipelines.

All file formats are dispatched by extension so that users can supply CSV,
TSV, or Excel inputs interchangeably.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".xlsm"}


def read_table(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """Read a CSV/TSV/Excel file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(
        f"Unsupported input format '{suffix}'. Expected one of {sorted(TABLE_SUFFIXES)}."
    )


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a DataFrame to CSV/TSV/Excel, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=index)
    elif suffix in {".xlsx", ".xlsm"}:
        df.to_excel(path, index=index)
    else:
        raise ValueError(f"Unsupported output format '{suffix}'.")
    return path


def write_excel_sheets(sheets: dict[str, pd.DataFrame], path: str | Path) -> Path:
    """Write several DataFrames to one workbook, one sheet per key."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return config


def config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a config, used to invalidate annotation caches."""
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """Raise a readable error if expected columns are missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{context}: missing required column(s) {missing}. "
            f"Columns present: {list(df.columns)}. "
            f"Rename your columns or edit the 'columns' block in config/analysis.yaml."
        )


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON run manifest recording provenance for a pipeline step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return path
