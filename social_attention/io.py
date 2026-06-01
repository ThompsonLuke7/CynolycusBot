"""Small IO helpers for parquet-first social attention data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: Path | str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def write_table(df: pd.DataFrame, path: Path | str) -> pd.DataFrame:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)
    return df


def merge_write_table(
    rows: pd.DataFrame,
    path: Path | str,
    *,
    dedupe_cols: list[str],
) -> pd.DataFrame:
    existing = read_table(path)
    if rows.empty:
        out = existing
    elif existing.empty:
        out = rows.copy()
    else:
        out = pd.concat([existing, rows], ignore_index=True)
    if not out.empty:
        out = out.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
    return write_table(out, path)


def read_json(path: Path | str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(data: Any, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")

