"""Small IO helpers shared across the forward-guidance pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_parent(path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: Path | str, default: Any | None = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(payload: Any, path: Path | str) -> Path:
    p = ensure_parent(path)
    p.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return p


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def flatten_dict(d: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        clean_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, prefix=f"{clean_key}_"))
        else:
            out[clean_key] = value
    return out


def write_dataframe(df: pd.DataFrame, path: Path | str) -> Path:
    p = ensure_parent(path)
    if str(p).lower().endswith(".csv"):
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)
    return p


def read_dataframe(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if str(p).lower().endswith(".csv"):
        return pd.read_csv(p)
    return pd.read_parquet(p)
