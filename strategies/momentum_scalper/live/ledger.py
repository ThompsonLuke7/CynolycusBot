"""Append-only JSONL shadow ledger for review and replay-parity investigation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _json_default(value: object) -> object:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "value"):
        return getattr(value, "value")
    raise TypeError(f"not JSON serializable: {type(value)!r}")


class ShadowLedger:
    """Write one record at a time; no aggregate file is ever overwritten."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, stream: str, payload: dict[str, Any]) -> Path:
        if not stream or any(part in {"", ".", ".."} for part in stream.split("/")):
            raise ValueError("stream must be a safe relative name")
        path = self.root / f"{stream}.jsonl"
        record = json.dumps(payload, default=_json_default, sort_keys=True, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")
        return path


__all__ = ["ShadowLedger"]
