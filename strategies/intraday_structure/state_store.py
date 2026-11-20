from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        # Compact on purpose: the engine state reached 188 MB pretty-printed and
        # was rewritten on every transition (18-20 s each, 2026-10-26); the
        # indent alone cost ~6 s of that. Nothing reads this file but restore().
        temp.write_text(json.dumps(state, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        temp.replace(self.path)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
