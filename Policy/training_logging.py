from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_name(raw: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(raw))
    return s.strip("._") or "run"


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")
    if isinstance(value, pd.Series):
        return {str(k): _json_safe(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return {
            "rows": int(len(value)),
            "columns": [str(c) for c in value.columns],
        }
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def log_training_run(
    *,
    run_name: str,
    output_dir: str | Path,
    hyperparameters: Mapping[str, Any] | None = None,
    train_metrics: Mapping[str, Any] | None = None,
    validation_metrics: Mapping[str, Any] | None = None,
    best_validation_metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    global_registry_path: str | Path = Path("Data") / "outputs" / "training_runs" / "training_runs.jsonl",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    run_id = f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{_safe_name(run_name)}"
    summary = {
        "run_id": run_id,
        "run_name": str(run_name),
        "created_at_utc": ts.isoformat().replace("+00:00", "Z"),
        "hyperparameters": _json_safe(hyperparameters or {}),
        "final_train_metrics": _json_safe(train_metrics or {}),
        "final_validation_metrics": _json_safe(validation_metrics or {}),
        "best_validation_metrics": _json_safe(best_validation_metrics or {}),
        "artifacts": _json_safe(artifacts or {}),
        "extra": _json_safe(extra or {}),
    }

    latest_path = out_dir / "training_run_summary.json"
    versioned_path = out_dir / f"training_run_summary_{run_id}.json"
    latest_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    versioned_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    registry_path = Path(global_registry_path)
    if not registry_path.is_absolute():
        registry_path = (Path.cwd() / registry_path).resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")

    return {
        "latest_path": latest_path,
        "versioned_path": versioned_path,
        "registry_path": registry_path,
    }
