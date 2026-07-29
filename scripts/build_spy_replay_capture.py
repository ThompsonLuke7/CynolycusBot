"""Materialize retained live SPY replay inputs from append-only JSONL logs.

The live runner records each received 1m bar and each Phase-4 decision input.
This command deduplicates those records into point-in-time parquet artifacts
that the buffer/grace replay can consume.  It never fetches or backfills bars.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_events(path: Path, event: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("event") == event:
            rows.append(row)
    return rows


def materialize_capture(capture_dir: Path, output_dir: Path) -> dict[str, int]:
    bars_rows = _read_events(capture_dir / "live_1m_bars.jsonl", "one_minute_bar")
    decision_rows = _read_events(capture_dir / "phase4_decisions.jsonl", "phase4_decision_input")
    bars = pd.DataFrame([{**(row.get("bar") or {}), "symbol": row.get("symbol")} for row in bars_rows])
    decisions = pd.DataFrame(
        [
            {
                **(row.get("bar") or {}), "symbol": row.get("symbol"),
                **(row.get("probs") or {}), **{f"thr_{k}": v for k, v in (row.get("thresholds") or {}).items()},
                "raw_action": row.get("raw_action"), "exec_pos": row.get("exec_pos"), "gate_status": row.get("gate_status"),
            }
            for row in decision_rows
        ]
    )
    for frame in (bars, decisions):
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame.dropna(subset=["timestamp"], inplace=True)
            frame.sort_values(["timestamp", "symbol"], inplace=True)
            frame.drop_duplicates(subset=["timestamp", "symbol"], keep="last", inplace=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bars.empty:
        bars.to_parquet(output_dir / "spy_intraday_1min.parquet", index=False)
    if not decisions.empty:
        decisions.to_parquet(output_dir / "phase4_signal_frame.parquet", index=False)
    return {"one_minute_bars": len(bars), "phase4_decisions": len(decisions)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build replay-ready SPY artifacts from live capture logs.")
    parser.add_argument("--capture-dir", default="Data/inference/spy/replay_capture")
    parser.add_argument("--output-dir", default="Data/inference/spy/replay_capture/artifacts")
    args = parser.parse_args()
    counts = materialize_capture(Path(args.capture_dir), Path(args.output_dir))
    print(f"materialized 1m={counts['one_minute_bars']:,} phase4={counts['phase4_decisions']:,} -> {args.output_dir}")


if __name__ == "__main__":
    main()
