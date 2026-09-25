"""Paired source-alert/control Intraday Structure entry-filter replay.

Each pair shares source symbol, direction, session, source evidence, and a
60-minute post-candidate observation window.  The control timestamp was chosen
without outcomes by ``select_ace_replay_controls.py``.  The engine is freshly
instantiated for each side of every pair, so their states cannot contaminate
one another.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.discord_ledger.build import read_jsonl, write_jsonl
from scripts.discord_ledger.run_ace_setup_replay import (
    _frame,
    _replay_entry,
    load_replay_input,
)
from strategies.intraday_structure.config import load_config
from dataclasses import replace


OBSERVATION_MINUTES = 60


def _source_key(row: dict) -> tuple[str, str, int]:
    return str(row["symbol"]), str(row["decision_at_utc"]), int(row["direction_sign"])


def load_pairs(source_path: Path, control_path: Path) -> list[tuple]:
    sources, _ = load_replay_input(source_path)
    controls, _ = load_replay_input(control_path)
    source_by_key = {(entry.symbol, entry.decision.isoformat(), entry.sign): entry for entry in sources}
    raw_controls = read_jsonl(control_path)
    raw_by_key = {_source_key(row): row for row in raw_controls}
    pairs = []
    for control in controls:
        raw = raw_by_key[(control.symbol, control.decision.isoformat(), control.sign)]
        paired_time = str(raw["paired_source_alert_decision_at_utc"])
        source = source_by_key.get((control.symbol, paired_time, control.sign))
        if source is None:
            raise ValueError(f"No source alert for paired control {control.symbol} {paired_time} {control.sign}")
        pairs.append((source, control))
    if len({(source.symbol, source.decision, source.sign) for source, _ in pairs}) != len(pairs):
        raise ValueError("Pair source identities are not unique")
    return sorted(pairs, key=lambda pair: (pair[0].decision, pair[0].symbol, pair[0].sign))


def _summary(rows: list[dict]) -> dict:
    def rate(side: str, field: str) -> float | None:
        values = [row[side][field] for row in rows]
        return None if not values else float(np.mean(values))
    return {"pair_count": len(rows), "source_setup_detected_rate": rate("source", "structure_setup_detected"),
            "control_setup_detected_rate": rate("control", "structure_setup_detected"),
            "source_explicit_confirmation_rate": rate("source", "structure_explicit_confirmation"),
            "control_explicit_confirmation_rate": rate("control", "structure_explicit_confirmation"),
            "source_modeled_entry_rate": None if not rows else float(np.mean([row["source"]["structure_modeled_trade_count"] > 0 for row in rows])),
            "control_modeled_entry_rate": None if not rows else float(np.mean([row["control"]["structure_modeled_trade_count"] > 0 for row in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-alerts", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    pairs = load_pairs(args.source_alerts, args.controls)
    shard = [pair for index, pair in enumerate(pairs) if index % args.shard_count == args.shard_index]
    config = replace(load_config(), execution=replace(load_config().execution, enabled=False))
    cache, rows, errors = {}, [], Counter()
    for source, control in shard:
        try:
            source_frame = _frame(source.source_file, source.symbol, cache)
            control_frame = _frame(control.source_file, control.symbol, cache)
            source_row = _replay_entry(source, source_frame, config, observation_minutes=OBSERVATION_MINUTES)
            control_row = _replay_entry(control, control_frame, config, observation_minutes=OBSERVATION_MINUTES)
            pair_id = hashlib.sha256(f"{source.symbol}|{source.decision.isoformat()}|{source.sign}".encode()).hexdigest()[:16]
            rows.append({"pair_id": pair_id, "source": source_row, "control": control_row})
        except (KeyError, ValueError) as exc:
            errors[type(exc).__name__] += 1
    payload = {"schema": "ace_paired_structure_control_replay_v1", "research_only": True,
               "observation_minutes": OBSERVATION_MINUTES, "shard": {"index": args.shard_index, "count": args.shard_count},
               "source_alerts_path": str(args.source_alerts), "controls_path": str(args.controls),
               "errors": dict(errors), "summary": _summary(rows)}
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "paired_replays.jsonl", rows)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), **payload["summary"], "errors": payload["errors"]}, indent=2))


if __name__ == "__main__":
    main()
