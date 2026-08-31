from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from strategies.intraday_structure.config import DEFAULT_CONFIG_PATH, load_config
from strategies.intraday_structure.evidence import (
    read_decision_events,
    render_decision_summary,
    summarize_decision_events,
)
from strategies.intraday_structure.models import Candidate
from strategies.intraday_structure.replay import EventReplay, write_replay_result
from strategies.intraday_structure.reporting import build_report, read_jsonl, render_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday Structure Engine research CLI (no order submission).")
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="Chronologically replay true 1-minute bars.")
    replay.add_argument("--bars", required=True, help="CSV or Parquet with symbol,timestamp,OHLCV.")
    replay.add_argument("--candidates", required=True, help="JSON/JSONL/CSV candidate records with availability timestamps.")
    replay.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    replay.add_argument("--output", default="Data/inference/intraday_structure/replay")

    report = sub.add_parser("report", help="Measure the closed-setup ledger. Read-only.")
    report.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    report.add_argument("--ledger", default=None, help="Override the configured ledger path.")
    report.add_argument("--abstentions", default=None, help="Override the configured abstention path.")
    report.add_argument("--json", action="store_true", help="Emit the report as JSON instead of text.")
    funnel = sub.add_parser("funnel-report", help="Summarize discovery through fixed-horizon outcomes.")
    funnel.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    funnel.add_argument("--events", default=None, help="Override the configured decision-event path.")
    funnel.add_argument("--json", action="store_true", help="Emit the report as JSON instead of text.")
    args = parser.parse_args()

    if args.command == "replay":
        bars = _read_frame(Path(args.bars))
        candidates = [Candidate.from_mapping(row) for row in _read_records(Path(args.candidates))]
        result = EventReplay(load_config(args.config)).run(bars, candidates)
        write_replay_result(result, args.output)
        print(json.dumps(result.metrics, indent=2, sort_keys=True))

    if args.command == "report":
        config = load_config(args.config)
        payload = build_report(
            read_jsonl(args.ledger or config.ledger_path),
            read_jsonl(args.abstentions or config.abstention_path),
        )
        print(json.dumps(payload, indent=2, sort_keys=True) if args.json else render_report(payload))

    if args.command == "funnel-report":
        config = load_config(args.config)
        payload = summarize_decision_events(
            read_decision_events(args.events or config.evidence.event_path)
        )
        print(json.dumps(payload, indent=2, sort_keys=True) if args.json else render_decision_summary(payload))


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else list(raw.get("candidates", []))


if __name__ == "__main__":
    main()
