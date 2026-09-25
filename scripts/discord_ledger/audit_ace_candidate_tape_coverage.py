"""Audit whether the available historical data can support an ACE candidate tape.

A valid retrospective candidate tape needs, at every decision timestamp, an
immutable point-in-time universe and enough pre-decision intraday data for
*every* eligible symbol in both directions.  This script never substitutes the
current universe and never manufactures absent symbol bars.  It reports a
coverage manifest instead of candidate rows whenever that prerequisite fails.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from scripts.discord_ledger.build import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALERTS = ROOT / "research/discord_ledger_2026-09-21/ace_setup_outcomes_20260923_02/source_alert_outcomes.jsonl"
DEFAULT_SNAPSHOTS = ROOT / "Data/shared/universe/snapshots"
SNAPSHOT_PREFIX = "shared_universe_"
SNAPSHOT_FORMAT = "%Y%m%dT%H%M%SZ"


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"timezone-aware timestamp required: {value!r}")
    return stamp.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_index(directory: Path) -> list[tuple[pd.Timestamp, Path]]:
    snapshots: list[tuple[pd.Timestamp, Path]] = []
    for path in directory.glob(f"{SNAPSHOT_PREFIX}*.csv.gz"):
        raw = path.name[len(SNAPSHOT_PREFIX):-len(".csv.gz")]
        try:
            taken = pd.Timestamp(datetime.strptime(raw, SNAPSHOT_FORMAT), tz=timezone.utc)
        except ValueError:
            continue
        snapshots.append((taken, path))
    return sorted(snapshots)


def snapshot_as_of(index: list[tuple[pd.Timestamp, Path]], decision: pd.Timestamp) -> tuple[pd.Timestamp, Path] | None:
    valid = [item for item in index if item[0] <= decision]
    return valid[-1] if valid else None


def eligible_symbols(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    if "ticker" not in frame.columns:
        raise ValueError(f"{path}: missing ticker")
    if "is_eligible" in frame.columns:
        frame = frame[frame["is_eligible"].astype(bool)]
    return set(frame["ticker"].astype(str).str.upper().str.strip())


def bar_symbols(path: Path) -> set[str]:
    frame = pd.read_parquet(path, columns=["symbol"])
    return set(frame["symbol"].astype(str).str.upper().str.strip())


def coverage_record(alert: dict[str, Any], *, snapshot: tuple[pd.Timestamp, Path] | None,
                    bars: set[str] | None) -> dict[str, Any]:
    decision = _utc(alert["decision_at_utc"])
    symbol = str(alert["symbol"]).upper()
    row = {
        "decision_at_utc": decision.isoformat().replace("+00:00", "Z"),
        "symbol": symbol,
        "direction_sign": int(alert["direction_sign"]),
        "source_message_ids": sorted(map(str, alert.get("source_message_ids") or [])),
        "source_session_file": str(alert.get("source_session_file") or ""),
        "snapshot_status": "resolved" if snapshot else "missing_pit_snapshot",
        "snapshot_taken_at_utc": None,
        "snapshot_path": None,
        "eligible_symbol_count": None,
        "bar_file_status": "not_checked",
        "bar_symbol_count": None,
        "eligible_symbols_with_bars": None,
        "eligible_bar_coverage": None,
        "alert_symbol_in_eligible_universe": None,
        "alert_symbol_has_bar": None,
        "full_two_direction_candidate_tape_possible": False,
    }
    if snapshot is None:
        return row
    taken, snapshot_path = snapshot
    universe = eligible_symbols(snapshot_path)
    row.update({
        "snapshot_taken_at_utc": taken.isoformat().replace("+00:00", "Z"),
        "snapshot_path": str(snapshot_path),
        "eligible_symbol_count": len(universe),
        "alert_symbol_in_eligible_universe": symbol in universe,
    })
    if bars is None:
        row["bar_file_status"] = "missing"
        return row
    covered = universe & bars
    row.update({
        "bar_file_status": "read",
        "bar_symbol_count": len(bars),
        "eligible_symbols_with_bars": len(covered),
        "eligible_bar_coverage": len(covered) / len(universe) if universe else None,
        "alert_symbol_has_bar": symbol in bars,
        "full_two_direction_candidate_tape_possible": bool(universe) and covered == universe,
    })
    return row


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    resolved = [row for row in rows if row["snapshot_status"] == "resolved"]
    readable = [row for row in resolved if row["bar_file_status"] == "read"]
    complete = [row for row in readable if row["full_two_direction_candidate_tape_possible"]]
    coverage = [float(row["eligible_bar_coverage"]) for row in readable if row["eligible_bar_coverage"] is not None]
    return {
        "source_alerts": len(rows),
        "pit_snapshot_resolved": len(resolved),
        "snapshot_and_bars_resolved": len(readable),
        "full_universe_two_direction_tape_possible": len(complete),
        "mean_eligible_bar_coverage": None if not coverage else float(sum(coverage) / len(coverage)),
        "max_eligible_bar_coverage": None if not coverage else float(max(coverage)),
        "conclusion": (
            "retrospective_full_candidate_tape_available" if len(complete) else
            "retrospective_full_candidate_tape_not_available_from_current_inputs"
        ),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    fmt = lambda value: "—" if value is None else f"{100 * float(value):.3f}%"
    lines = [
        "# ACE candidate-tape coverage audit", "",
        "A valid historical candidate tape requires an immutable universe snapshot at or before each ACE decision and pre-decision intraday bars for every eligible symbol, in both directions. This audit makes no current-universe substitution and emits no fake candidate rows.", "",
        "| Measure | Value |", "| --- | ---: |",
        f"| Source alerts | {summary['source_alerts']} |",
        f"| PIT snapshot resolved | {summary['pit_snapshot_resolved']} |",
        f"| Snapshot plus bar file resolved | {summary['snapshot_and_bars_resolved']} |",
        f"| Full-universe, two-direction tape possible | {summary['full_universe_two_direction_tape_possible']} |",
        f"| Mean eligible-symbol bar coverage | {fmt(summary['mean_eligible_bar_coverage'])} |",
        f"| Maximum eligible-symbol bar coverage | {fmt(summary['max_eligible_bar_coverage'])} |", "",
        "## Conclusion", "",
        f"`{summary['conclusion']}`. The existing session files may support alert-symbol analysis, but cannot be relabeled as an independent full-universe candidate tape unless the coverage manifest says every point-in-time eligible symbol is present.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    if not args.alerts.is_file() or not args.snapshot_dir.is_dir():
        raise FileNotFoundError("alerts or snapshot directory missing")
    index = snapshot_index(args.snapshot_dir)
    if not index:
        raise ValueError("no parseable point-in-time universe snapshots")
    bars_cache: dict[Path, set[str] | None] = {}
    rows: list[dict[str, Any]] = []
    for alert in read_jsonl(args.alerts):
        bar_path = Path(str(alert.get("source_session_file") or ""))
        if bar_path not in bars_cache:
            bars_cache[bar_path] = bar_symbols(bar_path) if bar_path.is_file() else None
        rows.append(coverage_record(alert, snapshot=snapshot_as_of(index, _utc(alert["decision_at_utc"])), bars=bars_cache[bar_path]))
    summary = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "coverage_manifest.jsonl", rows)
    (args.out / "results.json").write_text(json.dumps({
        "schema": "ace_candidate_tape_coverage_audit_v1", "research_only": True,
        "inputs": {str(args.alerts): _sha256(args.alerts)}, "summary": summary,
        "limitations": [
            "Coverage of a bar file does not certify original market-data availability or quote quality.",
            "A source session file contains retrospectively fetched bars and may be scoped to alerted symbols.",
            "No candidate rows are generated when full point-in-time universe coverage is absent.",
        ],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.out / "report.md", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
