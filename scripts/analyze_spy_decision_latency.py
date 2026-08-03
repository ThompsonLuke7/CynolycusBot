#!/usr/bin/env python3
"""Summarize SPY decision latency from a live run's decision-latency.jsonl.

Answers the question the 2026-07-30 review could not: *where* do the 2-20
minutes between a 10-minute bar closing and its decision being recorded go?

The breakdown attributes each decision's total lag to:

  close detection — the bucket cannot close until the first bar of the *next*
                    bucket arrives, so a gap in the 1-minute feed defers it
  handler queue   — time between that arrival and the handler picking it up
  inference       — model scoring
  order policy    — entry/exit policy evaluation

Usage:
    scripts/analyze_spy_decision_latency.py                     # newest live_spy run
    scripts/analyze_spy_decision_latency.py <path-to-jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_RUNS = REPO_ROOT / "Data/inference/live_runs"


def _newest_log() -> Path | None:
    candidates = sorted(
        (d for d in LIVE_RUNS.glob("*_live_spy") if (d / "decision-latency.jsonl").exists()),
        key=lambda d: d.stat().st_mtime,
    )
    return (candidates[-1] / "decision-latency.jsonl") if candidates else None


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return "     -"
    return f"{seconds / 60:6.1f}m" if abs(seconds) >= 60 else f"{seconds:6.1f}s"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _newest_log()
    if path is None or not path.exists():
        print("No decision-latency.jsonl found. It is written once a live SPY session")
        print("runs with the instrumentation in place (restart the server to enable).")
        return 1

    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        print(f"{path} is empty.")
        return 1

    print(f"source: {path}")
    print(f"decisions: {len(records)}\n")

    header = (
        f"{'bar close (UTC)':>16} {'total':>7} {'detect':>7} {'queue':>7} "
        f"{'infer':>7} {'policy':>7} {'bars':>7}"
    )
    print(header)
    print("-" * len(header))
    totals: list[float] = []
    detects: list[float] = []
    gappy = 0
    for rec in records:
        stages = rec.get("stages_sec") or {}
        total = rec.get("total_lag_after_close_sec")
        detect = rec.get("close_detection_lag_sec")
        bars = rec.get("bars_in_bucket")
        expected = rec.get("expected_bars_in_bucket")
        if isinstance(total, (int, float)):
            totals.append(float(total))
        if isinstance(detect, (int, float)):
            detects.append(float(detect))
        if isinstance(bars, (int, float)) and isinstance(expected, (int, float)) and bars < expected:
            gappy += 1
        close_utc = str(rec.get("bar_close_utc") or "")[11:19] or "?"
        print(
            f"{close_utc:>16} {_fmt(total)} {_fmt(detect)} "
            f"{_fmt(rec.get('handler_queue_sec'))} {_fmt(stages.get('inference'))} "
            f"{_fmt(stages.get('order_policy'))} "
            f"{str(bars) + '/' + str(expected) if bars is not None else '-':>7}"
        )

    print()
    if totals:
        print(
            f"total lag after close: median {_fmt(_pct(totals, 0.5))}  "
            f"p90 {_fmt(_pct(totals, 0.9))}  max {_fmt(max(totals))}"
        )
    if detects:
        share = (sum(detects) / sum(totals) * 100) if totals and sum(totals) else float("nan")
        print(
            f"close detection:       median {_fmt(_pct(detects, 0.5))}  "
            f"max {_fmt(max(detects))}  ({share:.0f}% of all lag)"
        )
        print()
        if share > 60:
            print("VERDICT: the lag is dominated by waiting for the next bucket's first")
            print("bar. That is feed sparsity, not compute — look at the IEX subscription")
            print("and whether a timer should close buckets on schedule instead.")
        else:
            print("VERDICT: close detection does NOT dominate. Compare the queue and")
            print("stage columns above to see which one does.")
    if gappy:
        print(f"\n{gappy}/{len(records)} buckets received fewer 1m bars than expected (feed gaps).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
