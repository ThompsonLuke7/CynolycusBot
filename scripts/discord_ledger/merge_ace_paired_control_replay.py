"""Integrity-check and summarize paired ACE/control entry-filter replay shards."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.discord_ledger.build import read_jsonl, write_jsonl


def _summary(rows: list[dict]) -> dict:
    def rate(side: str, field: str) -> float | None:
        values = [bool(row[side].get(field, False)) for row in rows]
        return None if not values else float(np.mean(values))
    source_detected = rate("source", "structure_setup_detected")
    control_detected = rate("control", "structure_setup_detected")
    source_confirmed = rate("source", "structure_explicit_confirmation")
    control_confirmed = rate("control", "structure_explicit_confirmation")
    source_entry = None if not rows else float(np.mean([row["source"]["structure_modeled_trade_count"] > 0 for row in rows]))
    control_entry = None if not rows else float(np.mean([row["control"]["structure_modeled_trade_count"] > 0 for row in rows]))
    return {"n": len(rows), "source_setup_detected_rate": source_detected, "control_setup_detected_rate": control_detected,
            "source_minus_control_detected_rate": None if source_detected is None else source_detected - control_detected,
            "source_explicit_confirmation_rate": source_confirmed, "control_explicit_confirmation_rate": control_confirmed,
            "source_minus_control_confirmation_rate": None if source_confirmed is None else source_confirmed - control_confirmed,
            "source_modeled_entry_rate": source_entry, "control_modeled_entry_rate": control_entry,
            "source_minus_control_modeled_entry_rate": None if source_entry is None else source_entry - control_entry}


def _render(summary: dict) -> str:
    lines = ["# ACE paired Intraday Structure control replay", "",
             "Each ACE source alert was compared with one deterministic same-symbol, same-direction, same-session non-alert control. The control was assigned without outcomes; both sides were replayed independently for 60 minutes with execution disabled.", "",
             "| Metric | Source alerts | Controls | Difference (pp) |", "| --- | ---: | ---: | ---: |"]
    for label, source, control, delta in (("Setup detected", "source_setup_detected_rate", "control_setup_detected_rate", "source_minus_control_detected_rate"),
                                          ("Explicit confirmation", "source_explicit_confirmation_rate", "control_explicit_confirmation_rate", "source_minus_control_confirmation_rate"),
                                          ("Modeled share entry", "source_modeled_entry_rate", "control_modeled_entry_rate", "source_minus_control_modeled_entry_rate")):
        value = lambda key: "n/a" if summary[key] is None else f"{summary[key] * 100:.1f}%"
        lines.append(f"| {label} | {value(source)} | {value(control)} | {value(delta)} |")
    lines.extend(["", "A positive difference would show that the current engine admits ACE-selected candidates more often than matched quiet windows. It remains conditional on ACE's ticker/direction choice and is not an options-P&L result or an independent universe-discovery test.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-controls", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    expected = {(row["symbol"], row["paired_source_alert_decision_at_utc"], row["direction_sign"])
                for row in read_jsonl(args.selected_controls)}
    paths = sorted(args.shards.glob("shard_*/paired_replays.jsonl"))
    rows = [row for path in paths for row in read_jsonl(path)]
    observed = {(row["source"]["symbol"], row["source"]["decision_at_utc"],
                 1 if row["source"]["direction"] == "long" else -1) for row in rows}
    duplicate_count = len(rows) - len(observed)
    missing, unexpected = expected - observed, observed - expected
    if duplicate_count or missing or unexpected:
        raise ValueError(f"Integrity failure: duplicate={duplicate_count}, missing={len(missing)}, unexpected={len(unexpected)}")
    groups = {"all": rows}
    for bucket in sorted({row["source"]["exclusive_bucket"] for row in rows}):
        groups[f"exclusive_bucket:{bucket}"] = [row for row in rows if row["source"]["exclusive_bucket"] == bucket]
    summaries = {name: _summary(group) for name, group in groups.items()}
    result = {"schema": "ace_paired_structure_control_replay_merged_v1", "research_only": True,
              "method": "One deterministic without-replacement matched control per source alert, selected before outcomes; fresh engine per pair side; 60-minute post-candidate window; execution disabled.",
              "integrity": {"expected_pairs": len(expected), "replay_rows": len(rows), "shard_files": len(paths),
                            "duplicate_rows": 0, "missing_rows": 0, "unexpected_rows": 0},
              "summaries": summaries}
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "paired_replays.jsonl", rows)
    (args.out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(_render(summaries["all"]), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "integrity": result["integrity"], "all": summaries["all"]}, indent=2))


if __name__ == "__main__":
    main()
