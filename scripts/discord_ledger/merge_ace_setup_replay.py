"""Merge immutable ACE Intraday Structure replay shards after completeness checks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.discord_ledger.build import read_jsonl, write_jsonl
from scripts.discord_ledger.run_ace_setup_replay import TAG_PATTERNS, summarize_replay_rows
from strategies.intraday_structure.config import load_config
from dataclasses import replace


def _identity(row: dict) -> tuple[str, str, str]:
    return str(row["symbol"]), str(row["decision_at_utc"]), str(row["direction"])


def _replay_by_tags(rows: list[dict]) -> dict:
    config = replace(load_config(), execution=replace(load_config().execution, enabled=False))
    result = {"exclusive_buckets": summarize_replay_rows(rows, config)["groups"], "literal_tags": {}}
    for tag in TAG_PATTERNS:
        subset = [row for row in rows if tag in row["source_tags"]]
        result["literal_tags"][tag] = summarize_replay_rows(subset, config)["groups"]["all"]
    return result


def _render(outcomes: dict, replay: dict) -> str:
    lines = ["# ACE source-tagged underlying outcomes and Intraday Structure replay", "",
             "Research only. The underlying analysis uses retrospective IEX trade bars. Intraday Structure received each ACE alert as an already-selected candidate; therefore replay measures confirmation/management compatibility, not independent universe discovery.", "",
             "## All alerts vs matched controls", "", "| Horizon | Alert n | Alert mean (%) | Control mean (%) | Difference (pp) |", "| --- | ---: | ---: | ---: | ---: |"]
    for horizon, item in outcomes["underlying_outcomes"]["all"]["outcomes"].items():
        alert, control = item["alerts"], item["matched_controls"]
        fmt = lambda value: "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {horizon} | {alert['n']} | {fmt(alert['mean_pct'])} | {fmt(control['mean_pct'])} | {fmt(item['alert_minus_control_mean_pct'])} |")
    all_replay = replay["exclusive_buckets"]["all"]
    lines.extend(["", "## Intraday Structure compatibility", "",
                  f"Complete isolated replays: {all_replay['n']}; detected: {all_replay['setup_detected_rate']:.1%}; "
                  f"modeled share trade: {all_replay['modeled_trade_rate']:.1%}.",
                  f"Detected setup families: `{all_replay['detected_setup_types']}`.", "",
                  "## Interpretation boundary", "",
                  "A detector overlap is not validation: ACE had already chosen both the ticker and direction. The replay does not include historical dealer context, full market/sector context where absent from the cached session file, or option execution. Negative matched-control underlying results mean this data does not yet support treating raw alert occurrence as a standalone directional entry policy.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True, help="Outcome-only results.json.")
    parser.add_argument("--eligible-alerts", type=Path, required=True, help="Outcome-only source_alert_outcomes.jsonl.")
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    expected_rows = read_jsonl(args.eligible_alerts)
    expected = {_identity({"symbol": row["symbol"], "decision_at_utc": row["decision_at_utc"],
                           "direction": row["direction"]}) for row in expected_rows}
    shard_paths = sorted(args.shards.glob("shard_*/intraday_structure_source_alert_replays.jsonl"))
    rows = [row for path in shard_paths for row in read_jsonl(path)]
    identities = [_identity(row) for row in rows]
    duplicate = [key for key, count in Counter(identities).items() if count > 1]
    observed = set(identities)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if duplicate or missing or unexpected:
        raise ValueError(f"Shard integrity failed: duplicate={len(duplicate)}, missing={len(missing)}, unexpected={len(unexpected)}")
    # These shards predate the correction that distinguished an explicit
    # CONFIRMED transition from a direct invalidation/closure.  The share-model
    # entry count is still valid; remove the ambiguous per-row field rather
    # than promote it into the canonical merged result.
    for row in rows:
        row["structure_confirmed_or_managed"] = None
        row["structure_confirmed_setup_types"] = []
        row["structure_explicit_confirmation_status"] = "not_reconstructed_from_legacy_shard; use_structure_modeled_trade_count"
    outcomes = json.loads(args.outcomes.read_text(encoding="utf-8"))
    replay = _replay_by_tags(rows)
    results = {"schema": "ace_setup_replay_merged_v1", "research_only": True,
               "outcomes_input": str(args.outcomes), "eligible_alerts_input": str(args.eligible_alerts),
               "shard_root": str(args.shards),
               "integrity": {"expected_source_alerts": len(expected), "replay_rows": len(rows),
                             "shard_files": len(shard_paths), "duplicate_rows": 0,
                             "missing_rows": 0, "unexpected_rows": 0},
               "underlying_outcomes": outcomes["underlying_outcomes"],
               "management_proxy_outcome_counts": outcomes["management_proxy_outcome_counts"],
               "management_proxy_warning": outcomes["management_proxy_warning"],
               "intraday_structure_replay": replay}
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "intraday_structure_source_alert_replays.jsonl", rows)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(_render(outcomes, replay), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "integrity": results["integrity"],
                      "replay_all": replay["exclusive_buckets"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
