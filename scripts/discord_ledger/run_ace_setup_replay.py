"""Research-only ACE setup-bucket and Intraday Structure replay.

This is deliberately an adapter, not a live-strategy integration.  Discord
entries are supplied to the Intraday Structure engine as *already selected*
candidates at their source timestamp.  The engine result therefore measures
whether its price-structure confirmation and management agree with ACE's
selected opportunities; it does not measure discovery of ACE's universe.

Only literal setup words in the source quote receive tags.  Unlabelled rows
remain unlabelled.  Underlying outcomes use retrospective IEX one-minute trade
bars and are never option P&L or evidence of fill quality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from scripts.discord_ledger.build import read_jsonl, write_jsonl
from scripts.discord_ledger.run_caller_underlying_experiment import _forward_return
from scripts.discord_ledger.spy_trigger_pilot import snapshot
from strategies.intraday_structure.config import load_config
from strategies.intraday_structure.models import Candidate, Direction
from strategies.intraday_structure.replay import EventReplay


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "research/discord_ledger_2026-09-21/underlying_proxy_20260922_07/underlying_proxy_ledger.jsonl"
DEFAULT_EVENTS = ROOT / "research/discord_ledger_2026-09-21/run_curated_20260921_05/events.jsonl"
HORIZONS_MINUTES = (5, 15, 30, 60, 120)
EXCLUSION_MINUTES = 30
CONTROLS_PER_ALERT = 12
TAG_PATTERNS = {
    "lotto": re.compile(r"\blotto\b", re.IGNORECASE),
    "scout": re.compile(r"\bscout\b", re.IGNORECASE),
    "swing": re.compile(r"\bswing(?:\s+trade)?\b", re.IGNORECASE),
    "intraday": re.compile(r"\bintraday\b|\bday\s*trade\b", re.IGNORECASE),
    "0dte": re.compile(r"\b0\s*dte\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class Entry:
    symbol: str
    decision: pd.Timestamp
    sign: int
    trade_ids: tuple[str, ...]
    message_ids: tuple[str, ...]
    source_file: Path
    source_quote: str
    source_tags: tuple[str, ...]
    exclusive_bucket: str
    proxy_outcome: str | None


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _tags(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in TAG_PATTERNS.items() if pattern.search(text or ""))


def _bucket(tags: tuple[str, ...]) -> str:
    """Mutually exclusive reporting bucket; retain all literal tags separately."""
    primary = [tag for tag in ("lotto", "scout", "swing", "intraday") if tag in tags]
    if len(primary) > 1:
        return "multi_tagged_" + "_".join(primary)
    if primary:
        return primary[0]
    return "0dte_unlabelled" if "0dte" in tags else "unlabelled"


def _direction(option_type: object) -> int | None:
    return {"call": 1, "put": -1}.get(str(option_type).lower())


def _event_text(events: dict[str, dict], message_ids: list[str]) -> str:
    return "\n".join(str(events.get(str(message_id), {}).get("source_quote") or "") for message_id in message_ids)


def load_entries(ledger_path: Path, event_path: Path) -> tuple[list[Entry], dict]:
    events = {str(row["message_id"]): row for row in read_jsonl(event_path)}
    raw = read_jsonl(ledger_path)
    deduped: dict[tuple[str, pd.Timestamp, int], Entry] = {}
    excluded = Counter()
    for row in raw:
        sign = _direction(row.get("option_type"))
        entry = row.get("entry_underlying_snapshot") or {}
        if sign is None:
            excluded["missing_call_put_direction"] += 1
            continue
        if entry.get("status") != "ok":
            excluded[f"entry_snapshot_{entry.get('status', 'missing')}"] += 1
            continue
        source_file = Path(str(entry.get("source_session_file") or ""))
        if not source_file.is_file():
            excluded["source_session_file_missing"] += 1
            continue
        symbol = str(entry.get("proxy_symbol") or "").upper()
        if not symbol:
            excluded["proxy_symbol_missing"] += 1
            continue
        decision = pd.Timestamp(row["entry_available_at_utc"]).tz_convert("UTC")
        message_ids = tuple(sorted(str(value) for value in row.get("source_message_ids", [])))
        text = _event_text(events, list(message_ids))
        tags = _tags(text)
        candidate = Entry(symbol=symbol, decision=decision, sign=sign,
                          trade_ids=(str(row["trade_id"]),), message_ids=message_ids,
                          source_file=source_file, source_quote=text, source_tags=tags,
                          exclusive_bucket=_bucket(tags), proxy_outcome=row.get("proxy_outcome"))
        key = (symbol, decision, sign)
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = candidate
            continue
        # A single source message can map to multiple reconciled lifecycles.
        # Preserve all evidence without treating it as multiple timing examples.
        merged_ids = tuple(sorted({*previous.message_ids, *candidate.message_ids}))
        merged_text = _event_text(events, list(merged_ids))
        merged_tags = _tags(merged_text)
        deduped[key] = Entry(symbol=symbol, decision=decision, sign=sign,
                             trade_ids=tuple(sorted({*previous.trade_ids, *candidate.trade_ids})),
                             message_ids=merged_ids, source_file=source_file, source_quote=merged_text,
                             source_tags=merged_tags, exclusive_bucket=_bucket(merged_tags),
                             proxy_outcome=previous.proxy_outcome or candidate.proxy_outcome)
        excluded["same_symbol_time_direction_lifecycle_deduplicated"] += 1
    result = sorted(deduped.values(), key=lambda entry: (entry.decision, entry.symbol, entry.sign))
    return result, {"raw_ledger_rows": len(raw), "deduplicated_entries": len(result), "excluded_or_deduplicated": dict(excluded)}


def load_replay_input(path: Path) -> tuple[list[Entry], dict]:
    """Read a preselected research-only candidate set, retaining its evidence."""
    entries = []
    for row in read_jsonl(path):
        entries.append(Entry(symbol=str(row["symbol"]).upper(),
                             decision=pd.Timestamp(row["decision_at_utc"]).tz_convert("UTC"),
                             sign=int(row["direction_sign"]),
                             trade_ids=tuple(str(value) for value in row.get("trade_ids", [])),
                             message_ids=tuple(str(value) for value in row.get("source_message_ids", [])),
                             source_file=Path(row["source_session_file"]),
                             source_quote=str(row.get("source_quote") or ""),
                             source_tags=tuple(str(value) for value in row.get("source_tags", [])),
                             exclusive_bucket=str(row.get("exclusive_bucket") or "unlabelled"),
                             proxy_outcome=None))
    if any(entry.sign not in {-1, 1} or not entry.source_file.is_file() for entry in entries):
        raise ValueError(f"Invalid replay input: {path}")
    return entries, {"replay_input_path": str(path), "replay_input_sha256": _digest(path), "replay_input_rows": len(entries)}


def _frame(path: Path, symbol: str, cache: dict[tuple[Path, str], pd.DataFrame]) -> pd.DataFrame:
    key = (path, symbol)
    if key not in cache:
        frame = pd.read_parquet(path)
        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        frame = frame[frame["symbol"].astype(str).str.upper() == symbol].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if frame.empty or frame["timestamp"].duplicated().any():
            raise ValueError(f"{path}: missing or duplicate {symbol} minute bars")
        frame["_timestamp_ns"] = frame["timestamp"].astype("int64")
        cache[key] = frame
    return cache[key]


def _session_close_return(frame: pd.DataFrame, base: dict, sign: int) -> float | None:
    if base.get("status") != "ok" or base.get("close") is None:
        return None
    latest = pd.Timestamp(base["latest_completed_bar_start_utc"]).tz_convert("America/New_York")
    expected = latest.normalize() + pd.Timedelta(hours=15, minutes=59)
    expected_utc = expected.tz_convert("UTC")
    row = frame.loc[frame["timestamp"] == expected_utc]
    if len(row) != 1:
        return None
    return float(sign * (float(row.iloc[0]["close"]) / float(base["close"]) - 1.0))


def _outcome_row(entry: Entry, frame: pd.DataFrame, decision: pd.Timestamp, *, row_kind: str) -> dict | None:
    base = snapshot(frame, decision)
    required = ("return_5m", "return_20m", "distance_above_session_vwap_pct",
                "volume_last5_vs_prior20", "distance_above_opening_range_high_pct")
    if base["status"] != "ok" or any(base.get(key) is None for key in required):
        return None
    row = {
        "row_kind": row_kind,
        "symbol": entry.symbol,
        "direction": "long" if entry.sign > 0 else "short",
        "direction_sign": entry.sign,
        "decision_at_utc": decision.isoformat(),
        "session_date_et": decision.tz_convert("America/New_York").date().isoformat(),
        "exclusive_bucket": entry.exclusive_bucket,
        "source_tags": list(entry.source_tags),
        "trade_ids": list(entry.trade_ids),
        "source_message_ids": list(entry.message_ids),
        "source_quote": entry.source_quote,
        "source_session_file": str(entry.source_file),
        "completed_bar_at_utc": base["latest_completed_bar_start_utc"],
        "entry_underlying_close": base["close"],
        "directional_return_5m": entry.sign * float(base["return_5m"]),
        "directional_return_20m": entry.sign * float(base["return_20m"]),
        "directional_vwap_distance_pct": entry.sign * float(base["distance_above_session_vwap_pct"]),
        "volume_last5_vs_prior20": float(base["volume_last5_vs_prior20"]),
        "directional_opening_range_high_distance_pct": entry.sign * float(base["distance_above_opening_range_high_pct"]),
    }
    for horizon in HORIZONS_MINUTES:
        row[f"directional_forward_return_{horizon}m"] = _forward_return(frame, base, entry.sign, horizon)
    row["directional_forward_return_session_close"] = _session_close_return(frame, base, entry.sign)
    return row


def _control_decisions(entries: list[Entry]) -> dict[tuple[Path, str, str], list[pd.Timestamp]]:
    groups: dict[tuple[Path, str, str], list[pd.Timestamp]] = defaultdict(list)
    for entry in entries:
        groups[(entry.source_file, entry.symbol, entry.decision.tz_convert("America/New_York").date().isoformat())].append(entry.decision)
    result = {}
    for (source_file, symbol, day), alert_times in groups.items():
        local_day = pd.Timestamp(day, tz="America/New_York")
        grid = pd.date_range(local_day + pd.Timedelta(hours=9, minutes=56),
                             local_day + pd.Timedelta(hours=15), freq="5min")
        eligible = [value.tz_convert("UTC") for value in grid
                    if not any(abs((value.tz_convert("UTC") - alert).total_seconds()) <= EXCLUSION_MINUTES * 60
                               for alert in alert_times)]
        if len(eligible) > CONTROLS_PER_ALERT:
            positions = np.linspace(0, len(eligible) - 1, CONTROLS_PER_ALERT, dtype=int)
            eligible = [eligible[position] for position in positions]
        result[(source_file, symbol, day)] = eligible
    return result


def build_outcomes(entries: list[Entry], *, include_controls: bool = True) -> tuple[list[dict], list[dict], dict]:
    frames: dict[tuple[Path, str], pd.DataFrame] = {}
    alerts, controls = [], []
    exclusions = Counter()
    eligible_entries: list[Entry] = []
    for entry in entries:
        frame = _frame(entry.source_file, entry.symbol, frames)
        row = _outcome_row(entry, frame, entry.decision, row_kind="source_alert")
        if row is None:
            exclusions["alert_missing_required_pre_alert_features"] += 1
        else:
            alerts.append(row)
            eligible_entries.append(entry)
    if include_controls:
        control_times = _control_decisions(entries)
        for entry in eligible_entries:
            day = entry.decision.tz_convert("America/New_York").date().isoformat()
            frame = _frame(entry.source_file, entry.symbol, frames)
            for decision in control_times[(entry.source_file, entry.symbol, day)]:
                row = _outcome_row(entry, frame, decision, row_kind="same_symbol_no_nearby_alert_control")
                if row is None:
                    exclusions["control_missing_required_pre_alert_features"] += 1
                    continue
                # Controls inherit this source-tagged alert's direction and bucket.
                # Thus each alert has the same deterministic matched control set.
                controls.append(row)
    return alerts, controls, {"alert_rows": len(alerts), "control_rows": len(controls),
                               "source_session_files_read": len({row["source_session_file"] for row in alerts}),
                               "excluded": dict(exclusions)}


def _metric(values: list[float | None]) -> dict:
    clean = np.array([value for value in values if value is not None], dtype=float)
    return {"n": int(len(clean)), "mean_pct": None if not len(clean) else float(clean.mean() * 100),
            "median_pct": None if not len(clean) else float(np.median(clean) * 100),
            "directional_hit_rate": None if not len(clean) else float((clean > 0).mean())}


def _outcome_report(alerts: list[dict], controls: list[dict]) -> dict:
    groups: dict[str, tuple[list[dict], list[dict]]] = {"all": (alerts, controls)}
    for bucket in sorted({row["exclusive_bucket"] for row in alerts}):
        groups[f"exclusive_bucket:{bucket}"] = ([row for row in alerts if row["exclusive_bucket"] == bucket],
                                                  [row for row in controls if row["exclusive_bucket"] == bucket])
    for tag in TAG_PATTERNS:
        groups[f"literal_tag:{tag}"] = ([row for row in alerts if tag in row["source_tags"]],
                                         [row for row in controls if tag in row["source_tags"]])
    result = {}
    for name, (group_alerts, group_controls) in groups.items():
        horizons = {}
        for label in [*(f"{minutes}m" for minutes in HORIZONS_MINUTES), "session_close"]:
            column = f"directional_forward_return_{label}"
            alert_metrics = _metric([row[column] for row in group_alerts])
            control_metrics = _metric([row[column] for row in group_controls])
            horizons[label] = {"alerts": alert_metrics, "matched_controls": control_metrics,
                               "alert_minus_control_mean_pct": (
                                   None if alert_metrics["mean_pct"] is None or control_metrics["mean_pct"] is None
                                   else alert_metrics["mean_pct"] - control_metrics["mean_pct"]),
                               "alert_minus_control_hit_rate": (
                                   None if alert_metrics["directional_hit_rate"] is None or control_metrics["directional_hit_rate"] is None
                                   else alert_metrics["directional_hit_rate"] - control_metrics["directional_hit_rate"])}
        result[name] = {"alert_rows": len(group_alerts), "control_rows": len(group_controls), "outcomes": horizons}
    return result


def _replay_entry(entry: Entry, frame: pd.DataFrame, config, *, observation_minutes: int | None = None) -> dict:
    candidate = Candidate(ticker=entry.symbol, timestamp=entry.decision.to_pydatetime(),
                          available_at=entry.decision.to_pydatetime(),
                          direction=Direction.LONG if entry.sign > 0 else Direction.SHORT,
                          sources=("discord_source_alert_research",), score=0.80,
                          metadata={"source_message_ids": list(entry.message_ids),
                                    "exclusive_bucket": entry.exclusive_bucket})
    if observation_minutes is not None:
        cutoff = entry.decision.floor("min") + pd.Timedelta(minutes=observation_minutes)
        frame = frame[frame["timestamp"] <= cutoff].copy()
    replay = EventReplay(config).run(frame, [candidate])
    transitions = replay.transitions
    states = set(transitions.get("to_state", pd.Series(dtype=str)).astype(str))
    detected = transitions[transitions.get("to_state", pd.Series(dtype=str)) == "SETUP_DETECTED"] if not transitions.empty else transitions
    # Invalidated/CLOSED can be reached directly from WATCHING or ARMED.  Only
    # an explicit CONFIRMED transition proves the engine admitted the setup.
    confirmed = transitions[transitions.get("to_state", pd.Series(dtype=str)) == "CONFIRMED"] if not transitions.empty else transitions
    return {
        "trade_ids": list(entry.trade_ids), "source_message_ids": list(entry.message_ids),
        "symbol": entry.symbol, "direction": "long" if entry.sign > 0 else "short",
        "decision_at_utc": entry.decision.isoformat(), "exclusive_bucket": entry.exclusive_bucket,
        "source_tags": list(entry.source_tags), "source_session_file": str(entry.source_file),
        "structure_setup_detected": bool(len(detected)),
        "structure_detected_setup_types": sorted(set(detected.get("setup_type", pd.Series(dtype=str)).astype(str))),
        "structure_explicit_confirmation": bool(len(confirmed)),
        "structure_confirmed_setup_types": sorted(set(confirmed.get("setup_type", pd.Series(dtype=str)).astype(str))),
        "structure_modeled_trade_count": int(replay.metrics["trade_count"]),
        "structure_transition_count": int(len(transitions)),
        "structure_metrics": replay.metrics,
    }


def summarize_replay_rows(rows: list[dict], config, exclusions: Counter | None = None,
                          observation_minutes: int | None = None) -> dict:
    groups = {"all": rows}
    for bucket in sorted({row["exclusive_bucket"] for row in rows}):
        groups[f"exclusive_bucket:{bucket}"] = [row for row in rows if row["exclusive_bucket"] == bucket]
    summary = {"replay_scope": "Source alerts are injected as already-selected candidates at alert availability time. This tests structure confirmation/management compatibility, not universe discovery or alert prediction.",
               "configuration": {"version": config.version, "paper_only": config.paper_only,
                                 "execution_enabled": config.execution.enabled, "time_exit_bars": config.target.time_exit_bars,
                                 "label_forward_bars": config.replay.label_forward_bars,
                                 "observation_minutes_after_candidate": observation_minutes},
               "excluded": dict(exclusions or {}), "groups": {}}
    for name, group in groups.items():
        setup_types = Counter(item for row in group for item in row["structure_detected_setup_types"])
        summary["groups"][name] = {"n": len(group),
                                     "setup_detected_rate": None if not group else float(np.mean([row["structure_setup_detected"] for row in group])),
                                     "explicit_confirmation_rate": None if not group else float(np.mean([row.get("structure_explicit_confirmation", False) for row in group])),
                                     "modeled_trade_rate": None if not group else float(np.mean([row["structure_modeled_trade_count"] > 0 for row in group])),
                                     "detected_setup_types": dict(setup_types)}
    return summary


def run_replays(entries: list[Entry], *, observation_minutes: int | None = None) -> tuple[list[dict], dict]:
    """Run only source alerts: this is a confirmation/management compatibility test."""
    config = replace(load_config(), execution=replace(load_config().execution, enabled=False))
    cache: dict[tuple[Path, str], pd.DataFrame] = {}
    rows, exclusions = [], Counter()
    for entry in entries:
        try:
            frame = _frame(entry.source_file, entry.symbol, cache)
            rows.append(_replay_entry(entry, frame, config, observation_minutes=observation_minutes))
        except (KeyError, ValueError) as exc:
            exclusions[f"replay_error:{type(exc).__name__}"] += 1
    return rows, summarize_replay_rows(rows, config, exclusions, observation_minutes)


def _management_summary(entries: list[Entry]) -> dict:
    groups: dict[str, list[Entry]] = {"all": entries}
    for bucket in sorted({entry.exclusive_bucket for entry in entries}):
        groups[f"exclusive_bucket:{bucket}"] = [entry for entry in entries if entry.exclusive_bucket == bucket]
    return {name: dict(Counter(entry.proxy_outcome or "unavailable" for entry in group)) for name, group in groups.items()}


def _render(results: dict) -> str:
    lines = ["# ACE source-tagged setup and Intraday Structure replay", "",
             "Research only. Underlying returns are based on retrospective IEX trade bars; no option prices, fills, or P&L are manufactured.", "",
             "## All source alerts vs matched controls", ""]
    if results["underlying_outcomes"] is None:
        lines.extend(["Outcome comparison omitted for this replay-only shard.", ""])
    else:
        all_metrics = results["underlying_outcomes"]["all"]["outcomes"]
        lines.extend(["| Horizon | Alert n | Alert mean (%) | Control mean (%) | Difference (pp) |", "| --- | ---: | ---: | ---: | ---: |"])
        for horizon, value in all_metrics.items():
            alert = value["alerts"]
            control = value["matched_controls"]
            difference = value["alert_minus_control_mean_pct"]
            fmt = lambda number: "n/a" if number is None else f"{number:.4f}"
            lines.append(f"| {horizon} | {alert['n']} | {fmt(alert['mean_pct'])} | {fmt(control['mean_pct'])} | {fmt(difference)} |")
    replay_summary = results["intraday_structure_replay"]
    if replay_summary.get("status") == "skipped":
        lines.extend(["", "Intraday Structure replay was intentionally skipped for this outcome-only run.", ""])
    else:
        replay = replay_summary["groups"]["all"]
        lines.extend(["", "## Intraday Structure compatibility", "",
                      f"Injected source-alert candidates: {replay['n']}; setup-detected rate: {replay['setup_detected_rate']:.1%}; "
                      f"modeled-share-trade rate: {replay['modeled_trade_rate']:.1%}.", "",
                      "This is conditional on ACE already selecting the candidate; it does not demonstrate that Intraday Structure independently discovers ACE’s universe or that its modeled share trades replicate options performance.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-replay", action="store_true", help="Write source-tagged outcome study without engine replay.")
    parser.add_argument("--replay-only", action="store_true", help="Write one replay shard without duplicating matched controls/outcomes.")
    parser.add_argument("--replay-eligible-outcomes", type=Path,
                        help="Existing source_alert_outcomes.jsonl to select point-in-time eligible alerts for replay-only shards.")
    parser.add_argument("--replay-input", type=Path,
                        help="Preselected research candidate JSONL. Only valid with --replay-only.")
    parser.add_argument("--replay-observation-minutes", type=int,
                        help="Research-only candidate observation window; omit for the full source-session replay.")
    parser.add_argument("--replay-shard-count", type=int, default=1)
    parser.add_argument("--replay-shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.replay_shard_count < 1 or not 0 <= args.replay_shard_index < args.replay_shard_count:
        raise ValueError("replay shard index must be in [0, replay shard count)")
    if args.skip_replay and args.replay_only:
        raise ValueError("--skip-replay and --replay-only cannot be combined")
    if args.replay_input and not args.replay_only:
        raise ValueError("--replay-input requires --replay-only")
    if args.replay_observation_minutes is not None and args.replay_observation_minutes < 1:
        raise ValueError("replay observation minutes must be positive")
    if args.replay_input:
        entries, input_summary = load_replay_input(args.replay_input)
        alerts, controls = [], []
        outcome_build = {"replay_input": str(args.replay_input), "control_rows": 0,
                         "excluded": "Input was preselected by a separate source-audited research step."}
        replay_entries = entries
    else:
        entries, input_summary = load_entries(args.ledger, args.events)
        if args.replay_only and args.replay_eligible_outcomes:
            eligible_rows = read_jsonl(args.replay_eligible_outcomes)
            eligible_keys = {(row["symbol"], pd.Timestamp(row["decision_at_utc"]), row["direction_sign"])
                             for row in eligible_rows}
            alerts, controls = [], []
            outcome_build = {"replay_eligible_outcomes_path": str(args.replay_eligible_outcomes),
                             "replay_eligible_outcomes_sha256": _digest(args.replay_eligible_outcomes),
                             "alert_rows": len(eligible_keys), "control_rows": 0,
                             "excluded": "Eligibility reused from source-tagged outcome study; no outcome fields recalculated."}
            replay_entries = [entry for entry in entries if (entry.symbol, entry.decision, entry.sign) in eligible_keys]
        else:
            alerts, controls, outcome_build = build_outcomes(entries, include_controls=not args.replay_only)
            # Replays use the same point-in-time eligible source alerts as outcome rows.
            eligible_keys = {(row["symbol"], pd.Timestamp(row["decision_at_utc"]), row["direction_sign"]) for row in alerts}
            replay_entries = [entry for entry in entries if (entry.symbol, entry.decision, entry.sign) in eligible_keys]
    replay_entries = [entry for index, entry in enumerate(replay_entries)
                      if index % args.replay_shard_count == args.replay_shard_index]
    if args.skip_replay:
        replay_rows, replay_summary = [], {"status": "skipped"}
    else:
        replay_rows, replay_summary = run_replays(replay_entries, observation_minutes=args.replay_observation_minutes)
    results = {
        "schema": "ace_setup_replay_v1", "research_only": True,
        "input": {"ledger_path": str(args.ledger), "ledger_sha256": _digest(args.ledger),
                  "events_path": str(args.events), "events_sha256": _digest(args.events), **input_summary},
        "source_tag_policy": "Tags are literal case-insensitive source-quote matches. Rows without a match remain unlabelled; 0DTE is retained as a contract-timing tag, not assumed to be a setup type.",
        "underlying_outcome_policy": {"horizons_minutes": list(HORIZONS_MINUTES), "session_close": "15:59 ET one-minute close", "control_policy": f"{CONTROLS_PER_ALERT} same-symbol, same-session times per source alert; every control is more than {EXCLUSION_MINUTES} minutes from any observed alert in that group and carries that alert's direction/tag."},
        "market_data_limitations": ["Retrospectively fetched IEX underlying trade bars; publication timing and original quote state are not certified.", "No option marks, option fills, quantities, or P&L are inferred."],
        "outcome_build": outcome_build,
        "underlying_outcomes": None if args.replay_only else _outcome_report(alerts, controls),
        "management_proxy_outcome_counts": _management_summary(entries),
        "management_proxy_warning": "Management-linked outcome labels exist only where an exported management link was reconstructed. They are source-selected and are not an unbiased profitability denominator.",
        "intraday_structure_replay": replay_summary,
        "replay_shard": {"index": args.replay_shard_index, "count": args.replay_shard_count,
                         "source_alerts_in_shard": len(replay_entries)},
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "source_alert_outcomes.jsonl", alerts)
    if not args.replay_only:
        write_jsonl(args.out / "matched_controls.jsonl", controls)
    write_jsonl(args.out / "intraday_structure_source_alert_replays.jsonl", replay_rows)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(_render(results), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "outcome_build": outcome_build,
                      "all_outcomes": None if results["underlying_outcomes"] is None else results["underlying_outcomes"]["all"],
                      "replay": replay_summary.get("groups", {}).get("all", replay_summary)}, indent=2))


if __name__ == "__main__":
    main()
