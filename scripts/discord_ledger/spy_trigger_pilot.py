"""Research-only SPY pre-alert feature snapshots and same-day comparison times.

The local final-bar cache lacks vendor publication metadata. Controls mean no
ACE SPY alert in this export near that time, not a proven non-trade or loss.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.discord_ledger.build import read_jsonl, write_jsonl

DEFAULT_BARS = Path("Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet")


def snapshot(frame: pd.DataFrame, decision: pd.Timestamp) -> dict:
    """Features from completed, contiguous one-minute bars only."""
    if decision.tzinfo is None:
        raise ValueError("decision time must be timezone-aware")
    decision = decision.tz_convert("UTC")
    local = decision.tz_convert("America/New_York")
    start = local.normalize() + pd.Timedelta(hours=9, minutes=30)
    start_utc = start.tz_convert("UTC")
    end = decision.floor("min") - pd.Timedelta(minutes=1)
    eligible = frame[(frame["timestamp"] >= start_utc) & (frame["timestamp"] <= end)]
    base = {"decision_at_utc": decision.isoformat(), "latest_completed_bar_start_utc": None,
            "status": "no_completed_regular_session_bar", "close": None,
            "return_5m": None, "return_20m": None, "volume_last5_vs_prior20": None,
            "distance_above_opening_range_high_pct": None,
            "distance_above_opening_range_low_pct": None,
            "distance_above_session_vwap_pct": None}
    if eligible.empty:
        return base
    last = eligible.iloc[-1]
    base["latest_completed_bar_start_utc"] = last["timestamp"].isoformat()
    if last["timestamp"] != end:
        base["status"] = "gap_at_decision"
        return base
    tail = eligible.tail(26)
    expected = pd.date_range(end=end, periods=len(tail), freq="min", tz="UTC")
    if not tail["timestamp"].reset_index(drop=True).equals(pd.Series(expected)):
        base["status"] = "gap_in_lookback"
        return base
    price = float(last["close"])
    base["status"] = "ok"
    base["close"] = price
    if len(tail) >= 6:
        base["return_5m"] = price / float(tail.iloc[-6]["close"]) - 1
    if len(tail) >= 21:
        base["return_20m"] = price / float(tail.iloc[-21]["close"]) - 1
    if len(tail) >= 25:
        prior = float(tail.iloc[-25:-5]["volume"].mean())
        base["volume_last5_vs_prior20"] = float(tail.iloc[-5:]["volume"].mean()) / prior if prior > 0 else None
    opening = eligible.iloc[:30]
    if len(opening) == 30 and opening.iloc[-1]["timestamp"] == start_utc + pd.Timedelta(minutes=29):
        base["distance_above_opening_range_high_pct"] = 100 * (price / float(opening["high"].max()) - 1)
        base["distance_above_opening_range_low_pct"] = 100 * (price / float(opening["low"].min()) - 1)
    if eligible["vwap"].notna().all() and eligible["volume"].sum() > 0:
        session_vwap = float((eligible["vwap"] * eligible["volume"]).sum() / eligible["volume"].sum())
        base["distance_above_session_vwap_pct"] = 100 * (price / session_vwap - 1)
    return base


def build(event_rows: list[dict], trades: list[dict], bars: pd.DataFrame) -> tuple[list[dict], dict]:
    events = {row["event_id"]: row for row in event_rows}
    alerts = [events[t["entry_event_id"]] for t in trades if t.get("entry_event_id")
              and t["author_name"] == "ACE" and t["symbol"] == "SPY"]
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    if bars["timestamp"].isna().any() or bars["timestamp"].duplicated().any():
        raise ValueError("Invalid/duplicate source minute-bar timestamps")
    if not bars["timestamp"].is_monotonic_increasing:
        raise ValueError("Source minute-bar timestamps are not ordered")
    if (bars["volume"] < 0).any() or (bars["close"] <= 0).any():
        raise ValueError("Invalid source minute-bar price or volume")
    out = []
    alert_times_by_day: dict[str, list[pd.Timestamp]] = {}
    for event in alerts:
        when = pd.Timestamp(event["available_at_utc"]).tz_convert("UTC")
        day = when.tz_convert("America/New_York").date().isoformat()
        alert_times_by_day.setdefault(day, []).append(when)
        row = snapshot(bars, when)
        row.update({"row_kind": "observed_ACE_alert", "entry_message_id": event["message_id"],
                    "source_message_ids": event["source_message_ids"],
                    "option_type": event.get("option_type"), "strike": event.get("strike"),
                    "reported_option_price": event.get("price"), "session_date_et": day})
        if row["close"] and event.get("strike") is not None:
            distance = 100 * (event["strike"] / row["close"] - 1)
            row["directional_otm_pct"] = distance if event.get("option_type") == "call" else -distance if event.get("option_type") == "put" else None
        else:
            row["directional_otm_pct"] = None
        out.append(row)
    for day, alert_times in alert_times_by_day.items():
        local_day = pd.Timestamp(day, tz="America/New_York")
        grid = pd.date_range(local_day + pd.Timedelta(hours=9, minutes=45),
                             local_day + pd.Timedelta(hours=15, minutes=45), freq="5min")
        for when_local in grid:
            when = when_local.tz_convert("UTC")
            if any(abs((when - alert).total_seconds()) <= 30 * 60 for alert in alert_times):
                continue
            row = snapshot(bars, when)
            if row["status"] != "ok":
                continue
            row.update({"row_kind": "same_day_no_nearby_ACE_SPY_alert_in_export",
                        "entry_message_id": None, "source_message_ids": [],
                        "option_type": None, "strike": None, "reported_option_price": None,
                        "session_date_et": day, "directional_otm_pct": None})
            out.append(row)
    controls_by_day: dict[str, list[float]] = {}
    for row in out:
        if row["row_kind"].startswith("same_day") and row["return_5m"] is not None:
            controls_by_day.setdefault(row["session_date_et"], []).append(row["return_5m"])
    comparable = [row for row in out if row["row_kind"] == "observed_ACE_alert"
                  and row["status"] == "ok" and row["option_type"] in {"call", "put"}
                  and row["return_5m"] is not None and controls_by_day.get(row["session_date_et"])]
    aligned = sum(row["return_5m"] > 0 if row["option_type"] == "call" else row["return_5m"] < 0
                  for row in comparable)
    control_expected = sum(sum(value > 0 if row["option_type"] == "call" else value < 0
                               for value in controls_by_day[row["session_date_et"]])
                           / len(controls_by_day[row["session_date_et"]]) for row in comparable)
    return out, {"ACE_SPY_alert_rows": sum(r["row_kind"] == "observed_ACE_alert" for r in out),
                 "ACE_SPY_alert_feature_status": dict(Counter(r["status"] for r in out if r["row_kind"] == "observed_ACE_alert")),
                 "same_day_comparison_rows": sum(r["row_kind"].startswith("same_day") for r in out),
                 "distinct_alert_days": len(alert_times_by_day),
                 "directional_5m_comparable_alerts": len(comparable),
                 "directional_5m_aligned_alerts": aligned,
                 "same_day_control_expected_alignment_rate": control_expected / len(comparable) if comparable else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bars", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.bars)
    required = {"timestamp", "open", "high", "low", "close", "volume", "vwap"}
    if not required <= set(frame):
        raise ValueError(f"Missing bar columns: {sorted(required - set(frame))}")
    rows, summary = build(read_jsonl(args.run / "events.jsonl"),
                          read_jsonl(args.run / "managed_trades.jsonl"), frame)
    bars_sha256 = hashlib.sha256(args.bars.read_bytes()).hexdigest()
    for row in rows:
        row["market_data_source_sha256"] = bars_sha256
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "snapshots.jsonl", rows)
    summary.update({"bars_path": str(args.bars),
                    "bars_sha256": bars_sha256,
                    "events_sha256": hashlib.sha256((args.run / "events.jsonl").read_bytes()).hexdigest(),
                    "trades_sha256": hashlib.sha256((args.run / "managed_trades.jsonl").read_bytes()).hexdigest(),
                    "source_availability": "retrospectively fetched final bars; publication time/feed/adjustment unverified",
                    "comparison_label_warning": "No nearby ACE SPY alert in this export is not a verified non-trade or losing trade."})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
