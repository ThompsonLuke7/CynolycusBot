"""Compare pre-alert intraday features for underlying-proxy winners and losers.

This consumes only bars ending before the Discord alert time. It is an
exploratory cohort comparison, not an ML training result or a trading rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from scripts.discord_ledger.build import read_jsonl, write_jsonl
from scripts.discord_ledger.spy_trigger_pilot import snapshot


FEATURES = ("directional_return_5m_pct", "directional_return_20m_pct", "directional_vwap_distance_pct",
            "volume_last5_vs_prior20", "directional_opening_range_high_distance_pct")


def feature_row(row: dict, cache: dict[str, pd.DataFrame]) -> dict:
    source = row.get("entry_underlying_snapshot") or {}
    path = source.get("source_session_file")
    sign = row.get("underlying_direction_sign")
    base = {"trade_id": row["trade_id"], "symbol": row["symbol"], "proxy_outcome": row["proxy_outcome"],
            "option_type": row["option_type"], "entry_message_id": row["entry_message_id"],
            "source_message_ids": row["source_message_ids"], "bar_source": path}
    if not path or sign not in {-1, 1}:
        return {**base, "feature_status": "missing_bar_source"}
    if path not in cache:
        cache[path] = pd.read_parquet(path)
        cache[path]["timestamp"] = pd.to_datetime(cache[path]["timestamp"], utc=True)
    symbol = source.get("proxy_symbol")
    frame = cache[path].loc[cache[path]["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
    snap = snapshot(frame, pd.Timestamp(row["entry_available_at_utc"]))
    if snap["status"] != "ok":
        return {**base, "feature_status": snap["status"]}
    return {**base, "feature_status": "ok", "entry_underlying_price": snap["close"],
            "directional_return_5m_pct": 100 * sign * snap["return_5m"] if snap["return_5m"] is not None else None,
            "directional_return_20m_pct": 100 * sign * snap["return_20m"] if snap["return_20m"] is not None else None,
            "directional_vwap_distance_pct": sign * snap["distance_above_session_vwap_pct"] if snap["distance_above_session_vwap_pct"] is not None else None,
            "volume_last5_vs_prior20": snap["volume_last5_vs_prior20"],
            "directional_opening_range_high_distance_pct": sign * snap["distance_above_opening_range_high_pct"] if snap["distance_above_opening_range_high_pct"] is not None else None}


def summary(rows: list[dict]) -> dict:
    good = [row for row in rows if row["feature_status"] == "ok"]
    result: dict = {"management_proxy_rows": len(rows), "feature_rows_ok": len(good),
                    "outcome_counts": dict(Counter(row["proxy_outcome"] for row in rows)),
                    "feature_medians": {},
                    "system_feature_mapping": {
                        "directional_vwap_distance_pct": "Same concept as intraday_structure `distance_to_vwap_atr` and multi_ticker_swing `dist_to_vwap`, but not ATR-normalized here.",
                        "volume_last5_vs_prior20": "Short-window analogue of intraday_structure relative-volume features and multi_ticker_swing `relative_volume_open_window`.",
                        "directional_return_5m_pct / directional_return_20m_pct": "Short-horizon momentum inputs; existing SPY feature pipeline already computes 5m/20m returns.",
                    },
                    "caveat": "Observed management anchors are source-selected and only 10 directional-proxy losers are available. Differences describe this export cohort and are not validated prediction rules."}
    for feature in FEATURES:
        groups = {}
        for outcome in ("proxy_winner", "proxy_loser"):
            values = pd.to_numeric(pd.Series([row.get(feature) for row in good if row["proxy_outcome"] == outcome]), errors="coerce").dropna()
            groups[outcome] = {"n": len(values), "median": float(values.median()) if len(values) else None,
                               "mean": float(values.mean()) if len(values) else None}
        if groups["proxy_winner"]["median"] is not None and groups["proxy_loser"]["median"] is not None:
            groups["winner_minus_loser_median"] = round(groups["proxy_winner"]["median"] - groups["proxy_loser"]["median"], 5)
        result["feature_medians"][feature] = groups
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ledger = read_jsonl(args.ledger)
    observed = [row for row in ledger if row["proxy_outcome"] in {"proxy_winner", "proxy_loser"}]
    cache: dict[str, pd.DataFrame] = {}
    rows = [feature_row(row, cache) for row in observed]
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "pre_alert_features.jsonl", rows)
    result = summary(rows)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
