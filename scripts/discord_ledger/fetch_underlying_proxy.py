"""Fetch auditable, retrospective underlying-bar snapshots for Discord targets.

The output is an underlying-direction proxy only. It intentionally does not
request, infer, or write option prices or place any broker orders.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import time as time_module
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from core.API.Alpaca_API.core.config import AlpacaConfig
from scripts.discord_ledger.build import read_jsonl, write_jsonl


NY = ZoneInfo("America/New_York")
# These are explicitly labelled proxies rather than silently pretending an
# index option is an ETF option. All other valid stock/ETF symbols retain their
# own underlying symbol.
PROXY_SYMBOL = {"SPX": "SPY", "SPXW": "SPY", "NDXP": "QQQ", "NDX": "QQQ"}


def target_date(target: dict) -> str:
    return pd.Timestamp(target["timestamp_utc"]).tz_convert(NY).date().isoformat()


def proxy_symbol(symbol: str | None) -> str | None:
    clean = str(symbol or "").strip().upper()
    return PROXY_SYMBOL.get(clean, clean) or None


def session_bounds(date_et: str) -> tuple[datetime, datetime]:
    day = datetime.fromisoformat(date_et).date()
    start = datetime.combine(day, time(9, 30), tzinfo=NY).astimezone(ZoneInfo("UTC"))
    # End is exclusive; the final regular-session minute starts at 15:59 ET.
    end = datetime.combine(day, time(16, 1), tzinfo=NY).astimezone(ZoneInfo("UTC"))
    return start, end


def fetch_day(client: StockHistoricalDataClient, date_et: str, symbols: list[str]) -> pd.DataFrame:
    start, end = session_bounds(date_et)
    request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                               start=start, end=end, adjustment=Adjustment.RAW, feed=DataFeed.IEX)
    response = client.get_stock_bars(request)
    frame = response.df
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "close", "volume"])
    frame = frame.reset_index()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def snapshot(target: dict, bars: pd.DataFrame, source_file: str | None) -> dict:
    timestamp = pd.Timestamp(target["timestamp_utc"])
    clean = proxy_symbol(target.get("symbol"))
    date_et = target_date(target)
    base = {"target_id": target["target_id"], "trade_id": target["trade_id"], "role": target["role"],
            "source_symbol": target.get("symbol"), "proxy_symbol": clean, "timestamp_utc": target["timestamp_utc"],
            "source_message_ids": target["source_message_ids"], "feed": "IEX", "adjustment": "raw",
            "certified_point_in_time": False,
            "limitation": "Retrospectively fetched final underlying trade bar; not a quote, fill, option price, or proof of original data availability."}
    local = timestamp.tz_convert(NY)
    if not clean:
        return {**base, "status": "unavailable_no_proxy_symbol", "price": None}
    if local.time() < time(9, 31) or local.time() > time(16, 1):
        return {**base, "status": "unavailable_outside_regular_session", "price": None}
    if bars.empty or not {"symbol", "timestamp", "close"}.issubset(bars.columns):
        return {**base, "status": "unavailable_market_data_request_failed", "price": None,
                "source_session_file": source_file}
    symbol_bars = bars.loc[bars["symbol"] == clean]
    # Bar labels are start times. Only a completed minute may influence the
    # snapshot, avoiding use of the in-progress alert minute.
    usable = symbol_bars.loc[symbol_bars["timestamp"] + pd.Timedelta(minutes=1) <= timestamp]
    if usable.empty:
        return {**base, "status": "unavailable_no_completed_bar", "price": None, "source_session_file": source_file}
    latest = usable.iloc[-1]
    return {**base, "status": "ok", "price": float(latest["close"]),
            "bar_start_utc": latest["timestamp"].isoformat(),
            "bar_completed_at_utc": (latest["timestamp"] + pd.Timedelta(minutes=1)).isoformat(),
            "source_session_file": source_file, "proxy_mapping": "index_to_etf" if clean != target.get("symbol") else "identity"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    targets = read_jsonl(args.targets)
    if len({row["target_id"] for row in targets}) != len(targets):
        raise ValueError("Duplicate target IDs require explicit reconciliation")
    args.out.mkdir(parents=True, exist_ok=False)
    raw_dir = args.out / "raw_session_bars"
    raw_dir.mkdir()
    groups: dict[str, list[dict]] = defaultdict(list)
    for target in targets:
        groups[target_date(target)].append(target)
    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(api_key=cfg.key_id, secret_key=cfg.secret_key)
    snapshots, failures, manifest_days = [], [], []
    for index, (date_et, group) in enumerate(sorted(groups.items())):
        symbols = sorted({proxy_symbol(row.get("symbol")) for row in group if proxy_symbol(row.get("symbol"))})
        bars, filename = pd.DataFrame(), None
        try:
            bars = fetch_day(client, date_et, symbols) if symbols else bars
            file = raw_dir / f"{date_et}.parquet"
            bars.to_parquet(file, index=False)
            filename = str(file)
            manifest_days.append({"date_et": date_et, "symbols_requested": symbols, "rows": len(bars),
                                  "path": filename, "sha256": sha256(file)})
        except Exception as exc:  # preserve all target rows and data-source failures
            failures.append({"date_et": date_et, "symbols_requested": symbols,
                             "error_type": type(exc).__name__, "error": str(exc)[:500]})
        snapshots.extend(snapshot(target, bars, filename) for target in group)
        # Alpaca applies request-rate limits. This only paces read-only market
        # retrieval; no live market data or trading endpoint is touched.
        if index < len(groups) - 1:
            time_module.sleep(0.15)
    write_jsonl(args.out / "snapshots.jsonl", snapshots)
    (args.out / "manifest.json").write_text(json.dumps({"source": "Alpaca historical stock bars", "feed": "IEX", "adjustment": "raw",
        "target_count": len(targets), "snapshot_ok": sum(row["status"] == "ok" for row in snapshots),
        "snapshot_failures": failures, "sessions": manifest_days,
        "limitation": "Historical underlying trade bars are a retrospective proxy, not options bid/ask, fills, or contemporaneous vendor capture."}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"targets": len(targets), "ok": sum(row["status"] == "ok" for row in snapshots),
                      "unavailable": sum(row["status"] != "ok" for row in snapshots), "request_failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
