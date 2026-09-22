"""Fetch dated IEX underlying quotes for reviewed follower cases (read-only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import requests

from core.API.Alpaca_API.core.config import AlpacaConfig
from scripts.discord_ledger.build import read_jsonl, write_jsonl

DATA_URL = "https://data.alpaca.markets"


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"Naive timestamp: {value}")
    return result.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def fetch_quotes(symbol: str, start: datetime, end: datetime, config: AlpacaConfig) -> list[dict]:
    url = f"{DATA_URL}/v2/stocks/{symbol}/quotes"
    headers = {"APCA-API-KEY-ID": config.key_id, "APCA-API-SECRET-KEY": config.secret_key}
    params = {"feed": "iex", "start": iso(start), "end": iso(end), "limit": 10000, "sort": "asc"}
    records = []
    for _ in range(20):
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Historical IEX quote fetch failed for {symbol}: HTTP {response.status_code}")
        data = response.json()
        records.extend(data.get("quotes") or [])
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    else:
        raise RuntimeError(f"Historical IEX quote pagination exceeded 20 pages for {symbol}")
    if not records:
        return []
    stamps = [parse_time(q["t"]) for q in records]
    if stamps != sorted(stamps):
        raise ValueError(f"Out-of-order quote timestamps for {symbol}; no silent sorting")
    if any(not start <= t <= end for t in stamps):
        raise ValueError(f"Out-of-window quote timestamp for {symbol}")
    return records


def snapshot_at(quotes: list[dict], target: datetime) -> dict:
    eligible = [q for q in quotes if parse_time(q["t"]) <= target]
    if not eligible:
        return {"status": "no_prior_quote", "target_timestamp_utc": iso(target), "quote": None}
    quote = eligible[-1]
    if len(eligible) > 1 and eligible[-2]["t"] == quote["t"] and eligible[-2] != quote:
        return {"status": "ambiguous_same_timestamp_quotes", "target_timestamp_utc": iso(target),
                "quote": None, "same_timestamp": quote["t"]}
    age = (target - parse_time(quote["t"])).total_seconds()
    bid, ask = quote.get("bp"), quote.get("ap")
    if age > 10 or bid is None or ask is None or bid <= 0 or ask < bid:
        return {"status": "stale_or_invalid", "target_timestamp_utc": iso(target),
                "quote": quote, "age_seconds": age}
    return {"status": "fresh_iex_underlying_quote_not_option_quote", "target_timestamp_utc": iso(target),
            "quote": quote, "age_seconds": age, "mid": (bid + ask) / 2,
            "spread": ask - bid}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reviewed = read_jsonl(args.reviewed)
    config = AlpacaConfig.from_env()
    args.out.mkdir(parents=True, exist_ok=False)
    all_quotes, cases = [], []
    for item in reviewed:
        alert = parse_time(item["alert_timestamp_utc"])
        symbol = item["symbol"]
        start, end = alert - timedelta(seconds=30), alert + timedelta(seconds=310)
        quotes = fetch_quotes(symbol, start, end, config)
        all_quotes.extend({"alert_message_id": item["alert_message_id"], "symbol": symbol,
                           "feed": "iex", "quote": q} for q in quotes)
        snapshots = {str(delay): snapshot_at(quotes, alert + timedelta(seconds=delay))
                     for delay in (0, 30, 60, 300)}
        cases.append({"alert_message_id": item["alert_message_id"], "symbol": symbol,
                      "alert_timestamp_utc": item["alert_timestamp_utc"],
                      "feed": "iex", "quote_count": len(quotes),
                      "request_start_utc": iso(start), "request_end_utc": iso(end),
                      "retrieved_at_utc": iso(datetime.now(timezone.utc)),
                      "snapshots": snapshots,
                      "limitation": "Retrospectively fetched IEX underlying quotes; not an options contract quote, NBBO, follower fill, or certified contemporaneous vendor capture.",
                      "source_message_ids": item["source_message_ids"]})
    write_jsonl(args.out / "raw_iex_quotes.jsonl", all_quotes)
    write_jsonl(args.out / "case_snapshots.jsonl", cases)
    manifest = {"source": "Alpaca historical stock quote API", "feed": "iex", "case_count": len(cases),
                "total_quote_rows": len(all_quotes), "credential_values_recorded": False,
                "reviewed_input_sha256": hashlib.sha256(args.reviewed.read_bytes()).hexdigest(),
                "raw_sha256": hashlib.sha256((args.out / "raw_iex_quotes.jsonl").read_bytes()).hexdigest(),
                "snapshot_sha256": hashlib.sha256((args.out / "case_snapshots.jsonl").read_bytes()).hexdigest()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "quote_rows": len(all_quotes),
                      "fresh_snapshots": sum(s["status"].startswith("fresh") for c in cases for s in c["snapshots"].values())}))


if __name__ == "__main__":
    main()
