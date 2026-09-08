#!/usr/bin/env python
"""Fetch real daily trade bars for the fixed multi-leg structure menu.

Read-only broker-data access. Results are resumable JSONL records; no orders are submitted.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import time

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient  # noqa: E402
from scripts.multileg_options_experiment.core import build_structures  # noqa: E402

SOURCE = REPO / "research/execution_quality/data/thesis_contract_paths.jsonl"
OUT_DIR = REPO / "research/multileg_options_experiment/data"
OUT = OUT_DIR / "structure_paths.jsonl"


def _contracts(response: object) -> list[dict]:
    if not isinstance(response, dict):
        return []
    rows = response.get("option_contracts", [])
    return [x for x in rows if isinstance(x, dict)]


def get_contracts(client: AlpacaOptionsClient, ticker: str, expiry: str) -> list[dict]:
    for status in ("inactive", "active"):
        response = client.get_option_contracts(
            underlying_symbol=ticker, expiration_date=expiry, status=status, limit=1000,
        )
        rows = _contracts(response)
        if rows:
            return rows
    return []


def candidate_fridays(signal_day: date, far_expiry: date):
    current = signal_day + timedelta(days=1)
    while current.weekday() != 4:
        current += timedelta(days=1)
    while current < far_expiry:
        dte = (current - signal_day).days
        if 7 <= dte <= 28:
            yield current
        current += timedelta(days=7)


def get_near_contracts(client: AlpacaOptionsClient, ticker: str, signal_day: date, far_expiry: date) -> list[dict]:
    # Prefer the longest near expiry: more remaining value and less terminal-expiry noise.
    choices = list(candidate_fridays(signal_day, far_expiry))
    for expiry in reversed(choices):
        rows = get_contracts(client, ticker, expiry.isoformat())
        if rows:
            return rows
    return []


def fetch_bars(client: AlpacaOptionsClient, symbols: list[str], start: str, end: str) -> dict[str, list[dict]]:
    if not symbols:
        return {}
    response = client._request(
        "GET", client._data_base + "/v1beta1/options/bars",
        params={"symbols": ",".join(symbols), "timeframe": "1Day", "start": start, "end": end, "limit": 10_000},
    )
    payload = response.get("bars", {}) if isinstance(response, dict) else {}
    return {symbol: list(payload.get(symbol, [])) for symbol in symbols}


def _clean_bars(rows: list[dict]) -> list[dict]:
    keys = ("t", "o", "h", "l", "c", "v", "n", "vw")
    return [{key: row.get(key) for key in keys} for row in rows]


def load_source() -> list[dict]:
    rows = [json.loads(line) for line in SOURCE.open() if line.strip()]
    # The far expiry must be complete. Future expiries cannot support an 8-session realized path.
    today = date.today()
    return [row for row in rows if row.get("bars") and date.fromisoformat(row["expiry"]) < today]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.08)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if OUT.exists() and not args.restart:
        for line in OUT.open():
            if line.strip():
                done.add(str(json.loads(line)["trade_key"]))
    mode = "w" if args.restart else "a"
    source = load_source()
    if args.limit is not None:
        source = source[: args.limit]
    client = AlpacaOptionsClient()
    wrote = 0
    with OUT.open(mode, encoding="utf-8") as handle:
        for i, row in enumerate(source, 1):
            key = f"{row['module']}|{row['ticker']}|{row['signal_date']}|{row['occ']}"
            if key in done:
                continue
            try:
                far = get_contracts(client, row["ticker"], row["expiry"])
                time.sleep(args.sleep)
                near = get_near_contracts(
                    client, row["ticker"], date.fromisoformat(row["signal_date"]), date.fromisoformat(row["expiry"]),
                )
                time.sleep(args.sleep)
                structures = build_structures(row["ticker"], float(row["spot_at_signal"]), row["occ"], far, near)
                symbols = sorted({leg.symbol for legs in structures.values() for leg in legs if leg.right != "S"})
                bars = fetch_bars(client, symbols, row["signal_date"], row["expiry"])
                time.sleep(args.sleep)
                payload = {
                    "trade_key": key,
                    "module": row["module"],
                    "ticker": row["ticker"],
                    "signal_date": row["signal_date"],
                    "signal_ts": row.get("signal_ts"),
                    "spot_at_signal": row["spot_at_signal"],
                    "far_expiry": row["expiry"],
                    "near_expiry": (str(near[0].get("expiration_date")) if near else None),
                    "structures": {
                        name: [leg.__dict__ for leg in legs] for name, legs in structures.items()
                    },
                    "bars": {symbol: _clean_bars(values) for symbol, values in bars.items()},
                }
            except Exception as exc:  # one failed ticker must not destroy a resumable research run
                payload = {
                    "trade_key": key, "module": row["module"], "ticker": row["ticker"],
                    "signal_date": row["signal_date"], "error": f"{type(exc).__name__}: {exc}",
                }
            handle.write(json.dumps(payload, allow_nan=False) + "\n")
            handle.flush()
            wrote += 1
            print(f"{i}/{len(source)} {row['ticker']} structures={len(payload.get('structures', {}))} error={payload.get('error')}", flush=True)
    print(f"wrote {wrote} new rows to {OUT}")


if __name__ == "__main__":
    main()

