#!/usr/bin/env python3
"""Propose additions to core/shared_universe/delisted_symbols.txt.

A universe symbol is treated as delisted when it is missing from Alpaca's list of
active US equities AND Alpaca reports the asset inactive or does not know it.
Read-only against Alpaca (two GET /v2/assets calls with the paper keys; no
orders, no account access).

Entries are only ever added. A listed symbol that Alpaca shows as active again
is reported for review and left alone: it may be a status that flapped back (IBOC
and OBK, 2026-10-19 -> 10-20; they kept trading and were removed by hand) or a new
security reusing the ticker (SATS, now a Strive preferred, which must stay listed
so its prices are not spliced onto the old company's history). Check its bars.

  .venv/bin/python scripts/build_delisted_symbols.py           # report only
  .venv/bin/python scripts/build_delisted_symbols.py --write   # append new entries
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.shared_universe.delisted import DELISTED_SYMBOLS_PATH, clean_symbol, parse_delisted_symbols  # noqa: E402

UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
BARS_1D = REPO / "Data/shared/bars/1d"


def classify(universe: list[str], *, active: set[str], inactive: dict[str, str]) -> dict[str, str]:
    """Universe symbols that no longer trade, with the evidence for each."""
    out: dict[str, str] = {}
    for symbol in sorted({clean_symbol(s) for s in universe}):
        if symbol in active:
            continue
        name = inactive.get(symbol)
        out[symbol] = "not found at Alpaca" if name is None else f"inactive at Alpaca; {name}".rstrip("; ")
    return out


def last_daily_bar(symbol: str, bars_dir: Path = BARS_1D) -> str:
    path = bars_dir / f"{symbol}.parquet"
    if not path.exists():
        return "none"
    ts = pd.read_parquet(path, columns=["timestamp"])["timestamp"]
    return pd.to_datetime(ts, utc=True).max().strftime("%Y-%m-%d") if len(ts) else "none"


def merge(existing_text: str, entries: dict[str, str]) -> tuple[str, list[str]]:
    """Append entries not already listed; existing lines are kept byte-for-byte."""
    listed = parse_delisted_symbols(existing_text)
    added = [s for s in sorted(entries) if s not in listed]
    if not added:
        return existing_text, []
    text = existing_text if existing_text.endswith("\n") or not existing_text else existing_text + "\n"
    text += f"# added {date.today().isoformat()} by scripts/build_delisted_symbols.py\n"
    text += "".join(f"{s:<6} # {entries[s]}\n" for s in added)
    return text, added


def _alpaca_get(path: str):
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    base = os.environ["APCA_API_BASE_URL_PAPER"].rstrip("/")
    base = base[:-3] if base.endswith("/v2") else base
    req = urllib.request.Request(
        f"{base}{path}",
        headers={"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _alpaca_assets(status: str) -> list[dict]:
    return _alpaca_get(f"/v2/assets?status={status}&asset_class=us_equity")


def _alpaca_asset(symbol: str) -> dict | None:
    """One asset by symbol, or None when Alpaca does not know it (404)."""
    import urllib.error

    try:
        return _alpaca_get(f"/v2/assets/{symbol}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="Append new entries to the list file.")
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--list-file", default=str(DELISTED_SYMBOLS_PATH))
    args = ap.parse_args(argv)

    universe = pd.read_csv(args.universe)["ticker"].dropna().astype(str).tolist()
    active = {clean_symbol(a["symbol"]) for a in _alpaca_assets("active")}
    inactive = {clean_symbol(a["symbol"]): str(a.get("name") or "") for a in _alpaca_assets("inactive")}
    if len(active) < 1000:
        raise SystemExit(f"Alpaca returned only {len(active)} active assets; refusing to classify against it")

    # The bulk inactive list does not carry every retired symbol; ask for the
    # rest one by one so "not found" really means Alpaca has no such asset.
    for symbol in sorted({clean_symbol(u) for u in universe} - active - set(inactive)):
        asset = _alpaca_asset(symbol)
        if asset is not None and str(asset.get("status")) == "active":
            active.add(symbol)
        elif asset is not None:
            inactive[symbol] = str(asset.get("name") or "")
    found = classify(universe, active=active, inactive=inactive)
    entries = {s: f"{note.split(';')[0]}; last daily bar {last_daily_bar(s)}" + (f"; {note.split('; ', 1)[1]}" if "; " in note else "")
               for s, note in found.items()}

    list_path = Path(args.list_file)
    existing = list_path.read_text(encoding="utf-8") if list_path.exists() else ""
    listed = parse_delisted_symbols(existing)
    text, added = merge(existing, entries)
    revived = sorted(listed & active)

    print(f"universe {len(set(map(clean_symbol, universe)))} | not active at Alpaca {len(found)} | already listed {len(listed)} | new {len(added)}")
    for s in added:
        print(f"  + {s:<6} {entries[s]}")
    for s in revived:
        print(f"  ! {s:<6} is listed but ACTIVE at Alpaca again -- review and remove by hand if it trades")
    if args.write and added:
        list_path.write_text(text, encoding="utf-8")
        print(f"wrote {list_path}")
    elif added:
        print("report only; rerun with --write to append")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
