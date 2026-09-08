"""Thesis test, step 1-2: pick the contract the THESIS would buy, fetch its real price path.

The thesis under test: enter on the module's signal, buy a ~35-45 DTE MONTHLY
call, hold 15-20 trading days, and let the right tail pay for the premium. The
live system instead buys ~21 DTE and is stopped out at a median of 2.0 trading
days by a -39% PREMIUM stop, which on a 7.8x-levered contract fires at roughly a
5% underlying move -- about 3.2x too tight for a hold that must tolerate a -16%
median adverse excursion. So the thesis has never actually run.

Everything here uses REAL historical option bars from Alpaca
(/v1beta1/options/bars). Nothing is modelled. That is deliberate: the 2026-07
retraction happened because option P&L was inferred rather than measured, and the
standing rule from it is that a derivative's price series must be verified against
its underlying before any P&L is computed. `validate_paths.py` does exactly that
and this script writes the raw material for it.

CAVEAT built into the design: option daily bars are TRADE bars, not marks. On a
thin contract a bar only exists on days something traded, and the "price" on other
days is a stale print. Coverage is therefore recorded per contract and the
validation step is what decides which contracts are usable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

DATA = REPO / "research/execution_quality/data"
OUT = DATA / "thesis_contract_paths.jsonl"
SPINE = DATA / "stage3_trade_metrics.jsonl"   # carries u_fill (underlying at entry)
BARS_1D = REPO / "Data/shared/bars/1d"

TARGET_DTE_LO = 30          # monthly-ish: far enough out that a 15-20 day hold
TARGET_DTE_HI = 60          # is not spent in terminal theta decay
DELTA_LO, DELTA_HI = 0.30, 0.60


_SPOT_CACHE: dict[str, pd.DataFrame | None] = {}


def _spot_for(row: dict, day: date) -> float | None:
    """Underlying price at the signal. Prefer the measured 1m fill price; fall
    back to the prior daily close so a missing 1m bar does not drop the entry."""
    for key in ("u_fill", "oa_underlying_price"):
        v = row.get(key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f) and f > 0:
            return f
    t = row["ticker"]
    if t not in _SPOT_CACHE:
        path = BARS_1D / f"{t}.parquet"
        d = None
        if path.exists():
            d = pd.read_parquet(path, columns=["timestamp", "close"])
            d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
                "America/New_York").dt.date
        _SPOT_CACHE[t] = d
    d = _SPOT_CACHE[t]
    if d is None:
        return None
    prior = d[d["date"] <= day]
    if prior.empty:
        return None
    v = float(prior["close"].iloc[-1])
    return v if np.isfinite(v) and v > 0 else None


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    fridays = [d + timedelta(days=i) for i in range(31)
               if (d + timedelta(days=i)).month == m and (d + timedelta(days=i)).weekday() == 4]
    return fridays[2]


def target_monthly(signal_day: date) -> date:
    """The monthly expiry sitting 30-60 days out from the signal."""
    for bump in (1, 2, 3):
        m = signal_day.month + bump
        y = signal_day.year + (m - 1) // 12
        exp = third_friday(y, (m - 1) % 12 + 1)
        if TARGET_DTE_LO <= (exp - signal_day).days <= TARGET_DTE_HI:
            return exp
    m = signal_day.month + 1
    return third_friday(signal_day.year + (m - 1) // 12, (m - 1) % 12 + 1)


def pick_contract(client, ticker: str, spot: float, expiry: date, side: str):
    """Nearest-to-target-delta contract, approximated by moneyness.

    Historical greeks are not available for expired contracts, so the delta band
    is approximated by strike distance: a ~0.40-delta call sits slightly OTM.
    Recorded as an approximation rather than presented as a delta.
    """
    cp = "call" if side == "long" else "put"
    lo, hi = (spot * 0.97, spot * 1.15) if cp == "call" else (spot * 0.85, spot * 1.03)
    # EXPIRED contracts are only returned under status="inactive"; the default
    # (active) silently returns an empty list for any past expiry, which reads
    # identically to "this name has no chain". Try inactive first because this is
    # a historical study, then active for expiries that have not passed yet.
    items = []
    for status in ("inactive", "active"):
        try:
            resp = client.get_option_contracts(
                underlying_symbol=ticker.upper(), type=cp,
                expiration_date=expiry.isoformat(), status=status,
                strike_price_gte=round(lo, 2), strike_price_lte=round(hi, 2),
                limit=200,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"contracts_error({type(exc).__name__})"
        got = resp.get("option_contracts", resp) if isinstance(resp, dict) else resp
        items = [c for c in (got or []) if isinstance(c, dict)]
        if items:
            break
    if not items:
        return None, "no_contracts"
    # ~0.40 delta proxy: 3% OTM for calls, 3% ITM-side for puts.
    want = spot * (1.03 if cp == "call" else 0.97)
    best = min(items, key=lambda c: abs(float(c.get("strike_price", 0)) - want))
    return best, "ok"


def fetch_bars(client, symbol: str, start: date, end: date):
    try:
        resp = client._request(
            "GET", client._data_base + "/v1beta1/options/bars",
            params={"symbols": symbol, "timeframe": "1Day",
                    "start": start.isoformat(), "end": end.isoformat(), "limit": 200},
        )
    except Exception as exc:  # noqa: BLE001
        return [], f"bars_error({type(exc).__name__})"
    return ((resp.get("bars") or {}).get(symbol) or []), "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap entries, for a smoke run")
    ap.add_argument("--sleep", type=float, default=0.06)
    ap.add_argument("--from-signals", action="store_true",
                    help="draw entries from the SIGNAL spine (every ranked target) rather "
                         "than only the ~200 that were actually traded. The traded set is "
                         "too small to bucket by option liquidity: it puts only 3-6 "
                         "contracts under a 10%% spread.")
    ap.add_argument("--max-spread", type=float, default=None,
                    help="only fetch names whose measured option spread is at or under this "
                         "(uses option_liquidity_by_name.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    allow = None
    if args.max_spread is not None:
        liq = json.loads((DATA / "option_liquidity_by_name.json").read_text())
        allow = {t for t, v in liq.items()
                 if v.get("spread_med") is not None and v["spread_med"] <= args.max_spread}
        print(f"names within {args.max_spread:.0%} spread: {len(allow)}")

    entries, seen = [], set()
    if args.from_signals:
        src = DATA / "stage2_signal_spine.jsonl"
        for line in src.open():
            r = json.loads(line)
            if not r.get("submit") or not r.get("available_at"):
                continue
            if allow is not None and r["ticker"] not in allow:
                continue
            key = (r["module"], r["ticker"], r["available_at"][:10])
            if key in seen:
                continue
            seen.add(key)
            entries.append({"module": r["module"], "ticker": r["ticker"],
                            "signal_available_at": r["available_at"],
                            "signal_side": r.get("side") or "long"})
    else:
        for line in SPINE.open():
            r = json.loads(line)
            if not r.get("module") or not r.get("signal_available_at"):
                continue
            if r["module"] not in ("momentum_expansion", "multi_ticker_swing_htf", "meta_ranker"):
                continue
            if allow is not None and r["ticker"] not in allow:
                continue
            key = (r["module"], r["ticker"], r["signal_available_at"][:10])
            if key in seen:
                continue
            seen.add(key)
            entries.append(r)
    entries.sort(key=lambda r: r["signal_available_at"])
    if args.limit:
        entries = entries[:args.limit]
    print(f"candidate entries: {len(entries)}")

    client = AlpacaOptionsClient()
    done = ok = 0
    out_path = Path(args.out) if args.out else OUT
    with out_path.open("w", encoding="utf-8") as fh:
        for r in entries:
            done += 1
            sig = datetime.fromisoformat(r["signal_available_at"].replace("Z", "+00:00"))
            day = sig.date()
            spot = _spot_for(r, day)
            if spot is None:
                fh.write(json.dumps({"module": r["module"], "ticker": r["ticker"],
                                     "signal_date": day.isoformat(), "skip": "no_spot"}) + "\n")
                continue
            exp = target_monthly(day)
            contract, why = pick_contract(client, r["ticker"], float(spot), exp,
                                          r.get("signal_side") or "long")
            time.sleep(args.sleep)
            if contract is None:
                fh.write(json.dumps({"module": r["module"], "ticker": r["ticker"],
                                     "signal_date": day.isoformat(), "skip": why}) + "\n")
                continue
            sym = str(contract.get("symbol"))
            bars, bwhy = fetch_bars(client, sym, day, exp)
            time.sleep(args.sleep)
            row = {
                "module": r["module"], "ticker": r["ticker"],
                "signal_date": day.isoformat(), "signal_ts": r["signal_available_at"],
                "side": r.get("signal_side") or "long",
                "spot_at_signal": float(spot),
                "occ": sym, "expiry": exp.isoformat(),
                "strike": float(contract.get("strike_price", 0)),
                "dte_at_entry": (exp - day).days,
                "bars": [{"t": b["t"][:10], "o": b["o"], "h": b["h"], "l": b["l"],
                          "c": b["c"], "v": b.get("v"), "n": b.get("n")} for b in bars],
                "n_bars": len(bars),
                "bars_status": bwhy,
            }
            fh.write(json.dumps(row) + "\n")
            if bars:
                ok += 1
            if done % 25 == 0:
                print(f"  {done}/{len(entries)}  with bars: {ok}", flush=True)
    print(f"wrote {out_path}  entries={done}  with option bars={ok}")


if __name__ == "__main__":
    main()
