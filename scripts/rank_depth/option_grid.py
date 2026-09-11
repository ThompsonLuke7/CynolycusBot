"""The same top-k question, expressed in OPTIONS, on real historical bars.

Standing rule from the 2026-07 retraction: a derivative's price series must be
validated against its underlying BEFORE any P&L is computed. This script does
that first and refuses to report a cell whose contracts fail it.

Design mirrors the share grid exactly so the two are comparable:
  entry = the option's OPEN on the first session after the signal (same session
          the share grid buys), exit = its CLOSE H trading days later.
  Both dates must have a REAL bar for the contract -- a missing bar is a day
  nothing traded, and its "price" would be a stale print, not a mark.
Costs: the measured live spread for that underlying's options, charged once as a
round trip (buy at the offer, sell at the bid). No modelled greeks anywhere.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
BARS = REPO / "Data/shared/bars/1d"
# Two sources, because neither alone covers the surface: Alpaca serves EXPIRED
# contracts and 403s live ones without the OPRA agreement; Schwab serves live
# ones and returns nothing for expired. See core/API/Schwab_API/option_bars.py.
PATHS = [DATA / "rank_top3_option_paths.jsonl",
         DATA / "rank_top3_option_paths_schwab.jsonl"]
HOLDS = [5, 8, 10, 15]
_c: dict[str, pd.DataFrame | None] = {}


def daily(t):
    if t not in _c:
        p = BARS / f"{t}.parquet"
        d = None
        if p.exists():
            d = pd.read_parquet(p, columns=["timestamp", "open", "close"])
            d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
                "America/New_York").dt.date
            d = d.reset_index(drop=True)
        _c[t] = d
    return _c[t]


def underlying_leg(t, signal_day, h):
    d = daily(t)
    if d is None:
        return None
    idx = d.index[d["date"] > signal_day]
    if len(idx) == 0:
        return None
    i0 = int(idx[0])
    w = d.iloc[i0:i0 + h]
    if len(w) < max(2, h // 2):
        return None
    e = float(d["open"].iloc[i0])
    if not (e > 0):
        return None
    return e, float(w["close"].iloc[-1]) / e - 1.0, d["date"].iloc[i0], \
        d["date"].iloc[min(i0 + h - 1, len(d) - 1)]


def option_leg(bars_by_date, entry_d, exit_d):
    """Real bars on BOTH dates or nothing -- no stale prints."""
    b0, b1 = bars_by_date.get(entry_d), bars_by_date.get(exit_d)
    if b0 is None or b1 is None:
        return None
    o, c = float(b0["o"]), float(b1["c"])
    if not (o > 0 and c >= 0):
        return None
    return o, c / o - 1.0, b0.get("v"), b1.get("v")


def main() -> None:
    liq = json.loads((DATA / "option_liquidity_by_name.json").read_text())
    spread = {t: v["spread_med"] for t, v in liq.items()
              if v.get("spread_med") is not None}

    rows, seen = [], set()
    for src in PATHS:
        if not src.exists():
            continue
        for line in src.open():
            r = json.loads(line)
            if r.get("skip") or not r.get("bars"):
                continue
            # A contract priced by both sources is kept once; the Schwab file only
            # holds contracts Alpaca could not price, so this is belt-and-braces.
            key = (r["module"], r["ticker"], r["signal_date"], r["occ"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    raw, rows = rows, []
    for r in raw:
        sd = date.fromisoformat(r["signal_date"])
        bbd = {date.fromisoformat(b["t"]): b for b in r["bars"]}
        for h in HOLDS:
            u = underlying_leg(r["ticker"], sd, h)
            if u is None:
                continue
            u_entry, u_ret, ed, xd = u
            o = option_leg(bbd, ed, xd)
            if o is None:
                continue
            o_entry, o_ret, v0, v1 = o
            rows.append({
                "module": r["module"], "rank": r.get("rank"), "ticker": r["ticker"],
                "signal_date": sd, "hold": h, "occ": r["occ"],
                "source": r.get("bars_source", "alpaca"),
                "expiry": r["expiry"],
                "dte": r["dte_at_entry"], "n_bars": r["n_bars"],
                "u_ret": u_ret, "o_ret": o_ret, "o_entry": o_entry,
                "spread": spread.get(r["ticker"]),
                "same_px": int(abs(o_ret) < 1e-9),
                "entry_vol": v0, "exit_vol": v1,
            })
    df = pd.DataFrame(rows)
    print(f"contract-hold observations with real bars on both dates: {len(df)}")
    print(f"distinct contracts: {df['occ'].nunique()}  "
          f"median bars per contract: {df['n_bars'].median():.0f}\n")

    print("=== VALIDATION (must pass before any P&L is read) ===")
    for h in HOLDS:
        s = df[df["hold"] == h]
        if len(s) < 20:
            continue
        c = np.corrcoef(s["u_ret"], s["o_ret"])[0, 1]
        print(f"  hold {h:2d}d  n={len(s):4d}  corr(option ret, underlying ret) = {c:+.3f}"
              f"   identical entry/exit price: {s['same_px'].mean():.1%}")
    ok = all(np.corrcoef(df[df['hold'] == h]["u_ret"],
                         df[df['hold'] == h]["o_ret"])[0, 1] > 0.6
             for h in HOLDS if (df["hold"] == h).sum() >= 20)
    print(f"  VERDICT: {'PASS -- prices track value, P&L is readable'
                        if ok else 'FAIL -- do not read the P&L below'}\n")
    if not ok:
        return

    print("=== BY EXPIRY CYCLE (top-3, all modules) ===")
    print(f"{'expiry':10s} {'src':8s} {'hold':>4s} {'n':>4s} {'undrl%':>7s} "
          f"{'gross%':>8s} {'net%':>8s} {'win%':>6s} {'med%':>8s}")
    for (exp, src), g0 in df.groupby(["expiry", "source"]):
        for h in HOLDS:
            s2 = g0[(g0["hold"] == h) & g0["spread"].notna()]
            if len(s2) < 10:
                continue
            net = (1 + s2["o_ret"]) * (1 - s2["spread"]) - 1
            print(f"{exp:10s} {src:8s} {h:4d} {len(s2):4d} {s2['u_ret'].mean()*100:7.2f} "
                  f"{s2['o_ret'].mean()*100:8.1f} {net.mean()*100:8.1f} "
                  f"{(net > 0).mean()*100:6.1f} {net.median()*100:8.1f}")
    print()

    print("=== OPTION RETURN BY RANK DEPTH (gross, then net of measured spread) ===")
    print(f"{'module':24s} {'hold':>4s} {'k':>2s} {'n':>4s} {'gross%':>8s} "
          f"{'net%':>8s} {'win%':>6s} {'med%':>8s} {'p90%':>8s} {'spread%':>8s}")
    for module in sorted(df["module"].unique()):
        for h in HOLDS:
            for k in [1, 2, 3]:
                s = df[(df["module"] == module) & (df["hold"] == h) &
                       (df["rank"] <= k) & df["spread"].notna()]
                if len(s) < 10:
                    continue
                g = s["o_ret"]
                net = (1 + s["o_ret"]) * (1 - s["spread"]) - 1
                print(f"{module:24s} {h:4d} {k:2d} {len(s):4d} {g.mean()*100:8.1f} "
                      f"{net.mean()*100:8.1f} {(net > 0).mean()*100:6.1f} "
                      f"{net.median()*100:8.1f} {net.quantile(0.9)*100:8.1f} "
                      f"{s['spread'].median()*100:8.1f}")
        print()
    df.to_parquet(DATA / "rank_depth_options.parquet")


if __name__ == "__main__":
    main()
