"""Measure per-NAME option-market liquidity, and check the share-price confound.

The thesis-test verdict was that long calls fail on our universe because a round
trip costs 32-76% of premium against an 11.9% break-even, while SPY-class names
cost 2-5%. This turns that into a usable filter by measuring every candidate
name's option spread on one consistent basis.

TWO THINGS THIS HAS TO GET RIGHT

1. **The price confound.** Spread as a % of mid is mechanically smaller on
   expensive options, and option premium scales with share price. So "% spread"
   partly just re-measures share price, and a filter on it could collapse into
   "only trade expensive stocks". This records share price alongside so the two
   can be separated, and reports spread WITHIN price buckets.

2. **Current quotes for historical trades.** Only live snapshots are available
   (historical option quotes 404 on this plan). Option liquidity per name is
   persistent enough for a name-level ranking, but this is a proxy and is
   labelled as one. Where a historical measurement exists in `order_audits`
   (limit=ask vs mid at order time) it is reported next to the live one so the
   proxy can be sanity-checked.

Only strikes with mid >= $0.20 count: a $0.05 option quoting $0.01/$0.09 is a
160% spread that says nothing about the name's tradeable liquidity.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient  # noqa: E402

DATA = REPO / "research/execution_quality/data"
OUT = DATA / "option_liquidity_by_name.json"
MIN_MID = 0.20
OCC = re.compile(r"^([A-Z]+)\d{6}[CP]\d{8}$")


def live_spread(client, under: str):
    try:
        r = client._request("GET", client._data_base + f"/v1beta1/options/snapshots/{under}",
                            params={"limit": 400})
    except Exception:  # noqa: BLE001
        return None
    vals, sizes = [], []
    for _sym, s in (r.get("snapshots") or {}).items():
        q = s.get("latestQuote") or {}
        bp, ap = q.get("bp"), q.get("ap")
        bs, asz = q.get("bs") or 0, q.get("as") or 0
        if bp and ap and bp > 0 and ap > 0 and ap >= bp:
            mid = (bp + ap) / 2.0
            if mid >= MIN_MID:
                vals.append((ap - bp) / mid)
                sizes.append(min(bs, asz))
    if len(vals) < 15:
        return None
    return {"n_quotes": len(vals),
            "spread_med": float(np.median(vals)),
            "spread_p25": float(np.percentile(vals, 25)),
            "depth_med": float(np.median(sizes)) if sizes else None}


def historical_spread():
    """(ask - mid)/mid * 2 at order time, from order_audits. Same units."""
    out = defaultdict(list)
    for m in ("momentum_expansion", "multi_ticker_swing_htf", "meta_ranker", "dealer_ranker"):
        p = REPO / f"Data/inference/{m}/live_signal_audit.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            if '"order_audits"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for sym, oa in (r.get("order_audits") or {}).items():
                if oa.get("side") != "buy":
                    continue
                lim, mid = oa.get("limit_price"), oa.get("mid_price")
                mm = OCC.match(str(sym))
                if lim and mid and float(mid) > 0 and mm:
                    out[mm.group(1)].append((float(lim) - float(mid)) / float(mid) * 2.0)
    return {k: float(np.median(v)) for k, v in out.items() if len(v) >= 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    names = set()
    for line in (DATA / "stage1_trade_spine.jsonl").open():
        r = json.loads(line)
        if r.get("module"):
            names.add(r["ticker"])
    for line in (DATA / "stage2_signal_spine.jsonl").open():
        names.add(json.loads(line)["ticker"])
    names = sorted(names)
    print(f"names to measure: {len(names)}")

    u = pd.read_csv(REPO / "Data/shared/universe/shared_universe.csv")
    px = dict(zip(u["ticker"].astype(str), pd.to_numeric(u["px"], errors="coerce")))
    dv = dict(zip(u["ticker"].astype(str), pd.to_numeric(u["avg_dollar_volume_20d"], errors="coerce")))

    client = AlpacaOptionsClient()
    hist = historical_spread()
    rows = {}
    for i, t in enumerate(names, 1):
        s = live_spread(client, t)
        time.sleep(args.sleep)
        if s is None:
            continue
        rows[t] = {**s, "price": None if pd.isna(px.get(t, np.nan)) else float(px[t]),
                   "dollar_vol": None if pd.isna(dv.get(t, np.nan)) else float(dv[t]),
                   "hist_spread": hist.get(t)}
        if i % 50 == 0:
            print(f"  {i}/{len(names)}  measured={len(rows)}", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"\nmeasured {len(rows)} / {len(names)} names -> {OUT}")

    d = pd.DataFrame(rows).T
    d["spread_med"] = d["spread_med"].astype(float)
    print("\nSPREAD DISTRIBUTION (median % of mid, per name)")
    for q in (5, 10, 25, 50, 75, 90):
        print(f"  p{q:<2d} {np.percentile(d['spread_med'], q):6.1%}")
    for thr in (0.02, 0.05, 0.10, 0.15, 0.20):
        n = int((d["spread_med"] <= thr).sum())
        print(f"  names at or under {thr:.0%} spread: {n:4d}  ({n/len(d):.0%})")

    print("\nPRICE CONFOUND — spread within share-price buckets")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    dd = d.dropna(subset=["price"])
    buckets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 1e9)]
    print(f"{'price bucket':>16s} {'n':>5s} {'median spread':>14s} {'p25':>8s} {'best name spread':>17s}")
    for lo, hi in buckets:
        sub = dd[(dd["price"] >= lo) & (dd["price"] < hi)]
        if len(sub) < 5:
            continue
        lab = f"${lo:g}-{hi:g}" if hi < 1e9 else f"${lo:g}+"
        print(f"{lab:>16s} {len(sub):5d} {np.median(sub['spread_med']):13.1%} "
              f"{np.percentile(sub['spread_med'], 25):7.1%} {sub['spread_med'].min():16.1%}")
    r = np.corrcoef(np.log(dd["price"].astype(float)), dd["spread_med"].astype(float))[0, 1]
    print(f"\n  corr(log price, spread) = {r:+.3f}")
    hv = d.dropna(subset=["hist_spread"])
    if len(hv) > 10:
        rr = np.corrcoef(hv["spread_med"].astype(float), hv["hist_spread"].astype(float))[0, 1]
        print(f"  corr(live spread, historical order-time spread) = {rr:+.3f}  (n={len(hv)})"
              "  <- how good the current-quote proxy is")


if __name__ == "__main__":
    main()
