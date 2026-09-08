"""Meta-labelling premise test: does the rules engine DECLINE trades that worked?

Meta-labelling only helps when the primary rule has good RECALL — it fires on
most of the moves worth taking, even if it also fires on junk, and ML then
supplies precision. It cannot help when the rule MISSES the moves, because ML
cannot rescue a trade that was never proposed.

So before committing to that direction, test the premise on the engine we have.

98.8% of the engine's 100,422 abstentions are ONE rule:
`invalidation_wider_than_max_atr` — the setup was declined because its stop would
have been too wide. This therefore reduces to a sharp, answerable question:
**do the setups rejected for a wide stop go on to make the move anyway?**

Method: for each declined setup and each confirmed setup, measure the forward
excursion in the setup's own direction from the decision minute, on SIP 1-minute
bars, normalised by the ticker's ATR. Compare like with like, plus a same-ticker
random-time control so "these names move a lot" cannot masquerade as a result.

Read: if declined setups move as well as confirmed ones, the gate is discarding
tradeable setups and RECALL is the problem — meta-labelling has nothing to filter.
If declined setups move worse, the gate is doing its job, recall is fine, and
precision (which is what ML supplies) is the available upside.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
BARS = DATA / "bars_1m"
EVENTS = REPO / "Data/inference/intraday_structure/decision_events.jsonl"
CLOSED = REPO / "Data/inference/intraday_structure/closed_setups.jsonl"
HORIZONS = (15, 30, 60)
_c: dict[str, pd.DataFrame | None] = {}


def bars(t):
    if t not in _c:
        p = BARS / f"{t}.parquet"
        d = None
        if p.exists():
            d = pd.read_parquet(p)
            d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
            d = d.sort_values("timestamp").set_index("timestamp")
        _c[t] = d
    return _c[t]


def atr_at(t, when):
    d = bars(t)
    if d is None:
        return None
    prior = d.loc[d.index < when].tail(390)
    if len(prior) < 60:
        return None
    tr = (prior["high"] - prior["low"]).rolling(14).mean().dropna()
    v = float(tr.iloc[-1]) * np.sqrt(30.0) if len(tr) else None   # ~30-min scale
    return v if v and np.isfinite(v) and v > 0 else None


def excursion(t, when, direction, minutes):
    """Forward favourable/adverse excursion in ATR units, from `when`."""
    d = bars(t)
    if d is None:
        return None
    a = atr_at(t, when)
    if a is None:
        return None
    w = d.loc[(d.index > when) & (d.index <= when + timedelta(minutes=minutes * 3))]
    if len(w) < 5:
        return None
    w = w.iloc[:minutes]
    ref_prior = d.loc[d.index <= when]
    if ref_prior.empty:
        return None
    ref = float(ref_prior["close"].iloc[-1])
    sign = 1 if str(direction).lower() == "long" else -1
    fav = (float(w["high"].max()) - ref) if sign > 0 else (ref - float(w["low"].min()))
    adv = (ref - float(w["low"].min())) if sign > 0 else (float(w["high"].max()) - ref)
    return {"mfe": fav / a, "mae": adv / a}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000)
    args = ap.parse_args()

    have = {p.stem for p in BARS.glob("*.parquet")}
    # DECLINED — dedupe to one row per (setup_id, day): the engine re-evaluates a
    # setup every bar, so raw abstention counts are re-evaluations, not decisions.
    seen, declined = set(), []
    for line in EVENTS.open():
        if '"setup_abstention"' not in line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event_type") != "setup_abstention":
            continue
        p = r.get("payload") or {}
        t = r.get("ticker")
        if t not in have:
            continue
        ts = pd.Timestamp(r["event_time"])
        key = (p.get("setup_id"), t, str(ts.date()))
        if key in seen:
            continue
        seen.add(key)
        declined.append({"ticker": t, "when": ts, "direction": p.get("direction"),
                         "reason": p.get("no_trade_reason")})
    random.Random(7).shuffle(declined)
    declined = declined[:args.sample]

    confirmed = []
    for line in CLOSED.open():
        if not line.strip():
            continue
        r = json.loads(line)
        t, et = r.get("ticker"), r.get("entry_time") or r.get("confirmed_at")
        if t in have and et:
            confirmed.append({"ticker": t, "when": pd.Timestamp(et),
                              "direction": r.get("direction")})

    # CONTROL — same tickers, random minutes, same direction mix.
    rng = random.Random(11)
    control = []
    for c in declined[:len(declined)]:
        d = bars(c["ticker"])
        if d is None or len(d) < 500:
            continue
        control.append({"ticker": c["ticker"],
                        "when": d.index[rng.randrange(200, len(d) - 200)],
                        "direction": c["direction"]})

    print(f"declined (deduped to one per setup per day): {len(declined):,}")
    print(f"confirmed setups with bars                 : {len(confirmed):,}")
    print(f"matched random-time controls               : {len(control):,}\n")

    print("Forward excursion in ATR units from the decision minute\n")
    print(f"{'group':12s} {'horizon':>8s} {'n':>6s} {'median MFE':>11s} {'median MAE':>11s} "
          f"{'MFE-MAE':>9s} {'share MFE>1':>12s}")
    results = {}
    for name, rows in (("DECLINED", declined), ("CONFIRMED", confirmed), ("CONTROL", control)):
        for h in HORIZONS:
            vals = [excursion(r["ticker"], r["when"], r["direction"], h) for r in rows]
            vals = [v for v in vals if v]
            if len(vals) < 20:
                continue
            mfe = np.array([v["mfe"] for v in vals])
            mae = np.array([v["mae"] for v in vals])
            results[(name, h)] = (mfe, mae)
            print(f"{name:12s} {h:7d}m {len(vals):6d} {np.median(mfe):11.3f} "
                  f"{np.median(mae):11.3f} {np.median(mfe - mae):9.3f} "
                  f"{np.mean(mfe > 1.0):11.0%}")
        print()

    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    for h in HORIZONS:
        if ("DECLINED", h) in results and ("CONFIRMED", h) in results:
            dm = np.median(results[("DECLINED", h)][0])
            cm = np.median(results[("CONFIRMED", h)][0])
            ctl = np.median(results[("CONTROL", h)][0]) if ("CONTROL", h) in results else float("nan")
            gap = cm - dm
            print(f"  {h:2d}m  confirmed MFE {cm:.3f} vs declined {dm:.3f} "
                  f"(gap {gap:+.3f}), control {ctl:.3f}")
    print("\n  gap > 0  -> the gate declines WORSE setups: recall is fine, precision is the")
    print("             upside, and meta-labelling has material to work with.")
    print("  gap <= 0 -> the gate declines setups as good as the ones it takes: RECALL is")
    print("             the problem and meta-labelling cannot fix it.")


if __name__ == "__main__":
    main()
