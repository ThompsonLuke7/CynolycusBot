"""Does the ranking work at the horizon it was TRAINED for?

Stage 4A tested 1d/3d/10d and found nothing. But the labels target longer:

  momentum_expansion   LABEL_CONFIG.forward_window_4h_bars = 25  -> ~15.4 trading days
  multi_ticker_swing_htf  PIVOT_LABEL_CONFIG forward 13-38 bars -> ~8-23, mid ~15.7

while the live policy exits at a median of 4-5 trading days. So Stage 4A may have
graded the models on a question they were never asked. This re-runs the same
within-decision test (top-3 vs the same bar's lower-ranked names) out to the
label's own horizon, using DAILY bars, which reach back far enough to evaluate a
15-20 day forward window on the whole sample.

Same controls as Stage 4A: within-decision pairing holds the day and the tape
fixed, and the drift control says what an average ranked name did.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "research/execution_quality/data"
DAILY = REPO_ROOT / "Data/shared/bars/1d"
HORIZONS = (1, 3, 5, 10, 15, 20, 25)
_cache: dict[str, pd.DataFrame | None] = {}


def daily(ticker: str):
    if ticker in _cache:
        return _cache[ticker]
    path = DAILY / f"{ticker}.parquet"
    df = None
    if path.exists():
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        prev = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                        (df["low"] - prev).abs()], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14, min_periods=5).mean()
        df["date"] = df["timestamp"].dt.tz_convert("America/New_York").dt.date
    _cache[ticker] = df
    return df


def forward(ticker: str, when: datetime, sign: int):
    """Forward MFE/MAE/return in ATR units at each horizon, from the session AFTER
    the decision — a same-session daily bar contains the decision's own morning."""
    df = daily(ticker)
    if df is None or len(df) < 30:
        return None
    day = when.date()
    idx = df.index[df["date"] > day]
    if len(idx) == 0:
        return None
    i0 = int(idx[0])
    atr = df["atr14"].iloc[max(0, i0 - 1)]
    ref = df["close"].iloc[max(0, i0 - 1)]
    if not (np.isfinite(atr) and atr > 0 and np.isfinite(ref) and ref > 0):
        return None
    out = {}
    for h in HORIZONS:
        w = df.iloc[i0:i0 + h]
        if len(w) < max(1, h // 2):
            continue
        if sign > 0:
            mfe = float(w["high"].max()) - ref
            mae = ref - float(w["low"].min())
        else:
            mfe = ref - float(w["low"].min())
            mae = float(w["high"].max()) - ref
        out[f"mfe_{h}d"] = mfe / atr
        out[f"mae_{h}d"] = mae / atr
        out[f"ret_{h}d"] = (float(w["close"].iloc[-1]) - ref) * sign / atr
    return out


def boot(a, b, n=4000, seed=11):
    a = np.array([x for x in a if x is not None and math.isfinite(x)])
    b = np.array([x for x in b if x is not None and math.isfinite(x)])
    if len(a) < 8 or len(b) < 8:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = [np.median(rng.choice(a, len(a), True)) - np.median(rng.choice(b, len(b), True))
         for _ in range(n)]
    return float(np.median(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    rows = []
    for line in (DATA / "stage2_signal_spine.jsonl").open():
        r = json.loads(line)
        if not r.get("submit"):
            continue
        when = datetime.fromisoformat(r["available_at"].replace("Z", "+00:00"))
        sign = -1 if str(r.get("side", "long")).lower() in ("short", "sell") else 1
        fwd = forward(r["ticker"], when, sign)
        if fwd:
            rows.append({**r, **fwd})
    print(f"signals with a daily forward path: {len(rows)}\n")

    print("=" * 96)
    print("Top-3 vs the SAME BAR's lower-ranked names — median forward MFE difference (ATR)")
    print("Label horizons: momentum ~15d, HTF ~16d.  Stage 4A only reached 10d.")
    print("=" * 96)
    hdr = "  ".join(f"{f'{h}d':>13s}" for h in HORIZONS)
    print(f"{'module':24s} {'n':>5s}  {hdr}")
    for m in sorted({r["module"] for r in rows}):
        sub = [r for r in rows if r["module"] == m]
        cells = []
        for h in HORIZONS:
            k = f"mfe_{h}d"
            bybar = defaultdict(list)
            for r in sub:
                if r.get(k) is not None and r.get("rank") is not None:
                    bybar[r["bar"]].append(r)
            top, rest = [], []
            for _, g in bybar.items():
                if len(g) < 4:
                    continue
                g.sort(key=lambda x: x["rank"])
                top += g[:3]
                rest += g[3:]
            d, lo, hi = boot([r[k] for r in top], [r[k] for r in rest])
            star = "*" if (not math.isnan(lo) and lo * hi > 0) else " "
            cells.append(f"{d:+7.3f}{star}     " if not math.isnan(d) else "      -      ")
        print(f"{m:24s} {len(sub):5d}  " + "  ".join(c[:13].ljust(13) for c in cells))

    print("\n" + "=" * 96)
    print("Drift control — what an AVERAGE ranked name did (all modules pooled)")
    print("=" * 96)
    for h in HORIZONS:
        mfe = [r.get(f"mfe_{h}d") for r in rows if r.get(f"mfe_{h}d") is not None]
        mae = [r.get(f"mae_{h}d") for r in rows if r.get(f"mae_{h}d") is not None]
        ret = [r.get(f"ret_{h}d") for r in rows if r.get(f"ret_{h}d") is not None]
        if not mfe:
            continue
        print(f"  {h:2d}d  n={len(mfe):5d}  MFE={np.median(mfe):6.3f}  MAE={np.median(mae):6.3f}  "
              f"ret={np.median(ret):+6.3f}  share ret>0 = {np.mean([x > 0 for x in ret]):.1%}")

    print("\n" + "=" * 96)
    print("What the exit gives up: MFE still available AFTER a median 4-5 day hold")
    print("=" * 96)
    for h in (5, 10, 15, 20, 25):
        got = [r.get("mfe_5d") for r in rows if r.get("mfe_5d") is not None and r.get(f"mfe_{h}d") is not None]
        full = [r.get(f"mfe_{h}d") for r in rows if r.get("mfe_5d") is not None and r.get(f"mfe_{h}d") is not None]
        if not got:
            continue
        print(f"  MFE by {h:2d}d = {np.median(full):.3f} ATR vs {np.median(got):.3f} by 5d "
              f"-> {np.median(full) - np.median(got):+.3f} ATR left on the table")


if __name__ == "__main__":
    main()
