"""What would taking only the top-k ranked names have yielded, in SHARES?

Every signal in the spine carries the rank the module assigned it that bar. The
live system took the top 10. This replays k = 1,2,3,5,10 on the same decisions.

Entry  = open of the first session strictly after the decision bar's availability
         (both decision bars are 2pm/4pm ET, so the next open is always the first
         executable price and carries no look-ahead).
Exit   = close of the session HOLD trading days later.
Also reported: MFE/MAE over the hold, in ATR units, ATR from the session before
entry.

No costs are applied here -- shares at Alpaca are commission-free and the spine's
measured equity entry slippage was -0.01 ATR (i.e. nil). The option overlay is a
separate script.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
BARS = REPO / "Data/shared/bars/1d"
HOLDS = [5, 8, 10, 15, 21]
DEPTHS = [1, 2, 3, 5, 10]
_c: dict[str, pd.DataFrame | None] = {}


def daily(t: str):
    if t not in _c:
        p = BARS / f"{t}.parquet"
        d = None
        if p.exists():
            d = pd.read_parquet(p, columns=["timestamp", "open", "high", "low", "close"])
            d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
                "America/New_York").dt.date
            prev = d["close"].shift(1)
            tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(),
                            (d["low"] - prev).abs()], axis=1).max(axis=1)
            d["atr"] = tr.rolling(14, min_periods=5).mean()
            d = d.reset_index(drop=True)
        _c[t] = d
    return _c[t]


def path(ticker: str, decision_day):
    """Entry/exit/MFE for one signal, or None if the bars cannot support it."""
    d = daily(ticker)
    if d is None or len(d) < 30:
        return None
    fwd = d.index[d["date"] > decision_day]
    if len(fwd) == 0:
        return None
    i0 = int(fwd[0])
    entry = float(d["open"].iloc[i0])
    atr = float(d["atr"].iloc[i0 - 1]) if i0 > 0 else np.nan
    if not (np.isfinite(entry) and entry > 0 and np.isfinite(atr) and atr > 0):
        return None
    out = {"entry": entry, "atr": atr, "entry_date": d["date"].iloc[i0]}
    for h in HOLDS:
        w = d.iloc[i0:i0 + h]
        if len(w) < max(2, h // 2):
            continue
        ex = float(w["close"].iloc[-1])
        out[f"ret_{h}"] = ex / entry - 1.0
        out[f"mfe_{h}"] = (float(w["high"].max()) - entry) / atr
        out[f"mae_{h}"] = (entry - float(w["low"].min())) / atr
        out[f"n_{h}"] = len(w)
    return out


def main() -> None:
    sigs = defaultdict(list)
    for line in (DATA / "stage2_signal_spine.jsonl").open():
        r = json.loads(line)
        if r.get("rank") is None or not r.get("submit"):
            continue
        day = datetime.fromisoformat(r["available_at"].replace("Z", "+00:00"))
        sigs[(r["module"], r["available_at"])].append(
            (int(r["rank"]), r["ticker"], day.astimezone().date(), r))

    rows = []
    for (module, avail), items in sigs.items():
        dd = datetime.fromisoformat(avail.replace("Z", "+00:00")).date()
        for rank, ticker, _d, raw in items:
            p = path(ticker, dd)
            if p is None:
                continue
            rows.append({"module": module, "avail": avail, "decision_day": dd,
                         "rank": rank, "ticker": ticker,
                         "was_traded": bool(raw.get("was_traded")),
                         "score": raw.get("score"), **p})
    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "rank_depth_shares.parquet")
    print(f"signals with usable forward bars: {len(df)}  "
          f"(of {sum(len(v) for v in sigs.values())})\n")

    for h in HOLDS:
        col = f"ret_{h}"
        if col not in df:
            continue
        print(f"===== HOLD {h} trading days =====")
        print(f"{'module':24s} {'top-k':>6s} {'n':>5s} {'mean%':>8s} {'med%':>8s} "
              f"{'hit%':>6s} {'p90%':>8s} {'sum$/1k':>9s} {'mfeATR':>7s} {'maeATR':>7s}")
        for module in sorted(df["module"].unique()):
            m = df[(df["module"] == module) & df[col].notna()]
            for k in DEPTHS:
                s = m[m["rank"] <= k]
                if len(s) < 10:
                    continue
                r = s[col]
                print(f"{module:24s} {k:6d} {len(r):5d} {r.mean()*100:8.2f} "
                      f"{r.median()*100:8.2f} {(r > 0).mean()*100:6.1f} "
                      f"{r.quantile(0.9)*100:8.2f} {r.sum()*1000:9.0f} "
                      f"{s[f'mfe_{h}'].median():7.2f} {s[f'mae_{h}'].median():7.2f}")
            print()
        print()


if __name__ == "__main__":
    main()
