"""
Event-driven exit-strategy backtest for the Meta Ranker (top-10 combo).

Unlike the earlier `fwd_close_return` numbers (which were a FIXED 10-day horizon from
entry), this opens a trade when a ticker ENTERS the combo top-K and closes it per the
exit rule, walking the actual 4H price path (close/high/low) and the per-bar top-K
membership. This makes the LIVE rebalance exit ("sell when it drops out") directly
comparable to fixed-horizon, take-profit, scale-out, stop, and trailing variants.

  PYTHONPATH=. python signals/meta_context/meta_ranker/backtest_exits.py

Stock returns are exact. Options returns are a leverage MODEL (see bottom) since we
have no historical option price paths.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[3]
BARS_4H = REPO / "Data/shared/bars/4h"
SCORED = Path("/tmp/meta_scored.parquet")
TOPK = 10
HOLDOUT = pd.Timestamp("2025-07-01", tz="UTC")
MAX_HOLD = 60  # hard cap (~24 trading days) so rebalance trades can't run forever


def _load_member() -> pd.DataFrame:
    df = pd.read_parquet(SCORED).dropna(subset=["s_combo"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["rk"] = df.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    df = df[df["timestamp"] >= HOLDOUT]
    df["in_top"] = df["rk"] <= TOPK
    return df[["timestamp", "ticker", "in_top"]]


def _ticker_path(ticker: str, ts_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    p = BARS_4H / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p, columns=["timestamp", "close", "high", "low"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    b = b.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return b


def simulate(member: pd.DataFrame, *, stop=None, target=None, scale_frac=1.0,
             trail=None, grace=0, horizon=None) -> dict:
    """Return aggregate stats for one exit policy across all tickers' top-K entries."""
    rets, holds = [], []
    for ticker, g in member.groupby("ticker"):
        g = g.sort_values("timestamp")
        bars = _ticker_path(ticker, g["timestamp"])
        if bars is None:
            continue
        m = g.set_index("timestamp")["in_top"].reindex(bars.index).fillna(False).astype(bool).values
        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values
        n = len(bars)
        i = 0
        while i < n - 1:
            if not m[i]:
                i += 1
                continue
            # open at this bar's close
            entry = close[i]
            if entry <= 0:
                i += 1
                continue
            peak = entry
            realized = 0.0      # banked from partial scale-out
            remaining = 1.0     # fraction still open
            trimmed = False
            out = 0
            j = i + 1
            exit_ret = None
            while j < n and (j - i) <= MAX_HOLD:
                peak = max(peak, high[j])
                lo_ret = low[j] / entry - 1
                hi_ret = high[j] / entry - 1
                # 1) hard stop (full)
                if stop is not None and lo_ret <= -stop:
                    exit_ret = -stop
                    break
                # 2) trailing stop on the open remainder (full)
                if trail is not None and low[j] <= peak * (1 - trail):
                    exit_ret = peak * (1 - trail) / entry - 1
                    break
                # 3) target / scale-out
                if target is not None and not trimmed and hi_ret >= target:
                    if scale_frac >= 1.0:
                        exit_ret = target
                        break
                    realized += scale_frac * target
                    remaining = 1.0 - scale_frac
                    trimmed = True
                # 4) rebalance exit (with grace for dip-out-then-back-in)
                out = out + 1 if not m[j] else 0
                if grace is not None and out > grace:
                    exit_ret = close[j] / entry - 1
                    break
                # 5) fixed horizon
                if horizon is not None and (j - i) >= horizon:
                    exit_ret = close[j] / entry - 1
                    break
                j += 1
            if exit_ret is None:  # hit MAX_HOLD or ran out of data
                jj = min(j, n - 1)
                exit_ret = close[jj] / entry - 1
            total = realized + remaining * exit_ret
            rets.append(total)
            holds.append(min(j, n - 1) - i)
            i = min(j, n - 1) + 1  # no overlap: next scan after exit
    r = np.array(rets)
    h = np.array(holds)
    if len(r) == 0:
        return {}
    return {
        "n": len(r), "mean": r.mean(), "median": np.median(r), "win": (r > 0).mean(),
        "std": r.std(), "ret_std": r.mean() / r.std() if r.std() else 0,
        "avg_hold": h.mean(), "ret_per_bar": (r / np.maximum(h, 1)).mean(),
    }


def main():
    member = _load_member()
    print(f"holdout {HOLDOUT.date()}+  ticker-bars={len(member):,}  tickers={member.ticker.nunique()}\n")
    strategies = {
        "fixed horizon 25 (label basis)":      dict(horizon=25, grace=None),
        "rebalance, grace=0 (CURRENT LIVE)":   dict(grace=0),
        "rebalance, grace=1 (dip-out 1 bar)":  dict(grace=1),
        "rebalance, grace=2":                  dict(grace=2),
        "rebalance, grace=3":                  dict(grace=3),
        "stop -15% + rebalance g=1":           dict(stop=0.15, grace=1),
        "trailing -15% (from peak)":           dict(trail=0.15, grace=None),
        "target +20% full exit":               dict(target=0.20, scale_frac=1.0, grace=None),
        "SCALEOUT 50%@+20% + rest rebal g=1":  dict(target=0.20, scale_frac=0.5, grace=1),
        "SCALEOUT 50%@+20% + rest horizon25":  dict(target=0.20, scale_frac=0.5, horizon=25, grace=None),
        "SCALEOUT 50%@+20% + rest trail-15%":  dict(target=0.20, scale_frac=0.5, trail=0.15, grace=1),
    }
    print(f"{'exit policy':38} {'n':>5} {'mean':>7} {'med':>6} {'win':>5} {'ret/std':>7} {'hold':>5} {'ret/bar':>7}")
    for name, kw in strategies.items():
        s = simulate(member, **kw)
        if not s:
            continue
        print(f"{name:38} {s['n']:5} {s['mean']:7.4f} {s['median']:6.3f} {s['win']:5.2f} "
              f"{s['ret_std']:7.3f} {s['avg_hold']:5.1f} {s['ret_per_bar']:7.4f}")

    print("\n# OPTIONS = leverage model on the stock return per trade:")
    print("#   ~0.45-delta monthly call early in life ≈ 4-7x leverage + theta drag.")
    print("#   A +8% stock move ≈ +30-55% option; a -15% stock move ≈ -60-90% option (convex but")
    print("#   capped at -100%). => options need the scale-out MORE (bank the pop) and a wider")
    print("#   effective stop; rank/exit *ordering* is identical, magnitudes are amplified.")


if __name__ == "__main__":
    main()
