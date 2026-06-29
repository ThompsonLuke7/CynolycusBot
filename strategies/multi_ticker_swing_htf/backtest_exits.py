"""Event-driven exit/parameter backtest for the standalone HTF Swing harness.

Mirrors signals/meta_context/meta_ranker/backtest_exits.py, but the signal is the
HTF scorer's ``htf_score`` (the same per-(timestamp,ticker) value that feeds the
Meta matrix). A trade opens when a ticker ENTERS the top-K by htf_score and closes
per the exit rule, walking the actual 4H close/high/low path and per-bar top-K
membership — so it directly compares hold/rebalance/stop/trail/scale-out variants
and the top-K width.

  PYTHONPATH=. python strategies/multi_ticker_swing_htf/backtest_exits.py
  PYTHONPATH=. python strategies/multi_ticker_swing_htf/backtest_exits.py --since 2025-07-01 --topks 5 10 15 20
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
BARS_4H = REPO / "Data/shared/bars/4h"
MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
MAX_HOLD = 60  # hard cap (~24 trading days) so rebalance trades can't run forever


def _membership(matrix: Path, since: pd.Timestamp, top_k: int) -> pd.DataFrame:
    df = pd.read_parquet(matrix, columns=None).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["htf_score"])
    df = df[df["timestamp"] >= since]
    df["rk"] = df.groupby("timestamp")["htf_score"].rank(ascending=False, method="first")
    df["in_top"] = df["rk"] <= top_k
    return df[["timestamp", "ticker", "in_top"]]


def _ticker_path(ticker: str) -> pd.DataFrame | None:
    p = BARS_4H / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p, columns=["timestamp", "close", "high", "low"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    return b.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def _load_paths(member: pd.DataFrame) -> dict:
    """Read each ticker's 4H path ONCE (the slow part) so policy sweeps are cheap."""
    return {t: _ticker_path(t) for t in member["ticker"].unique()}


def simulate(member: pd.DataFrame, paths: dict, *, stop=None, target=None, scale_frac=1.0,
             trail=None, grace=0, horizon=None) -> dict:
    """Aggregate stats for one exit policy across all tickers' top-K entries
    (no-overlap: the next scan resumes after each exit). Returns are exact stock
    returns; the options harness is a leverage model on top (see report notes)."""
    rets, holds = [], []
    for ticker, g in member.groupby("ticker"):
        g = g.sort_values("timestamp")
        bars = paths.get(ticker)
        if bars is None:
            continue
        m = g.set_index("timestamp")["in_top"].reindex(bars.index).fillna(False).astype(bool).values
        close, high, low = bars["close"].values, bars["high"].values, bars["low"].values
        n = len(bars)
        i = 0
        while i < n - 1:
            if not m[i]:
                i += 1
                continue
            entry = close[i]
            if entry <= 0:
                i += 1
                continue
            peak = entry
            realized, remaining, trimmed, out = 0.0, 1.0, False, 0
            j, exit_ret = i + 1, None
            while j < n and (j - i) <= MAX_HOLD:
                peak = max(peak, high[j])
                lo_ret, hi_ret = low[j] / entry - 1, high[j] / entry - 1
                if stop is not None and lo_ret <= -stop:
                    exit_ret = -stop; break
                if trail is not None and low[j] <= peak * (1 - trail):
                    exit_ret = peak * (1 - trail) / entry - 1; break
                if target is not None and not trimmed and hi_ret >= target:
                    if scale_frac >= 1.0:
                        exit_ret = target; break
                    realized += scale_frac * target
                    remaining = 1.0 - scale_frac
                    trimmed = True
                out = out + 1 if not m[j] else 0
                if grace is not None and out > grace:
                    exit_ret = close[j] / entry - 1; break
                if horizon is not None and (j - i) >= horizon:
                    exit_ret = close[j] / entry - 1; break
                j += 1
            if exit_ret is None:
                exit_ret = close[min(j, n - 1)] / entry - 1
            rets.append(realized + remaining * exit_ret)
            holds.append(min(j, n - 1) - i)
            i = min(j, n - 1) + 1
    r, h = np.array(rets), np.array(holds)
    if len(r) == 0:
        return {}
    return {"n": len(r), "mean": r.mean(), "median": np.median(r), "win": (r > 0).mean(),
            "ret_std": r.mean() / r.std() if r.std() else 0.0,
            "avg_hold": h.mean(), "ret_per_bar": (r / np.maximum(h, 1)).mean()}


_POLICIES = {
    "fixed horizon 25":                    dict(horizon=25, grace=None),
    "rebalance, grace=0":                  dict(grace=0),
    "rebalance, grace=1":                  dict(grace=1),
    "rebalance, grace=2":                  dict(grace=2),
    "rebalance, grace=3":                  dict(grace=3),
    "stop -15% + rebalance g=1":           dict(stop=0.15, grace=1),
    "trailing -15% (from peak)":           dict(trail=0.15, grace=None),
    "target +20% full exit":               dict(target=0.20, scale_frac=1.0, grace=None),
    "SCALE 50%@+20% + rest rebal g=1":     dict(target=0.20, scale_frac=0.5, grace=1),
    "SCALE 50%@+20% + rest horizon25":     dict(target=0.20, scale_frac=0.5, horizon=25, grace=None),
    "SCALE 50%@+20% + rest trail-15%":     dict(target=0.20, scale_frac=0.5, trail=0.15, grace=1),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-07-01")
    ap.add_argument("--topks", type=int, nargs="*", default=[5, 10, 15, 20])
    ap.add_argument("--matrix", default=str(MATRIX))
    args = ap.parse_args()
    since = pd.Timestamp(args.since, tz="UTC")

    best = []
    for k in args.topks:
        member = _membership(Path(args.matrix), since, k)
        paths = _load_paths(member)
        print(f"\n===== top-{k} by htf_score  |  holdout {since.date()}+  "
              f"ticker-bars={len(member):,}  tickers={member.ticker.nunique()} =====")
        print(f"{'exit policy':36} {'n':>5} {'mean':>7} {'med':>6} {'win':>5} {'ret/std':>7} {'hold':>5} {'ret/bar':>8}")
        for name, kw in _POLICIES.items():
            s = simulate(member, paths, **kw)
            if not s:
                continue
            print(f"{name:36} {s['n']:5} {s['mean']:7.4f} {s['median']:6.3f} {s['win']:5.2f} "
                  f"{s['ret_std']:7.3f} {s['avg_hold']:5.1f} {s['ret_per_bar']:8.4f}")
            best.append({"top_k": k, "policy": name, **s})

    bdf = pd.DataFrame(best)
    if not bdf.empty:
        print("\n##### best configs #####")
        print("by mean/trade:   ", bdf.loc[bdf['mean'].idxmax(), ['top_k', 'policy', 'mean', 'win', 'avg_hold']].to_dict())
        print("by ret/std:      ", bdf.loc[bdf['ret_std'].idxmax(), ['top_k', 'policy', 'mean', 'ret_std', 'avg_hold']].to_dict())
        print("by ret/bar:      ", bdf.loc[bdf['ret_per_bar'].idxmax(), ['top_k', 'policy', 'mean', 'ret_per_bar', 'avg_hold']].to_dict())
    print("\n# OPTIONS = leverage model on the stock return per trade (~4-7x + theta);")
    print("#   ordering of policies is identical, magnitudes amplified — scale-out matters more.")


if __name__ == "__main__":
    main()
