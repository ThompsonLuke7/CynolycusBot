"""
Demo backtest of the OOF long+short rankers with the v2 per-side order policy on a few
tickers, plotted with the shared plotter (plot_trades.plot_ticker_trades).

Directional logic mirrors the live RankerSwingScanner: per (timestamp, ticker) compute calibrated
P_long / P_short, normalise to p_long_dir / p_short_dir, and go with the stronger side when it
clears the gate. Execution uses BACKTEST_CONFIG_V2 (per-side TP/SL, macro filter off).

Usage:
  python -m strategies.multi_ticker_swing.backtest.backtest_ranker_demo --tickers AAPL NVDA AMD
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import (
    BACKTEST_CONFIG_V2, MODULE_ROOT, TRADING_BLACKLIST,
)
from strategies.multi_ticker_swing.backtest.simulate import simulate
from strategies.multi_ticker_swing.backtest.plot_trades import plot_ticker_trades, load_raw

BUNDLE = MODULE_ROOT / "models" / "oof_ranker_20260618"
BDIR   = MODULE_ROOT / "data" / "bundle"
TESTMX = MODULE_ROOT / "backtest" / "competition_20260615_test_matrix.parquet"
OUT    = MODULE_ROOT / "backtest" / "results_oof" / "demo_v2"


def build_directional(plot_tickers: list[str], gate: float, top_k: int = 10) -> tuple[Path, dict[str, pd.DataFrame]]:
    """Full-universe directional selection (the model's real edge is cross-sectional):
    each bar, rank ALL names by conviction = max(p_long_dir, p_short_dir), take the top_k that
    clear the gate, and trade them in their stronger direction. plot_tickers are only used to
    extract per-ticker proba for plotting their actually-selected trades."""
    cl = joblib.load(BUNDLE / "calib_long.joblib")
    cs = joblib.load(BUNDLE / "calib_short.joblib")
    def _load(p, name):
        d = pd.read_parquet(p, columns=["timestamp", "ticker", "score"]).rename(columns={"score": name})
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        return d[d.timestamp >= "2025-08-01"]
    df = _load(BDIR / "oof_preds_long.parquet", "score_long").merge(
         _load(BDIR / "oof_preds_short.parquet", "score_short"), on=["timestamp", "ticker"], how="inner")
    df = df[~df.ticker.str.upper().isin({t.upper() for t in TRADING_BLACKLIST})]

    p_long  = np.clip(cl.predict(df["score_long"].to_numpy()),  1e-9, 1.0)
    p_short = np.clip(cs.predict(df["score_short"].to_numpy()), 1e-9, 1.0)
    df["p_long"], df["p_short"] = p_long, p_short          # raw calibrated (for the plotter)
    # Rank by RAW conviction (max calibrated P(win)), NOT the directional ratio. The ratio
    # P_long/(P_long+P_short) is pure noise on neutral bars (both raw probs ~0); raw P stays
    # low there so junk never makes the top-K. Direction = the stronger raw side.
    df["conviction"] = np.maximum(p_long, p_short)
    df["dir"] = np.where(p_long >= p_short, 1, -1)

    rank = df.groupby("timestamp")["conviction"].rank(ascending=False, method="first")
    df["signal"] = np.where((rank <= top_k) & (df["conviction"] >= gate), df["dir"], 0).astype(int)
    df["score"] = df["conviction"]

    atr = pd.read_parquet(TESTMX, columns=["timestamp", "ticker", "atr_pct_14"])
    atr["timestamp"] = pd.to_datetime(atr["timestamp"], utc=True)
    df = df.merge(atr, on=["timestamp", "ticker"], how="left")

    OUT.mkdir(parents=True, exist_ok=True)
    scores_path = OUT / "directional_scores.parquet"
    df.to_parquet(scores_path, index=False)
    proba_by_ticker = {t: df[df.ticker == t][["timestamp", "p_long", "p_short"]].copy() for t in plot_tickers}
    s = df["signal"]
    print(f"universe directional top-{top_k}: long={int((s==1).sum())} short={int((s==-1).sum())} (gate {gate})")
    return scores_path, proba_by_ticker


def _plot_dense_window(ticker: str, tt: pd.DataFrame, proba: pd.DataFrame, out_path: Path, window_days: int = 12) -> int:
    """Plot the single window_days span that contains the most trades for this ticker (zoomed in)."""
    raw = load_raw(ticker)
    # Trades enter/exit on 5m bars but the plot uses 30m candles; snap to the 30m grid so the
    # markers land on a bar (plot_ticker_trades maps by exact timestamp and silently drops misses).
    tt = tt.copy()
    tt["entry_time"] = pd.to_datetime(tt["entry_time"], utc=True).dt.floor("30min")
    tt["exit_time"] = pd.to_datetime(tt["exit_time"], utc=True).dt.floor("30min")
    tt = tt.sort_values("entry_time")
    win = pd.Timedelta(days=window_days)
    # densest window: start at each trade, count trades within window_days, pick best
    starts = tt["entry_time"].to_numpy()
    best_start, best_n = tt["entry_time"].iloc[0], 0
    for s in tt["entry_time"]:
        n = int(((tt["entry_time"] >= s) & (tt["entry_time"] < s + win)).sum())
        if n > best_n:
            best_n, best_start = n, s
    lo, hi = best_start - pd.Timedelta(days=1), best_start + win + pd.Timedelta(days=1)
    raw_w = raw[(raw["timestamp"] >= lo) & (raw["timestamp"] <= hi)]
    plot_ticker_trades(ticker, raw_w, tt, proba, out_path, tail_bars=len(raw_w))
    return best_n


def main() -> None:
    ap = argparse.ArgumentParser()
    # default to well-traded names (these actually make the top-K often); --tickers to override
    ap.add_argument("--tickers", nargs="+", default=["PLUG", "OPEN", "SMCI"])
    ap.add_argument("--gate", type=float, default=0.30, help="raw-conviction floor (P(win))")
    ap.add_argument("--window-days", type=int, default=12, help="zoom window for plots")
    args = ap.parse_args()

    scores_path, proba_by_ticker = build_directional(args.tickers, args.gate)

    trades, equity = simulate(
        scores_path=scores_path, selection="precomputed", cfg=BACKTEST_CONFIG_V2,
        results_dir=OUT, test_start_str="2025-08-01", test_end_str="2026-06-30", force=True,
    )
    if trades.empty:
        print("no trades"); return
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    print(trades.groupby("direction").agg(n=("pnl_pct", "size"),
          wr=("pnl_pct", lambda s: round((s > 0).mean() * 100, 1)),
          pnl=("pnl_dollar", lambda s: round(s.sum()))).to_string())

    for t in args.tickers:
        tt = trades[trades.ticker == t]
        if tt.empty:
            print(f"{t}: no trades"); continue
        try:
            n = _plot_dense_window(t, tt, proba_by_ticker[t], OUT / f"trades_{t}.png", args.window_days)
            print(f"plotted {t}: {len(tt)} total trades, densest {args.window_days}d window={n} -> {OUT / f'trades_{t}.png'}")
        except Exception as exc:
            print(f"{t}: plot failed: {exc}")


if __name__ == "__main__":
    main()
