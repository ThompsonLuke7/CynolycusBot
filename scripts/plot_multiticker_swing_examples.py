from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from shared_plotting import (
    DEFAULT_THEME,
    apply_mpl_defaults,
    make_price_probability_figure,
    plot_candles_from_frame,
    plot_direction_probabilities,
    save_figure,
    setup_datetime_axis,
    to_mpl_time,
)


ROOT = Path("multi_ticker_swing/backtest/results/sweep_v4_shared_20260606")
PROBA_PATH = Path("multi_ticker_swing/models/p_swing_probs.parquet")
RAW_5M = Path("multi_ticker_swing/data/raw/5m")
RAW_30M = Path("multi_ticker_swing/data/raw/30m")
OUT_DIR = Path("UI/swing_audit/backtest_examples_20260606")

apply_mpl_defaults()


def _load_bars(ticker: str, timeframe: str) -> pd.DataFrame:
    path = (RAW_5M if timeframe == "5m" else RAW_30M) / f"{ticker}.parquet"
    df = pd.read_parquet(path)
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns and df.index.name == "timestamp":
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_probs(ticker: str) -> pd.DataFrame:
    cols = ["ticker", "timestamp", "p_long", "p_short"]
    df = pd.read_parquet(PROBA_PATH, columns=cols)
    df = df[df["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    p_dir = (df["p_long"] + df["p_short"]).clip(lower=1e-8)
    df["p_long_dir"] = df["p_long"] / p_dir
    df["p_short_dir"] = df["p_short"] / p_dir
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


def _trade_times(trade: pd.Series, bars5: pd.DataFrame | None, bars30: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    signal_idx = int(trade["signal_idx"])
    exit_idx = int(trade["exit_idx"])
    entry_price = float(trade["entry_price"])
    holding = int(trade["holding_bars"])
    signal_ts = bars30.iloc[signal_idx]["timestamp"]

    if bars5 is not None and holding > 0:
        after_signal = bars5.index[bars5["timestamp"] > signal_ts]
        candidates = [
            idx for idx in after_signal[:8]
            if abs(float(bars5.iloc[idx]["open"]) - entry_price) <= max(0.01, entry_price * 0.002)
        ]
        if candidates:
            entry_idx = int(candidates[0])
            exit_5m_idx = min(entry_idx + holding, len(bars5) - 1)
            return bars5.iloc[entry_idx]["timestamp"], bars5.iloc[exit_5m_idx]["timestamp"], "5m"

    return bars30.iloc[signal_idx]["timestamp"], bars30.iloc[min(exit_idx, len(bars30) - 1)]["timestamp"], "30m"


def _plot_trade(trade: pd.Series, tier: str, rank: int) -> Path | None:
    ticker = str(trade["ticker"]).upper()
    bars30 = _load_bars(ticker, "30m")
    bars5 = _load_bars(ticker, "5m") if (RAW_5M / f"{ticker}.parquet").exists() else None
    entry_ts, exit_ts, tf = _trade_times(trade, bars5, bars30)
    bars = bars5 if tf == "5m" else bars30
    probs = _load_probs(ticker)

    start = entry_ts - pd.Timedelta(days=2)
    end = exit_ts + pd.Timedelta(days=2)
    view = bars[(bars["timestamp"] >= start) & (bars["timestamp"] <= end)].copy()
    pview = probs[(probs["timestamp"] >= start) & (probs["timestamp"] <= end)].copy()
    if view.empty:
        return None

    direction = int(trade["direction"])
    side = "LONG" if direction == 1 else "SHORT"
    pnl = float(trade["pnl_pct"])
    theme = DEFAULT_THEME
    color = theme.win if pnl >= 0 else theme.loss
    entry_price = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])
    title = (
        f"{tier.upper()} #{rank}: {ticker} {side} "
        f"{pnl * 100:.1f}% underlying, exit={trade['exit_reason']}"
    )

    fig, ax_price, ax_prob = make_price_probability_figure(figsize=(13, 7), theme=theme)

    plot_candles_from_frame(ax_price, view, time_col="timestamp", compressed=False, theme=theme)
    entry_x, exit_x = to_mpl_time([entry_ts, exit_ts])
    ax_price.axvspan(entry_x, exit_x, color=color, alpha=0.12)
    ax_price.scatter([entry_x], [entry_price], marker="^" if direction == 1 else "v", s=110, color=theme.blue, zorder=5, label="entry")
    ax_price.scatter([exit_x], [exit_price], marker="x", s=110, color=color, zorder=5, label="exit")
    ax_price.axhline(entry_price, color=theme.blue, lw=1, ls="--", alpha=0.8)
    ax_price.axhline(exit_price, color=color, lw=1, ls=":", alpha=0.8)
    ax_price.set_title(title)
    ax_price.set_ylabel("Underlying price")
    ax_price.legend(loc="best")

    if not pview.empty:
        p_x = to_mpl_time(pview["timestamp"])
        plot_direction_probabilities(
            ax_prob,
            p_x,
            pview["p_long_dir"],
            pview["p_short_dir"],
            theme=theme,
            thresholds=(0.6, 0.7),
        )
    ax_prob.axvspan(entry_x, exit_x, color=color, alpha=0.08)
    ax_prob.legend(loc="best", ncol=2)
    setup_datetime_axis(ax_prob)

    fig.autofmt_xdate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tier}_{rank:02d}_{ticker}_{side.lower()}_{pnl*100:.1f}pct.png"
    save_figure(fig, out, dpi=160, close=True)
    return out


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-tier", type=int, default=3)
    parser.add_argument("--require-5m", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)

    outputs: list[Path] = []
    for tier in ("tier1", "tier2"):
        trades = pd.read_parquet(ROOT / tier / f"best_trades_{tier}.parquet")
        trades = trades[(trades["pnl_pct"] > 0) & (trades["entry_price"] >= 2.0)].copy()
        if args.require_5m:
            tickers_with_5m = {
                path.stem.upper()
                for path in RAW_5M.glob("*.parquet")
            }
            trades = trades[trades["ticker"].astype(str).str.upper().isin(tickers_with_5m)]
        trades = trades.sort_values(["pnl_pct", "holding_bars"], ascending=[False, False]).head(args.per_tier)
        for rank, (_, trade) in enumerate(trades.iterrows(), 1):
            out = _plot_trade(trade, tier, rank)
            if out is not None:
                outputs.append(out)

    print("created", len(outputs), "plots")
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
