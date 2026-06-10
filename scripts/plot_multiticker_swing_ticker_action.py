from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from strategies.multi_ticker_swing.backtest.sweep_v4 import (
    CONFIRM_MAX_5M,
    TRADE_COLS,
    TickerData,
    load_raw_30m,
    load_raw_5m,
    simulate_ticker_5m,
)
from strategies.multi_ticker_swing.config.pipeline_config import RAW_30M_DIR
from core.shared_plotting import (
    DEFAULT_THEME,
    apply_mpl_defaults,
    apply_time_ticks,
    compute_time_ticks,
    plot_candles_from_frame,
    save_figure,
    style_figure,
    time_to_position,
)


ET = ZoneInfo("America/New_York")
DEFAULT_PROBA = Path("strategies/multi_ticker_swing/models/p_swing_probs.parquet")
DEFAULT_UNIVERSE = Path("strategies/multi_ticker_swing/config/trading_universe.json")
DEFAULT_OUT_DIR = Path("UI/swing_audit/ticker_action_20260607")


def _load_probs(ticker: str, proba_path: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(proba_path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[df["ticker"].eq(ticker.upper())].copy()
    if split != "all":
        df = df[df["split"].eq(split)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    p_dir = (df["p_long"] + df["p_short"]).clip(lower=1e-8)
    df["p_long_dir"] = df["p_long"] / p_dir
    df["p_short_dir"] = df["p_short"] / p_dir
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def _load_config(ticker: str, universe_path: Path) -> dict:
    universe = json.loads(universe_path.read_text())
    cfg = universe.get(ticker.upper())
    if not cfg:
        raise SystemExit(f"{ticker.upper()} is not in {universe_path}")
    return cfg


def _exit_cfg(cfg: dict) -> dict:
    return {
        "name": str(cfg.get("combo", "")),
        "arm_pct": 0.025,
        "giveback_pct": 0.25,
        "sl_atr": float(cfg.get("sl_atr", 4.0) or 0.0),
        "np_n_bars": None if cfg.get("np_n_bars") is None else int(cfg.get("np_n_bars")),
        "np_mfe_atr": None if cfg.get("np_mfe_atr") is None else float(cfg.get("np_mfe_atr")),
    }


def _infer_trade_times(trades: pd.DataFrame, bars30: pd.DataFrame, bars5: pd.DataFrame | None) -> pd.DataFrame:
    out = trades.copy()
    stamps30 = pd.to_datetime(bars30["timestamp"], utc=True).reset_index(drop=True)
    out["signal_time"] = stamps30.iloc[out["signal_idx"].clip(0, len(stamps30) - 1).to_numpy()].to_list()
    out["exit_time"] = stamps30.iloc[out["exit_idx"].clip(0, len(stamps30) - 1).to_numpy()].to_list()
    out["entry_time"] = out["signal_time"]

    if bars5 is None or bars5.empty:
        return out

    bars5 = bars5.copy()
    bars5["timestamp"] = pd.to_datetime(bars5["timestamp"], utc=True)
    for idx, trade in out.iterrows():
        after_signal = bars5.index[bars5["timestamp"] > trade["signal_time"]]
        candidates = [
            int(i)
            for i in after_signal[:8]
            if abs(float(bars5.iloc[i]["open"]) - float(trade["entry_price"]))
            <= max(0.01, float(trade["entry_price"]) * 0.002)
        ]
        if not candidates:
            continue
        entry_idx = candidates[0]
        exit_idx = min(entry_idx + int(trade["holding_bars"]), len(bars5) - 1)
        out.at[idx, "entry_time"] = bars5.iloc[entry_idx]["timestamp"]
        out.at[idx, "exit_time"] = bars5.iloc[exit_idx]["timestamp"]
    return out


def _simulate_ticker(ticker: str, cfg: dict, probs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    bars30 = load_raw_30m(ticker)
    bars5 = load_raw_5m(ticker)
    td = TickerData(ticker, bars30, bars5, probs)
    trades = simulate_ticker_5m(td, float(cfg.get("entry_threshold", 0.6)), CONFIRM_MAX_5M, _exit_cfg(cfg))
    trades_df = pd.DataFrame(trades, columns=TRADE_COLS)
    if trades_df.empty:
        trades_df["entry_time"] = []
        trades_df["signal_time"] = []
        trades_df["exit_time"] = []
        return trades_df, bars30, bars5
    return _infer_trade_times(trades_df, bars30, bars5), bars30, bars5


def _window_ts(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(start, tz=ET).tz_convert("UTC"), pd.Timestamp(end, tz=ET).tz_convert("UTC")


def _normalized_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    path = RAW_30M_DIR / f"{ticker.upper()}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns and df.index.name == "timestamp":
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    view = df[df["timestamp"].between(start, end)].dropna(subset=["timestamp", "close"]).copy()
    if view.empty:
        return None
    first = float(view["close"].iloc[0])
    view["norm"] = (view["close"].astype(float) / first - 1.0) * 100.0
    return view[["timestamp", "norm"]]


def plot_ticker_action(
    *,
    ticker: str,
    cfg: dict,
    bars30: pd.DataFrame,
    probs: pd.DataFrame,
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    save_path: Path,
    benchmarks: list[str],
) -> Path:
    ticker = ticker.upper()
    theme = DEFAULT_THEME
    apply_mpl_defaults(theme, font_size=10)
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.05, 1.0])
    ax_price = fig.add_subplot(gs[0])
    ax_prob = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rel = fig.add_subplot(gs[2], sharex=ax_price)
    style_figure(fig, [ax_price, ax_prob, ax_rel], theme)

    bars = bars30.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    view = bars[bars["timestamp"].between(start, end)].dropna(subset=["timestamp"]).reset_index(drop=True)
    if view.empty:
        raise SystemExit(f"No {ticker} 30m bars in requested window")

    candle = plot_candles_from_frame(
        ax_price,
        view.set_index("timestamp", drop=False),
        compressed=True,
        theme=theme,
        width=0.65,
    )
    index = pd.DatetimeIndex(view["timestamp"])
    tick_pos, tick_labels = compute_time_ticks(index, candle.x, max_ticks=12, fmt="%m-%d")
    for ax in (ax_price, ax_prob, ax_rel):
        apply_time_ticks(ax, tick_pos, tick_labels, color=theme.muted_text, fontsize=9)

    wtrades = trades[
        pd.to_datetime(trades["entry_time"], utc=True, errors="coerce").between(start, end)
        | pd.to_datetime(trades["exit_time"], utc=True, errors="coerce").between(start, end)
    ].copy()
    wtrades["entry_time"] = pd.to_datetime(wtrades["entry_time"], utc=True, errors="coerce")
    wtrades["exit_time"] = pd.to_datetime(wtrades["exit_time"], utc=True, errors="coerce")
    wtrades = wtrades.dropna(subset=["entry_time", "exit_time"]).sort_values("entry_time")

    for trade_num, (_, trade) in enumerate(wtrades.iterrows(), 1):
        entry_x = float(time_to_position(index, pd.Series([trade["entry_time"]])).iloc[0])
        exit_x = float(time_to_position(index, pd.Series([trade["exit_time"]])).iloc[0])
        direction = int(trade["direction"])
        pnl = float(trade["pnl_pct"])
        color = theme.win if pnl >= 0 else theme.loss
        side = "L" if direction == 1 else "S"
        ax_price.axvspan(entry_x, exit_x, color=color, alpha=0.08, zorder=0.2)
        ax_price.plot(
            [entry_x, exit_x],
            [float(trade["entry_price"]), float(trade["exit_price"])],
            color=color,
            lw=1.2,
            ls="--",
            alpha=0.9,
            zorder=4,
        )
        ax_price.scatter(
            [entry_x],
            [float(trade["entry_price"])],
            marker="^" if direction == 1 else "v",
            s=72,
            color=theme.blue if direction == 1 else theme.loss,
            edgecolor="#f8fafc",
            linewidth=0.6,
            zorder=5,
        )
        ax_price.scatter(
            [exit_x],
            [float(trade["exit_price"])],
            marker="X",
            s=68,
            color=color,
            edgecolor="#f8fafc",
            linewidth=0.6,
            zorder=5,
        )
        ax_price.text(
            exit_x,
            float(trade["exit_price"]),
            f"{trade_num}{side} {pnl * 100:+.1f}% {trade['exit_reason']}",
            color=theme.text,
            fontsize=7.5,
            ha="left",
            va="bottom" if pnl >= 0 else "top",
            zorder=6,
        )

    pview = probs[probs["timestamp"].between(start, end)].copy()
    if not pview.empty:
        px = time_to_position(index, pview["timestamp"])
        edge = (pview["p_long_dir"] - pview["p_short_dir"]).astype(float)
        smooth_edge = edge.rolling(6, min_periods=1).mean()
        threshold = float(cfg.get("entry_threshold", 0.6))
        edge_threshold = threshold - (1.0 - threshold)
        ax_prob.plot(px, edge, color=theme.neutral, lw=0.65, alpha=0.22, label="raw edge")
        ax_prob.plot(px, smooth_edge, color=theme.blue, lw=1.8, label="6-bar edge")
        ax_prob.fill_between(px, 0, smooth_edge, where=smooth_edge >= 0, color=theme.long, alpha=0.14)
        ax_prob.fill_between(px, 0, smooth_edge, where=smooth_edge < 0, color=theme.short, alpha=0.14)
        ax_prob.axhline(edge_threshold, color=theme.long, lw=0.85, ls="--", alpha=0.55)
        ax_prob.axhline(-edge_threshold, color=theme.short, lw=0.85, ls="--", alpha=0.55)
    ax_prob.axhline(0, color=theme.neutral, lw=0.8, alpha=0.65)
    ax_prob.set_ylim(-1, 1)
    ax_prob.set_ylabel("Long-short edge")
    ax_prob.legend(loc="upper left", ncol=3)

    rel_symbols = [ticker] + [b.upper() for b in benchmarks if b.upper() != ticker]
    rel_colors = [theme.text, theme.blue, "#a3e635", "#fbbf24", "#f472b6"]
    for symbol, color in zip(rel_symbols, rel_colors, strict=False):
        rel = _normalized_close(symbol, start, end)
        if rel is None:
            continue
        rx = time_to_position(index, rel["timestamp"])
        ax_rel.plot(rx, rel["norm"], lw=1.6 if symbol == ticker else 1.2, color=color, label=symbol)
    ax_rel.axhline(0, color=theme.neutral, lw=0.8, alpha=0.6)
    ax_rel.set_ylabel("Window return %")
    ax_rel.legend(loc="upper left", ncol=min(4, len(rel_symbols)))

    pnl_sum = float(wtrades["pnl_pct"].sum() * 100.0) if not wtrades.empty else 0.0
    win_rate = float((wtrades["pnl_pct"] > 0).mean() * 100.0) if not wtrades.empty else 0.0
    long_count = int((wtrades["direction"] == 1).sum()) if not wtrades.empty else 0
    short_count = int((wtrades["direction"] == -1).sum()) if not wtrades.empty else 0
    start_label = start.tz_convert(ET).strftime("%Y-%m-%d")
    end_label = (end.tz_convert(ET) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ax_price.set_title(
        (
            f"{ticker} 30m Model Action | {start_label} to {end_label} | "
            f"tier {cfg.get('tier')} {cfg.get('combo')} | "
            f"{len(wtrades)} trades, {long_count} long/{short_count} short, {pnl_sum:+.1f} pp, {win_rate:.0f}% win"
        ),
        loc="left",
        fontsize=14,
        weight="bold",
    )
    ax_price.set_ylabel("Price")
    ax_rel.set_xlabel("Date")
    save_figure(fig, save_path, dpi=170, tight=False, close=True)
    return save_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot one ticker's 30m swing-model trades, probabilities, and rotation context.")
    parser.add_argument("--ticker", default="APLD")
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-06-06", help="Exclusive ET end date.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--proba", type=Path, default=DEFAULT_PROBA)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--benchmarks", nargs="*", default=["QQQ", "XLV"])
    args = parser.parse_args()

    ticker = args.ticker.upper()
    cfg = _load_config(ticker, args.universe)
    probs = _load_probs(ticker, args.proba, args.split)
    trades, bars30, _bars5 = _simulate_ticker(ticker, cfg, probs)
    start, end = _window_ts(args.start, args.end)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker.lower()}_30m_action_{start:%Y%m%d}_{(end - pd.Timedelta(days=1)):%Y%m%d}"
    trades_out = args.out_dir / f"{stem}_trades.csv"
    plot_out = args.out_dir / f"{stem}.png"
    window_trades = trades.copy()
    window_trades["entry_time"] = pd.to_datetime(window_trades["entry_time"], utc=True, errors="coerce")
    window_trades["exit_time"] = pd.to_datetime(window_trades["exit_time"], utc=True, errors="coerce")
    window_trades = window_trades[
        window_trades["entry_time"].between(start, end) | window_trades["exit_time"].between(start, end)
    ].copy()
    window_trades.to_csv(trades_out, index=False)
    out = plot_ticker_action(
        ticker=ticker,
        cfg=cfg,
        bars30=bars30,
        probs=probs,
        trades=trades,
        start=start,
        end=end,
        save_path=plot_out,
        benchmarks=list(args.benchmarks),
    )
    print(out)
    print(trades_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
