from __future__ import annotations

import argparse
import json
import math
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

from multi_ticker_swing.backtest.sweep_v4 import (
    CONFIRM_MAX_5M,
    TRADE_COLS,
    TickerData,
    load_raw_30m,
    load_raw_5m,
    simulate_ticker_5m,
)
from multi_ticker_swing.config.pipeline_config import RAW_30M_DIR, RAW_5M_DIR, TRADING_BLACKLIST
from shared_plotting import DEFAULT_THEME, apply_mpl_defaults, save_figure, style_figure


ET = ZoneInfo("America/New_York")
DEFAULT_PROBA = Path("multi_ticker_swing/models/p_swing_probs.parquet")
DEFAULT_UNIVERSE = Path("multi_ticker_swing/config/trading_universe.json")
DEFAULT_OUT_DIR = Path("UI/swing_audit/backtest_may_20260607")


def _read_universe(path: Path, tiers: set[int]) -> dict[str, dict]:
    data = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for ticker, cfg in data.items():
        ticker = str(ticker).upper()
        if ticker in TRADING_BLACKLIST:
            continue
        try:
            tier = int(cfg.get("tier", 0))
        except Exception:
            continue
        if tier in tiers:
            out[ticker] = cfg
    return out


def _load_proba(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if split != "all":
        df = df[df["split"].eq(split)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    p_dir = (df["p_long"] + df["p_short"]).clip(lower=1e-8)
    df["p_long_dir"] = df["p_long"] / p_dir
    df["p_short_dir"] = df["p_short"] / p_dir
    return df.dropna(subset=["timestamp"])


def _stamps(raw_30m: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(raw_30m["timestamp"], utc=True, errors="coerce").reset_index(drop=True)


def _ticker_exit_cfg(cfg: dict) -> dict:
    return {
        "name": str(cfg.get("combo", "")),
        "arm_pct": 0.025,
        "giveback_pct": 0.25,
        "sl_atr": float(cfg.get("sl_atr", 4.0) or 0.0),
        "np_n_bars": None if cfg.get("np_n_bars") is None else int(cfg.get("np_n_bars")),
        "np_mfe_atr": None if cfg.get("np_mfe_atr") is None else float(cfg.get("np_mfe_atr")),
    }


def _simulate_available(
    *,
    universe: dict[str, dict],
    proba: pd.DataFrame,
    min_fresh_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[pd.DataFrame] = []
    coverage = {
        "configured": len(universe),
        "missing_30m": 0,
        "missing_5m": 0,
        "stale_30m": 0,
        "stale_5m": 0,
        "missing_proba": 0,
        "stale_proba": 0,
        "simulated": 0,
    }
    proba_groups = dict(tuple(proba.groupby("ticker", sort=False)))

    for ticker, cfg in universe.items():
        raw_30m_path = RAW_30M_DIR / f"{ticker}.parquet"
        raw_5m_path = RAW_5M_DIR / f"{ticker}.parquet"
        if not raw_30m_path.exists():
            coverage["missing_30m"] += 1
            continue
        if not raw_5m_path.exists():
            coverage["missing_5m"] += 1
            continue

        raw_30m = load_raw_30m(ticker)
        raw_5m = load_raw_5m(ticker)
        stamps = _stamps(raw_30m)
        if stamps.empty or stamps.max() < min_fresh_ts:
            coverage["stale_30m"] += 1
            continue
        stamps_5m = pd.to_datetime(raw_5m["timestamp"], utc=True, errors="coerce")
        if stamps_5m.empty or stamps_5m.max() < min_fresh_ts:
            coverage["stale_5m"] += 1
            continue
        pt = proba_groups.get(ticker)
        if pt is None or pt.empty:
            coverage["missing_proba"] += 1
            continue
        if pd.to_datetime(pt["timestamp"], utc=True, errors="coerce").max() < min_fresh_ts:
            coverage["stale_proba"] += 1
            continue

        td = TickerData(ticker, raw_30m, raw_5m, pt)
        trades = simulate_ticker_5m(
            td,
            float(cfg.get("entry_threshold", 0.6)),
            CONFIRM_MAX_5M,
            _ticker_exit_cfg(cfg),
        )
        if not trades:
            continue

        tdf = pd.DataFrame(trades, columns=TRADE_COLS)
        signal_idx = pd.to_numeric(tdf["signal_idx"], errors="coerce").fillna(-1).astype(int).clip(0, len(stamps) - 1)
        exit_idx = pd.to_numeric(tdf["exit_idx"], errors="coerce").fillna(-1).astype(int).clip(0, len(stamps) - 1)
        tdf["signal_time"] = stamps.iloc[signal_idx.to_numpy()].to_list()
        tdf["exit_time"] = stamps.iloc[exit_idx.to_numpy()].to_list()
        tdf["tier"] = int(cfg.get("tier"))
        tdf["combo"] = str(cfg.get("combo", ""))
        rows.append(tdf)
        coverage["simulated"] += 1

    if not rows:
        return pd.DataFrame(columns=TRADE_COLS + ["signal_time", "exit_time", "tier", "combo"]), coverage
    return pd.concat(rows, ignore_index=True), coverage


def _filter_window(trades: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=ET).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz=ET).tz_convert("UTC")
    out = trades.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["pnl_pct"] = pd.to_numeric(out["pnl_pct"], errors="coerce")
    return out[out["signal_time"].ge(start_ts) & out["signal_time"].lt(end_ts)].dropna(subset=["pnl_pct", "exit_time"])


def _metrics(trades: pd.DataFrame, label: str) -> dict:
    pnl = pd.to_numeric(trades["pnl_pct"], errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else math.inf
    sharpe = pnl.mean() / pnl.std() * math.sqrt(252) if len(pnl) > 1 and pnl.std() > 0 else 0.0
    curve = pnl.cumsum() * 100.0
    dd = curve - curve.cummax()
    return {
        "bucket": label,
        "trades": int(len(pnl)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "profit_factor": float(pf),
        "sharpe": float(sharpe),
        "avg_trade_pp": float(pnl.mean() * 100.0) if len(pnl) else 0.0,
        "total_pnl_pp": float(pnl.sum() * 100.0) if len(pnl) else 0.0,
        "max_dd_pp": float(dd.min()) if len(dd) else 0.0,
        "avg_hold_5m_bars": float(pd.to_numeric(trades["holding_bars"], errors="coerce").mean()) if len(trades) else 0.0,
    }


def _summary(trades: pd.DataFrame, coverage: dict[str, int]) -> pd.DataFrame:
    frames = [
        ("combined", trades),
        ("calls_only", trades[trades["direction"].eq(1)]),
        ("puts_only", trades[trades["direction"].eq(-1)]),
    ]
    for tier in sorted(trades["tier"].dropna().astype(int).unique()):
        tdf = trades[trades["tier"].eq(tier)]
        frames.extend(
            [
                (f"tier{tier}_combined", tdf),
                (f"tier{tier}_calls", tdf[tdf["direction"].eq(1)]),
                (f"tier{tier}_puts", tdf[tdf["direction"].eq(-1)]),
            ]
        )
    summary = pd.DataFrame([_metrics(df, label) for label, df in frames])
    for key, value in coverage.items():
        summary[key] = value
    return summary


def _ticker_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ticker, tier), group in trades.groupby(["ticker", "tier"], sort=False):
        rows.append(
            {
                "ticker": ticker,
                "tier": int(tier),
                "trades": int(len(group)),
                "total_pnl_pp": float(group["pnl_pct"].sum() * 100.0),
                "avg_trade_pp": float(group["pnl_pct"].mean() * 100.0),
                "win_rate": float((group["pnl_pct"] > 0).mean()),
                "calls": int((group["direction"] == 1).sum()),
                "puts": int((group["direction"] == -1).sum()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ticker", "tier", "trades", "total_pnl_pp", "avg_trade_pp", "win_rate", "calls", "puts"])
    return pd.DataFrame(rows).sort_values(["total_pnl_pp", "win_rate"], ascending=[False, False])


def _plot_report(trades: pd.DataFrame, summary: pd.DataFrame, save_path: Path, title: str) -> Path:
    theme = DEFAULT_THEME
    apply_mpl_defaults(theme, font_size=10)
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0])
    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])
    style_figure(fig, [ax, ax_tbl], theme)

    colors = {
        "combined": theme.text,
        "calls_only": theme.long,
        "puts_only": theme.loss,
    }
    if trades.empty:
        ax.text(
            0.5,
            0.5,
            "No trades found in this replay window.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=theme.muted_text,
            fontsize=13,
        )
    for label, df in (
        ("combined", trades),
        ("calls_only", trades[trades["direction"].eq(1)]),
        ("puts_only", trades[trades["direction"].eq(-1)]),
    ):
        if df.empty:
            continue
        curve = df.sort_values("exit_time").copy()
        curve["cum_pp"] = curve["pnl_pct"].cumsum() * 100.0
        x = curve["exit_time"].dt.tz_convert(ET).dt.tz_localize(None)
        ax.plot(x, curve["cum_pp"], lw=2.0 if label == "combined" else 1.55, color=colors[label], label=label)

    ax.axhline(0, color=theme.neutral, lw=0.8, alpha=0.55)
    ax.set_title(title, loc="left", fontsize=14, weight="bold")
    ax.set_ylabel("Cumulative underlying return pp")
    ax.legend(loc="upper left", ncol=3)

    ax_tbl.axis("off")
    rows = []
    table_df = summary[summary["bucket"].isin(["combined", "calls_only", "puts_only", "tier1_combined", "tier2_combined"])].copy()
    for _, row in table_df.iterrows():
        pf = float(row["profit_factor"])
        rows.append(
            [
                str(row["bucket"]),
                f"{int(row['trades']):,}",
                f"{float(row['win_rate']) * 100:.1f}%",
                f"{pf:.2f}" if math.isfinite(pf) else "inf",
                f"{float(row['sharpe']):.2f}",
                f"{float(row['avg_trade_pp']):+.2f}",
                f"{float(row['total_pnl_pp']):+,.1f}",
                f"{float(row['max_dd_pp']):+,.1f}",
            ]
        )
    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Bucket", "Trades", "Win", "PF", "Sharpe", "Avg pp", "Total pp", "Max DD"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor(theme.spine)
        cell.set_linewidth(0.5)
        cell.set_facecolor("#1f2937" if r == 0 else theme.axes_bg)
        cell.get_text().set_color("#f9fafb" if r == 0 else theme.text)

    save_figure(fig, save_path, dpi=170, tight=False, close=True)
    return save_path


def _window_metadata(start: str, end: str) -> tuple[str, str]:
    start_ts = pd.Timestamp(start, tz=ET)
    end_excl = pd.Timestamp(end, tz=ET)
    last_ts = end_excl - pd.Timedelta(days=1)
    stem = f"multiticker_swing_{start_ts:%Y%m%d}_{last_ts:%Y%m%d}_replay"
    if start_ts.year == last_ts.year and start_ts.month == last_ts.month:
        month = start_ts.strftime("%B")
        if start_ts.day == 1:
            display = f"{month} {start_ts.year}" if last_ts.day >= 28 else f"{month} {start_ts.day}-{last_ts.day}, {start_ts.year}"
        else:
            display = f"{month} {start_ts.day}-{last_ts.day}, {start_ts.year}"
    else:
        display = f"{start_ts:%Y-%m-%d} to {last_ts:%Y-%m-%d}"
    return stem, display


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay current multi-ticker swing configs over a recent date window.")
    parser.add_argument("--proba", type=Path, default=DEFAULT_PROBA)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", default="2026-05-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()

    universe = _read_universe(args.universe, set(args.tiers))
    proba = _load_proba(args.proba, args.split)
    min_fresh_ts = pd.Timestamp(args.start, tz=ET).tz_convert("UTC")
    all_trades, coverage = _simulate_available(universe=universe, proba=proba, min_fresh_ts=min_fresh_ts)
    window = _filter_window(all_trades, args.start, args.end)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem, display_window = _window_metadata(args.start, args.end)
    trades_out = args.out_dir / f"{stem}_trades.csv"
    summary_out = args.out_dir / f"{stem}_summary.csv"
    ticker_out = args.out_dir / f"{stem}_tickers.csv"
    plot_out = args.out_dir / f"{stem}_direction_report.png"

    window.sort_values("exit_time").to_csv(trades_out, index=False)
    summary = _summary(window, coverage)
    summary.to_csv(summary_out, index=False)
    tickers = _ticker_summary(window)
    tickers.to_csv(ticker_out, index=False)
    _plot_report(
        window,
        summary,
        plot_out,
        (
            f"Multi-Ticker Swing 30m {display_window} Replay | "
            f"Current Tier 1/2 Configs, {int(summary['simulated'].iloc[0])} Symbols With Window Proba"
        ),
    )

    print(plot_out)
    print(summary_out)
    print(trades_out)
    print(ticker_out)
    print(summary.to_string(index=False))
    if not tickers.empty:
        print("\nTop tickers:")
        print(tickers.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
