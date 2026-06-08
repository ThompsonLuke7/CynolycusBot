from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared_plotting import DEFAULT_THEME, apply_mpl_defaults, save_figure, style_figure


DEFAULT_SWEEP_DIR = Path("multi_ticker_swing/backtest/results/sweep_v4_shared_20260606")
DEFAULT_RAW_30M_DIR = Path("multi_ticker_swing/data/raw/30m")
DEFAULT_OUT_DIR = Path("UI/swing_audit/backtest_overview_20260606")
DEFAULT_OUT = DEFAULT_OUT_DIR / "multiticker_swing_30m_backtest_overview.png"
DEFAULT_METRICS_OUT = DEFAULT_OUT_DIR / "multiticker_swing_30m_backtest_overview_metrics.csv"

TIER_COLORS = {
    "tier1": DEFAULT_THEME.long,
    "tier2": DEFAULT_THEME.amber,
    "tier3": DEFAULT_THEME.purple,
    "combined": DEFAULT_THEME.text,
}


def _load_raw_timestamps(ticker: str, raw_dir: Path) -> pd.Series | None:
    path = raw_dir / f"{ticker.upper()}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns and df.index.name and str(df.index.name).lower() == "timestamp":
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns:
        return None
    return pd.to_datetime(df["timestamp"], utc=True, errors="coerce").reset_index(drop=True)


def _attach_times(trades: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    out = trades.copy()
    out["signal_time"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["exit_time"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["has_time"] = False
    cache: dict[str, pd.Series | None] = {}

    for ticker, idx in out.groupby("ticker", sort=False).groups.items():
        ticker_key = str(ticker).upper()
        if ticker_key not in cache:
            cache[ticker_key] = _load_raw_timestamps(ticker_key, raw_dir)
        stamps = cache[ticker_key]
        if stamps is None or stamps.empty:
            continue

        loc = list(idx)
        group = out.loc[loc]
        signal_idx = pd.to_numeric(group["signal_idx"], errors="coerce").fillna(-1).astype(int).clip(0, len(stamps) - 1)
        exit_idx = pd.to_numeric(group["exit_idx"], errors="coerce").fillna(-1).astype(int).clip(0, len(stamps) - 1)
        out.loc[loc, "signal_time"] = stamps.iloc[signal_idx.to_numpy()].to_list()
        out.loc[loc, "exit_time"] = stamps.iloc[exit_idx.to_numpy()].to_list()
        out.loc[loc, "has_time"] = True

    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    return out.dropna(subset=["exit_time"])


def _load_best_trades(sweep_dir: Path, raw_dir: Path, tiers: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tier in tiers:
        path = sweep_dir / tier / f"best_trades_{tier}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["tier"] = tier
        frames.append(_attach_times(df, raw_dir))
    if not frames:
        raise FileNotFoundError(f"No best_trades parquet files found under {sweep_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["pnl_pct"] = pd.to_numeric(out["pnl_pct"], errors="coerce")
    out["holding_bars"] = pd.to_numeric(out["holding_bars"], errors="coerce")
    out = out.dropna(subset=["pnl_pct", "exit_time"])
    return out.sort_values("exit_time").reset_index(drop=True)


def _equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.sort_values("exit_time").copy()
    df["cum_pnl_pp"] = df["pnl_pct"].cumsum() * 100.0
    df["peak_pp"] = df["cum_pnl_pp"].cummax()
    df["drawdown_pp"] = df["cum_pnl_pp"] - df["peak_pp"]
    return df


def _metrics(trades: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    pnl = pd.to_numeric(trades["pnl_pct"], errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    curve = pnl.cumsum() * 100.0
    dd = curve - curve.cummax()
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else math.inf
    sharpe = pnl.mean() / pnl.std() * math.sqrt(252) if len(pnl) > 1 and pnl.std() > 0 else 0.0
    longs = trades[trades["direction"] == 1]
    shorts = trades[trades["direction"] == -1]
    return {
        "tier": label,
        "trades": int(len(pnl)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(pf),
        "sharpe": float(sharpe),
        "avg_trade_pp": float(pnl.mean() * 100.0),
        "total_pnl_pp": float(pnl.sum() * 100.0),
        "max_dd_pp": float(dd.min()) if len(dd) else 0.0,
        "avg_hold_30m_bars": float(pd.to_numeric(trades["holding_bars"], errors="coerce").mean()),
        "long_wr": float((longs["pnl_pct"] > 0).mean()) if len(longs) else 0.0,
        "short_wr": float((shorts["pnl_pct"] > 0).mean()) if len(shorts) else 0.0,
    }


def _local_naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.tz_localize(None)


def _format_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100.0:.{digits}f}%"


def _plot_metrics_panel(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    theme = DEFAULT_THEME
    ax.axis("off")
    cols = [
        ("tier", "Tier"),
        ("trades", "Trades"),
        ("win_rate", "Win"),
        ("profit_factor", "PF"),
        ("sharpe", "Sharpe"),
        ("avg_trade_pp", "Avg pp"),
        ("total_pnl_pp", "Total pp"),
        ("max_dd_pp", "Max DD"),
    ]
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            [
                str(row["tier"]),
                f"{int(row['trades']):,}",
                _format_pct(float(row["win_rate"])),
                f"{float(row['profit_factor']):.2f}" if math.isfinite(float(row["profit_factor"])) else "inf",
                f"{float(row['sharpe']):.2f}",
                f"{float(row['avg_trade_pp']):+.2f}",
                f"{float(row['total_pnl_pp']):+,.0f}",
                f"{float(row['max_dd_pp']):+,.0f}",
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=[label for _, label in cols],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(theme.spine)
        cell.set_linewidth(0.5)
        cell.set_facecolor(theme.axes_bg if row_idx else "#1f2937")
        cell.get_text().set_color(theme.text if row_idx else "#f9fafb")
    ax.set_title("Best Config Metrics", loc="left", pad=10, fontsize=12, weight="bold")


def plot_overview(trades: pd.DataFrame, metrics: pd.DataFrame, save_path: Path) -> Path:
    theme = DEFAULT_THEME
    apply_mpl_defaults(theme, font_size=10)
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.25, 1.05, 1.25], width_ratios=[1.35, 1.0])
    ax_equity = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_equity)
    ax_quarter = fig.add_subplot(gs[2, 0])
    ax_table = fig.add_subplot(gs[2, 1])
    style_figure(fig, [ax_equity, ax_dd, ax_quarter, ax_table], theme)

    all_curves: dict[str, pd.DataFrame] = {}
    for tier in ["tier1", "tier2", "tier3"]:
        tier_trades = trades[trades["tier"] == tier]
        if tier_trades.empty:
            continue
        all_curves[tier] = _equity_curve(tier_trades)
    all_curves["combined"] = _equity_curve(trades)

    for label, curve in all_curves.items():
        x = _local_naive(curve["exit_time"])
        lw = 2.2 if label == "combined" else 1.45
        alpha = 0.96 if label == "combined" else 0.78
        ax_equity.plot(x, curve["cum_pnl_pp"], color=TIER_COLORS[label], lw=lw, alpha=alpha, label=label)
        ax_dd.plot(x, curve["drawdown_pp"], color=TIER_COLORS[label], lw=lw, alpha=alpha, label=label)

    ax_equity.set_title(
        "Multi-Ticker Swing 30m Model Backtest | Shared-Universe Sweep v4 Best Configs",
        loc="left",
        fontsize=15,
        weight="bold",
    )
    ax_equity.set_ylabel("Cumulative PnL\n(sum of trade return pp)")
    ax_equity.legend(loc="upper left", ncol=4, frameon=True)
    ax_equity.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")

    ax_dd.axhline(0, color=theme.neutral, lw=0.8, alpha=0.5)
    ax_dd.set_ylabel("Drawdown pp")
    ax_dd.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_equity.get_xticklabels(), visible=False)

    quarter = trades.copy()
    quarter["quarter"] = _local_naive(quarter["exit_time"]).dt.to_period("Q").astype(str)
    q_pnl = (
        quarter.groupby(["quarter", "tier"])["pnl_pct"]
        .sum()
        .mul(100.0)
        .unstack(fill_value=0.0)
        .sort_index()
    )
    xq = np.arange(len(q_pnl))
    bottom = np.zeros(len(q_pnl))
    for tier in ["tier1", "tier2", "tier3"]:
        if tier not in q_pnl.columns:
            continue
        vals = q_pnl[tier].to_numpy(dtype=float)
        ax_quarter.bar(xq, vals, bottom=bottom, width=0.74, color=TIER_COLORS[tier], alpha=0.82, label=tier)
        bottom += vals
    ax_quarter.axhline(0, color=theme.neutral, lw=0.8, alpha=0.6)
    step = max(1, int(math.ceil(len(q_pnl) / 12)))
    ax_quarter.set_xticks(xq[::step])
    ax_quarter.set_xticklabels(q_pnl.index[::step], rotation=35, ha="right")
    ax_quarter.set_title("Quarterly PnL Contribution", loc="left", fontsize=12, weight="bold")
    ax_quarter.set_ylabel("PnL pp")
    ax_quarter.legend(loc="upper left", ncol=3, fontsize=8)

    _plot_metrics_panel(ax_table, metrics)
    save_figure(fig, save_path, dpi=170, tight=False, close=True)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a readable overview plot for the 30m multi-ticker swing backtest.")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--raw-30m-dir", type=Path, default=DEFAULT_RAW_30M_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--tiers", nargs="+", default=["tier1", "tier2", "tier3"])
    args = parser.parse_args()

    trades = _load_best_trades(args.sweep_dir, args.raw_30m_dir, list(args.tiers))
    metrics_rows = [_metrics(trades[trades["tier"] == tier], tier) for tier in args.tiers if not trades[trades["tier"] == tier].empty]
    metrics_rows.append(_metrics(trades, "combined"))
    metrics = pd.DataFrame(metrics_rows)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_out, index=False)
    out = plot_overview(trades, metrics, args.out)
    print(out)
    print(args.metrics_out)


if __name__ == "__main__":
    main()
