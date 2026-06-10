from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DAILY_BARS_PATH, OUTPUT_DIR, PLOTS_DIR, REPORT_DIR, ensure_dirs


DEEP_DIR = OUTPUT_DIR / "deep_theme_rotation"
BACKTEST_DIR = OUTPUT_DIR / "backtests"
PLOT_PATH = PLOTS_DIR / "active_theme_strategy_backtest.png"
SUMMARY_PATH = REPORT_DIR / "active_theme_strategy_backtest_summary.csv"
PERIOD_PATH = REPORT_DIR / "active_theme_strategy_period_summary.csv"

STRATEGY_FILES = {
    "top5 leaders full": DEEP_DIR / "top5_leaders_full_daily.parquet",
    "top5 leaders cash risk-off": DEEP_DIR / "top5_leaders_cash_risk_off_daily.parquet",
    "existing top3 scaled": DEEP_DIR / "existing_like_top3_scaled_daily.parquet",
    "original rule-based": BACKTEST_DIR / "rule_based_theme_rotation_daily.parquet",
}

BENCHMARKS = ("SPY", "QQQ")
TRADING_DAYS = 252.0


def load_strategy(path: Path, name: str, start: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(start)].sort_values("date").copy()
    if "strategy_return" not in df.columns:
        raise ValueError(f"{path} missing strategy_return")
    if "equity" not in df.columns:
        df["equity"] = (1.0 + df["strategy_return"].fillna(0.0)).cumprod()
    df["strategy"] = name
    df["equity_rebased"] = df["equity"] / df["equity"].iloc[0]
    df["drawdown"] = df["equity_rebased"] / df["equity_rebased"].cummax() - 1.0
    return df


def load_benchmark_curves(dates: pd.Series) -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    wide = bars[bars["ticker"].isin(BENCHMARKS)].pivot(index="date", columns="ticker", values="px").sort_index()
    wide = wide.reindex(pd.DatetimeIndex(dates)).ffill()
    returns = wide.pct_change().fillna(0.0)
    curves = (1.0 + returns).cumprod()
    curves = curves / curves.iloc[0]
    curves.index.name = "date"
    return curves


def perf_stats(name: str, returns: pd.Series, equity: pd.Series, turnover: pd.Series | None = None) -> dict[str, object]:
    returns = returns.fillna(0.0)
    equity = equity.ffill().fillna(1.0)
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(returns) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    final_equity = float(equity.iloc[-1])
    cagr = final_equity ** (1.0 / years) - 1.0 if final_equity > 0 else np.nan
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    loss_sum = abs(losers.sum())
    return {
        "strategy": name,
        "total_return": final_equity - 1.0,
        "cagr": cagr,
        "sharpe": np.sqrt(TRADING_DAYS) * returns.mean() / returns.std() if returns.std() else 0.0,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else np.nan,
        "avg_turnover": float(turnover.fillna(0.0).mean()) if turnover is not None else 0.0,
    }


def summarize_periods(strategies: dict[str, pd.DataFrame], benchmark_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in strategies.items():
        frame = df.copy()
        frame["period"] = frame["date"].dt.to_period("Q").astype(str)
        for period, group in frame.groupby("period", sort=True):
            if period < "2025Q1":
                continue
            spy_ret = benchmark_returns.reindex(group["date"])["SPY"].fillna(0.0)
            rows.append(
                {
                    "period": period,
                    "strategy": name,
                    "strategy_return": float((1.0 + group["strategy_return"].fillna(0.0)).prod() - 1.0),
                    "spy_return": float((1.0 + spy_ret).prod() - 1.0),
                    "excess_vs_spy": float((1.0 + group["strategy_return"].fillna(0.0)).prod() - (1.0 + spy_ret).prod()),
                    "max_drawdown": float(((1.0 + group["strategy_return"].fillna(0.0)).cumprod() / (1.0 + group["strategy_return"].fillna(0.0)).cumprod().cummax() - 1.0).min()),
                    "avg_active_themes": float(group.get("n_active_themes", pd.Series(0.0, index=group.index)).fillna(0.0).mean()),
                    "avg_positions": float(group.get("n_positions", group.get("n_active_tickers", pd.Series(0.0, index=group.index))).fillna(0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def top_theme_labels(df: pd.DataFrame, min_days: int = 8) -> pd.DataFrame:
    if "active_themes" not in df.columns:
        return pd.DataFrame(columns=["period", "label"])
    rows = []
    frame = df[["date", "active_themes"]].copy()
    frame["period"] = frame["date"].dt.to_period("Q").astype(str)
    for period, group in frame.groupby("period", sort=True):
        counts: dict[str, int] = {}
        for text in group["active_themes"].dropna():
            for theme in str(text).split("|"):
                theme = theme.strip()
                if theme:
                    counts[theme] = counts.get(theme, 0) + 1
        top = [f"{theme} ({days}d)" for theme, days in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:3] if days >= min_days]
        rows.append({"period": period, "label": ", ".join(top)})
    return pd.DataFrame(rows)


def build_plot(strategies: dict[str, pd.DataFrame], benchmarks: pd.DataFrame, summary: pd.DataFrame) -> None:
    primary = strategies["top5 leaders full"]
    dates = primary["date"]
    theme_notes = top_theme_labels(primary)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.2, 1.35, 1.2, 0.9], hspace=0.16)
    ax = fig.add_subplot(gs[0])
    dd_ax = fig.add_subplot(gs[1], sharex=ax)
    exposure_ax = fig.add_subplot(gs[2], sharex=ax)
    table_ax = fig.add_subplot(gs[3])

    colors = {
        "top5 leaders full": "#111111",
        "top5 leaders cash risk-off": "#1f77b4",
        "existing top3 scaled": "#ff7f0e",
        "original rule-based": "#2ca02c",
        "SPY": "#777777",
        "QQQ": "#9467bd",
    }

    for name, df in strategies.items():
        linewidth = 2.8 if name == "top5 leaders full" else 1.8
        ax.plot(df["date"], df["equity_rebased"], label=name, color=colors.get(name), linewidth=linewidth)
    for ticker in BENCHMARKS:
        if ticker in benchmarks:
            ax.plot(benchmarks.index, benchmarks[ticker], label=ticker, color=colors.get(ticker), linewidth=1.5, alpha=0.8)
    ax.set_title("Active Theme Rotation Backtest: Enter Top 5, Exit Below 12, Hold Top Leaders")
    ax.set_ylabel("Growth of 1.0")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    for name, df in strategies.items():
        linewidth = 2.2 if name == "top5 leaders full" else 1.3
        dd_ax.plot(df["date"], df["drawdown"], label=name, color=colors.get(name), linewidth=linewidth)
    dd_ax.set_ylabel("Drawdown")
    dd_ax.grid(True, alpha=0.22)
    dd_ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    if "n_active_themes" in primary.columns:
        exposure_ax.plot(dates, primary["n_active_themes"], label="active themes", color="#111111", linewidth=1.8)
    position_col = "n_positions" if "n_positions" in primary.columns else "n_active_tickers"
    if position_col in primary.columns:
        exposure_ax.plot(dates, primary[position_col], label="positions", color="#1f77b4", linewidth=1.3)
    exposure_ax.set_ylabel("Count")
    exposure_ax.grid(True, alpha=0.22)
    exposure_ax.legend(loc="upper left", fontsize=9)

    y_min, y_max = ax.get_ylim()
    for idx, note in theme_notes.iterrows():
        if not note["label"]:
            continue
        period = pd.Period(note["period"], freq="Q")
        x = period.end_time.normalize()
        point = primary[primary["date"] <= x]
        if point.empty or note["period"] < "2025Q1":
            continue
        y = float(point["equity_rebased"].iloc[-1])
        ax.annotate(
            f"{note['period']}\n{note['label']}",
            xy=(point["date"].iloc[-1], y),
            xytext=(0, 34 if idx % 2 == 0 else -64),
            textcoords="offset points",
            fontsize=7.5,
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#888888", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.7},
        )
    ax.set_ylim(y_min, y_max)

    table_ax.axis("off")
    display = summary.copy()
    display = display[display["strategy"].isin(["top5 leaders full", "top5 leaders cash risk-off", "existing top3 scaled", "SPY", "QQQ"])]
    display = display.assign(
        total_return=display["total_return"].map(lambda value: f"{value:.0%}"),
        cagr=display["cagr"].map(lambda value: f"{value:.1%}"),
        sharpe=display["sharpe"].map(lambda value: f"{value:.2f}"),
        max_drawdown=display["max_drawdown"].map(lambda value: f"{value:.1%}"),
        avg_turnover=display["avg_turnover"].map(lambda value: f"{value:.2f}"),
    )
    cols = ["strategy", "total_return", "cagr", "sharpe", "max_drawdown", "avg_turnover"]
    table = table_ax.table(cellText=display[cols].values, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.32)

    for axis in (ax, dd_ax, exposure_ax):
        axis.set_xlim(dates.min(), dates.max())
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(dd_ax.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=170)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    start = "2023-01-01"
    missing = [str(path) for path in STRATEGY_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing strategy daily outputs: {missing}")

    strategies = {name: load_strategy(path, name, start) for name, path in STRATEGY_FILES.items()}
    primary_dates = strategies["top5 leaders full"]["date"]
    benchmarks = load_benchmark_curves(primary_dates)
    benchmark_returns = benchmarks.pct_change().fillna(0.0)

    rows = []
    for name, df in strategies.items():
        rows.append(perf_stats(name, df["strategy_return"], df["equity_rebased"], df.get("turnover")))
    for ticker in BENCHMARKS:
        rows.append(perf_stats(ticker, benchmark_returns[ticker], benchmarks[ticker]))
    summary = pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    period = summarize_periods(strategies, benchmark_returns)
    period.to_csv(PERIOD_PATH, index=False)
    build_plot(strategies, benchmarks, summary)

    print(f"saved plot -> {PLOT_PATH}")
    print(f"saved summary -> {SUMMARY_PATH}")
    print(f"saved periods -> {PERIOD_PATH}")
    print(summary[["strategy", "total_return", "cagr", "sharpe", "max_drawdown", "avg_turnover"]].to_string(index=False))


if __name__ == "__main__":
    main()
