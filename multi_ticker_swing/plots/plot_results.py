"""
Evaluation plots for the multi-ticker swing pipeline.

Generates:
  1. Equity curve
  2. Trade outcome histogram (PnL distribution)
  3. Win rate by exit reason / direction
  4. Model probability calibration curve
  5. Feature importance bar chart
  6. Prediction vs actual return scatter
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed — plots will not be generated.")

from multi_ticker_swing.config.pipeline_config import (
    BACKTEST_RESULTS_DIR,
    EVAL_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    PLOTS_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# 1. Equity curve
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity_df: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ts  = pd.to_datetime(equity_df["timestamp"])
    eq  = equity_df["equity"].values
    ret = np.diff(eq, prepend=eq[0]) / np.where(eq > 0, eq, 1)

    # Equity
    axes[0].plot(ts, eq, color="#2196F3", linewidth=1.5, label="Portfolio Equity")
    axes[0].fill_between(ts, eq, eq.min(), alpha=0.08, color="#2196F3")
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title("Portfolio Equity Curve")
    axes[0].legend(loc="upper left")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[0].grid(alpha=0.3)

    # Drawdown
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100
    axes[1].fill_between(ts, dd, 0, color="#F44336", alpha=0.6, label="Drawdown %")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _savefig(fig, output_dir / "equity_curve.png", "equity_curve")


# ---------------------------------------------------------------------------
# 2. Trade outcome histogram
# ---------------------------------------------------------------------------

def plot_trade_histogram(
    trades_df: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    if not HAS_MPL or trades_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pnl_pct = trades_df["pnl_pct"].dropna() * 100

    # All trades histogram
    axes[0].hist(pnl_pct, bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="black", linewidth=1.5, linestyle="--")
    axes[0].axvline(pnl_pct.mean(), color="#FF9800", linewidth=1.5, linestyle="-", label=f"Mean={pnl_pct.mean():.2f}%")
    axes[0].set_xlabel("Return per Trade (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Trade Return Distribution  (n={len(pnl_pct)})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # By direction
    for direction, label, color in [(1, "Long", "#4CAF50"), (-1, "Short", "#F44336")]:
        subset = trades_df.loc[trades_df["direction"] == direction, "pnl_pct"].dropna() * 100
        if not subset.empty:
            axes[1].hist(subset, bins=30, alpha=0.6, label=f"{label} (n={len(subset)})", color=color, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Return per Trade (%)")
    axes[1].set_title("Return Distribution by Direction")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _savefig(fig, output_dir / "trade_histogram.png", "trade_histogram")


# ---------------------------------------------------------------------------
# 3. Win rate breakdown
# ---------------------------------------------------------------------------

def plot_win_rate_breakdown(
    trades_df: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> None:
    if not HAS_MPL or trades_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By exit reason
    wr_by_reason = (
        trades_df.groupby("exit_reason")["pnl_pct"]
        .agg(win_rate=lambda x: (x > 0).mean(), count="count")
        .reset_index()
    )
    colors = ["#4CAF50" if w > 0.5 else "#F44336" for w in wr_by_reason["win_rate"]]
    axes[0].bar(wr_by_reason["exit_reason"], wr_by_reason["win_rate"] * 100, color=colors, edgecolor="white")
    axes[0].axhline(50, color="black", linewidth=1, linestyle="--")
    for _, r in wr_by_reason.iterrows():
        axes[0].text(r.name, r["win_rate"] * 100 + 1, f"n={r['count']}", ha="center", fontsize=9)
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_title("Win Rate by Exit Reason")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.3, axis="y")

    # Monthly win rate
    trades_df = trades_df.copy()
    trades_df["month"] = pd.to_datetime(trades_df["entry_time"]).dt.to_period("M")
    monthly = trades_df.groupby("month")["pnl_pct"].agg(win_rate=lambda x: (x > 0).mean()).reset_index()
    month_labels = monthly["month"].astype(str)
    axes[1].bar(range(len(monthly)), monthly["win_rate"] * 100,
                color=["#4CAF50" if w > 0.5 else "#F44336" for w in monthly["win_rate"]], edgecolor="white")
    axes[1].axhline(50, color="black", linewidth=1, linestyle="--")
    step = max(1, len(monthly) // 8)
    axes[1].set_xticks(range(0, len(monthly), step))
    axes[1].set_xticklabels(month_labels[::step], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_title("Monthly Win Rate")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    _savefig(fig, output_dir / "win_rate_breakdown.png", "win_rate_breakdown")


# ---------------------------------------------------------------------------
# 4. Probability calibration
# ---------------------------------------------------------------------------

def plot_calibration(
    y_true: np.ndarray,
    proba: np.ndarray,
    output_dir: Path = PLOTS_DIR,
    n_bins: int = 10,
) -> None:
    """
    y_true: integer class labels (0/1/2)
    proba:  (N, 3) probability array
    """
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_names = ["Short (0)", "Neutral (1)", "Long (2)"]
    colors = ["#F44336", "#9E9E9E", "#4CAF50"]

    for cls_idx, (ax, cname, color) in enumerate(zip(axes, class_names, colors)):
        p   = proba[:, cls_idx]
        hit = (y_true == cls_idx).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        act_rates = []
        counts    = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() > 0:
                act_rates.append(hit[mask].mean())
                counts.append(mask.sum())
            else:
                act_rates.append(np.nan)
                counts.append(0)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.scatter(bin_centers, act_rates, s=[max(c, 5) for c in counts],
                   color=color, zorder=5, label="Observed rate")
        ax.plot(bin_centers, act_rates, color=color, linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Rate")
        ax.set_title(f"Calibration: {cname}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _savefig(fig, output_dir / "calibration.png", "calibration")


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    fi_path: Path = FEATURE_IMPORTANCE_PATH,
    output_dir: Path = PLOTS_DIR,
    top_n: int = 30,
) -> None:
    if not HAS_MPL or not fi_path.exists():
        return

    fi = pd.read_csv(fi_path).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    bars = ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#2196F3", edgecolor="white")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Features")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    _savefig(fig, output_dir / "feature_importance.png", "feature_importance")


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_plots(
    trades_df: pd.DataFrame | None = None,
    equity_df: pd.DataFrame | None = None,
    y_true: np.ndarray | None = None,
    proba: np.ndarray | None = None,
    output_dir: Path = PLOTS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load from disk if not passed
    if trades_df is None:
        p = BACKTEST_RESULTS_DIR / "trades.parquet"
        if p.exists():
            trades_df = pd.read_parquet(p)
    if equity_df is None:
        p = BACKTEST_RESULTS_DIR / "equity_curve.parquet"
        if p.exists():
            equity_df = pd.read_parquet(p)

    if equity_df is not None and not equity_df.empty:
        plot_equity_curve(equity_df, output_dir)

    if trades_df is not None and not trades_df.empty:
        plot_trade_histogram(trades_df, output_dir)
        plot_win_rate_breakdown(trades_df, output_dir)

    if y_true is not None and proba is not None:
        plot_calibration(y_true, proba, output_dir)

    plot_feature_importance(FEATURE_IMPORTANCE_PATH, output_dir)

    logger.info("All plots saved to %s", output_dir)
