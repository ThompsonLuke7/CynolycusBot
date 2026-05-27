from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import BACKTEST_DIR, OUTPUT_DIR, PLOTS_DIR, THEME_SCORES_PATH, TOP_N_THEMES, ensure_dirs


PNL_CHART_PATH = PLOTS_DIR / "theme_strategy_pnl_annotated.png"
LABELS_PATH = OUTPUT_DIR / "theme_strategy_period_labels.csv"
RANK_BUCKET_PATH = OUTPUT_DIR / "theme_rank_bucket_forward_returns.csv"
SHORT_DIAGNOSTICS_PATH = OUTPUT_DIR / "short_theme_diagnostics.csv"


def load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_script("07_backtest_ranked_themes.py", "baseline_backtest")
experiments = load_script("19_theme_rotation_experiments.py", "theme_experiments")


def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(THEME_SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    if "is_tradable" in scores.columns:
        scores = scores[scores["is_tradable"].fillna(False).astype(bool)].copy()
    return scores.dropna(subset=["theme_regime_rank"]).copy()


def build_period_labels(scores: pd.DataFrame, start: str = "2023-01-01") -> pd.DataFrame:
    data = scores[scores["date"] >= pd.Timestamp(start)].copy()
    data["period"] = data["date"].dt.to_period("Q")
    rows = []
    for period, group in data.groupby("period", sort=True):
        summary = (
            group.groupby("theme")
            .agg(
                avg_rank=("theme_regime_rank", "mean"),
                avg_return_5d=("theme_return_5d", "mean"),
                top5_days=("theme_regime_rank", lambda s: int((s <= 5).sum())),
                days=("theme_regime_rank", "size"),
            )
            .reset_index()
        )
        top = summary.sort_values(["avg_rank", "avg_return_5d"], ascending=[True, False]).iloc[0]
        worst = summary.sort_values(["avg_rank", "avg_return_5d"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "period": str(period),
                "period_end": period.end_time.normalize(),
                "top_theme": top["theme"],
                "top_avg_rank": float(top["avg_rank"]),
                "top_avg_5d_return": float(top["avg_return_5d"]),
                "worst_theme": worst["theme"],
                "worst_avg_rank": float(worst["avg_rank"]),
                "worst_avg_5d_return": float(worst["avg_return_5d"]),
            }
        )
    return pd.DataFrame(rows)


def build_rank_bucket_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    data = scores.sort_values(["theme", "date"]).copy()
    data["fwd_5d_theme_return"] = data.groupby("theme")["theme_return_5d"].shift(-5)
    data["rank_bucket"] = pd.cut(
        data["theme_regime_rank"],
        bins=[0, 3, 5, 10, 25, 50, 75, 200],
        labels=["1-3", "4-5", "6-10", "11-25", "26-50", "51-75", "76+"],
        include_lowest=True,
    )
    return (
        data.groupby("rank_bucket", observed=True)
        .agg(
            observations=("fwd_5d_theme_return", "count"),
            avg_fwd_5d_theme_return=("fwd_5d_theme_return", "mean"),
            median_fwd_5d_theme_return=("fwd_5d_theme_return", "median"),
            pct_negative_fwd_5d=("fwd_5d_theme_return", lambda s: float((s < 0).mean())),
            avg_rank=("theme_regime_rank", "mean"),
        )
        .reset_index()
    )


def build_short_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    returns = baseline.load_stock_returns()
    next_1d = returns.shift(-1)
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    next_5d = curve.shift(-5) / curve - 1.0
    panel = experiments.build_all_member_ranks(scores)
    panel_by_date = {date: frame for date, frame in panel.groupby("date", sort=False)}
    dates = baseline.build_trade_calendar(scores, returns)

    rows = []
    for date in dates:
        if date not in panel_by_date:
            continue
        day_scores = scores[scores["date"].eq(date)]
        if day_scores.empty:
            continue
        worst_themes = day_scores.nlargest(TOP_N_THEMES, "theme_regime_rank")["theme"].tolist()
        best_themes = day_scores.nsmallest(TOP_N_THEMES, "theme_regime_rank")["theme"].tolist()
        day_panel = panel_by_date[date]
        short_pool = day_panel[day_panel["theme"].isin(worst_themes)].copy()
        short_rows = (
            short_pool.sort_values(["theme", "leader_rank"], ascending=[True, False])
            .groupby("theme", group_keys=False)
            .head(3)
        )
        tickers = sorted(short_rows["ticker"].dropna().unique())
        if not tickers:
            continue
        selected_same_day_return = float(returns.loc[date].reindex(tickers).fillna(0.0).mean()) if date in returns.index else np.nan
        next_day_return = float(next_1d.loc[date].reindex(tickers).fillna(0.0).mean()) if date in next_1d.index else np.nan
        future_5d_return = float(next_5d.loc[date].reindex(tickers).fillna(0.0).mean()) if date in next_5d.index else np.nan
        rows.append(
            {
                "date": date,
                "best_themes": "|".join(best_themes),
                "short_themes": "|".join(worst_themes),
                "short_tickers": "|".join(tickers),
                "selected_short_basket_same_day_long_return": selected_same_day_return,
                "selected_short_basket_next_1d_long_return": next_day_return,
                "selected_short_basket_next_5d_long_return": future_5d_return,
                "short_pnl_next_1d_before_cost": -next_day_return,
                "short_pnl_next_5d_before_cost": -future_5d_return,
                "short_basket_next_1d_positive_return": bool(next_day_return > 0),
                "short_basket_next_5d_positive_return": bool(future_5d_return > 0),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["short_pnl_equity_before_cost"] = (1.0 + out["short_pnl_next_1d_before_cost"]).cumprod()
    return out


def plot_pnl(labels: pd.DataFrame) -> None:
    bt = pd.read_parquet(BACKTEST_DIR / "rule_based_theme_rotation_daily.parquet")
    bt["date"] = pd.to_datetime(bt["date"])
    bt = bt[bt["date"] >= pd.Timestamp("2023-01-01")].copy()
    bench = baseline.benchmark_returns(bt["date"])
    curves = pd.DataFrame({"date": bt["date"], "Theme rotation": bt["equity"] / bt["equity"].iloc[0]})
    for ticker in ("SPY", "QQQ"):
        if ticker in bench:
            curves[ticker] = (1.0 + bench[ticker].fillna(0.0)).cumprod().values

    fig, ax = plt.subplots(figsize=(14, 8))
    for column, linewidth in [("Theme rotation", 2.8), ("SPY", 1.4), ("QQQ", 1.4)]:
        if column in curves:
            ax.plot(curves["date"], curves[column], label=column, linewidth=linewidth)

    y_min, y_max = ax.get_ylim()
    label_offset = (y_max - y_min) * 0.04
    for idx, row in labels.iterrows():
        period_end = pd.Timestamp(row["period_end"])
        nearby = curves[curves["date"] <= period_end]
        if nearby.empty:
            continue
        point = nearby.iloc[-1]
        y = point["Theme rotation"]
        text = f"{row['period']}\nTop: {row['top_theme']}\nWeak: {row['worst_theme']}"
        ax.annotate(
            text,
            xy=(point["date"], y),
            xytext=(0, 28 if idx % 2 == 0 else -58),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#777", "alpha": 0.82},
            arrowprops={"arrowstyle": "-", "color": "#777", "lw": 0.8},
        )

    ax.set_title("Theme Rotation Strategy PnL with Quarterly Theme Leaders and Laggards")
    ax.set_ylabel("Growth of $1, rebased to 2023-01-01")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PNL_CHART_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    scores = load_scores()
    labels = build_period_labels(scores)
    buckets = build_rank_bucket_diagnostics(scores)
    shorts = build_short_diagnostics(scores)

    labels.to_csv(LABELS_PATH, index=False)
    buckets.to_csv(RANK_BUCKET_PATH, index=False)
    shorts.to_csv(SHORT_DIAGNOSTICS_PATH, index=False)
    plot_pnl(labels)

    print(f"saved labels -> {LABELS_PATH}")
    print(f"saved rank buckets -> {RANK_BUCKET_PATH}")
    print(f"saved short diagnostics -> {SHORT_DIAGNOSTICS_PATH}")
    print(f"saved chart -> {PNL_CHART_PATH}")
    print("\nrank bucket forward returns")
    print(buckets.to_string(index=False))
    if not shorts.empty:
        print("\nshort side summary")
        cols = [
            "selected_short_basket_same_day_long_return",
            "selected_short_basket_next_1d_long_return",
            "selected_short_basket_next_5d_long_return",
            "short_pnl_next_1d_before_cost",
            "short_pnl_next_5d_before_cost",
        ]
        print(shorts[cols].describe().to_string())
        print(f"short basket rose next day on {shorts['short_basket_next_1d_positive_return'].mean():.1%} of days")
        print(f"short basket rose next 5d on {shorts['short_basket_next_5d_positive_return'].mean():.1%} of days")


if __name__ == "__main__":
    main()
