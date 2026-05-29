from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    DAILY_BARS_PATH,
    PLOTS_DIR,
    REPORT_DIR,
    THEME_DAILY_PATH,
    THEME_SCORES_PATH,
    clean_ticker,
    ensure_dirs,
    load_theme_memberships,
)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def normalize(series: pd.Series) -> pd.Series:
    first = series.dropna().iloc[0] if series.notna().any() else np.nan
    if pd.isna(first) or first == 0:
        return series * np.nan
    return series / first * 100.0


def load_price_index(tickers: list[str]) -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars[bars["ticker"].isin(tickers)].copy()
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    return bars.pivot(index="date", columns="ticker", values="px").sort_index()


def find_membership_themes(ticker: str) -> list[str]:
    memberships = load_theme_memberships()
    themes = memberships.loc[memberships["ticker"].eq(ticker), "theme"].drop_duplicates().sort_values().tolist()
    if not themes:
        raise SystemExit(f"no theme memberships found for {ticker}")
    return themes


def summarize_rank_cycles(scores: pd.DataFrame, themes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for theme, frame in scores[scores["theme"].isin(themes)].groupby("theme"):
        frame = frame.sort_values("date").copy()
        top5 = frame["theme_regime_rank"].le(5).fillna(False)
        top10 = frame["theme_regime_rank"].le(10).fillna(False)
        bottom_half = frame["theme_regime_rank"].gt(50).fillna(False)
        top5_entries = (top5 & ~top5.shift(1, fill_value=False)).sum()
        top10_entries = (top10 & ~top10.shift(1, fill_value=False)).sum()
        rows.append(
            {
                "theme": theme,
                "days": int(len(frame)),
                "top5_days": int(top5.sum()),
                "top10_days": int(top10.sum()),
                "bottom_half_days": int(bottom_half.sum()),
                "top5_entries": int(top5_entries),
                "top10_entries": int(top10_entries),
                "best_rank": float(frame["theme_regime_rank"].min()),
                "worst_rank": float(frame["theme_regime_rank"].max()),
                "avg_rank": float(frame["theme_regime_rank"].mean()),
                "latest_rank": float(frame["theme_regime_rank"].dropna().iloc[-1]) if frame["theme_regime_rank"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["latest_rank", "theme"], na_position="last")


def plot_ticker_theme_rotation(ticker: str, start: str, end: str | None) -> tuple[Path, Path]:
    ticker = clean_ticker(ticker)
    themes = find_membership_themes(ticker)

    theme_daily = pd.read_parquet(THEME_DAILY_PATH)
    scores = pd.read_parquet(THEME_SCORES_PATH)
    for frame in (theme_daily, scores):
        frame["date"] = pd.to_datetime(frame["date"])

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else scores["date"].max()
    theme_daily = theme_daily[theme_daily["theme"].isin(themes) & theme_daily["date"].between(start_ts, end_ts)].copy()
    scores = scores[scores["theme"].isin(themes) & scores["date"].between(start_ts, end_ts)].copy()

    prices = load_price_index([ticker, "SPY"])
    prices = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)].copy()
    dates = pd.Index(sorted(set(prices.index).union(set(theme_daily["date"]))))

    price_plot = pd.DataFrame(index=dates)
    if ticker in prices:
        price_plot[ticker] = prices[ticker].reindex(dates).ffill()
    if "SPY" in prices:
        price_plot["SPY"] = prices["SPY"].reindex(dates).ffill()
    for theme in themes:
        series = theme_daily[theme_daily["theme"].eq(theme)].set_index("date")["theme_index"].sort_index()
        if not series.empty:
            price_plot[theme] = series.reindex(dates).ffill()
    price_plot = price_plot.dropna(how="all").apply(normalize)

    rank_plot = scores.pivot(index="date", columns="theme", values="theme_regime_rank").reindex(dates).ffill()
    rank_plot_smooth = rank_plot.rolling(5, min_periods=1).median()
    top_flag = pd.DataFrame(index=dates)
    for theme in themes:
        if theme in rank_plot:
            top_flag[f"{theme}_top5"] = rank_plot[theme].le(5)
            top_flag[f"{theme}_top10"] = rank_plot[theme].le(10)

    summary = summarize_rank_cycles(scores, themes)
    summary_path = REPORT_DIR / f"{safe_name(ticker)}_theme_membership_rotation_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, (ax, rank_ax, heat_ax) = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        sharex=False,
        gridspec_kw={"height_ratios": [3.0, 2.0, 0.8]},
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(themes))))
    ax.plot(price_plot.index, price_plot[ticker], color="#111111", linewidth=2.8, label=ticker)
    if "SPY" in price_plot:
        ax.plot(price_plot.index, price_plot["SPY"], color="#777777", linewidth=2.1, label="SPY")
    for idx, theme in enumerate(themes):
        if theme in price_plot:
            ax.plot(price_plot.index, price_plot[theme], color=colors[idx], linewidth=1.8, alpha=0.88, label=theme)
    ax.set_title(f"{ticker} Versus Its Theme Memberships", loc="left", fontsize=15, fontweight="bold")
    ax.set_ylabel("Growth of 100")
    ax.set_xlim(dates.min(), dates.max())
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.tick_params(labelbottom=False)

    for idx, theme in enumerate(themes):
        if theme in rank_plot_smooth:
            rank_ax.plot(rank_plot_smooth.index, rank_plot_smooth[theme], color=colors[idx], linewidth=1.9, label=theme)
    rank_ax.axhspan(0.5, 5.5, color="#2ca02c", alpha=0.08, label="top 5 zone")
    rank_ax.axhspan(5.5, 10.5, color="#1f77b4", alpha=0.06, label="top 10 zone")
    rank_ax.invert_yaxis()
    rank_ax.set_ylabel("Theme regime rank")
    rank_ax.set_xlim(dates.min(), dates.max())
    rank_ax.grid(True, alpha=0.22)
    rank_ax.legend(loc="upper left", ncol=2, fontsize=8)
    rank_ax.tick_params(labelbottom=False)

    heat = rank_plot.copy()
    heat = heat[themes]
    heat_ax.imshow(heat.T, aspect="auto", interpolation="nearest", cmap="viridis_r")
    heat_ax.set_yticks(range(len(themes)), labels=themes)
    heat_ax.set_ylabel("Rank heat")
    heat_ax.set_xlabel("Date")
    if len(heat.index) > 0:
        ticks = np.linspace(0, len(heat.index) - 1, 8).astype(int)
        heat_ax.set_xticks(ticks, labels=[heat.index[i].strftime("%Y-%m-%d") for i in ticks], rotation=0)

    note_parts = []
    for row in summary.itertuples(index=False):
        note_parts.append(f"{row.theme}: top5 entries {row.top5_entries}, top10 entries {row.top10_entries}, latest rank {row.latest_rank:.0f}")
    fig.text(0.01, 0.01, " | ".join(note_parts), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = PLOTS_DIR / f"{safe_name(ticker)}_theme_membership_rotation.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a ticker against all themes it belongs to.")
    parser.add_argument("--ticker", default="NVDA", help="Ticker to inspect.")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD.")
    args = parser.parse_args()

    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path, summary_path = plot_ticker_theme_rotation(args.ticker, args.start, args.end)
    print(f"saved plot -> {out_path}")
    print(f"saved summary -> {summary_path}")
    print(pd.read_csv(summary_path).to_string(index=False))


if __name__ == "__main__":
    main()
