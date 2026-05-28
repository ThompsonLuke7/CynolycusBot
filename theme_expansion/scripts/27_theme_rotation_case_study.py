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
    OUTPUT_DIR,
    PLOTS_DIR,
    REPORT_DIR,
    THEME_DAILY_PATH,
    THEME_LEADERS_PATH,
    THEME_SIGNAL_LABELS_PATH,
    ensure_dirs,
)


CASE_STUDY_SUMMARY_PATH = REPORT_DIR / "theme_rotation_case_study_summary.csv"
CASE_STUDY_CANDIDATES_PATH = REPORT_DIR / "theme_rotation_case_study_candidates.csv"


def clean_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def normalize_at_first(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    first = series.dropna().iloc[0] if series.notna().any() else np.nan
    return series / first * 100.0 if pd.notna(first) and first != 0 else series * np.nan


def load_prices(tickers: list[str]) -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars[bars["ticker"].isin(tickers)].copy()
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    return bars.pivot(index="date", columns="ticker", values="px").sort_index()


def forward_return(series: pd.Series, date: pd.Timestamp, horizon: int) -> float:
    series = series.dropna().sort_index()
    if date not in series.index:
        return np.nan
    loc = series.index.get_loc(date)
    if isinstance(loc, slice) or loc + horizon >= len(series):
        return np.nan
    return float(series.iloc[loc + horizon] / series.iloc[loc] - 1.0)


def choose_candidate(labels: pd.DataFrame, theme: str | None, date: str | None) -> pd.Series:
    labels = labels.sort_values(["theme", "date"]).copy()
    labels["fwd_theme_return_40d"] = labels.groupby("theme")["theme_index"].shift(-40) / labels["theme_index"] - 1.0

    if theme:
        subset = labels[labels["theme"].eq(theme)].copy()
        if subset.empty:
            raise SystemExit(f"no rows found for theme {theme}")
        if date:
            day = subset[subset["date"].eq(pd.Timestamp(date))]
            if day.empty:
                raise SystemExit(f"no row found for {theme} on {date}")
            return day.iloc[0]
        subset = subset[subset["signal_continuation_long_flag"].fillna(False)]
        if subset.empty:
            subset = labels[labels["theme"].eq(theme)].copy()
    elif date:
        subset = labels[labels["date"].eq(pd.Timestamp(date))].copy()
    else:
        subset = labels[
            labels["signal_continuation_long_flag"].fillna(False)
            & labels["fwd_theme_return_40d"].notna()
            & labels["fwd_theme_excess_spy_20d"].notna()
            & labels["date"].between(pd.Timestamp("2023-01-01"), labels["date"].max() - pd.Timedelta(days=80))
            & labels["theme_regime_rank"].le(10)
        ].copy()

    if subset.empty:
        raise SystemExit("no candidate rows found")

    subset["case_score"] = (
        0.45 * subset["fwd_theme_excess_spy_20d"].fillna(-1.0)
        + 0.35 * subset["fwd_theme_return_40d"].fillna(-1.0)
        + 0.10 * subset["signal_theme_continuation_score"].fillna(0.0)
        + 0.10 * subset["signal_rank_percentile"].fillna(0.0)
    )

    by_theme = subset.sort_values("case_score", ascending=False).drop_duplicates("theme")
    by_theme.to_csv(CASE_STUDY_CANDIDATES_PATH, index=False)
    return by_theme.sort_values("case_score", ascending=False).iloc[0]


def top_leaders_on_date(leaders: pd.DataFrame, theme: str, date: pd.Timestamp, n: int = 3) -> list[str]:
    day = leaders[(leaders["theme"].eq(theme)) & (leaders["date"].eq(date))].copy()
    if day.empty:
        prior = leaders[(leaders["theme"].eq(theme)) & (leaders["date"].le(date))].sort_values("date")
        if prior.empty:
            return []
        date = prior["date"].max()
        day = prior[prior["date"].eq(date)].copy()
    return day.sort_values("leader_rank")["ticker"].head(n).tolist()


def build_case_study(theme: str | None, date: str | None, pre_days: int, post_days: int) -> tuple[Path, Path, pd.DataFrame]:
    labels = pd.read_parquet(THEME_SIGNAL_LABELS_PATH)
    theme_daily = pd.read_parquet(THEME_DAILY_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    for frame in (labels, theme_daily, leaders):
        frame["date"] = pd.to_datetime(frame["date"])

    candidate = choose_candidate(labels, theme, date)
    signal_date = pd.Timestamp(candidate["date"])
    theme_name = str(candidate["theme"])
    leader_tickers = top_leaders_on_date(leaders, theme_name, signal_date)
    tickers = ["SPY", *leader_tickers]
    prices = load_prices(tickers)

    theme_series = theme_daily[theme_daily["theme"].eq(theme_name)].set_index("date")["theme_index"].sort_index()
    all_dates = theme_series.index.union(prices.index).sort_values()
    if signal_date not in all_dates:
        raise SystemExit(f"signal date {signal_date.date()} not found in price index")
    signal_loc = all_dates.get_loc(signal_date)
    start = all_dates[max(0, signal_loc - pre_days)]
    end = all_dates[min(len(all_dates) - 1, signal_loc + post_days)]
    window = pd.date_range(start, end, freq="D")

    plot_df = pd.DataFrame(index=window)
    plot_df[f"{theme_name} theme"] = theme_series.reindex(window).ffill()
    for ticker in tickers:
        if ticker in prices:
            plot_df[ticker] = prices[ticker].reindex(window).ffill()
    plot_df = plot_df.dropna(how="all")
    normalized = plot_df.apply(normalize_at_first)

    rank_window = labels[labels["theme"].eq(theme_name)].set_index("date").sort_index()
    rank_window = rank_window.reindex(window).ffill()

    fwd_rows = []
    for label, series in plot_df.items():
        for horizon in (20, 40):
            fwd_rows.append(
                {
                    "series": label,
                    "horizon": horizon,
                    "forward_return": forward_return(series, signal_date, horizon),
                }
            )
    summary = pd.DataFrame(fwd_rows)
    summary["theme"] = theme_name
    summary["signal_date"] = signal_date.date().isoformat()
    summary["theme_regime_rank"] = float(candidate.get("theme_regime_rank", np.nan))
    summary["signal_market_playbook"] = str(candidate.get("signal_market_playbook", ""))
    summary["signal_pair_trade_side"] = str(candidate.get("signal_pair_trade_side", ""))
    summary["signal_theme_continuation_score"] = float(candidate.get("signal_theme_continuation_score", np.nan))
    summary["leader_tickers"] = "|".join(leader_tickers)
    case_summary_path = REPORT_DIR / f"theme_rotation_case_study_summary_{clean_name(theme_name)}_{signal_date.date()}.csv"
    summary.to_csv(case_summary_path, index=False)
    summary.to_csv(CASE_STUDY_SUMMARY_PATH, index=False)

    fig, (ax, rank_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.1]},
    )

    theme_label = f"{theme_name} theme"
    ax.plot(normalized.index, normalized[theme_label], color="#1f77b4", linewidth=2.8, label=theme_label)
    if "SPY" in normalized:
        ax.plot(normalized.index, normalized["SPY"], color="#555555", linewidth=2.2, label="SPY")

    colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for idx, ticker in enumerate(leader_tickers):
        if ticker in normalized:
            ax.plot(
                normalized.index,
                normalized[ticker],
                linewidth=1.8,
                linestyle="--",
                color=colors[idx % len(colors)],
                label=f"{ticker} leader",
            )

    ax.axvline(signal_date, color="#111111", linestyle=":", linewidth=1.7)
    ax.axvspan(normalized.index.min(), signal_date, color="#b0b0b0", alpha=0.10)
    ax.axvspan(signal_date, normalized.index.max(), color="#1f77b4", alpha=0.07)
    ax.set_ylabel("Growth of 100")
    ax.set_title(
        f"Theme Rotation Case Study: {theme_name} ranked #{int(candidate['theme_regime_rank'])} on {signal_date.date()}",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        signal_date,
        ax.get_ylim()[1],
        " signal date",
        va="top",
        ha="left",
        fontsize=9,
        color="#111111",
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    rank_ax.plot(
        rank_window.index,
        rank_window["theme_regime_rank"],
        color="#d62728",
        linewidth=2.0,
        label="theme regime rank",
    )
    score_ax = rank_ax.twinx()
    score_ax.plot(
        rank_window.index,
        rank_window["signal_theme_continuation_score"] * 100,
        color="#1f77b4",
        linewidth=1.5,
        alpha=0.75,
        label="continuation score x100",
    )
    rank_ax.axvline(signal_date, color="#111111", linestyle=":", linewidth=1.5)
    rank_ax.invert_yaxis()
    rank_ax.set_ylabel("Theme regime rank")
    score_ax.set_ylabel("Continuation score")
    score_ax.set_ylim(0, 100)
    rank_ax.grid(True, alpha=0.22)
    handles_1, labels_1 = rank_ax.get_legend_handles_labels()
    handles_2, labels_2 = score_ax.get_legend_handles_labels()
    rank_ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left", fontsize=9)

    theme_20 = summary[(summary["series"].eq(theme_label)) & (summary["horizon"].eq(20))]["forward_return"].iloc[0]
    spy_20 = summary[(summary["series"].eq("SPY")) & (summary["horizon"].eq(20))]["forward_return"].iloc[0]
    theme_40 = summary[(summary["series"].eq(theme_label)) & (summary["horizon"].eq(40))]["forward_return"].iloc[0]
    spy_40 = summary[(summary["series"].eq("SPY")) & (summary["horizon"].eq(40))]["forward_return"].iloc[0]
    note = (
        f"Signal: {candidate['signal_market_playbook']} / {candidate['signal_pair_trade_side']} | "
        f"20d: theme {theme_20:.1%}, SPY {spy_20:.1%} | "
        f"40d: theme {theme_40:.1%}, SPY {spy_40:.1%} | "
        f"leaders: {', '.join(leader_tickers)}"
    )
    fig.text(0.01, 0.01, note, fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    out_path = PLOTS_DIR / f"theme_rotation_case_study_{clean_name(theme_name)}_{signal_date.date()}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path, case_summary_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a concrete theme rotation case study.")
    parser.add_argument("--theme", help="Optional theme to plot. Defaults to best continuation case found.")
    parser.add_argument("--date", help="Optional signal date YYYY-MM-DD.")
    parser.add_argument("--pre-days", type=int, default=20, help="Trading days before signal to show.")
    parser.add_argument("--post-days", type=int, default=45, help="Trading days after signal to show.")
    args = parser.parse_args()

    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path, summary_path, summary = build_case_study(args.theme, args.date, args.pre_days, args.post_days)
    print(f"saved plot -> {out_path}")
    print(f"saved summary -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
