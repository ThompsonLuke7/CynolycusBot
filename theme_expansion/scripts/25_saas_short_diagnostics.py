from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
THEME_DIR = ROOT / "theme_expansion"
DATA_DIR = THEME_DIR / "data"
OUTPUT_DIR = THEME_DIR / "outputs"
OUT_DIR = OUTPUT_DIR / "deep_theme_rotation"

THEME_MAP_PATH = (
    DATA_DIR / "theme_map_v3_1.csv"
    if (DATA_DIR / "theme_map_v3_1.csv").exists()
    else DATA_DIR / "theme_map_v2.csv"
    if (DATA_DIR / "theme_map_v2.csv").exists()
    else DATA_DIR / "default_theme_map.csv"
)

SOFTWARE_THEMES = [
    "enterprise_software",
    "cloud",
    "workflow_software",
    "business_apps",
    "dev_tools",
    "creative_software",
    "vertical_software",
]


def load_memberships() -> pd.DataFrame:
    raw = pd.read_csv(THEME_MAP_PATH)
    theme_cols = [c for c in ["theme_1", "theme_2", "theme_3", "theme"] if c in raw.columns]
    id_cols = [c for c in raw.columns if c not in theme_cols]
    long = raw.melt(id_vars=id_cols, value_vars=theme_cols, value_name="theme")
    long["theme"] = long["theme"].fillna("").astype(str).str.strip().str.lower()
    long = long[long["theme"].ne("")]
    long["ticker"] = long["ticker"].astype(str).str.upper().str.strip()
    return long.drop_duplicates(["ticker", "theme"])


def period_return_from_index(frame: pd.DataFrame, theme: str, start: str, end: str) -> float | None:
    sub = frame[(frame["theme"] == theme) & (frame["date"].between(start, end))].sort_values("date")
    sub = sub.dropna(subset=["theme_index"])
    if len(sub) < 2:
        return None
    return float(sub.iloc[-1]["theme_index"] / sub.iloc[0]["theme_index"] - 1.0)


def ticker_period_return(ticker: str, start: str, end: str) -> float | None:
    path = DATA_DIR / "daily_bars" / f"{ticker}.parquet"
    if not path.exists():
        return None
    bars = pd.read_parquet(path)
    bars["date"] = pd.to_datetime(bars["date"])
    sub = bars[bars["date"].between(start, end)].sort_values("date")
    if len(sub) < 2 or "close" not in sub.columns:
        return None
    return float(sub.iloc[-1]["close"] / sub.iloc[0]["close"] - 1.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(OUTPUT_DIR / "theme_scores.parquet")
    daily = pd.read_parquet(OUTPUT_DIR / "theme_daily.parquet")
    scores["date"] = pd.to_datetime(scores["date"])
    daily["date"] = pd.to_datetime(daily["date"])
    members = load_memberships()

    latest_date = scores["date"].max()
    start = pd.Timestamp("2025-01-02")
    end = latest_date

    rows: list[dict[str, object]] = []
    for theme in SOFTWARE_THEMES:
        theme_scores = scores[(scores["theme"] == theme) & (scores["date"].between(start, end))].copy()
        if theme_scores.empty:
            continue
        latest = theme_scores.sort_values("date").iloc[-1]
        ranks = theme_scores["theme_regime_rank"].dropna()
        heat_ranks = theme_scores["theme_heat_rank"].dropna()
        ticker_list = sorted(members.loc[members["theme"] == theme, "ticker"].unique())
        ticker_returns = [ticker_period_return(t, str(start.date()), str(end.date())) for t in ticker_list]
        ticker_returns = [r for r in ticker_returns if r is not None]
        rows.append(
            {
                "theme": theme,
                "is_tradable_latest": bool(latest.get("is_tradable", False)),
                "is_watchlist_only_latest": bool(latest.get("is_watchlist_only", False)),
                "mapped_tickers": "|".join(ticker_list),
                "ticker_count": len(ticker_list),
                "theme_return_2025_to_latest": period_return_from_index(daily, theme, str(start.date()), str(end.date())),
                "avg_member_return_2025_to_latest": sum(ticker_returns) / len(ticker_returns) if ticker_returns else None,
                "median_member_return_2025_to_latest": pd.Series(ticker_returns).median() if ticker_returns else None,
                "pct_members_negative_2025_to_latest": pd.Series(ticker_returns).lt(0).mean() if ticker_returns else None,
                "avg_regime_rank_2025_to_latest": ranks.mean() if not ranks.empty else None,
                "worst_regime_rank_2025_to_latest": ranks.max() if not ranks.empty else None,
                "pct_days_bottom_25_regime_rank": ranks.ge(76).mean() if not ranks.empty else None,
                "pct_days_bottom_10_regime_rank": ranks.ge(86).mean() if not ranks.empty else None,
                "avg_heat_rank_2025_to_latest": heat_ranks.mean() if not heat_ranks.empty else None,
                "latest_regime_rank": latest.get("theme_regime_rank"),
                "latest_heat_rank": latest.get("theme_heat_rank"),
                "latest_theme_return_20d": latest.get("theme_return_20d"),
                "latest_theme_vs_spy_20d": latest.get("theme_vs_spy_20d"),
            }
        )

    result = pd.DataFrame(rows).sort_values("theme")
    out_path = OUT_DIR / "saas_software_short_diagnostics.csv"
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
