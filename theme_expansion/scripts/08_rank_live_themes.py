from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import LEADERS_PER_THEME, LIVE_RANKING_PATH, THEME_LEADERS_PATH, THEME_SCORES_PATH, ensure_dirs


def leader_columns(leaders: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    day = leaders[leaders["date"].eq(as_of)].sort_values(["theme", "leader_rank"])
    rows = []
    for theme, frame in day.groupby("theme"):
        tickers = frame.head(LEADERS_PER_THEME)["ticker"].tolist()
        rows.append(
            {
                "theme": theme,
                "top_leader_1": tickers[0] if len(tickers) > 0 else "",
                "top_leader_2": tickers[1] if len(tickers) > 1 else "",
                "top_leader_3": tickers[2] if len(tickers) > 2 else "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the latest rule-based theme rotation radar.")
    parser.add_argument("--date", help="Optional YYYY-MM-DD as-of date. Defaults to latest score date.")
    parser.add_argument("--top", type=int, default=25, help="Number of themes to print.")
    args = parser.parse_args()

    ensure_dirs()
    scores = pd.read_parquet(THEME_SCORES_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    leaders["date"] = pd.to_datetime(leaders["date"])
    as_of = pd.Timestamp(args.date) if args.date else scores["date"].max()

    day = scores[scores["date"].eq(as_of)].sort_values("theme_rank").copy()
    if day.empty:
        raise SystemExit(f"no theme scores for {as_of.date()}")

    out = day[
        [
            "date",
            "theme",
            "theme_rank",
            "theme_score",
            "theme_rank_change_5d",
            "theme_return_5d",
            "theme_vs_spy_5d",
            "theme_rvol",
            "theme_breadth",
        ]
    ].rename(
        columns={
            "theme_rank_change_5d": "rank_change_5d",
            "theme_breadth": "breadth",
        }
    )
    out = out.merge(leader_columns(leaders, as_of), on="theme", how="left")
    out = out.head(args.top)
    out.to_csv(LIVE_RANKING_PATH, index=False)
    print(out.to_string(index=False))
    print(f"saved -> {LIVE_RANKING_PATH}")


if __name__ == "__main__":
    main()
