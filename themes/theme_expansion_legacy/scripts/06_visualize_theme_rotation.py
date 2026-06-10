from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PLOTS_DIR, THEME_LEADERS_PATH, THEME_SCORES_PATH, TOP_N_THEMES, ensure_dirs


def save_leaderboard(scores: pd.DataFrame) -> None:
    latest_dates = sorted(scores["date"].unique())[-60:]
    leaderboard = scores[scores["date"].isin(latest_dates)].sort_values(["date", "theme_rank"])
    leaderboard.to_csv(PLOTS_DIR / "theme_leaderboard_by_date.csv", index=False)
    latest = scores["date"].max()
    day = scores[scores["date"].eq(latest)].nsmallest(20, "theme_rank").sort_values("theme_score")
    day.plot.barh(x="theme", y="theme_score", legend=False, figsize=(9, 7), title=f"Theme Leaderboard {latest.date()}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "theme_leaderboard_latest.png", dpi=150)
    plt.close()


def save_heatmap(scores: pd.DataFrame) -> None:
    recent = scores[scores["date"].isin(sorted(scores["date"].unique())[-90:])]
    pivot = recent.pivot(index="theme", columns="date", values="theme_vs_spy_5d").fillna(0.0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(5, len(pivot) * 0.28)))
    im = ax.imshow(pivot, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.15)
    step = max(1, len(pivot.columns) // 12)
    ax.set_xticks(range(0, len(pivot.columns), step), labels=[d.strftime("%Y-%m-%d") for d in pivot.columns[::step]], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    fig.colorbar(im, ax=ax, label="theme_vs_spy_5d")
    ax.set_title("Theme Relative Strength Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "theme_vs_spy_5d_heatmap.png", dpi=150)
    plt.close()


def save_bubble_chart(scores: pd.DataFrame, leaders: pd.DataFrame) -> None:
    latest = scores["date"].max()
    day = scores[scores["date"].eq(latest)].copy()
    leader_counts = leaders[leaders["date"].eq(latest)].groupby("theme")["ticker"].nunique().rename("theme_num_leaders")
    day = day.merge(leader_counts, on="theme", how="left")
    day["theme_num_leaders"] = day["theme_num_leaders"].fillna(0)
    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = (day["theme_num_leaders"] + 1) * 80
    ax.scatter(day["theme_vs_spy_5d"], day["theme_rvol"], s=sizes, alpha=0.55)
    for _, row in day.nsmallest(12, "theme_rank").iterrows():
        ax.annotate(row["theme"], (row["theme_vs_spy_5d"], row["theme_rvol"]), fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("theme_vs_spy_5d")
    ax.set_ylabel("theme_rvol")
    ax.set_title(f"Theme Rotation Bubble Chart {latest.date()}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "theme_rotation_bubble.png", dpi=150)
    plt.close()


def save_rank_change(scores: pd.DataFrame) -> None:
    latest = scores["date"].max()
    day = scores[scores["date"].eq(latest)].nsmallest(20, "theme_rank").sort_values("theme_rank_change_5d")
    day.plot.barh(x="theme", y="theme_rank_change_5d", legend=False, figsize=(9, 7), title="5d Theme Rank Change")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "theme_rank_change_5d.png", dpi=150)
    plt.close()


def save_top_themes_over_time(scores: pd.DataFrame) -> None:
    top = scores[scores["theme_rank"] <= TOP_N_THEMES].groupby("theme").size().sort_values(ascending=False)
    top.head(25).sort_values().plot.barh(figsize=(9, 7), title="Top Themes Over Time")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_themes_over_time.png", dpi=150)
    plt.close()


def save_top_leaders(leaders: pd.DataFrame) -> None:
    latest = leaders["date"].max()
    day = leaders[leaders["date"].eq(latest)].sort_values(["theme", "leader_rank"])
    day.to_csv(PLOTS_DIR / "top_leaders_per_theme.csv", index=False)
    top = day.sort_values("leader_score", ascending=False).head(30).copy()
    top["label"] = top["theme"] + " / " + top["ticker"]
    top.sort_values("leader_score").plot.barh(
        x="label",
        y="leader_score",
        legend=False,
        figsize=(10, 8),
        title=f"Top Theme Leaders {latest.date()}",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_leaders_per_theme.png", dpi=150)
    plt.close()


def main() -> None:
    argparse.ArgumentParser(description="Create rule-based theme rotation visualizations.").parse_args()
    ensure_dirs()
    scores = pd.read_parquet(THEME_SCORES_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    leaders["date"] = pd.to_datetime(leaders["date"])
    save_leaderboard(scores)
    save_heatmap(scores)
    save_bubble_chart(scores, leaders)
    save_rank_change(scores)
    save_top_themes_over_time(scores)
    save_top_leaders(leaders)
    print(f"saved plots and tables -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()
