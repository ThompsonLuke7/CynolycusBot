from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import BACKTEST_DIR, OUTPUT_DIR, THEME_DAILY_PATH, THEME_SCORES_PATH, ensure_dirs

REPORT_DIR = OUTPUT_DIR / "reports"


def save_count_distribution(scores: pd.DataFrame) -> None:
    latest = scores["date"].max()
    day = scores[scores["date"].eq(latest)].copy()
    cols = ["theme", "theme_num_constituents", "is_tradable", "is_watchlist_only"]
    out = day[[c for c in cols if c in day.columns]].sort_values("theme_num_constituents", ascending=False)
    out.to_csv(REPORT_DIR / "theme_count_distribution.csv", index=False)
    out.plot.bar(
        x="theme",
        y="theme_num_constituents",
        legend=False,
        figsize=(13, 6),
        title=f"Theme Constituent Count {latest.date()}",
    )
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "theme_count_distribution.png", dpi=150)
    plt.close()


def save_entropy_distribution(scores: pd.DataFrame) -> None:
    latest = scores["date"].max()
    day = scores[scores["date"].eq(latest)].copy()
    out = day[["theme", "entropy_score", "theme_effective_breadth", "leader_concentration"]].sort_values(
        "entropy_score",
        ascending=False,
    )
    out.to_csv(REPORT_DIR / "theme_entropy_distribution.csv", index=False)
    out.plot.bar(
        x="theme",
        y="entropy_score",
        legend=False,
        figsize=(13, 6),
        title=f"Theme Entropy {latest.date()}",
    )
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "theme_entropy_distribution.png", dpi=150)
    plt.close()


def save_return_vs_constituents(scores: pd.DataFrame) -> None:
    latest = scores["date"].max()
    day = scores[scores["date"].eq(latest)].copy()
    out = day[
        [
            "theme",
            "theme_return_5d",
            "theme_vs_spy_5d",
            "theme_num_constituents",
            "entropy_score",
            "is_tradable",
            "is_watchlist_only",
        ]
    ].sort_values("theme_vs_spy_5d", ascending=False)
    out.to_csv(REPORT_DIR / "theme_return_vs_constituents.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(out["theme_num_constituents"], out["theme_vs_spy_5d"], alpha=0.7)
    for _, row in out.head(12).iterrows():
        ax.annotate(row["theme"], (row["theme_num_constituents"], row["theme_vs_spy_5d"]), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("theme_num_constituents")
    ax.set_ylabel("theme_vs_spy_5d")
    ax.set_title(f"Theme Return vs Constituents {latest.date()}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "theme_return_vs_constituents.png", dpi=150)
    plt.close()


def save_v1_v2_comparison() -> None:
    candidates = {
        "old_v2": REPORT_DIR / "old_v2_backtest_metrics.json",
        "v2": BACKTEST_DIR / "rule_based_backtest_metrics.json",
    }
    rows = []
    for label, path in candidates.items():
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        strategy = metrics.get("strategy", {})
        rows.append({"universe": label, **strategy})
    if len(rows) < 2:
        return
    pd.DataFrame(rows).to_csv(REPORT_DIR / "v1_vs_v2_backtest_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v2 theme breadth and entropy reports.")
    args = parser.parse_args()
    _ = args
    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(THEME_SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    save_count_distribution(scores)
    save_entropy_distribution(scores)
    save_return_vs_constituents(scores)
    save_v1_v2_comparison()
    print(f"saved reports -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
