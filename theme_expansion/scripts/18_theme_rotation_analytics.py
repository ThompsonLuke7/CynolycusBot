from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR, THEME_SCORES_PATH


PARTICIPATION_PATH = OUTPUT_DIR / "theme_regime_participation.csv"
CORRELATION_PATH = OUTPUT_DIR / "theme_signal_correlations.csv"


def _first_present(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def build_participation(scores: pd.DataFrame) -> pd.DataFrame:
    data = scores.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["is_tradable"].fillna(False).astype(bool)].copy()
    data = data.dropna(subset=["theme_regime_rank"])
    data = data.sort_values(["theme", "date"])
    data["top5"] = data["theme_regime_rank"] <= 5
    data["top10"] = data["theme_regime_rank"] <= 10
    data["fwd_5d_theme_return"] = data.groupby("theme")["theme_return_5d"].shift(-5)

    rows: list[dict[str, object]] = []
    for theme, group in data.groupby("theme", sort=True):
        top5 = group[group["top5"]]
        top10 = group[group["top10"]]
        rows.append(
            {
                "theme": theme,
                "days": len(group),
                "top5_days": int(group["top5"].sum()),
                "top10_days": int(group["top10"].sum()),
                "top5_pct": float(group["top5"].mean()),
                "top10_pct": float(group["top10"].mean()),
                "avg_rank": float(group["theme_regime_rank"].mean()),
                "median_rank": float(group["theme_regime_rank"].median()),
                "best_rank": int(group["theme_regime_rank"].min()),
                "worst_rank": int(group["theme_regime_rank"].max()),
                "latest_rank": int(group.iloc[-1]["theme_regime_rank"]),
                "avg_fwd_5d": float(group["fwd_5d_theme_return"].mean()),
                "avg_fwd_5d_when_top5": float(top5["fwd_5d_theme_return"].mean()) if len(top5) else None,
                "avg_fwd_5d_when_top10": float(top10["fwd_5d_theme_return"].mean()) if len(top10) else None,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["top5_days", "avg_rank"], ascending=[True, False])


def build_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    data = scores.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["is_tradable"].fillna(False).astype(bool)].copy()
    data = data.sort_values(["theme", "date"])
    data["fwd_5d_theme_return"] = data.groupby("theme")["theme_return_5d"].shift(-5)

    columns = set(data.columns)
    candidate_signals = [
        "theme_score",
        "theme_regime_score",
        "theme_regime_rank",
        "theme_heat_score",
        "theme_heat_rank",
        "rank_5d",
        "rank_10d",
        "rank_20d",
        "rank_change_5d",
        "rank_change_10d",
        "rank_change_20d",
        "rank_stability_5d",
        "rank_stability_10d",
        "rank_stability_20d",
        "theme_breadth",
        "theme_breadth_10d",
        "theme_breadth_20d",
        "theme_rvol",
        "theme_rvol_10d",
        "entropy_score",
        "leader_concentration",
        "theme_effective_breadth",
        "theme_vs_spy_5d",
        "theme_vs_spy_10d",
        "theme_vs_spy_20d",
        "theme_return_5d",
        "theme_return_10d",
        "theme_return_20d",
    ]
    signals = [col for col in candidate_signals if col in columns]
    if not signals:
        return pd.DataFrame(columns=["signal", "corr_to_fwd_5d_return"])

    rows = []
    target = data["fwd_5d_theme_return"]
    for signal in signals:
        value = pd.to_numeric(data[signal], errors="coerce")
        valid = value.notna() & target.notna()
        if valid.sum() < 50:
            continue
        rows.append(
            {
                "signal": signal,
                "corr_to_fwd_5d_return": float(value[valid].corr(target[valid])),
                "observations": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("corr_to_fwd_5d_return", ascending=False)


def main() -> None:
    scores = pd.read_parquet(THEME_SCORES_PATH)
    participation = build_participation(scores)
    correlations = build_correlations(scores)

    PARTICIPATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    participation.to_csv(PARTICIPATION_PATH, index=False)
    correlations.to_csv(CORRELATION_PATH, index=False)

    weak = participation[(participation["top5_days"] == 0) | ((participation["top5_days"] < 5) & (participation["avg_rank"] > 50))]
    biotech = participation[participation["theme"].str.contains("biotech", case=False, na=False)]

    print(f"saved {len(participation):,} theme participation rows -> {PARTICIPATION_PATH}")
    print(f"saved {len(correlations):,} signal correlation rows -> {CORRELATION_PATH}")
    print("\nweak / never-top themes")
    print(weak.head(20).to_string(index=False))
    print("\nbiotech themes")
    print(biotech.to_string(index=False))
    print("\ntop positive correlations")
    print(correlations.head(12).to_string(index=False))
    print("\ntop negative correlations")
    print(correlations.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
