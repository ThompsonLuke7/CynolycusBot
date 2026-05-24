from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MATRIX = Path("momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_OUT = Path("momentum_expansion/data/processed/label_variant_experiment")


WEIGHTS = {
    "fwd_max_alpha": 0.40,
    "fwd_atr_adj_return": 0.25,
    "trend_persistence": 0.20,
    "fwd_max_drawdown": 0.15,
}


def _date_level(df: pd.DataFrame) -> str | int:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex with timestamp and ticker levels")
    if "timestamp" in df.index.names:
        return "timestamp"
    return 0


def _rank_by_date(df: pd.DataFrame, col: str, *, ascending: bool = True) -> pd.Series:
    ts_level = _date_level(df)
    return df.groupby(level=ts_level)[col].rank(pct=True, ascending=ascending)


def _raw_cross_sectional_score(df: pd.DataFrame) -> pd.Series:
    alpha_r = _rank_by_date(df, "fwd_max_alpha")
    atr_r = _rank_by_date(df, "fwd_atr_adj_return")
    persist_r = _rank_by_date(df, "trend_persistence")
    drawdown_good_r = _rank_by_date(df, "fwd_max_drawdown", ascending=False)
    return (
        WEIGHTS["fwd_max_alpha"] * alpha_r
        + WEIGHTS["fwd_atr_adj_return"] * atr_r
        + WEIGHTS["trend_persistence"] * persist_r
        + WEIGHTS["fwd_max_drawdown"] * drawdown_good_r
    )


def _summarize_selection(df: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, float | int | str]:
    g = df.loc[mask.fillna(False)]
    base_rate_20 = float((df["fwd_max_return"] >= 0.20).mean())
    selected_rate_20 = float((g["fwd_max_return"] >= 0.20).mean()) if len(g) else np.nan
    close_winners = g.loc[g["fwd_close_return"] > 0, "fwd_close_return"]
    close_losers = g.loc[g["fwd_close_return"] <= 0, "fwd_close_return"]
    return {
        "experiment": name,
        "rows": int(len(g)),
        "row_share": float(len(g) / max(len(df), 1)),
        "tickers": int(g.index.get_level_values("ticker").nunique()) if len(g) else 0,
        "avg_fwd_max_return": float(g["fwd_max_return"].mean()) if len(g) else np.nan,
        "median_fwd_max_return": float(g["fwd_max_return"].median()) if len(g) else np.nan,
        "avg_fwd_close_return": float(g["fwd_close_return"].mean()) if len(g) else np.nan,
        "avg_drawdown": float(g["fwd_max_drawdown"].mean()) if len(g) else np.nan,
        "pct_gt_20": selected_rate_20,
        "pct_gt_25": float((g["fwd_max_return"] >= 0.25).mean()) if len(g) else np.nan,
        "pct_gt_40": float((g["fwd_max_return"] >= 0.40).mean()) if len(g) else np.nan,
        "gt20_lift_vs_all": float(selected_rate_20 / base_rate_20) if base_rate_20 > 0 else np.nan,
        "avg_close_winner": float(close_winners.mean()) if len(close_winners) else np.nan,
        "avg_close_loser": float(close_losers.mean()) if len(close_losers) else np.nan,
        "close_win_rate": float((g["fwd_close_return"] > 0).mean()) if len(g) else np.nan,
        "avg_tickers_per_bar": float(g.groupby(level=_date_level(g)).size().mean()) if len(g) else 0.0,
    }


def _score_buckets(df: pd.DataFrame, score_col: str, name: str) -> pd.DataFrame:
    work = df[[score_col, "fwd_max_return", "fwd_close_return", "fwd_max_drawdown"]].dropna().copy()
    ranked = work[score_col].rank(method="first")
    work["decile"] = pd.qcut(ranked, 10, labels=False) + 1
    rows = []
    base_rate_20 = float((work["fwd_max_return"] >= 0.20).mean())
    for decile, g in work.groupby("decile"):
        rate_20 = float((g["fwd_max_return"] >= 0.20).mean())
        rows.append(
            {
                "experiment": name,
                "bucket": f"decile_{int(decile)}",
                "rows": int(len(g)),
                "score_min": float(g[score_col].min()),
                "score_mean": float(g[score_col].mean()),
                "score_max": float(g[score_col].max()),
                "avg_fwd_max_return": float(g["fwd_max_return"].mean()),
                "avg_fwd_close_return": float(g["fwd_close_return"].mean()),
                "avg_drawdown": float(g["fwd_max_drawdown"].mean()),
                "pct_gt_20": rate_20,
                "gt20_lift_vs_all": float(rate_20 / base_rate_20) if base_rate_20 > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(matrix_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(matrix_path)
    needed = [
        "fwd_max_return",
        "fwd_max_alpha",
        "fwd_atr_adj_return",
        "fwd_max_drawdown",
        "fwd_close_return",
        "trend_persistence",
        "expansion_score",
        "expansion_target",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=needed).copy()

    df["exp_b_current_score_xsec_top10"] = (
        _rank_by_date(df, "expansion_score") >= 0.90
    ).astype(float)
    df["raw_xsec_expansion_score"] = _raw_cross_sectional_score(df)
    df["exp_c_raw_xsec_top10"] = (
        _rank_by_date(df, "raw_xsec_expansion_score") >= 0.90
    ).astype(float)

    summaries = pd.DataFrame(
        [
            _summarize_selection(df, pd.Series(True, index=df.index), "ALL_candidate_rows"),
            _summarize_selection(df, df["expansion_target"] > 0, "EXP_A_current_ticker_p80"),
            _summarize_selection(df, df["exp_b_current_score_xsec_top10"] > 0, "EXP_B_current_score_date_top10"),
            _summarize_selection(df, df["exp_c_raw_xsec_top10"] > 0, "EXP_C_raw_xsec_score_date_top10"),
        ]
    )
    summaries.to_csv(out_dir / "summary.csv", index=False)

    buckets = pd.concat(
        [
            _score_buckets(df, "expansion_score", "current_expansion_score"),
            _score_buckets(df, "raw_xsec_expansion_score", "raw_xsec_expansion_score"),
        ],
        ignore_index=True,
    )
    buckets.to_csv(out_dir / "score_deciles.csv", index=False)

    overlap = pd.DataFrame(
        {
            "metric": [
                "A_and_B_share_of_A",
                "A_and_C_share_of_A",
                "B_and_C_share_of_B",
                "A_only_rows",
                "B_only_rows",
                "C_only_rows",
            ],
            "value": [
                float(((df["expansion_target"] > 0) & (df["exp_b_current_score_xsec_top10"] > 0)).sum() / max((df["expansion_target"] > 0).sum(), 1)),
                float(((df["expansion_target"] > 0) & (df["exp_c_raw_xsec_top10"] > 0)).sum() / max((df["expansion_target"] > 0).sum(), 1)),
                float(((df["exp_b_current_score_xsec_top10"] > 0) & (df["exp_c_raw_xsec_top10"] > 0)).sum() / max((df["exp_b_current_score_xsec_top10"] > 0).sum(), 1)),
                int(((df["expansion_target"] > 0) & ~(df["exp_b_current_score_xsec_top10"] > 0) & ~(df["exp_c_raw_xsec_top10"] > 0)).sum()),
                int((~(df["expansion_target"] > 0) & (df["exp_b_current_score_xsec_top10"] > 0) & ~(df["exp_c_raw_xsec_top10"] > 0)).sum()),
                int((~(df["expansion_target"] > 0) & ~(df["exp_b_current_score_xsec_top10"] > 0) & (df["exp_c_raw_xsec_top10"] > 0)).sum()),
            ],
        }
    )
    overlap.to_csv(out_dir / "overlap.csv", index=False)

    print("Summary")
    print(summaries.to_string(index=False))
    print()
    print("Top score deciles")
    top = buckets[buckets["bucket"].isin(["decile_9", "decile_10"])]
    print(top.to_string(index=False))
    print()
    print("Overlap")
    print(overlap.to_string(index=False))
    print()
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.matrix, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
