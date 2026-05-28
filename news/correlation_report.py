"""Correlation / validation report for catalyst news features.

Answers two questions:

1. Which catalyst cohorts (family, subtype, source, cluster, similarity bucket,
   sentiment bucket, sector, market cap) actually predict forward returns?
2. Does that predictive power survive an out-of-sample split?

The report intentionally does **not** train any model. It computes cohort
statistics — count, mean forward 5d return, hit rate at +10% within 10d,
median, std, simple t-statistic of the mean vs zero — and reports them with a
fixed in-sample / out-of-sample split.

Inputs (already on disk):
    - news/data/processed/news_records.parquet
    - news/data/processed/news_labels.parquet
    - news/data/processed/news_scores.parquet
    - news/data/processed/news_embeddings.parquet (for finbert sentiment buckets)

Outputs:
    - news/data/processed/correlation_report/{cohort}.csv
    - news/data/processed/correlation_report/summary.md

Run:
    python -m news.correlation_report
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from news.config import (
    NEWS_EMBEDDINGS_PATH,
    NEWS_LABELS_PATH,
    NEWS_RECORDS_PATH,
    NEWS_SCORES_PATH,
    PROCESSED_DIR,
    ensure_data_dirs,
)


# Inclusive train cutoff, then 2025-01-01 onward is OOS.
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
EXPANSION_THRESHOLD = 0.10
MIN_COHORT_SIZE = 20  # below this, t-stats are too noisy to report a verdict


def _load_joined() -> pd.DataFrame:
    news = pd.read_parquet(NEWS_RECORDS_PATH)
    labels = pd.read_parquet(NEWS_LABELS_PATH)
    scores = pd.read_parquet(NEWS_SCORES_PATH)
    emb = pd.read_parquet(NEWS_EMBEDDINGS_PATH)
    keep_emb_cols = [
        c for c in ("record_id", "ticker", "timestamp", "finbert_positive_score", "finbert_negative_score", "finbert_neutral_score", "news_cluster_id")
        if c in emb.columns
    ]
    df = (
        news.merge(labels, on=["record_id", "ticker", "timestamp"], how="left")
        .merge(scores[[c for c in scores.columns if c in ("record_id", "ticker", "timestamp", "news_similarity_score", "news_similarity_max", "news_similarity_neighbor_count", "realized_news_score")]], on=["record_id", "ticker", "timestamp"], how="left")
        .merge(emb[keep_emb_cols], on=["record_id", "ticker", "timestamp"], how="left", suffixes=("", "_emb"))
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["split"] = np.where(df["timestamp"] < TRAIN_END, "train_2023_2024", "eval_2025_2026")
    return df


def _stat_block(group: pd.DataFrame) -> pd.Series:
    """Per-cohort metric block."""
    fwd_5d = group["forward_5d_return"].astype(float)
    fwd_10d = group["forward_10d_return"].astype(float)
    label = group["expansion_label"].astype(float)
    valid = fwd_5d.notna()
    n = int(valid.sum())
    if n == 0:
        return pd.Series(
            {
                "count": int(len(group)),
                "labeled_count": 0,
                "mean_fwd_5d": np.nan,
                "median_fwd_5d": np.nan,
                "std_fwd_5d": np.nan,
                "tstat_fwd_5d": np.nan,
                "win_rate_5d_pos": np.nan,
                "expansion_hit_rate": np.nan,
                "mean_fwd_10d": np.nan,
            }
        )
    arr = fwd_5d[valid].to_numpy()
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    tstat = float(mean / (std / np.sqrt(n))) if std and std > 0 else float("nan")
    return pd.Series(
        {
            "count": int(len(group)),
            "labeled_count": n,
            "mean_fwd_5d": mean,
            "median_fwd_5d": float(np.median(arr)),
            "std_fwd_5d": std,
            "tstat_fwd_5d": tstat,
            "win_rate_5d_pos": float(np.mean(arr > 0)),
            "expansion_hit_rate": float(label[valid].mean()) if label[valid].notna().any() else float("nan"),
            "mean_fwd_10d": float(fwd_10d[valid].dropna().mean()) if fwd_10d[valid].dropna().size else float("nan"),
        }
    )


def _cohort_table(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    keys = list(group_cols)
    out_blocks = []
    for split_value, split_df in df.groupby("split", sort=True):
        block = (
            split_df.groupby(keys, dropna=False, observed=False)
            .apply(_stat_block, include_groups=False)
            .reset_index()
        )
        if "split" in block.columns:
            block = block.drop(columns=["split"])
        block.insert(0, "split", split_value)
        out_blocks.append(block)
    if not out_blocks:
        return pd.DataFrame()
    out = pd.concat(out_blocks, ignore_index=True)
    sort_keys = [c for c in ["split", *keys] if c in out.columns]
    return out.sort_values(sort_keys)


def _bucket(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series.astype(float), bins=[-np.inf, *edges, np.inf], labels=labels, include_lowest=True)


def _similarity_bucket(score: pd.Series) -> pd.Series:
    edges = [-0.30, -0.10, 0.0, 0.10, 0.30]
    labels = ["very_neg", "neg", "weak_neg", "weak_pos", "pos", "very_pos"]
    return _bucket(score, edges, labels)


def _sentiment_bucket(pos: pd.Series, neg: pd.Series) -> pd.Series:
    net = pos.fillna(0.0) - neg.fillna(0.0)
    edges = [-0.40, -0.10, 0.10, 0.40]
    labels = ["bearish", "lean_bearish", "neutral", "lean_bullish", "bullish"]
    return _bucket(net, edges, labels)


def _ic_by_split(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Rank correlation (Spearman) of score vs forward 5d return per split."""
    rows = []
    for split_value, split_df in df.groupby("split", sort=True):
        joined = split_df[[score_col, "forward_5d_return"]].dropna()
        if len(joined) < MIN_COHORT_SIZE:
            rho = float("nan")
            n = len(joined)
        else:
            rho = float(joined.corr(method="spearman").iloc[0, 1])
            n = len(joined)
        rows.append({"split": split_value, "score_col": score_col, "n": n, "spearman_ic": rho})
    return pd.DataFrame(rows)


def _format_md_table(df: pd.DataFrame, *, max_rows: int = 25, format_floats: bool = True) -> str:
    if df.empty:
        return "_(no rows)_"
    out = df.copy().head(max_rows)
    if format_floats:
        for col in out.select_dtypes(include=["float", "float64", "float32"]).columns:
            out[col] = out[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    out = out.astype(str)
    headers = list(out.columns)
    sep = ["---"] * len(headers)
    lines = [" | ".join(headers), " | ".join(sep)]
    for _, row in out.iterrows():
        lines.append(" | ".join(row.tolist()))
    return "\n".join("| " + line + " |" for line in lines)


def build_correlation_report(*, output_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Build the correlation report and write CSVs + a markdown summary."""
    ensure_data_dirs()
    output_dir = Path(output_dir or PROCESSED_DIR / "correlation_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_joined()
    df["news_similarity_bucket"] = _similarity_bucket(df["news_similarity_score"])
    df["sentiment_bucket"] = _sentiment_bucket(df.get("finbert_positive_score", pd.Series(0.0, index=df.index)), df.get("finbert_negative_score", pd.Series(0.0, index=df.index)))

    results: dict[str, pd.DataFrame] = {}

    overall = df.groupby("split").apply(_stat_block, include_groups=False).reset_index()
    results["overall"] = overall

    cohorts = {
        "by_family": ["catalyst_family"],
        "by_subtype": ["catalyst_family", "catalyst_subtype"],
        "by_source": ["source"],
        "by_relation_type": ["relation_type"],
        "by_impact_role": ["impact_role"],
        "by_cluster": ["catalyst_family", "news_cluster_id"],
        "by_similarity_bucket": ["news_similarity_bucket"],
        "by_sentiment_bucket": ["sentiment_bucket"],
        "by_family_x_sentiment": ["catalyst_family", "sentiment_bucket"],
        "by_direct_catalyst": ["is_direct_catalyst"],
    }
    for name, cols in cohorts.items():
        table = _cohort_table(df, cols)
        table.to_csv(output_dir / f"{name}.csv", index=False)
        results[name] = table

    # Rank-correlation report for continuous score columns.
    ic_rows = []
    for score_col in ("news_similarity_score", "news_similarity_max", "earnings_guidance_score", "earnings_beat_miss_score", "earnings_language_score", "finbert_positive_score", "finbert_negative_score", "relation_confidence", "is_direct_catalyst"):
        if score_col in df.columns:
            ic_rows.append(_ic_by_split(df, score_col))
    ic = pd.concat(ic_rows, ignore_index=True) if ic_rows else pd.DataFrame()
    ic.to_csv(output_dir / "rank_correlations.csv", index=False)
    results["rank_correlations"] = ic

    # Per-family top-of-similarity tail: does pushing the threshold pick a real edge?
    tails = []
    for fam, fam_df in df.groupby("catalyst_family", dropna=False):
        scored = fam_df.dropna(subset=["news_similarity_score", "forward_5d_return"])
        if len(scored) < MIN_COHORT_SIZE * 2:
            continue
        top = scored.loc[scored["news_similarity_score"] >= scored["news_similarity_score"].quantile(0.75)]
        bot = scored.loc[scored["news_similarity_score"] <= scored["news_similarity_score"].quantile(0.25)]
        tails.append(
            {
                "catalyst_family": fam,
                "top_quartile_n": int(len(top)),
                "bot_quartile_n": int(len(bot)),
                "top_quartile_mean_fwd_5d": float(top["forward_5d_return"].mean()),
                "bot_quartile_mean_fwd_5d": float(bot["forward_5d_return"].mean()),
                "spread_top_minus_bot": float(top["forward_5d_return"].mean() - bot["forward_5d_return"].mean()),
            }
        )
    tail_df = pd.DataFrame(tails).sort_values("spread_top_minus_bot", ascending=False)
    tail_df.to_csv(output_dir / "similarity_score_tail_spread.csv", index=False)
    results["similarity_score_tail_spread"] = tail_df

    # Markdown summary
    summary = []
    summary.append("# News / Catalyst Correlation Report\n")
    summary.append(
        f"_Generated from {len(df):,} news records spanning "
        f"{df['timestamp'].min()} → {df['timestamp'].max()}._\n"
    )
    summary.append(
        f"\n**Split**: in-sample = records before `{TRAIN_END.date()}`; out-of-sample = `{TRAIN_END.date()}` onward.\n"
    )
    summary.append("\n## 1. Overall forward-return statistics by split\n")
    summary.append(_format_md_table(overall))
    summary.append("\n\n## 2. By catalyst family\n")
    summary.append(_format_md_table(results["by_family"]))
    summary.append("\n\n## 3. By source\n")
    summary.append(_format_md_table(results["by_source"]))
    summary.append("\n\n## 4. By relation type\n")
    summary.append(_format_md_table(results["by_relation_type"]))
    summary.append("\n\n## 5. By similarity-score bucket\n")
    summary.append("Bucket boundaries: −0.30 / −0.10 / 0 / 0.10 / 0.30 on `news_similarity_score`.\n\n")
    summary.append(_format_md_table(results["by_similarity_bucket"]))
    summary.append("\n\n## 6. By sentiment (FinBERT pos − neg) bucket\n")
    summary.append(_format_md_table(results["by_sentiment_bucket"]))
    summary.append("\n\n## 7. Top-vs-bottom similarity quartile spread, per family\n")
    summary.append(_format_md_table(tail_df, max_rows=30))
    summary.append("\n\n## 8. Score → return rank correlation (Spearman IC)\n")
    summary.append(_format_md_table(ic, max_rows=40))
    summary.append("\n\n## 9. By direct-catalyst flag\n")
    summary.append(_format_md_table(results["by_direct_catalyst"]))
    summary.append(
        "\n\n## Reading guide\n"
        "- `expansion_hit_rate` is the share of records whose max forward return within 10 days reached +10%.\n"
        f"- A cohort with `labeled_count < {MIN_COHORT_SIZE}` should be ignored — t-stats are unstable.\n"
        "- A cohort whose `tstat_fwd_5d` is significant in both `train_2023_2024` and `eval_2025_2026` is a genuinely useful filter; one that's only significant in train is overfit-by-luck.\n"
        "- `news_similarity_score` is leak-free as of this build: priors only contribute when their 10-day label was realized by the prediction time. If you used the old artifact, results may have looked stronger than reality.\n"
    )

    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return results


if __name__ == "__main__":
    out = build_correlation_report()
    print("wrote correlation report to news/data/processed/correlation_report/")
    print(f"  overall: {len(out['overall'])} split rows")
    for name in ("by_family", "by_source", "by_similarity_bucket", "by_sentiment_bucket", "by_direct_catalyst", "similarity_score_tail_spread", "rank_correlations"):
        print(f"  {name}: {len(out[name])} rows")
