"""Phase 4: unsupervised clustering on signal/entry-time + regime features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "p_dir", "ev_score", "atr_pct_of_entry", "ref_range_pct", "entry_vs_ref",
    "confirm_bars_watched", "signal_to_entry_secs",
    "option_dte", "option_strike_distance_pct",
    "entry_minute_of_day", "day_of_week",
    "qqq_ret_30m", "qqq_ret_1h", "iwm_ret_30m", "iwm_ret_1h",
    "vix_ret_30m", "vix_ret_1h", "qqq_iwm_relstr_1h",
]
CATEGORICAL = ["direction", "tier"]


def _basic(s):
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "win_rate": float((s > 0).mean()),
        "sum": float(s.sum()),
    }


def _profile_clusters(closed, labels, baseline_wr, baseline_pnl, emit):
    clos = closed.copy()
    clos["cluster"] = labels
    stats = []
    for c, sub in clos.groupby("cluster"):
        st = _basic(sub["underlying_pnl_pct"])
        prof = {}
        for feat in FEATURES:
            v = sub[feat].dropna()
            if len(v) == 0:
                continue
            prof[feat] = float(v.mean())
        dom_tier = sub["tier"].value_counts(normalize=True).round(2).to_dict()
        dom_dir = sub["direction"].value_counts(normalize=True).round(2).to_dict()
        top_tickers = sub["ticker"].value_counts().head(5).to_dict()
        top_tod = sub.assign(
            tod=pd.cut(
                sub["entry_minute_of_day"],
                bins=[-1, 600, 660, 780, 870, 1000],
                labels=["pre10", "10-11", "11-13", "13-1430", "1430-16"],
            )
        )["tod"].value_counts(normalize=True).round(2).to_dict()
        stats.append((c, st, prof, dom_tier, dom_dir, top_tickers, top_tod))
    stats.sort(key=lambda x: x[1]["mean"], reverse=True)
    for c, st, prof, dom_tier, dom_dir, top_tickers, top_tod in stats:
        emit(f"\n### Cluster {int(c)}")
        emit(
            f"- n = {st['n']}, mean PnL = {100*st['mean']:.2f}%, median = {100*st['median']:.2f}%, "
            f"WR = {100*st['win_rate']:.1f}%, sum = {100*st['sum']:.1f}%"
        )
        dwr = st["win_rate"] - baseline_wr
        dm = st["mean"] - baseline_pnl
        flag = ""
        if st["n"] >= 30:
            if dwr < -0.10 and dm < -0.005:
                flag = "  🚩 VETO candidate"
            elif dwr > 0.07 and dm > 0.005:
                flag = "  ⭐ HIGH-QUALITY"
        emit(f"- vs baseline: dWR={100*dwr:+.1f}pp, dMean={100*dm:+.3f}%{flag}")
        emit(f"- dominant tier: {dom_tier}, direction: {dom_dir}")
        emit(f"- time-of-day mix: {top_tod}")
        emit(f"- top tickers (count): {top_tickers}")
        feat_lines = []
        for f, m in prof.items():
            overall_m = closed[f].mean()
            overall_std = closed[f].std() or 1
            z = (m - overall_m) / overall_std
            if abs(z) > 0.5:
                feat_lines.append((abs(z), f"  {f}: cluster_mean={m:.4f} (overall={overall_m:.4f}, z={z:+.2f})"))
        feat_lines.sort(reverse=True)
        for _, ln in feat_lines[:8]:
            emit(ln)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default="local_artifacts/swing_analysis_20260525/trades_with_regime.parquet")
    p.add_argument("--out", default="local_artifacts/swing_analysis_20260525/phase4_clusters.md")
    args = p.parse_args()

    df = pd.read_parquet(args.trades)
    closed = df[df["exit_price"].notna() & df["underlying_pnl_pct"].notna()].copy()
    print(f"Closed trades for clustering: {len(closed)}")

    X_num = closed[FEATURES].copy()
    for c in X_num.columns:
        if X_num[c].isna().any():
            X_num[c] = X_num[c].fillna(X_num[c].median())
    X_cat = pd.get_dummies(closed[CATEGORICAL].astype(str), drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    feat_names = list(X.columns)
    Xs = StandardScaler().fit_transform(X)

    baseline_pnl = closed["underlying_pnl_pct"].mean()
    baseline_wr = (closed["underlying_pnl_pct"] > 0).mean()

    md = []
    def emit(s=""):
        print(s)
        md.append(s)

    emit("# Phase 4 — Unsupervised Clustering\n")
    emit(f"- n = {len(closed)} closed trades")
    emit(f"- features ({len(feat_names)}): {feat_names}")
    emit(f"- baseline WR = {100*baseline_wr:.1f}%  mean PnL = {100*baseline_pnl:.3f}%\n")

    emit("## K-Means k sweep\n")
    emit(f"{'k':>3}  {'silhouette':>10}  per-cluster summary")
    best = {"k": None, "score": -1, "labels": None}
    all_labels = {}
    for k in (3, 4, 5, 6, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        labels = km.labels_
        score = silhouette_score(Xs, labels)
        clos = closed.copy()
        clos["c"] = labels
        per = (
            clos.groupby("c")["underlying_pnl_pct"]
            .agg(n="count", mean="mean", wr=lambda s: (s > 0).mean())
            .sort_values("mean")
        )
        summary = "; ".join(
            f"c{int(c)}: n={int(r['n'])} mean={100*r['mean']:.2f}% wr={100*r['wr']:.0f}%"
            for c, r in per.iterrows()
        )
        emit(f"{k:>3}  {score:>10.4f}  {summary}")
        all_labels[k] = labels
        if score > best["score"]:
            best.update(k=k, score=score, labels=labels)
    emit("")

    # Profile both the silhouette-best k and k=8 (richer structure)
    detail_ks = sorted({best["k"], 8})
    for k in detail_ks:
        emit(f"\n## K-Means k={k} cluster profiles\n")
        _profile_clusters(closed, all_labels[k], baseline_wr, baseline_pnl, emit)

    emit("\n## HDBSCAN (density-based)\n")
    h = HDBSCAN(min_cluster_size=20, min_samples=5).fit(Xs)
    labels = h.labels_
    noise = (labels == -1).sum()
    n_clust = len(set(labels)) - (1 if noise > 0 else 0)
    emit(f"- found {n_clust} clusters; noise points: {noise}/{len(closed)}")
    clos = closed.copy()
    clos["h"] = labels
    per = (
        clos.groupby("h")["underlying_pnl_pct"]
        .agg(n="count", mean="mean", wr=lambda s: (s > 0).mean())
        .sort_values("mean")
    )
    emit(per.to_string())
    if n_clust >= 2:
        emit("\n### HDBSCAN cluster profiles\n")
        # only profile non-noise clusters
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > 0:
            _profile_clusters(closed[non_noise_mask], labels[non_noise_mask], baseline_wr, baseline_pnl, emit)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
