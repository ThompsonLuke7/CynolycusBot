"""Phase 5: supervised winner-vs-loser separators (logistic + shallow tree)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

NUMERIC_FEATURES = [
    "p_dir", "ev_score", "atr_pct_of_entry", "ref_range_pct", "entry_vs_ref",
    "confirm_bars_watched", "signal_to_entry_secs",
    "option_dte", "option_strike_distance_pct",
    "entry_minute_of_day", "day_of_week",
    "qqq_ret_30m", "qqq_ret_1h", "iwm_ret_30m", "iwm_ret_1h",
    "vix_ret_30m", "vix_ret_1h", "qqq_iwm_relstr_1h",
]
CATEGORICAL = ["direction", "tier"]


def build_matrix(closed: pd.DataFrame):
    Xn = closed[NUMERIC_FEATURES].copy()
    for c in Xn.columns:
        if Xn[c].isna().any():
            Xn[c] = Xn[c].fillna(Xn[c].median())
    Xc = pd.get_dummies(closed[CATEGORICAL].astype(str), drop_first=False)
    X = pd.concat([Xn, Xc], axis=1)
    return X


def fit_logistic(X, y, scale: bool = True):
    Xs = StandardScaler().fit_transform(X) if scale else X.values
    lr = LogisticRegression(penalty="l1", C=0.5, solver="liblinear", max_iter=500).fit(Xs, y)
    coefs = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=lambda s: s.abs(), ascending=False)
    return lr, coefs


def fit_tree(X, y, max_depth=3, min_samples_leaf=15):
    t = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42,
        criterion="entropy",
    ).fit(X.values, y)
    return t


def evaluate_rule(closed: pd.DataFrame, mask: pd.Series, label: str):
    sub = closed[mask]
    rest = closed[~mask]
    n_sub = len(sub); n_rest = len(rest)
    if n_sub == 0:
        return None
    return {
        "rule": label,
        "n_match": n_sub,
        "wr_match": float((sub["underlying_pnl_pct"] > 0).mean()),
        "mean_match": float(sub["underlying_pnl_pct"].mean()),
        "sum_match": float(sub["underlying_pnl_pct"].sum()),
        "n_rest": n_rest,
        "wr_rest": float((rest["underlying_pnl_pct"] > 0).mean()) if n_rest else float("nan"),
        "mean_rest": float(rest["underlying_pnl_pct"].mean()) if n_rest else float("nan"),
        "sum_rest": float(rest["underlying_pnl_pct"].sum()) if n_rest else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default="local_artifacts/swing_analysis_20260525/trades_with_regime.parquet")
    p.add_argument("--out", default="local_artifacts/swing_analysis_20260525/phase5_supervised.md")
    args = p.parse_args()

    df = pd.read_parquet(args.trades)
    closed = df[df["exit_price"].notna() & df["underlying_pnl_pct"].notna()].copy()
    print(f"Closed: {len(closed)}")

    md = []
    def emit(s=""):
        print(s)
        md.append(s)

    emit("# Phase 5 — Supervised Separators: Winner vs Loser\n")
    emit(f"- n = {len(closed)}")
    baseline_wr = (closed["underlying_pnl_pct"] > 0).mean()
    baseline_pnl = closed["underlying_pnl_pct"].mean()
    emit(f"- baseline WR = {100*baseline_wr:.1f}%, mean PnL = {100*baseline_pnl:.3f}%\n")

    X = build_matrix(closed)
    y_u = (closed["underlying_pnl_pct"] > 0).astype(int).values

    # ---- Logistic regression (underlying win)
    emit("## Logistic regression (L1, target = underlying win)\n")
    lr, coefs = fit_logistic(X, y_u, scale=True)
    nonzero = coefs[coefs.abs() > 1e-6]
    emit(f"- non-zero features (after L1): {len(nonzero)}")
    emit("- coefficients (+ pushes toward winning, − toward losing):\n")
    emit("```")
    for f, v in nonzero.items():
        emit(f"  {f:35s} {v:+.4f}")
    emit("```\n")

    # ---- Shallow decision tree
    emit("## Shallow decision tree (depth=3, entropy, min_leaf=15)\n")
    t = fit_tree(X, y_u, max_depth=3, min_samples_leaf=15)
    tree_text = export_text(t, feature_names=list(X.columns), max_depth=5)
    emit("```")
    emit(tree_text)
    emit("```\n")

    # ---- Hand-built candidate rules from descriptive + cluster findings, scored
    emit("## Candidate filter rules — historical impact on the 5/14–5/22 window\n")
    emit("For each rule, 'match' = trades the filter would VETO; 'rest' = trades that survive the filter.\n")
    emit("A *good* veto rule has low WR / negative mean on 'match' and lifts WR / mean on 'rest'.\n")

    rules = []
    # Time-of-day
    rules.append(("Veto: entry after 14:30 ET",
                  closed["entry_minute_of_day"] >= 14 * 60 + 30))
    rules.append(("Veto: entry after 13:00 ET",
                  closed["entry_minute_of_day"] >= 13 * 60))
    rules.append(("Veto: Tier 2 shorts",
                  (closed["tier"] == 2) & (closed["direction"] == -1)))
    rules.append(("Veto: Tier 2 only",
                  closed["tier"] == 2))
    rules.append(("Veto: Tuesday entries (day_of_week=1)",
                  closed["day_of_week"] == 1))
    rules.append(("Veto: VIX rising >0.3% in 30m before entry",
                  closed["vix_ret_30m"].fillna(0) > 0.003))
    rules.append(("Veto: 3-4 DTE contracts",
                  closed["option_dte"].between(3, 4)))
    rules.append(("Veto: confirm_bars_watched in {2,4,5}",
                  closed["confirm_bars_watched"].isin([2, 4, 5])))
    rules.append(("Veto: Tier 2 shorts AFTER 13:00",
                  (closed["tier"] == 2) & (closed["direction"] == -1) & (closed["entry_minute_of_day"] >= 13*60)))
    rules.append(("Veto: shorts AFTER 14:30 ET",
                  (closed["direction"] == -1) & (closed["entry_minute_of_day"] >= 14*60+30)))
    rules.append(("Veto: bottom-10 tickers by historical mean PnL",
                  closed["ticker"].isin([
                      "RGTI", "KLAC", "CRWV", "IREN", "STNE", "ENPH", "SMR", "PDD", "SNOW", "BABA"
                  ])))
    rules.append(("Veto: cross-session matches (positions held overnight)",
                  closed["match_type"] == "cross_session_optsym"))
    # Combined "tighten"
    tighten_mask = (
        (closed["tier"] == 1)
        & (closed["entry_minute_of_day"] < 13*60)
        & (closed["vix_ret_30m"].fillna(0) <= 0.003)
    )
    rules.append(("KEEP-ONLY: Tier 1 AND entry < 13:00 AND VIX not rising fast (everything else vetoed)",
                  ~tighten_mask))  # match = trades that would be vetoed

    results = []
    for label, mask in rules:
        r = evaluate_rule(closed, mask, label)
        if r:
            results.append(r)
    emit("```")
    emit(f"{'rule':<70} {'n_v':>5} {'wr_v':>6} {'mn_v':>7} {'sum_v':>7}  {'n_k':>5} {'wr_k':>6} {'mn_k':>7} {'sum_k':>7}  {'dWR':>5} {'dMean':>6}")
    for r in results:
        dWR = (r["wr_rest"] - baseline_wr) * 100
        dMean = (r["mean_rest"] - baseline_pnl) * 100
        emit(
            f"{r['rule']:<70} {r['n_match']:>5} {100*r['wr_match']:>5.1f}% {100*r['mean_match']:>+6.2f}% {100*r['sum_match']:>+6.1f}%  "
            f"{r['n_rest']:>5} {100*r['wr_rest']:>5.1f}% {100*r['mean_rest']:>+6.2f}% {100*r['sum_rest']:>+6.1f}%  "
            f"{dWR:>+4.1f} {dMean:>+5.2f}"
        )
    emit("```\n")
    emit("Legend: `v`=vetoed/match group; `k`=kept group; `dWR`=lift in win rate on kept; `dMean`=lift in mean PnL on kept (pp).\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
