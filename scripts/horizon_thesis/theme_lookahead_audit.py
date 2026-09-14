"""Does the backfilled theme membership leak the future into Meta's training data?

`themes/dynamic_theme/backfill_theme_features.py` says it outright: it labels
every historical date with TODAY's clusters and today's text embeddings. Only the
co-movement half of the embedding is rebuilt as-of each date. So a company that
was described as an AI-infrastructure name in 2026 is an AI-infrastructure name
on its 2023 rows too, and the theme's 2023 "heat" is computed over members that
were chosen partly because of what they became.

Three checks, cheapest first:

1. CHURN -- how fast does membership actually move? If primary themes churn
   materially in weeks, a 3.5-year backfill cannot be point-in-time. Measured
   between the real weekly PIT snapshots (ticker_theme_membership_history, which
   only starts 2026-08-24) and against the backfill's own last date.

2. IC BY YEAR -- the look-ahead signature. Membership is most anachronistic in
   the earliest years, so a leak should make theme features look BEST early and
   decay toward the present. `mom_score`, which is genuinely walk-forward, is the
   control: it absorbs "some years are easier to rank than others".

3. ABLATION -- train the same model with and without the theme block on an
   identical split, and compare. If the theme features carry a real edge it
   should survive into the test period; if it is backfill an it should be
   concentrated where the anachronism is largest.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/horizon_thesis"))

from run_horizon_grid import META_DROP, N_ROUNDS, PARAMS, per_bar_spearman, split_bars  # noqa: E402

DATA = REPO / "research/execution_quality/data"
OUTPUTS = REPO / "themes/dynamic_theme/outputs"
MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix_research.parquet"
THEME_FEATURES = ["primary_theme_rank", "theme_heat_score", "theme_breadth", "theme_acceleration",
                  "theme_strength", "membership_score", "related_theme_heat", "related_theme_rank",
                  "theme_age_days", "theme_newness_score", "theme_days_since_refresh",
                  "within_theme_mom_rank", "theme_crowding_frac"]
TARGET = "fwd_atr_adj_return"
N_TICKERS = 500
SEED = 17


def churn() -> dict:
    print("=" * 90 + "\n1. MEMBERSHIP CHURN\n" + "=" * 90)
    out: dict = {}
    hist_path = OUTPUTS / "ticker_theme_membership_history.parquet"
    h = pd.read_parquet(hist_path)
    top = (h.sort_values("membership_score", ascending=False)
           .drop_duplicates(["as_of", "ticker"])[["as_of", "ticker", "theme"]])
    snaps = sorted(top["as_of"].unique())
    print(f"PIT snapshots available: {[str(pd.Timestamp(s).date()) for s in snaps]}")
    for a, b in zip(snaps, snaps[1:]):
        ta = top[top["as_of"] == a].set_index("ticker")["theme"]
        tb = top[top["as_of"] == b].set_index("ticker")["theme"]
        common = ta.index.intersection(tb.index)
        chg = float((ta.loc[common] != tb.loc[common]).mean())
        days = (pd.Timestamp(b) - pd.Timestamp(a)).days
        # A renamed cluster is not churn, so also measure whether a ticker keeps
        # the same GROUP: Jaccard of its co-member set across the two snapshots.
        names_a, names_b = set(ta.unique()), set(tb.unique())
        ga = ta.loc[common].groupby(ta.loc[common]).groups
        gb = tb.loc[common].groupby(tb.loc[common]).groups
        jac = [len(set(ga[ta[t]]) & set(gb[tb[t]])) / len(set(ga[ta[t]]) | set(gb[tb[t]])) for t in common]
        print(f"  {pd.Timestamp(a).date()} -> {pd.Timestamp(b).date()} ({days:2d}d): "
              f"{chg:6.1%} of {len(common):,} tickers changed primary theme; "
              f"theme NAMES overlap {len(names_a & names_b) / max(1, len(names_a)):.0%}; "
              f"co-membership Jaccard median {np.median(jac):.3f}")
        out[f"churn_{pd.Timestamp(a).date()}_{pd.Timestamp(b).date()}"] = dict(
            days=days, churn=chg, n=len(common), name_overlap=len(names_a & names_b) / max(1, len(names_a)),
            jaccard_median=float(np.median(jac)), jaccard_mean=float(np.mean(jac)))

    bf = OUTPUTS / "ticker_theme_features_history.parquet"
    if bf.exists():
        b = pd.read_parquet(bf, columns=["date", "ticker", "primary_theme"])
        last = b[b["date"] == b["date"].max()].set_index("ticker")["primary_theme"]
        newest = top[top["as_of"] == snaps[-1]].set_index("ticker")["theme"]
        common = last.index.intersection(newest.index)
        chg = float((last.loc[common] != newest.loc[common]).mean())
        gap = (pd.Timestamp(snaps[-1]) - pd.Timestamp(b["date"].max())).days
        print(f"  backfill last date {pd.Timestamp(b['date'].max()).date()} vs PIT "
              f"{pd.Timestamp(snaps[-1]).date()} ({gap}d): {chg:.1%} of {len(common):,} differ")
        print(f"  backfill history spans {pd.Timestamp(b['date'].min()).date()} .. "
              f"{pd.Timestamp(b['date'].max()).date()}, all labelled with one registry")
        out["backfill_vs_pit"] = dict(days=gap, churn=chg, n=len(common))
    return out


def load_matrix() -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in pq.ParquetFile(MATRIX).schema.names if c not in META_DROP]
    df = pd.read_parquet(MATRIX, columns=cols + [TARGET])
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    tk = pd.Series(sorted(df.index.get_level_values(1).unique()))
    if len(tk) > N_TICKERS:
        df = df[df.index.get_level_values(1).isin(set(tk.sample(N_TICKERS, random_state=SEED)))]
    return df.dropna(subset=[TARGET]).sort_index(), cols


def ic_by_year(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 90 + "\n2. IC BY YEAR (within-bar Spearman vs fwd_atr_adj_return)\n" + "=" * 90)
    bars = df.index.get_level_values(0)
    year = bars.year
    y = df[TARGET].to_numpy()
    feats = [c for c in THEME_FEATURES if c in df.columns] + ["mom_score", "htf_score"]
    rows = {}
    for f in feats:
        per_year = {}
        for yr in sorted(set(year)):
            m = year == yr
            if m.sum() < 5000:
                continue
            r = per_bar_spearman(bars[m].to_numpy(), df[f].to_numpy()[m], y[m])
            per_year[int(yr)] = float(r.mean()) if len(r) else np.nan
        rows[f] = per_year
    table = pd.DataFrame(rows).T
    print(table.round(4).to_string())
    theme_cols = [c for c in THEME_FEATURES if c in table.index]
    years = sorted(table.columns)
    early, late = years[0], years[-1]
    print(f"\n  mean |IC| over theme features: {early} = {table.loc[theme_cols, early].abs().mean():.4f}, "
          f"{late} = {table.loc[theme_cols, late].abs().mean():.4f}")
    print(f"  control mom_score:              {early} = {abs(table.loc['mom_score', early]):.4f}, "
          f"{late} = {abs(table.loc['mom_score', late]):.4f}")
    print("  A leak would show theme IC decaying toward the present FASTER than the control.")
    return {f: {str(k): v for k, v in d.items()} for f, d in rows.items()}


def ablation(df: pd.DataFrame, cols: list[str]) -> dict:
    print("\n" + "=" * 90 + "\n3. ABLATION: same split, theme block in vs out\n" + "=" * 90)
    train, val, test, bounds = split_bars(df)
    print(f"  train={len(train):,} val={len(val):,} test={len(test):,}  {bounds}")
    label = df[TARGET].groupby(level=0).rank(pct=True)
    feats_all = [c for c in cols if c != TARGET]
    feats_no = [c for c in feats_all if c not in THEME_FEATURES]
    out = {}
    preds = {}
    for tag, feats in (("with_theme", feats_all), ("no_theme", feats_no)):
        ytr, yva = label.reindex(train.index), label.reindex(val.index)
        bst = xgb.train(PARAMS, xgb.DMatrix(train[feats], label=ytr), num_boost_round=N_ROUNDS,
                        evals=[(xgb.DMatrix(val[feats], label=yva), "v")],
                        early_stopping_rounds=30, verbose_eval=False)
        p = bst.predict(xgb.DMatrix(test[feats]))
        preds[tag] = p
        r = per_bar_spearman(test.index.get_level_values(0).to_numpy(), p, test[TARGET].to_numpy())
        out[tag] = dict(test_rho=float(r.mean()), n_features=len(feats), bars=int(len(r)))
        print(f"  {tag:11s} features={len(feats):3d} test rho={r.mean():+.4f} over {len(r)} bars")
    bars_te = test.index.get_level_values(0).to_numpy()
    ra = per_bar_spearman(bars_te, preds["no_theme"], test[TARGET].to_numpy())
    rb = per_bar_spearman(bars_te, preds["with_theme"], test[TARGET].to_numpy())
    common = ra.index.intersection(rb.index)
    d = (rb.loc[common] - ra.loc[common]).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = [float(np.mean(rng.choice(d, len(d), True))) for _ in range(3000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  theme block contributes {d.mean():+.4f} rho  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({'distinguishable' if lo * hi > 0 else 'NOT distinguishable'} from noise)")
    out["delta"] = dict(mean=float(d.mean()), ci=[float(lo), float(hi)], bars=int(len(common)))
    return out


def main() -> None:
    res = {"churn": churn()}
    df, cols = load_matrix()
    print(f"\nmatrix rows {len(df):,}, bars {df.index.get_level_values(0).nunique():,}")
    res["ic_by_year"] = ic_by_year(df)
    res["ablation"] = ablation(df, cols)
    out = DATA / "theme_lookahead_audit.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
