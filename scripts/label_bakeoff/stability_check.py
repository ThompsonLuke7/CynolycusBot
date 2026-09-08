"""Are the bake-off differences distinguishable from noise?

The winning margins are small in absolute terms (rho 0.04-0.10), and a single
test split can order candidates by luck. This re-fits the two comparisons that
would actually drive a decision, keeps the PER-BAR correlations, and bootstraps
the PAIRED difference over test bars — paired because both candidates are scored
on the identical set of bars, so the bar-to-bar noise is shared and differencing
removes it.
"""
from __future__ import annotations

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
sys.path.insert(0, str(REPO / "scripts/label_bakeoff"))

from run_bakeoff import (EMBARGO_BARS, EVAL, N_ROUNDS, N_TICKERS, PARAMS, SEED,  # noqa: E402
                         meta_candidates, momentum_candidates, split_with_embargo)
from strategies.momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H  # noqa: E402


def per_bar_rho(index, pred, y):
    tmp = pd.DataFrame({"p": pred, "y": y}, index=index.get_level_values(0)).dropna()
    out = {}
    for ts, g in tmp.groupby(level=0):
        if len(g) < 5 or g["p"].std() == 0 or g["y"].std() == 0:
            continue
        out[ts] = np.corrcoef(g["p"].rank(), g["y"].rank())[0, 1]
    return pd.Series(out)


def load(matrix, feature_cols, label_inputs):
    schema = set(pq.ParquetFile(matrix).schema.names)
    feats = [c for c in feature_cols if c in schema]
    df = pd.read_parquet(matrix, columns=sorted(set(feats) | (set(label_inputs) & schema)))
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    tk = df.index.get_level_values(1).unique()
    if len(tk) > N_TICKERS:
        keep = set(pd.Series(sorted(tk)).sample(N_TICKERS, random_state=SEED))
        df = df[df.index.get_level_values(1).isin(keep)]
    ev = pd.read_parquet(EVAL, columns=["eval_mfe_10b", "eval_clean_10b"])
    ev = ev[ev.index.get_level_values(1).isin(set(df.index.get_level_values(1).unique()))]
    df = df.join(ev, how="inner").dropna(subset=["eval_mfe_10b"])
    return df, feats


def fit_predict(df, feats, series, train, val, test):
    ytr, yva = series.reindex(train.index), series.reindex(val.index)
    ok, okv = ytr.notna().to_numpy(), yva.notna().to_numpy()
    bst = xgb.train(PARAMS, xgb.DMatrix(train[feats][ok], label=ytr[ok]),
                    num_boost_round=N_ROUNDS,
                    evals=[(xgb.DMatrix(val[feats][okv], label=yva[okv]), "v")],
                    early_stopping_rounds=30, verbose_eval=False)
    return bst.predict(xgb.DMatrix(test[feats]))


def compare(tag, df, feats, cands, a, b, target="eval_mfe_10b"):
    train, val, test = split_with_embargo(df)
    ra = per_bar_rho(test.index, fit_predict(df, feats, cands[a], train, val, test),
                     test[target].to_numpy())
    rb = per_bar_rho(test.index, fit_predict(df, feats, cands[b], train, val, test),
                     test[target].to_numpy())
    common = ra.index.intersection(rb.index)
    d = (rb.loc[common] - ra.loc[common]).to_numpy()
    rng = np.random.default_rng(3)
    boot = [np.mean(rng.choice(d, len(d), True)) for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    win = float(np.mean(d > 0))
    print(f"\n{tag}   ({target}, {len(common)} test bars)")
    print(f"  {a:22s} mean per-bar rho = {ra.loc[common].mean():+.4f}")
    print(f"  {b:22s} mean per-bar rho = {rb.loc[common].mean():+.4f}")
    print(f"  paired difference       = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  bars where {b} wins      = {win:.1%}")
    print(f"  VERDICT: {'distinguishable from noise' if lo * hi > 0 else 'NOT distinguishable from noise'}")


def main() -> None:
    mom_inputs = ["fwd_max_alpha", "fwd_atr_adj_return", "trend_persistence",
                  "fwd_max_drawdown", "fwd_max_return", "beta_spy_60"]
    df, feats = load(REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet",
                     FEATURE_COLUMNS_4H, mom_inputs)
    c = momentum_candidates(df)
    compare("MOMENTUM: current vs pure ATR-adjusted", df, feats, c, "M0_current", "M3_pure_atr_adj")
    compare("MOMENTUM: current vs pure ATR-adjusted (clean target)", df, feats, c,
            "M0_current", "M3_pure_atr_adj", target="eval_clean_10b")
    del df, c

    meta = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
    cols = list(pq.ParquetFile(meta).schema.names)
    drop = {"meta_label", "trade_quality", "meta_good", "meta_upside", "fwd_max_return",
            "fwd_max_alpha", "fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return",
            "trend_persistence", "theme", "date", "timestamp", "ticker",
            "__index_level_0__", "__index_level_1__"}
    mi = ["fwd_max_return", "fwd_max_alpha", "fwd_max_drawdown", "fwd_close_return",
          "fwd_atr_adj_return", "trend_persistence", "beta_spy_60", "dollar_vol_pctile_252"]
    df, feats = load(meta, [x for x in cols if x not in drop], mi)
    c = meta_candidates(df)
    compare("META: deployed meta_good vs rank-composite", df, feats, c,
            "T0_meta_good", "T3_rank_composite")


if __name__ == "__main__":
    main()
