"""Label bake-off — which label design makes the model rank TRADEABLE outcome best?

The question is not "which label is easiest to predict". Removing the volatility
component from a label removes the easy, autocorrelated part of it, so measured
learnability falls BY CONSTRUCTION (the current momentum score correlates +0.17
with ticker ATR% against +0.19 with its own target). A candidate wins only if its
PREDICTIONS order realised forward outcome better than today's do.

So every candidate is scored on the same two independent targets, computed from
bars and equal to none of the candidates:

  eval_mfe_10b    forward MFE in ATR over ~5 trading days (the horizon actually traded)
  eval_clean_10b  the same minus adverse excursion (prefers clean moves)

Protocol, identical for every candidate:
  * same feature set, same rows, same split, same hyperparameters -- only the
    training target changes
  * time-ordered split with an EMBARGO on both boundaries wide enough to cover
    the longest label look-forward (the capstone leakage audit flagged that the
    production splits have none)
  * scored by WITHIN-BAR rank correlation on test, so a market-wide up day
    cannot masquerade as ranking skill, plus top-decile lift
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from strategies.momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H  # noqa: E402

DATA = REPO / "research/execution_quality/data"
EVAL = DATA / "eval_targets_4h.parquet"
EMBARGO_BARS = 60          # > the longest label look-forward (38 x 4H for HTF)
# This box has a 19GB ceiling and the repo has an OOM incident on record, so the
# matrices are read with column projection and a ticker subsample rather than
# whole. The subsample keeps FULL history for the tickers it keeps, so the time
# split and embargo are unaffected; it only reduces cross-sectional width.
N_TICKERS = 500
SEED = 17

PARAMS = dict(max_depth=6, learning_rate=0.06, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=20, tree_method="hist", n_jobs=8,
              reg_lambda=2.0, objective="reg:squarederror")
N_ROUNDS = 300


# ---------------------------------------------------------------- candidates

def rank_in_bar(df, col, ascending=True):
    return df.groupby(level=0)[col].rank(pct=True, ascending=ascending)


def momentum_candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Momentum builds its composite from RANKS, so the weights are honest and the
    only defect is that `fwd_max_alpha` is a within-bar no-op (rho=1.000 with raw
    return) that therefore ranks volatility."""
    r_alpha = rank_in_bar(df, "fwd_max_alpha")
    r_atr = rank_in_bar(df, "fwd_atr_adj_return")
    r_pers = rank_in_bar(df, "trend_persistence")
    r_dd = rank_in_bar(df, "fwd_max_drawdown", ascending=False)
    # Beta-adjusted alpha, from columns already in the matrix:
    # bench_fwd_max = fwd_max_return - fwd_max_alpha (a per-bar constant), so
    # beta-adjusted alpha = fwd_max_return - beta * bench.
    bench = df["fwd_max_return"] - df["fwd_max_alpha"]
    beta = df["beta_spy_60"].clip(-3, 5).fillna(1.0)
    r_balpha = rank_in_bar(pd.DataFrame({"x": df["fwd_max_return"] - beta * bench},
                                        index=df.index), "x")
    return {
        # as deployed today
        "M0_current": 0.40 * r_alpha + 0.25 * r_atr + 0.20 * r_pers + 0.15 * r_dd,
        # same shape, alpha made genuinely market-relative
        "M1_beta_alpha": 0.40 * r_balpha + 0.25 * r_atr + 0.20 * r_pers + 0.15 * r_dd,
        # alpha dropped, its weight moved to the volatility-neutral component
        "M2_drop_alpha": 0.65 * r_atr + 0.20 * r_pers + 0.15 * r_dd,
        # the tradeable quantity alone
        "M3_pure_atr_adj": r_atr,
        # tradeable, with a real drawdown penalty
        "M4_atr_dd": 0.50 * r_atr + 0.30 * r_dd + 0.20 * r_pers,
        # beta-alpha leading, tradeable second (the re-weight I proposed)
        "M5_reweighted": 0.45 * r_atr + 0.25 * r_balpha + 0.15 * r_dd + 0.15 * r_pers,
    }


def htf_candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    """HTF sums RAW components with no normalisation, so its stated 35/25/25/15
    is fiction: measured effective share is alpha 3.9%, atr_adjusted 91.6%,
    drawdown 1.3%, persistence 3.2%."""
    atr_pct = df["atr_pct_14"].replace(0, np.nan)
    alpha = df["fwd_best_high_return"]                        # bench is a per-bar constant
    atr_adj = (df["fwd_best_high_return"] / atr_pct).clip(-50, 50)
    dd = (-df["fwd_worst_low_return"]).clip(lower=0)
    pers = df["long_persistence"].fillna(0.0)
    raw = pd.DataFrame({"alpha": alpha, "atr_adj": atr_adj, "dd": dd, "pers": pers},
                       index=df.index)

    def z(col):
        s = raw[col]
        return (s - s.mean()) / (s.std() if s.std() else 1.0)

    def rk(col, ascending=True):
        return rank_in_bar(raw, col, ascending)

    return {
        # as deployed: raw, unnormalised -- atr_adj swamps everything
        "H0_current": 0.35 * alpha + 0.25 * atr_adj - 0.25 * dd + 0.15 * pers,
        # same weights, components z-scored so the stated design becomes real
        "H1_zscored": 0.35 * z("alpha") + 0.25 * z("atr_adj") - 0.25 * z("dd") + 0.15 * z("pers"),
        # same weights, components ranked within bar (momentum's construction)
        "H2_ranked": 0.35 * rk("alpha") + 0.25 * rk("atr_adj") + 0.25 * rk("dd", False) + 0.15 * rk("pers"),
        # what H0 already effectively is, stated honestly
        "H3_pure_atr_adj": rk("atr_adj"),
        # ranked, with the drawdown penalty actually mattering
        "H4_rank_dd_heavy": 0.45 * rk("atr_adj") + 0.35 * rk("dd", False) + 0.20 * rk("pers"),
    }


def meta_candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Meta uses RAW components. Its regression target is accidentally near
    beta-neutral (+0.035) but the DEPLOYED binary `meta_good` is 2x beta-biased
    (high-beta 19.3% vs low-beta 9.9% pass rate)."""
    r_atr = rank_in_bar(df, "fwd_atr_adj_return")
    r_dd = rank_in_bar(df, "fwd_max_drawdown", ascending=False)
    r_pers = rank_in_bar(df, "trend_persistence")
    bench = df["fwd_max_return"] - df["fwd_max_alpha"]
    beta = df["beta_spy_60"].clip(-3, 5).fillna(1.0)
    balpha = df["fwd_max_return"] - beta * bench
    r_balpha = rank_in_bar(pd.DataFrame({"x": balpha}, index=df.index), "x")
    liq = df.get("dollar_vol_pctile_252", pd.Series(1.0, index=df.index)).fillna(0.0)
    giveback = (df["fwd_max_return"] - df["fwd_close_return"]).clip(lower=0.0)
    return {
        # deployed binary gate
        "T0_meta_good": ((df["fwd_max_return"] >= 0.12) & (df["fwd_max_drawdown"] <= 0.08)
                         & (df["fwd_max_alpha"] > 0.0) & (liq >= 0.40)).astype(float),
        # same gate, beta-neutral instead of alpha>0
        "T1_good_beta_neutral": ((df["fwd_max_return"] >= 0.12) & (df["fwd_max_drawdown"] <= 0.08)
                                 & (balpha > 0.0) & (liq >= 0.40)).astype(float),
        # continuous regression target as deployed
        "T2_trade_quality": (1.0 * df["fwd_max_alpha"] + 0.25 * df["trend_persistence"].fillna(0)
                             - 1.0 * df["fwd_max_drawdown"] - 0.25 * giveback),
        # rank-composite, momentum's construction applied to Meta
        "T3_rank_composite": 0.45 * r_atr + 0.25 * r_balpha + 0.15 * r_dd + 0.15 * r_pers,
        # the tradeable quantity alone
        "T4_pure_atr_adj": r_atr,
    }


# ---------------------------------------------------------------- evaluation

def within_bar_rho(df: pd.DataFrame, pred: np.ndarray, target: str) -> float:
    """Mean within-bar Spearman. Holds the tape fixed so a market-wide move
    cannot read as ranking skill."""
    tmp = pd.DataFrame({"p": pred, "y": df[target].to_numpy()},
                       index=df.index.get_level_values(0))
    tmp = tmp.dropna()
    out = []
    for _, g in tmp.groupby(level=0):
        if len(g) < 5 or g["p"].std() == 0 or g["y"].std() == 0:
            continue
        out.append(np.corrcoef(g["p"].rank(), g["y"].rank())[0, 1])
    return float(np.mean(out)) if out else float("nan")


def decile_lift(df: pd.DataFrame, pred: np.ndarray, target: str) -> float:
    """Mean target in the top-decile-by-prediction of each bar, minus the bar mean."""
    tmp = pd.DataFrame({"p": pred, "y": df[target].to_numpy()},
                       index=df.index.get_level_values(0)).dropna()
    out = []
    for _, g in tmp.groupby(level=0):
        if len(g) < 10:
            continue
        k = max(1, len(g) // 10)
        top = g.nlargest(k, "p")["y"].mean()
        out.append(top - g["y"].mean())
    return float(np.mean(out)) if out else float("nan")


def split_with_embargo(df: pd.DataFrame):
    ts = df.index.get_level_values(0)
    uniq = np.array(sorted(ts.unique()))
    n = len(uniq)
    tr_end, va_end = uniq[int(n * 0.60)], uniq[int(n * 0.78)]
    gap = pd.Timedelta(hours=4 * EMBARGO_BARS)
    train = df[ts <= tr_end]
    val = df[(ts > tr_end + gap) & (ts <= va_end)]
    test = df[ts > va_end + gap]
    return train, val, test


def run_module(name: str, matrix: Path, cand_fn, feature_cols, label_inputs, out_rows) -> None:
    print(f"\n{'=' * 100}\n{name}\n{'=' * 100}", flush=True)
    import pyarrow.parquet as pq
    schema = set(pq.ParquetFile(matrix).schema.names)
    feats = [c for c in feature_cols if c in schema]
    want = sorted(set(feats) | (set(label_inputs) & schema))
    df = pd.read_parquet(matrix, columns=want)
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")

    # Ticker subsample: full history each, so only cross-sectional width shrinks.
    tickers = df.index.get_level_values(1).unique()
    if len(tickers) > N_TICKERS:
        keep = set(pd.Series(sorted(tickers)).sample(N_TICKERS, random_state=SEED))
        df = df[df.index.get_level_values(1).isin(keep)]
    print(f"  subsampled to {df.index.get_level_values(1).nunique()} tickers, "
          f"{len(df):,} rows", flush=True)

    ev = pd.read_parquet(EVAL, columns=["eval_mfe_10b", "eval_clean_10b"])
    ev = ev[ev.index.get_level_values(1).isin(set(df.index.get_level_values(1).unique()))]
    df = df.join(ev, how="inner")
    del ev
    df = df.dropna(subset=["eval_mfe_10b"])
    df = df.dropna(subset=feats, how="all")
    print(f"rows={len(df):,}  features={len(feats)}  "
          f"{str(df.index.get_level_values(0).min())[:10]} -> "
          f"{str(df.index.get_level_values(0).max())[:10]}", flush=True)

    cands = cand_fn(df)
    train, val, test = split_with_embargo(df)
    print(f"split: train={len(train):,}  val={len(val):,}  test={len(test):,}  "
          f"(embargo {EMBARGO_BARS} bars each boundary)\n", flush=True)

    Xtr, Xva, Xte = train[feats], val[feats], test[feats]
    print(f"{'candidate':22s} {'learnability':>13s} | {'rho vs MFE_5d':>14s} {'lift MFE_5d':>12s} "
          f"{'rho vs clean':>13s} {'lift clean':>11s}", flush=True)
    print("-" * 100, flush=True)
    results = []
    for label, series in cands.items():
        ytr = series.reindex(train.index)
        yva = series.reindex(val.index)
        ok = ytr.notna()
        if ok.sum() < 5000:
            print(f"{label:22s}   too few labelled rows ({int(ok.sum())})")
            continue
        dtr = xgb.DMatrix(Xtr[ok.to_numpy()], label=ytr[ok.to_numpy()])
        okv = yva.notna()
        dva = xgb.DMatrix(Xva[okv.to_numpy()], label=yva[okv.to_numpy()])
        bst = xgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS,
                        evals=[(dva, "val")], early_stopping_rounds=30, verbose_eval=False)
        pte = bst.predict(xgb.DMatrix(Xte))
        # learnability: how well the model predicts THIS label on test
        yte = series.reindex(test.index)
        learn = within_bar_rho(test.assign(_lab=yte.to_numpy()), pte, "_lab")
        r_mfe = within_bar_rho(test, pte, "eval_mfe_10b")
        l_mfe = decile_lift(test, pte, "eval_mfe_10b")
        r_cln = within_bar_rho(test, pte, "eval_clean_10b")
        l_cln = decile_lift(test, pte, "eval_clean_10b")
        print(f"{label:22s} {learn:13.3f} | {r_mfe:14.3f} {l_mfe:12.3f} "
              f"{r_cln:13.3f} {l_cln:11.3f}", flush=True)
        results.append(dict(module=name, label=label, learnability=learn,
                            rho_mfe_5d=r_mfe, lift_mfe_5d=l_mfe,
                            rho_clean=r_cln, lift_clean=l_cln,
                            best_iter=int(bst.best_iteration or 0)))
    out_rows.extend(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="momentum,htf,meta")
    args = ap.parse_args()
    rows: list[dict] = []
    want = set(args.modules.split(","))
    mom_inputs = ["fwd_max_alpha", "fwd_atr_adj_return", "trend_persistence",
                  "fwd_max_drawdown", "fwd_max_return", "beta_spy_60"]
    htf_inputs = ["fwd_best_high_return", "fwd_worst_low_return", "long_persistence",
                  "atr_pct_14"]
    if "momentum" in want:
        run_module("MOMENTUM  (rank-composite label)",
                   REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet",
                   momentum_candidates, FEATURE_COLUMNS_4H, mom_inputs, rows)
    if "htf" in want:
        run_module("HTF  (raw-sum label, same feature set as momentum)",
                   REPO / "strategies/multi_ticker_swing_htf/data/processed/training_matrix_4h.parquet",
                   htf_candidates, FEATURE_COLUMNS_4H, htf_inputs, rows)
    if "meta" in want:
        meta = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
        import pyarrow.parquet as pq
        cols = list(pq.ParquetFile(meta).schema.names)
        # `timestamp`/`ticker` are the index, not features; the meta matrix lists
        # them in its parquet schema and they must not enter the feature set.
        drop = {"meta_label", "trade_quality", "meta_good", "meta_upside", "fwd_max_return",
                "fwd_max_alpha", "fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return",
                "trend_persistence", "theme", "date", "timestamp", "ticker",
                "__index_level_0__", "__index_level_1__"}
        meta_inputs = ["fwd_max_return", "fwd_max_alpha", "fwd_max_drawdown", "fwd_close_return",
                       "fwd_atr_adj_return", "trend_persistence", "beta_spy_60",
                       "dollar_vol_pctile_252"]
        run_module("META  (raw-value label, stacked over mom/htf OOF)", meta,
                   meta_candidates, [c for c in cols if c not in drop], meta_inputs, rows)
    out = DATA / "label_bakeoff_results.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
