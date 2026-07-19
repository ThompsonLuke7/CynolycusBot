"""
Phase 4 of the segmented entry/exit-policy study: is a LEARNED entry-admission
model better than the rules-based frontier?

Stated hypothesis going in (to be disproven): given the confluence null result
and Meta's feature-importance concentration in 1-3 features, a learned
admission policy will NOT robustly beat the simple mom_xs_rank rule.

Setup:
  - Trades = chained entries under the FIXED id4 exit (the ship candidate),
    per module stream, same val/test split as the whole thread.
  - Features = entry-time-safe meta_ranker_matrix columns (same exclusion
    list as exit_policy_entry_quality.py). No forward-looking columns.
  - Model = XGBRegressor with FIXED small hyperparameters (no tuning at all,
    stated up front: max_depth=3, n_estimators=200, lr=0.05, subsample=0.8).
  - Embargo: training uses val trades entered BEFORE 2025-12-10 only, so no
    training label's outcome window (53 bars ~ 22 trading days) can overlap
    the test period, and an internal early/late val split checks
    generalization before the single frozen test read.
  - Admission rule: keep trades with predicted ret above the TRAIN 20th
    percentile of predictions (matches the incumbent skip-bottom-20% rule's
    keep fraction). Comparators, same keep protocol:
      (i) no filter, (ii) mom_xs_rank > val 20th pct (incumbent rule).
  - Trades with missing features are admitted by default under BOTH filters.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_learned_admission.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/capstone"))
from exit_policy_segmentation import (  # noqa: E402
    VAL_START, VAL_END, TEST_END, OUT_DIR, POLICIES, all_trades, member_from_stream,
)

ID4 = POLICIES["id4"]
EMBARGO_TRAIN_END = pd.Timestamp("2025-12-10", tz="UTC")
INTERNAL_SPLIT = pd.Timestamp("2025-10-15", tz="UTC")
KEEP_FRAC = 0.80

LEAK_COLS = {"meta_label", "fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return",
             "trend_persistence", "trade_quality", "meta_good", "meta_upside",
             "fwd_max_return", "fwd_max_alpha"}
NON_FEATURES = {"theme", "date", "sector_id", "market_cap_bucket", "asset_type", "is_etf"}


def load_features() -> tuple[pd.DataFrame, list[str]]:
    matrix = pd.read_parquet(REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet")
    feat_cols = [c for c in matrix.columns if c not in LEAK_COLS and c not in NON_FEATURES
                 and pd.api.types.is_numeric_dtype(matrix[c])]
    m = matrix[feat_cols].reset_index()
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    return m, feat_cols


def trades_for(module: str, start, end) -> pd.DataFrame:
    stream = pd.read_parquet(OUT_DIR / f"stream_{module}.parquet")
    tr = all_trades(member_from_stream(stream, start, end), **ID4)
    tr = tr.rename(columns={"entry_ts": "timestamp"})
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
    return tr


def summarize(name: str, tr: pd.DataFrame, keep: pd.Series) -> dict:
    kept, skipped = tr[keep], tr[~keep]
    d = dict(filter=name, n_base=len(tr), n_kept=len(kept),
             mean_base=tr["ret"].mean(), mean_kept=kept["ret"].mean(),
             win_base=(tr["ret"] > 0).mean(), win_kept=(kept["ret"] > 0).mean(),
             mean_skipped=skipped["ret"].mean() if len(skipped) else np.nan,
             total_base=tr["ret"].sum(), total_kept=kept["ret"].sum())
    print(f"  {name:22s} kept {d['n_kept']}/{d['n_base']}  mean {d['mean_base']:.4f}->{d['mean_kept']:.4f}  "
          f"win {d['win_base']:.3f}->{d['win_kept']:.3f}  skipped-mean {d['mean_skipped']:.4f}")
    return d


def main() -> None:
    from xgboost import XGBRegressor

    feats, feat_cols = load_features()
    cohorts = pd.read_csv(OUT_DIR / "ticker_tail_cohorts.csv")[["ticker", "tail_cohort"]]
    rows = []

    for module in ("momentum", "htf", "meta"):
        print(f"\n===== {module} (exit fixed = id4) =====")
        val_tr = trades_for(module, VAL_START, VAL_END).merge(
            feats, on=["timestamp", "ticker"], how="left").merge(cohorts, on="ticker", how="left")
        test_tr = trades_for(module, VAL_END, TEST_END).merge(
            feats, on=["timestamp", "ticker"], how="left").merge(cohorts, on="ticker", how="left")
        has_f_val = val_tr[feat_cols].notna().any(axis=1)
        has_f_test = test_tr[feat_cols].notna().any(axis=1)
        print(f"val trades {len(val_tr)} ({has_f_val.mean():.0%} with features), "
              f"test trades {len(test_tr)} ({has_f_test.mean():.0%} with features)")

        train = val_tr[(val_tr["timestamp"] < EMBARGO_TRAIN_END) & has_f_val]

        model = XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=7,
                             n_jobs=4, verbosity=0)

        # ---- internal generalization check inside val (early -> late), before test
        tr_early = train[train["timestamp"] < INTERNAL_SPLIT]
        tr_late = train[train["timestamp"] >= INTERNAL_SPLIT]
        if len(tr_early) > 100 and len(tr_late) > 50:
            m0 = XGBRegressor(**model.get_params())
            m0.fit(tr_early[feat_cols], tr_early["ret"])
            p_late = m0.predict(tr_late[feat_cols])
            rho = pd.Series(p_late).corr(tr_late["ret"].reset_index(drop=True), method="spearman")
            print(f"internal val check (train<{INTERNAL_SPLIT.date()}, predict later val): "
                  f"Spearman(pred, ret) = {rho:.3f} (n={len(tr_late)})")

        # ---- final: train on embargoed val, single frozen test read
        model.fit(train[feat_cols], train["ret"])
        thr_pred = float(np.quantile(model.predict(train[feat_cols]), 1 - KEEP_FRAC))
        mom_thr = float(train["mom_xs_rank"].quantile(1 - KEEP_FRAC))

        imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print("top-8 learned-model gain features:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(8).items()))

        for wname, tr, hasf in (("val", val_tr, has_f_val), ("test", test_tr, has_f_test)):
            print(f" -- {wname} --")
            pred = pd.Series(np.nan, index=tr.index)
            pred[hasf] = model.predict(tr.loc[hasf, feat_cols])
            keep_ml = (pred >= thr_pred) | ~hasf
            keep_mom = (tr["mom_xs_rank"] > mom_thr) | tr["mom_xs_rank"].isna()
            for nm, keep in (("no_filter", pd.Series(True, index=tr.index)),
                             ("mom_xs_rank_rule", keep_mom), ("xgb_admission", keep_ml)):
                d = summarize(nm, tr, keep)
                d.update(module=module, window=wname)
                rows.append(d)
            # per-segment: does the ML filter beat the rule inside any cohort?
            for nm, keep in (("mom_xs_rank_rule", keep_mom), ("xgb_admission", keep_ml)):
                seg = (tr.assign(kept=keep).groupby("tail_cohort", observed=True)
                       .apply(lambda g: pd.Series(dict(
                           n=len(g), kept_frac=g["kept"].mean(),
                           mean_kept=g.loc[g["kept"], "ret"].mean(),
                           mean_all=g["ret"].mean())), include_groups=False))
                print(f"    {nm} by cohort:\n" + seg.round(4).to_string())

    pd.DataFrame(rows).to_csv(OUT_DIR / "phase4_learned_admission.csv", index=False)
    print(f"\nsaved {OUT_DIR / 'phase4_learned_admission.csv'}")


if __name__ == "__main__":
    main()
