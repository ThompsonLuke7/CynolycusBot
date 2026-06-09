"""
Build the Meta Ranker training matrix + labels.

Unit:   (ticker, 4H bar).  Horizon: momentum's ~25x4H (~10 trading day) window.
Label:  drawdown-penalized forward return (risk-adjusted).

Stacked-generalization design: the base-model signals used as features are the
models' OUT-OF-FOLD predictions (leakage-free), so the meta model never sees an
in-sample base prediction. Daily theme context is joined AS-OF the PRIOR trading
day so an intraday 4H bar cannot see that day's end-of-day theme aggregates.

Feature groups:
  base scores   - momentum / HTF / theme(ML) out-of-fold scores
  theme context - rotation state for the ticker's theme (rank, dRank, regime,
                  breadth, persistence, leader concentration) [prior day]
  cross-context - cross-sectional ranks, signal agreement, within-theme rank,
                  theme crowding
  ticker meta   - sector, cap bucket, liquidity, beta, asset type, is_etf
  regime        - SPY trend / ret, VIX z / high
  stubs         - catalyst_score, options_score (NaN placeholders for v2)

Outputs: meta_context/meta_ranker/{meta_ranker_matrix.parquet, manifest.json}
Run:     python meta_context/build_meta_ranker_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "meta_context/meta_ranker"

MOM_OOF = REPO / "momentum_expansion/data/training_import/bundle/oof_preds.parquet"
HTF_OOF = REPO / "multi_ticker_swing_htf/data/bundle/oof_preds.parquet"
THEME_OOF = REPO / "theme_expansion/models/bundle/oof_preds.parquet"
THEME_SCORES = REPO / "theme_expansion/outputs/theme_scores.parquet"
THEME_MAP = REPO / "theme_expansion/data/theme_map_v4.csv"
FEATURES_4H = REPO / "momentum_expansion/data/processed/features_4h.parquet"

DRAWDOWN_PENALTY = 1.0   # meta_label = fwd_close_return - penalty * fwd_max_drawdown

META_COLS = ["sector_id", "market_cap_bucket", "asset_type", "is_etf",
             "beta_spy_60", "dollar_vol_pctile_252",
             "regime_spy_trend", "regime_spy_ret_20", "regime_vix_z", "regime_vix_high"]
THEME_CTX = {"theme_score": "theme_ctx_score", "theme_rank": "theme_rank",
             "theme_rank_change_1d": "theme_rank_chg_1d", "theme_rank_change_5d": "theme_rank_chg_5d",
             "theme_regime_rank": "theme_regime_rank", "theme_heat_rank": "theme_heat_rank",
             "theme_persistence_score": "theme_persistence", "theme_breadth": "theme_breadth",
             "theme_vs_spy_20d": "theme_vs_spy_20d", "leader_concentration": "theme_leader_conc",
             "theme_num_constituents": "theme_n_constituents"}


def _norm_date(ts: pd.Series) -> pd.Series:
    """tz-aware UTC 4H timestamp -> naive midnight date (matches theme daily)."""
    return pd.to_datetime(ts, utc=True).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()


def _asof_prior_day(spine: pd.DataFrame, right: pd.DataFrame, by="theme") -> pd.DataFrame:
    """merge_asof on 'date' within each theme, strictly prior day (no lookahead)."""
    s = spine.sort_values("date").reset_index()  # keep original index in a column
    r = right.sort_values("date")
    merged = pd.merge_asof(s, r, on="date", by=by, direction="backward", allow_exact_matches=False)
    return merged.set_index("index").sort_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- spine: momentum OOF (score + forward outcome columns for the label)
    print("loading momentum OOF (spine) ...")
    mom = pd.read_parquet(MOM_OOF).reset_index()
    mom = mom.rename(columns={mom.columns[0]: "timestamp"})
    mom["timestamp"] = pd.to_datetime(mom["timestamp"], utc=True)
    # base OOF files carry a few duplicate (timestamp,ticker) rows at walk-forward
    # fold boundaries; keep the latest fold's prediction so the spine is unique.
    mom = mom.drop_duplicates(subset=["timestamp", "ticker"], keep="last")
    spine = mom[["timestamp", "ticker", "score", "fwd_close_return",
                 "fwd_max_drawdown", "fwd_atr_adj_return", "trend_persistence"]].copy()
    spine = spine.rename(columns={"score": "mom_score"})
    spine["date"] = _norm_date(spine["timestamp"])
    print(f"  spine rows: {len(spine):,}  bars: {spine['timestamp'].nunique():,}  "
          f"range {spine['timestamp'].min()} -> {spine['timestamp'].max()}")

    # ---- HTF OOF score
    print("joining HTF OOF score ...")
    htf = pd.read_parquet(HTF_OOF, columns=["score"]).reset_index()
    htf = htf.rename(columns={htf.columns[0]: "timestamp", "score": "htf_score"})
    htf["timestamp"] = pd.to_datetime(htf["timestamp"], utc=True)
    htf = htf.drop_duplicates(subset=["timestamp", "ticker"], keep="last")
    spine = spine.merge(htf[["timestamp", "ticker", "htf_score"]], on=["timestamp", "ticker"], how="left")

    # ---- theme membership (primary theme = theme_1)
    tmap = pd.read_csv(THEME_MAP)[["ticker", "theme_1"]].rename(columns={"theme_1": "theme"})
    spine = spine.merge(tmap, on="ticker", how="left")

    # ---- theme ML OOF score (prior day, by theme)
    print("joining theme ML OOF score (prior day) ...")
    th_oof = pd.read_parquet(THEME_OOF)[["date", "theme", "score"]].copy()
    th_oof["date"] = pd.to_datetime(th_oof["date"]).dt.normalize()
    th_oof = th_oof.rename(columns={"score": "theme_ml_score"})
    spine = _asof_prior_day(spine, th_oof[["date", "theme", "theme_ml_score"]])

    # ---- theme context (prior day, by theme)
    print("joining theme rotation context (prior day) ...")
    tsc = pd.read_parquet(THEME_SCORES, columns=["date", "theme"] + list(THEME_CTX))
    tsc["date"] = pd.to_datetime(tsc["date"]).dt.normalize()
    tsc = tsc.rename(columns=THEME_CTX)
    # raw daily theme rank swings ~7-8 positions/day (of ~85) -> too noisy as a
    # feature. Add a 10-day smoothed rank (keep the raw + change cols too).
    tsc = tsc.sort_values(["theme", "date"])
    tsc["theme_rank_smooth_10"] = tsc.groupby("theme")["theme_rank"].transform(
        lambda s: s.rolling(10, min_periods=3).mean())
    spine = _asof_prior_day(spine, tsc)

    # ---- ticker meta + regime (point-in-time, exact 4H join)
    print("joining ticker meta + regime from features_4h ...")
    cut = spine["timestamp"].min().to_pydatetime()
    f4 = pd.read_parquet(FEATURES_4H, columns=META_COLS,
                         filters=[("timestamp", ">=", cut)])
    f4 = f4.reset_index()
    f4 = f4.rename(columns={f4.columns[0]: "timestamp"}) if "timestamp" not in f4.columns else f4
    f4["timestamp"] = pd.to_datetime(f4["timestamp"], utc=True)
    spine = spine.merge(f4, on=["timestamp", "ticker"], how="left")

    # ---- cross-context (per 4H bar)
    print("deriving cross-context features ...")
    g = spine.groupby("timestamp")
    spine["mom_xs_rank"] = g["mom_score"].rank(pct=True)
    spine["htf_xs_rank"] = g["htf_score"].rank(pct=True)
    spine["signal_agreement"] = spine["mom_xs_rank"] * spine["htf_xs_rank"]
    # within-theme momentum rank + crowding (top-quintile share) per bar+theme
    gt = spine.groupby(["timestamp", "theme"])
    spine["within_theme_mom_rank"] = gt["mom_score"].rank(pct=True)
    spine["_hot"] = (spine["mom_xs_rank"] > 0.8).astype(float)
    spine["theme_crowding_frac"] = gt["_hot"].transform("mean")
    spine = spine.drop(columns=["_hot"])

    # ---- stubs for v2 modules
    spine["catalyst_score"] = np.nan
    spine["options_score"] = np.nan

    # ---- label: drawdown-penalized forward return (risk-adjusted)
    spine["meta_label"] = spine["fwd_close_return"] - DRAWDOWN_PENALTY * spine["fwd_max_drawdown"]

    # ---- assemble + manifest
    id_cols = ["timestamp", "ticker", "theme", "date"]
    label_cols = ["meta_label", "fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return", "trend_persistence"]
    base_scores = ["mom_score", "htf_score", "theme_ml_score"]
    theme_ctx = list(THEME_CTX.values()) + ["theme_rank_smooth_10"]
    cross = ["mom_xs_rank", "htf_xs_rank", "signal_agreement", "within_theme_mom_rank", "theme_crowding_frac"]
    stubs = ["catalyst_score", "options_score"]
    feature_cols = base_scores + theme_ctx + cross + META_COLS + stubs

    out = spine[id_cols + feature_cols + label_cols].copy()
    out = out.dropna(subset=["meta_label"]).set_index(["timestamp", "ticker"]).sort_index()
    out.to_parquet(OUT / "meta_ranker_matrix.parquet")

    cov = {c: float(out[c].notna().mean()) for c in feature_cols}
    manifest = {
        "unit": "(ticker, 4H bar)",
        "horizon": "momentum 25x4H (~10 trading days)",
        "label_column": "meta_label",
        "label_definition": "fwd_close_return - %.1f * fwd_max_drawdown" % DRAWDOWN_PENALTY,
        "id_columns": id_cols,
        "feature_columns": feature_cols,
        "base_score_columns": base_scores,
        "theme_context_columns": theme_ctx,
        "cross_context_columns": cross,
        "ticker_meta_columns": META_COLS,
        "stub_columns": stubs,
        "label_columns": label_cols,
        "leakage_controls": [
            "base scores are out-of-fold (stacked generalization)",
            "daily theme context joined as-of PRIOR trading day",
            "ticker meta/regime are point-in-time 4H features",
        ],
        "n_rows": int(len(out)),
        "n_bars": int(out.index.get_level_values("timestamp").nunique()),
        "n_tickers": int(out.index.get_level_values("ticker").nunique()),
        "date_min": str(out.index.get_level_values("timestamp").min()),
        "date_max": str(out.index.get_level_values("timestamp").max()),
        "feature_coverage_non_null": cov,
        "categorical_columns": ["sector_id", "market_cap_bucket", "asset_type"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nwrote {OUT/'meta_ranker_matrix.parquet'}  rows={len(out):,}  features={len(feature_cols)}")
    print("coverage (non-null):")
    for c in feature_cols:
        print(f"  {c:24s} {cov[c]:6.1%}")
    print(f"\nlabel meta_label: mean {out['meta_label'].mean():+.4f}  std {out['meta_label'].std():.4f}")


if __name__ == "__main__":
    main()
