"""Stage 1 of the 4H feature study: training-free feature screen.

Plan: docs/superpowers/plans/2026-09-25-4h-feature-study-plan.md

For every deployed feature and every candidate block feature, per decision bar:
  * univariate IC   -- Spearman(feature, label), via Gaussian-rank Pearson;
  * incremental IC  -- partial correlation after projecting BOTH the feature and
    the label on the controls (deployed OOF score, lagged daily ATR%, 4H ATR%,
    60-bar beta, dollar-volume percentile, log price). A feature that only
    restates volatility or the existing model scores ~0 here.
Aggregated as the mean over bars, with a week-block bootstrap CI and p-value,
per-fold sign consistency, and BH-FDR across the whole family.

Decision bars: the LAST 4H bar of each session (it contains the close). The
executable outcome enters at the NEXT session's open and exits at the close
h sessions later; the corporate-action guard drops windows with a non-organic
split/recap gap. Reads production parquets READ-ONLY. Writes the cached panel to
Data/research/feature_screen_2026-09/ and results to ablation/results/feature_screen/.

Scope/limits: the OOF window is 2022-11..2026-05 (the deployed walk-forward
folds). Rows after 2026-05-14 are NOT in the matrix and are reserved as the
untouched final test for Stage 2/3. The daily cache is survivor-shaped
(delisted names missing), which flatters levels more than within-bar ranks.

Usage:
    .venv/bin/python -m strategies.momentum_expansion.ablation.run_feature_screen --model momentum
    .venv/bin/python -m strategies.momentum_expansion.ablation.run_feature_screen --model htf
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import ndtri

from core.corporate_actions import suspect_sessions
from strategies.momentum_expansion.ablation.bootstrap import week_block_bootstrap_ci, week_of
from strategies.momentum_expansion.ablation.candidate_features import CANDIDATE_BLOCKS, daily_candidate_features
from strategies.momentum_expansion.ablation.label_arms import build_label_arms
from strategies.momentum_expansion.ablation.screen import apply_bh_fdr

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
PANEL_DIR = REPO / "Data" / "research" / "feature_screen_2026-09"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "feature_screen"
DAILY_DIR = REPO / "Data" / "shared" / "bars" / "1d"

MODELS = {
    "momentum": {
        "oof": REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet",
        "matrix": REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet",
        "manifest": REPO / "strategies/momentum_expansion/models/expansion_v1/feature_manifest.json",
        "mfe": "fwd_max_return",
    },
    "htf": {
        "oof": REPO / "strategies/multi_ticker_swing_htf/models/oof_preds.parquet",
        "matrix": REPO / "strategies/multi_ticker_swing_htf/data/processed/training_matrix_4h.parquet",
        "manifest": REPO / "strategies/multi_ticker_swing_htf/models/feature_manifest.json",
        "mfe": "fwd_best_high_return",
    },
}
HORIZONS = (5, 10)
PRIMARY_H = 10
MIN_NAMES_PER_BAR = 200
CONTROL_COLS = ["score", "daily_atr_pct", "atr_pct_14", "beta_spy_60", "dollar_vol_pctile_252", "log_price"]
# Folds = the deployed walk-forward 6-month test windows (WALK_FORWARD_CONFIG).
FOLD_EDGES = pd.to_datetime(
    ["2022-11-01", "2023-05-01", "2023-11-01", "2024-05-01", "2024-11-01", "2025-05-01",
     "2025-11-01", "2026-06-01"], utc=True,
)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def _daily_block(ticker: str) -> pd.DataFrame | None:
    """Candidate features + executable forward returns for one ticker, keyed by
    NY session date. Row D: features through D-1; entry open D+1; exit close D+h."""
    path = DAILY_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if len(d) < 60:
        return None
    feats = daily_candidate_features(d)
    out = feats.copy()
    out["date"] = d["timestamp"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    out["close_d"] = d["close"]
    # Liquidity for subset buckets only; use per-date RANKS (IEX->SIP volume switch).
    out["adv20_d"] = (d["close"] * d["volume"]).rolling(20, min_periods=10).mean()
    for h, ret in exec_forward_returns(d, HORIZONS).items():
        out[f"fret_{h}"] = ret
    out["ticker"] = ticker
    return out


def exec_forward_returns(d: pd.DataFrame, horizons) -> dict[int, pd.Series]:
    """{h: return from the NEXT session's open to the close h sessions ahead}.

    Row D enters at open D+1 and exits at close D+h, so prices span rows
    D+1..D+h. A non-organic corporate-action flag at row p is a gap between
    p-1 and p; one at D+2..D+h falls inside the window and voids it (a flag at
    D+1 precedes the entry and does not).
    """
    flags = suspect_sessions(d)
    flag_pos = np.zeros(len(d), dtype=bool)
    if not flags.empty:
        flags = flags[~flags["organic"].astype(bool)]
        flag_pos[flags["idx"].astype(int).to_numpy()] = True
    entry = d["open"].shift(-1)
    out: dict[int, pd.Series] = {}
    for h in horizons:
        ret = d["close"].shift(-h) / entry - 1.0
        if h > 1:
            bad = pd.Series(flag_pos).rolling(h - 1, min_periods=1).max().shift(-h).fillna(0).astype(bool)
            ret = ret.mask(bad.to_numpy())
        out[int(h)] = ret
    return out


def assert_oof_unfiltered(oof_ts: pd.Series, matrix_ts: pd.Series, *, model: str,
                          min_coverage: float = 0.90) -> float:
    """Fail fast if the OOF holds only a subset of the matrix rows in its window.

    The deployed HTF OOF keeps only rows whose regression target
    (``htf_swing_score``) exists -- rows inside swing zones defined by pivots that
    need 3 FUTURE bars. That sample is selected on the future: recent losers
    "bounce" by construction (ret_3 IC -0.10 there vs ~0 on all rows). The live
    model scores every row, so no screen or backtest may run on such an OOF.
    """
    lo, hi = oof_ts.min(), oof_ts.max()
    in_window = int(((matrix_ts >= lo) & (matrix_ts <= hi)).sum())
    coverage = len(oof_ts) / max(in_window, 1)
    if coverage < min_coverage:
        raise ValueError(
            f"{model} OOF covers {coverage:.1%} of matrix rows in {lo.date()}..{hi.date()}: "
            "it is a filtered sample (e.g. rows kept only where a future-defined target "
            "exists) and would bias every IC/top-k result. Regenerate OOF over all rows."
        )
    return coverage


def build_panel(model: str, *, workers: int = 5, refresh: bool = False) -> pd.DataFrame:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    cache = PANEL_DIR / f"panel_{model}.parquet"
    if cache.exists() and not refresh:
        logger.info("Loading cached panel %s", cache)
        return pd.read_parquet(cache)
    spec = MODELS[model]
    features = json.loads(spec["manifest"].read_text())["feature_columns"]

    oof = pd.read_parquet(spec["oof"], columns=["score", spec["mfe"]]).reset_index()
    oof["timestamp"] = pd.to_datetime(oof["timestamp"], utc=True)
    mat_idx = pd.read_parquet(spec["matrix"], columns=[]).reset_index()
    assert_oof_unfiltered(oof["timestamp"], pd.to_datetime(mat_idx["timestamp"], utc=True), model=model)
    del mat_idx
    oof["date"] = oof["timestamp"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    # last 4H bar of each session per ticker
    oof = oof.sort_values("timestamp").groupby(["ticker", "date"], as_index=False).tail(1)
    logger.info("OOF decision rows: %d (%d sessions)", len(oof), oof["date"].nunique())

    have = set(pq.ParquetFile(spec["matrix"]).schema_arrow.names)
    need = sorted((set(features) | {"daily_atr_pct", "atr_pct_14", "beta_spy_60", "dollar_vol_pctile_252",
                                    "low_price_flag", "realized_vol_20", "regime_spy_trend", "regime_vix_z",
                                    "market_cap_bucket"}) & have)
    missing = sorted(set(features) - have)
    if missing:
        raise ValueError(f"{model} matrix lacks deployed features: {missing}")
    mat = pd.read_parquet(spec["matrix"], columns=need).reset_index()
    mat["timestamp"] = pd.to_datetime(mat["timestamp"], utc=True)
    panel = oof.merge(mat, on=["timestamp", "ticker"], how="inner")
    del mat
    logger.info("OOF x matrix rows: %d", len(panel))

    tickers = sorted(panel["ticker"].unique())
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        blocks = [b for b in ex.map(_daily_block, tickers, chunksize=16) if b is not None]
    daily = pd.concat(blocks, ignore_index=True)
    logger.info("Daily blocks for %d/%d tickers in %.0fs", len(blocks), len(tickers), time.time() - t0)
    panel = panel.merge(daily, on=["ticker", "date"], how="left")

    spy = _daily_block("SPY")[["date"] + [f"fret_{h}" for h in HORIZONS]]
    spy = spy.rename(columns={f"fret_{h}": f"spy_fret_{h}" for h in HORIZONS})
    panel = panel.merge(spy, on="date", how="left")
    for h in HORIZONS:
        panel[f"fxret_{h}"] = panel[f"fret_{h}"] - panel[f"spy_fret_{h}"]
    panel["log_price"] = np.log(panel["close_d"].where(panel["close_d"] > 0))

    counts = panel.groupby("timestamp")["ticker"].transform("size")
    panel = panel[counts >= MIN_NAMES_PER_BAR].reset_index(drop=True)
    fcols = panel.select_dtypes("float64").columns
    panel[fcols] = panel[fcols].astype("float32")
    panel.attrs["features"] = features
    panel.to_parquet(cache)
    (PANEL_DIR / f"panel_{model}_features.json").write_text(json.dumps(features))
    logger.info("Panel %s: %d rows, %d bars, %d tickers -> %s", model, len(panel),
                panel["timestamp"].nunique(), panel["ticker"].nunique(), cache)
    return panel


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

def _gauss_rank(x: np.ndarray) -> np.ndarray:
    """Column-wise Gaussian rank (NaN stays NaN)."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = ~np.isnan(col)
        k = int(ok.sum())
        if k < 3:
            continue
        r = pd.Series(col[ok]).rank(method="average").to_numpy()
        out[ok, j] = ndtri((r - 0.5) / k)
    return out


def _col_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson of each column of ``a`` with vector ``b`` (both centred inside)."""
    a = a - a.mean(axis=0)
    b = b - b.mean()
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b[:, None]).sum(axis=0) / den


def per_bar_ics(panel: pd.DataFrame, feats: list[str], labels: list[str], controls: list[str]) -> dict:
    """{(label, kind): DataFrame[bars x feats]} for kind in {uni, inc}, plus coverage."""
    res = {(lab, kind): [] for lab in labels for kind in ("uni", "inc")}
    cover, bar_index = [], []
    for ts, g in panel.groupby("timestamp", sort=True):
        g = g.dropna(subset=labels + controls)
        if len(g) < MIN_NAMES_PER_BAR:
            continue
        F = _gauss_rank(g[feats].to_numpy(np.float64))
        cover.append((~np.isnan(F)).mean(axis=0))
        F = np.nan_to_num(F, nan=0.0)  # within-bar median impute; coverage reported
        C = _gauss_rank(g[controls].to_numpy(np.float64))
        C = np.column_stack([np.ones(len(g)), C])
        # projection residual maker
        Q, _ = np.linalg.qr(C)
        F_res = F - Q @ (Q.T @ F)
        Y = _gauss_rank(g[labels].to_numpy(np.float64))
        Y_res = Y - Q @ (Q.T @ Y)
        for i, lab in enumerate(labels):
            res[(lab, "uni")].append(_col_corr(F, Y[:, i]))
            res[(lab, "inc")].append(_col_corr(F_res, Y_res[:, i]))
        bar_index.append(ts)
    idx = pd.DatetimeIndex(bar_index)
    out = {k: pd.DataFrame(np.vstack(v), index=idx, columns=feats) for k, v in res.items()}
    out["coverage"] = pd.DataFrame(np.vstack(cover), index=idx, columns=feats)
    return out


def _boot_summary(ic: pd.Series, n_boot: int, seed: int = 42) -> dict:
    ic = ic.dropna()
    if len(ic) < 20:
        # Bar-constant columns (calendar, market regime) have no cross-sectional
        # variance, so their per-bar IC is undefined -- structurally, not a null.
        return {"mean_ic": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
                "n_bars": len(ic), "n_weeks": 0, "folds_same_sign": 0, "n_folds": 0, "worst_fold": np.nan}
    weeks = week_of(pd.Series(ic.index))
    vals = pd.Series(ic.to_numpy())
    b = week_block_bootstrap_ci(vals, weeks.reset_index(drop=True), n_boot=n_boot, seed=seed)
    # two-sided bootstrap p (same construction as screen.rank_ic_with_bootstrap)
    rng = np.random.default_rng(seed + 1)
    by_week = {w: g.to_numpy() for w, g in vals.groupby(weeks.to_numpy())}
    keys = np.array(list(by_week))
    boots = np.array([np.mean(np.concatenate([by_week[w] for w in rng.choice(keys, len(keys))]))
                      for _ in range(n_boot)])
    p = float(min(1.0, 2 * min((boots >= 0).mean(), (boots <= 0).mean())))
    folds = pd.cut(ic.index, FOLD_EDGES, right=False)
    fold_means = ic.groupby(folds, observed=True).mean()
    sign = np.sign(b["point"])
    return {"mean_ic": b["point"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "p_value": p,
            "n_bars": len(ic), "n_weeks": b["n_weeks"],
            "folds_same_sign": int((np.sign(fold_means) == sign).sum()), "n_folds": int(len(fold_means)),
            "worst_fold": float(fold_means.min() if sign > 0 else fold_means.max())}


def summarize(ics: dict, feats: list[str], labels: list[str], block_of: dict[str, str], n_boot: int) -> pd.DataFrame:
    rows = []
    cov = ics["coverage"].mean()
    for lab in labels:
        for kind in ("uni", "inc"):
            mat = ics[(lab, kind)]
            for f in feats:
                s = _boot_summary(mat[f], n_boot)
                s.update({"feature": f, "block": block_of.get(f, "deployed"), "label": lab, "kind": kind,
                          "coverage": float(cov[f])})
                rows.append(s)
    return apply_bh_fdr(rows)


def verdict(tab: pd.DataFrame, label: str = "L2") -> pd.DataFrame:
    """keep / candidate / drop per feature from the incremental IC on ``label``.

    keep/candidate needs BH-FDR survival AND sign agreement in >=5 of 7 folds;
    'keep' additionally needs |IC| >= 0.005 (below that it cannot move top-k).
    """
    inc = tab[(tab["label"] == label) & (tab["kind"] == "inc")].set_index("feature")
    uni = tab[(tab["label"] == label) & (tab["kind"] == "uni")].set_index("feature")
    out = inc[["block", "mean_ic", "ci_lo", "ci_hi", "q_fdr", "folds_same_sign", "n_folds", "coverage"]].copy()
    out.columns = ["block", "inc_ic", "inc_lo", "inc_hi", "inc_q", "inc_folds", "n_folds", "coverage"]
    out["uni_ic"] = uni["mean_ic"]
    out["uni_q"] = uni["q_fdr"]
    robust = (out["inc_q"] <= 0.10) & (out["inc_folds"] >= 5)
    out["verdict"] = np.where(robust & (out["inc_ic"].abs() >= 0.005), "keep",
                              np.where(robust, "candidate", "drop"))
    out.loc[out["inc_ic"].isna(), "verdict"] = "undefined_bar_constant"
    for f, why in LEAK_RISK.items():
        if f in out.index:
            out.loc[f, "verdict"] = f"leak_risk:{why}"
    return out.sort_values("inc_ic", key=np.abs, ascending=False)


# Features whose screen result cannot be trusted on this universe/data.
LEAK_RISK = {
    "market_cap_bucket": "static_current_snapshot",   # one 2026 value stamped on all history
    "low_price_flag": "survivor_universe",            # dead sub-$5 names are absent from the cache
}


def beyond_deployed(panel: pd.DataFrame, cand: list[str], v: pd.DataFrame, label: str,
                    n_boot: int) -> pd.DataFrame:
    """Incremental IC of candidate features after ALSO controlling for every
    deployed feature that earned 'keep' -- i.e. what a candidate adds that the
    deployed set already carries (the screen's first pass controls only for the
    score + vol, and the score ignores trend features because L0 rewarded vol)."""
    keeps = [f for f in v.index[(v["verdict"] == "keep") & (v["block"] == "deployed")]]
    ctrls = CONTROL_COLS + [f for f in keeps if f not in CONTROL_COLS]
    ics = per_bar_ics(panel, cand, [label], ctrls)
    rows = []
    for f in cand:
        srow = _boot_summary(ics[(label, "inc")][f], n_boot)
        srow.update({"feature": f, "n_controls": len(ctrls)})
        rows.append(srow)
    out = apply_bh_fdr(rows).set_index("feature")
    out["verdict"] = np.where((out["q_fdr"] <= 0.10) & (out["folds_same_sign"] >= 5)
                              & (out["mean_ic"].abs() >= 0.005), "adds", "redundant_or_null")
    return out.sort_values("mean_ic", key=np.abs, ascending=False)


def run(model: str, *, n_boot: int = 300, refresh: bool = False) -> pd.DataFrame:
    out_dir = RESULTS_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = build_panel(model, refresh=refresh)
    deployed = json.loads((PANEL_DIR / f"panel_{model}_features.json").read_text())
    cand = [f for blk in CANDIDATE_BLOCKS.values() for f in blk]
    block_of = {f: b for b, fs in CANDIDATE_BLOCKS.items() for f in fs}
    feats = [f for f in deployed if f in panel.columns] + cand
    feats = [f for f in feats if panel[f].notna().any()]

    arms = build_label_arms(panel, mfe_col=MODELS[model]["mfe"], ret_col=f"fxret_{PRIMARY_H}")
    panel = pd.concat([panel, arms], axis=1)
    labels = ["L0", "L1", "L2", "R"]
    logger.info("Screening %d features x %d labels on %d rows", len(feats), len(labels), len(panel))
    t0 = time.time()
    ics = per_bar_ics(panel, feats, labels, CONTROL_COLS)
    logger.info("Per-bar ICs over %d bars in %.0fs", len(ics["coverage"]), time.time() - t0)
    tab = summarize(ics, feats, labels, block_of, n_boot)
    tab.to_csv(out_dir / "screen_all.csv", index=False)
    for lab in ("L2", "L1"):
        v = verdict(tab, lab)
        v.to_csv(out_dir / f"verdict_{lab}.csv")
        beyond_deployed(panel, cand, v, lab, n_boot).to_csv(out_dir / f"candidates_beyond_deployed_{lab}.csv")
    # per-bar series of the control score for reference / later stages
    ics[("L2", "uni")].to_parquet(PANEL_DIR / f"ic_L2_uni_{model}.parquet")
    logger.info("Wrote %s", out_dir)
    return tab


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="momentum")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--refresh", action="store_true", help="rebuild the cached panel")
    a = ap.parse_args()
    run(a.model, n_boot=a.n_boot, refresh=a.refresh)


if __name__ == "__main__":
    main()
