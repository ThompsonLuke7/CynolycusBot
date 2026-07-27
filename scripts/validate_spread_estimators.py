#!/usr/bin/env python
"""Validate research/options_lab/spread_estimators.py against real fills.

Follow-up to Gate G1c (research/options_experiment/01_gate_g1_verdict.md):
fills.py's regression spread model has cross-sectional R^2 ~ 0.10 and
predicts approximately the median spread for every contract, which blocks
any multi-leg-vs-single-leg conclusion (leg count multiplies exactly the
cost that regression cannot tell apart contract-by-contract). This script
answers the only question that matters: do the new per-contract empirical
estimators (Roll / Corwin-Schultz / clustering, in research/options_lab/
spread_estimators.py) actually do better, against real fills, or not.

Population: research/options_experiment/data/g1c_calibration_inputs.csv --
the 401-of-470 G1b-repriceable real trades that ALSO have the liquidity
features (moneyness, DTE, OI, contract volume, underlying ADV) needed to
even evaluate the regression baseline on the same population. Using this
population (rather than the full 470) keeps the before/after comparison on
IDENTICAL trades -- see 02_spread_model.md for why that matters and what it
costs in coverage.

Ground truth (registered in the task, not invented here): the realized
half-spread implied by each real trade against our own modeled mid --

    entry_half_spread_pct = (actual_entry - modeled_entry) / modeled_entry
    exit_half_spread_pct  = (modeled_exit - actual_exit)  / modeled_exit
    realized_half_spread_pct = mean(entry_half_spread_pct, exit_half_spread_pct)

**Caveat stated loudly, not buried:** these residuals also contain our own
mid-pricing model's error (bar-close staleness, timestamp-alignment gap,
etc.), not spread alone. Treat every R^2/Spearman number below as an upper
bound on how well an estimator would score against TRUE bid/ask, not a
clean measurement of it -- there is no clean bid/ask anywhere in this
repo's data (see fills.py's module docstring).

Run: .venv/bin/python scripts/validate_spread_estimators.py [--sample N]
Outputs: research/options_experiment/data/spread_estimator_validation.csv
         printed summary (captured by hand into
         research/options_experiment/02_spread_model.md)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.options_lab import chain_cache, fills, spread_estimators as se  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "research" / "options_experiment" / "data"
CALIB_INPUTS_PATH = DATA_DIR / "g1c_calibration_inputs.csv"
OUT_PATH = DATA_DIR / "spread_estimator_validation.csv"

TRADES_PAD_MIN = 30  # window around entry_time for Roll / clustering, real trade prints
SESSION_START = pd.Timedelta(hours=13, minutes=30)  # 09:30 ET in UTC (fixed offset -- see caveat below)
SESSION_END = pd.Timedelta(hours=20, minutes=0)      # 16:00 ET in UTC

ESTIMATOR_COLS = ["roll_pct", "corwin_schultz_pct", "clustering_pct", "regression_pct", "combined_pct"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_population(sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(CALIB_INPUTS_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    return df


def add_realized_half_spread(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["entry_half_spread_pct"] = (
        (df["actual_entry_price"] - df["modeled_entry_price"]) / df["modeled_entry_price"]
    )
    df["exit_half_spread_pct"] = (
        (df["modeled_exit_price"] - df["actual_exit_price"]) / df["modeled_exit_price"]
    )
    df["realized_half_spread_pct"] = df[["entry_half_spread_pct", "exit_half_spread_pct"]].mean(axis=1)
    return df


# --------------------------------------------------------------------------
# Per-row estimator computation (real network calls, disk-cached by
# chain_cache -- reruns after the first are fast).
# --------------------------------------------------------------------------

def compute_row_estimates(row: pd.Series) -> dict:
    sym = row["symbol"]
    entry_time = row["entry_time"]

    t_start = (entry_time - pd.Timedelta(minutes=TRADES_PAD_MIN)).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_end = (entry_time + pd.Timedelta(minutes=TRADES_PAD_MIN)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        trades = chain_cache.fetch_trades([sym], t_start, t_end)
    except Exception as exc:
        logger.warning("fetch_trades failed for %s: %s", sym, exc)
        trades = pd.DataFrame()
    prices = trades.loc[trades["osi_symbol"] == sym, "p"].dropna().tolist() if not trades.empty else []
    roll_pct = se.roll_effective_spread_pct(prices)
    clustering_pct = se.price_clustering_spread_pct(prices)

    date = entry_time.normalize()
    b_start = (date + SESSION_START).strftime("%Y-%m-%dT%H:%M:%SZ")
    b_end = (date + SESSION_END).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        bars = chain_cache.fetch_bars([sym], "30Min", b_start, b_end)
    except Exception as exc:
        logger.warning("fetch_bars failed for %s: %s", sym, exc)
        bars = pd.DataFrame()
    cs_pct = None
    if not bars.empty:
        sub = bars.loc[bars["osi_symbol"] == sym].copy()
        if not sub.empty:
            sub["t"] = pd.to_datetime(sub["t"], utc=True)
            sub = sub.sort_values("t")
            cs_pct = se.corwin_schultz_spread_pct(sub["h"].tolist(), sub["l"].tolist())

    regression_pct = fills.estimate_spread(
        row["moneyness"], row["dte_at_entry"], row["open_interest"],
        row["contract_volume"], row["underlying_adv"],
    )

    combined = se.combine_spread_estimates(
        roll_pct=roll_pct, corwin_schultz_pct=cs_pct, clustering_pct=clustering_pct,
        regression_pct=regression_pct,
    )

    return {
        "roll_pct": roll_pct,
        "corwin_schultz_pct": cs_pct,
        "clustering_pct": clustering_pct,
        "regression_pct": regression_pct,
        "combined_pct": combined.spread_pct,
        "combined_method": combined.method,
        "n_trades": len(prices),
        "n_bar_windows": 0 if bars.empty else len(bars.loc[bars["osi_symbol"] == sym]),
    }


def run(sample: int | None, seed: int) -> pd.DataFrame:
    df = load_population(sample, seed)
    df = add_realized_half_spread(df)
    logger.info("population: %d trades", len(df))

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 25 == 0:
            logger.info("progress: %d/%d", i, len(df))
        rows.append(compute_row_estimates(row))
    est_df = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), est_df.reset_index(drop=True)], axis=1)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    logger.info("wrote %s (%d rows)", OUT_PATH, len(out))
    return out


# --------------------------------------------------------------------------
# Cross-sectional scoring
# --------------------------------------------------------------------------

def score_estimator(y_true: pd.Series, y_pred_full_spread: pd.Series) -> dict:
    """y_pred_full_spread is the estimator's FULL round-trip spread_pct;
    compare its implied HALF spread (pred/2) against the realized half
    spread ground truth. Cross-sectional R^2 is 1 - SS_res/SS_tot of a
    plain OLS fit of y_true ~ y_pred_half (one predictor, matching how the
    R^2=0.10 regression baseline's own fit quality was reported in G1c)."""
    mask = y_pred_full_spread.notna() & y_true.notna()
    n = int(mask.sum())
    if n < 5:
        return {"n": n, "coverage": mask.mean(), "r2": float("nan"), "spearman": float("nan")}
    yt = y_true[mask].to_numpy()
    yp = (y_pred_full_spread[mask] / 2.0).to_numpy()
    # Simple linear R^2 (slope+intercept OLS of yt on yp), not R^2 of yp==yt
    # directly -- consistent with how G1c's own regression R^2 was computed
    # (fit quality of the predictor, not raw agreement).
    if np.std(yp) == 0:
        r2 = float("nan")
    else:
        slope, intercept = np.polyfit(yp, yt, 1)
        y_hat = slope * yp + intercept
        ss_res = float(np.sum((yt - y_hat) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho, _ = stats.spearmanr(yp, yt)
    return {"n": n, "coverage": float(mask.mean()), "r2": r2, "spearman": float(rho)}


def summarize(out: pd.DataFrame) -> None:
    print("\n=== Cross-sectional fit vs realized half-spread ===")
    print(f"population n={len(out)}")
    for col, label in [
        ("roll_pct", "Roll (1984)"),
        ("corwin_schultz_pct", "Corwin-Schultz (2012)"),
        ("clustering_pct", "Price clustering"),
        ("regression_pct", "Regression baseline (fills.py, R^2~0.10 reported)"),
        ("combined_pct", "Combined ladder"),
    ]:
        s = score_estimator(out["realized_half_spread_pct"], out[col])
        print(
            f"{label:52s} coverage={s['coverage']:.1%} n={s['n']:3d} "
            f"R^2={s['r2']:.4f} spearman={s['spearman']:.4f}"
        )

    print("\ncombined ladder method mix:")
    print(out["combined_method"].value_counts().to_string())

    print("\n=== G1b total P&L, recomputed over this same population ===")
    total_actual = float(((out["actual_exit_price"] - out["actual_entry_price"]) * out["qty"] * 100.0).sum())
    total_mid = float(((out["modeled_exit_price"] - out["modeled_entry_price"]) * out["qty"] * 100.0).sum())

    FLAT_PESSIMISTIC_SPREAD = 0.256  # calibrated median round-trip spread, per 01_gate_g1_verdict.md

    def total_pnl(spread_col: str | None, flat_spread: float | None, assumption: str) -> tuple[float, int]:
        n_priced = 0
        pnl = 0.0
        for _, r in out.iterrows():
            sp = flat_spread if flat_spread is not None else r.get(spread_col)
            if sp is None or not np.isfinite(sp):
                continue
            entry_fill = fills.apply_fill(r["modeled_entry_price"], "buy", float(sp), assumption)
            exit_fill = fills.apply_fill(r["modeled_exit_price"], "sell", float(sp), assumption)
            pnl += (exit_fill - entry_fill) * r["qty"] * 100.0
            n_priced += 1
        return pnl, n_priced

    flat_pess_pnl, flat_pess_n = total_pnl(None, FLAT_PESSIMISTIC_SPREAD, "pessimistic")
    combined_cal_pnl, combined_cal_n = total_pnl("combined_pct", None, "calibrated")
    combined_pess_pnl, combined_pess_n = total_pnl("combined_pct", None, "pessimistic")
    regression_cal_pnl, regression_cal_n = total_pnl("regression_pct", None, "calibrated")

    print(f"actual (real fills):                          ${total_actual:>12,.2f}  (n={len(out)})")
    print(f"mid-priced (optimistic):                       ${total_mid:>12,.2f}  error={total_mid-total_actual:>+12,.2f}")
    print(f"flat 25.6% spread (pessimistic, full-cross):    ${flat_pess_pnl:>12,.2f}  error={flat_pess_pnl-total_actual:>+12,.2f}  (n_priced={flat_pess_n})")
    print(f"regression baseline (calibrated, half-cross):   ${regression_cal_pnl:>12,.2f}  error={regression_cal_pnl-total_actual:>+12,.2f}  (n_priced={regression_cal_n})")
    print(f"new combined ladder (calibrated, half-cross):   ${combined_cal_pnl:>12,.2f}  error={combined_cal_pnl-total_actual:>+12,.2f}  (n_priced={combined_cal_n})")
    print(f"new combined ladder (pessimistic, full-cross):  ${combined_pess_pnl:>12,.2f}  error={combined_pess_pnl-total_actual:>+12,.2f}  (n_priced={combined_pess_n})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="subsample size (default: full g1c_calibration_inputs population)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out = run(args.sample, args.seed)
    summarize(out)


if __name__ == "__main__":
    main()
