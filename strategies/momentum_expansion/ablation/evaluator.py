"""The frozen evaluator (year-two checklist item 0.3).

One locked way to judge a score. It refuses to run without a validated manifest
(``manifest.py``), refuses validation input that reaches into the sealed test window,
opens the test only on an explicit, logged request, and reports the same scorecard for
every experiment, per sealed chronological block and pooled:

  within-bar rank IC (and the beta/liquidity/volatility-neutral IC) with week-block CIs and
  the MDE; precision@k; net expected return@k after the manifest's cost model; large-winner
  recall@k against its random expectation; first passage, MFE and MAE of the picks;
  calibration of a probability; the IC delta against the noise-trained twin (``__perm``);
  neutral top-k; results by regime; finite-capital P&L marked daily when daily closes are
  supplied.

Input panel: one row per (decision bar, ticker) with ``timestamp`` (UTC), ``ticker``,
``score`` and ``label``. Optional columns switch on the matching metrics: ``fwd_ret``,
``mfe``, ``mae``, ``first_passage``, ``prob``, ``perm_score``, ``regime`` and the neutral
controls ``atr_pct``, ``beta``, ``adv_rank``, ``price``.

    .venv/bin/python -m strategies.momentum_expansion.ablation.evaluator \\
        --manifest m.json --panel panel.parquet --split validation
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.ablation import metrics as M
from strategies.momentum_expansion.ablation.bootstrap import mde_from_ci, week_block_bootstrap_ci
from strategies.momentum_expansion.ablation.export_stage2 import FINAL_TEST_START
from strategies.momentum_expansion.ablation.manifest import (
    Manifest, ManifestError, git_head, load_manifest, log_test_read, manifest_hash, validate)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "evaluator"
TEST_START = pd.Timestamp(FINAL_TEST_START, tz="UTC")      # sealed; never used to select anything
# Frozen before any year-two experiment ran (2026-11-23). Regime and month dominate seed
# variance, so every metric is reported per block; the blocks are not tunable.
VALIDATION_BLOCKS = (
    ("pre_2022", None, "2022-01-01"),
    ("2022_bear", "2022-01-01", "2023-01-01"),
    ("2023_recovery", "2023-01-01", "2024-01-01"),
    ("2024_trend", "2024-01-01", "2025-01-01"),
    ("2025_chop", "2025-01-01", "2026-01-01"),
    ("2026_rotation", "2026-01-01", FINAL_TEST_START),
)
REQUIRED_COLS = ("timestamp", "ticker", "score", "label")
NEUTRAL_CONTROLS = ("atr_pct", "beta", "adv_rank", "price")
LIQUIDITY_CONTROLS = ("beta", "adv_rank")


class ManifestRequired(RuntimeError):
    """evaluate() was called without a manifest."""


class SealedTestError(RuntimeError):
    """The sealed test window was touched without an explicit, logged opening."""


def block_of(ts: pd.Series) -> pd.Series:
    out = pd.Series("unassigned", index=ts.index, dtype=object)
    for name, lo, hi in VALIDATION_BLOCKS:
        m = ts < pd.Timestamp(hi, tz="UTC")
        if lo is not None:
            m &= ts >= pd.Timestamp(lo, tz="UTC")
        out[m] = name
    out[ts >= TEST_START] = "test"
    return out


def prepare(panel: pd.DataFrame, m: Manifest, *, split: str) -> tuple[pd.DataFrame, dict]:
    """Validated, gated, thin-bar-free panel plus the counts of what was removed and why."""
    missing = [c for c in REQUIRED_COLS if c not in panel.columns]
    if missing:
        raise ValueError(f"panel lacks required columns {missing}")
    df = panel.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    counts = {"rows_in": int(len(df))}
    # Only the columns the scorecard needs may drop a row: a drop-any over every column
    # once threw away every row with a NaN earnings field.
    keep = df[list(REQUIRED_COLS)].notna().all(axis=1)
    counts["rows_missing_required"] = int((~keep).sum())
    df = df[keep]
    cutoff = pd.Timestamp(m.dataset_cutoff, tz="UTC") + pd.Timedelta(days=1)
    if (df["timestamp"] >= cutoff).any():
        raise ManifestError(f"panel has rows after the manifest's dataset_cutoff {m.dataset_cutoff}")
    in_test = df["timestamp"] >= TEST_START
    if split == "validation":
        if in_test.any():
            raise SealedTestError(f"{int(in_test.sum())} validation rows fall in the sealed test window "
                                  f"(>= {TEST_START.date()}); cut them before evaluating")
    elif split == "test":
        df = df[in_test]
    else:
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    names = df.groupby("timestamp")["ticker"].transform("size")
    thin = names < m.min_names_per_bar
    counts["thin_bars"] = int(df.loc[thin, "timestamp"].nunique())
    counts["rows_in_thin_bars"] = int(thin.sum())
    df = df[~thin].copy()
    counts["rows_used"], counts["bars_used"] = int(len(df)), int(df["timestamp"].nunique())
    df["block"] = block_of(df["timestamp"])
    return df.reset_index(drop=True), counts


ALPHA = 0.05            # 95% CIs; MDE at 80% power (the Stage 2/3 convention)


def horizon_blocks(ts: pd.Series, horizon_sessions: int) -> pd.Series:
    """Bootstrap block key: calendar weeks grouped by ceil(h/5), so an h-session label
    never spans more than two blocks (eval_stage2._blocks). Plain weeks would treat
    overlapping 10-20 session outcomes as independent and narrow the CI."""
    weeks = max(1, math.ceil(int(horizon_sessions) / 5))
    t = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert(None)
    return pd.Series(t.to_period("W").asi8 // weeks)


def _ci(per_bar: pd.Series, *, seed: int, horizon: int) -> dict:
    s = per_bar.dropna()
    if s.empty:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_bars": 0, "mde": float("nan")}
    ci = week_block_bootstrap_ci(s.reset_index(drop=True), horizon_blocks(pd.Series(s.index), horizon),
                                 seed=seed, alpha=ALPHA)
    return {"point": ci["point"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "n_bars": int(len(s)),
            "n_blocks": ci["n_weeks"], "excludes_zero": ci["excludes_zero"], "mde": mde_from_ci(ci, alpha=ALPHA)}


def scorecard(df: pd.DataFrame, m: Manifest) -> dict:
    """The fixed metric set on one slice of a prepared panel."""
    seed, h = int(m.seed), int(m.label_horizon_sessions)
    out: dict = {"rows": int(len(df)), "bars": int(df["timestamp"].nunique())}
    if df.empty:
        return out
    ic = M.within_bar_rank_ic(df, score_col="score", label_col="label")
    out["ic"] = _ci(ic, seed=seed, horizon=h)
    controls = [c for c in NEUTRAL_CONTROLS if c in df.columns]
    if controls:
        df = df.assign(score_neutral=M.residualise_within_bar(df, score_col="score", control_cols=controls))
        out["neutral_ic"] = {**_ci(M.within_bar_rank_ic(df, score_col="score_neutral", label_col="label"),
                                   seed=seed, horizon=h), "controls": controls}
    if "perm_score" in df.columns:
        perm = M.within_bar_rank_ic(df, score_col="perm_score", label_col="label")
        out["perm_delta_ic"] = _ci((ic - perm).dropna(), seed=seed, horizon=h)
    cost = m.round_trip_cost
    liq = [c for c in LIQUIDITY_CONTROLS if c in df.columns]
    if liq:
        df = df.assign(score_liq=M.residualise_within_bar(df, score_col="score", control_cols=liq))
    topk = {}
    for k in m.top_k:
        mask = M.top_k_mask(df, score_col="score", k=int(k))
        row: dict = {"precision": _ci(M.per_bar_mean(df.assign(_hit=(df["label"] > 0).astype(float)), mask, "_hit"),
                                      seed=seed, horizon=h)}
        if "fwd_ret" in df.columns:
            net = df.assign(_net=df["fwd_ret"] - cost)
            row["net_ret"] = _ci(M.per_bar_mean(net, mask, "_net"), seed=seed, horizon=h)
            rec, rnd = M.large_winner_recall(df, mask, ret_col="fwd_ret")
            row["large_winner_recall"] = {"recall": float(rec.mean()), "random": float(rnd.mean())}
            if liq:
                lmask = M.top_k_mask(df, score_col="score_liq", k=int(k))
                row["net_ret_liquidity_neutral"] = _ci(M.per_bar_mean(net, lmask, "_net"), seed=seed, horizon=h)
        for col in ("first_passage", "mfe", "mae"):
            if col in df.columns:
                row[col] = float(df.loc[mask, col].mean())
        topk[str(k)] = row
    out["top_k"] = topk
    if "prob" in df.columns:
        out["calibration"] = M.calibration_report(df["prob"], (df["label"] > 0).astype(float))
    if "regime" in df.columns:
        out["by_regime"] = {str(r): {"ic": _ci(M.within_bar_rank_ic(g, score_col="score", label_col="label"),
                                               seed=seed, horizon=h), "rows": int(len(g))}
                            for r, g in df.groupby("regime")}
    return out


def finite_capital_curve(df: pd.DataFrame, daily_close: pd.DataFrame, *, k: int, hold_sessions: int,
                         cost: float) -> dict:
    """Equal-weight top-``k``, ``hold_sessions`` staggered sleeves (one rebalances each
    session, so every phase offset is held at once), marked to the daily close.

    Entry and exit are at the decision day's close (an approximation of the label's
    executable entry, stated in the report); a name with no close that day is skipped;
    a held name whose prices stop is carried at its last close."""
    closes = daily_close.sort_index().ffill()
    closes.index = pd.DatetimeIndex(closes.index).normalize()
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    d = df.assign(_day=df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize())
    first_bar = d.groupby("_day")["timestamp"].transform("min") == d["timestamp"]
    d = d[first_bar & M.top_k_mask(d, score_col="score", k=int(k))]
    picks = d.groupby("_day")["ticker"].apply(list)
    days = closes.index[(closes.index >= picks.index.min()) & (closes.index <= picks.index.max())]
    if len(days) < 2:
        return {"days": int(len(days)), "note": "fewer than two marked days"}
    sleeves = [{"cash": 1.0 / hold_sessions, "shares": {}} for _ in range(hold_sessions)]
    equity, names_held = [], []
    for i, day in enumerate(days):
        px = closes.loc[day]
        s = sleeves[i % hold_sessions]
        if s["shares"]:
            value = sum(n * px.get(t, np.nan) for t, n in s["shares"].items() if np.isfinite(px.get(t, np.nan)))
            s["cash"] += value * (1 - cost / 2)
            s["shares"] = {}
        todays = [t for t in picks.get(day, []) if t in px.index and np.isfinite(px[t]) and px[t] > 0]
        if todays:
            alloc = s["cash"] * (1 - cost / 2) / len(todays)
            s["shares"] = {t: alloc / px[t] for t in todays}
            s["cash"] = 0.0
        equity.append(sum(x["cash"] + sum(n * px.get(t, 0.0) for t, n in x["shares"].items()) for x in sleeves))
        names_held.append(sum(len(x["shares"]) for x in sleeves))
    eq = pd.Series(equity, index=days)
    r = eq.pct_change().dropna()
    years = max((days[-1] - days[0]).days / 365.25, 1e-9)
    return {"days": int(len(days)), "total_return": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
            "sharpe": M.sharpe_ratio(r, periods_per_year=252), "max_drawdown_daily_marked": M.max_drawdown(r),
            "avg_names_held": float(np.mean(names_held)), "k": int(k), "hold_sessions": int(hold_sessions),
            "entry_basis": "decision-day close (approximation)"}


def evaluate(panel: pd.DataFrame, manifest: Manifest | None, *, split: str = "validation",
             open_test: bool = False, test_read_reason: str | None = None,
             daily_close: pd.DataFrame | None = None, check_git: bool = True) -> dict:
    """The scorecard for ``panel`` under ``manifest``. No manifest, no evaluation."""
    if not isinstance(manifest, Manifest):
        raise ManifestRequired("the frozen evaluator needs a Manifest (strategies/momentum_expansion/"
                               "ablation/manifest.py); write the experiment down before looking at it")
    validate(manifest, check_git=check_git)
    if split == "test":
        if not open_test:
            raise SealedTestError("the test window is sealed; pass open_test=True with a reason to read it")
        read_number = log_test_read(manifest, reason=test_read_reason or "")
    df, counts = prepare(panel, manifest, split=split)
    head, dirty = git_head()
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "split": split,
        "manifest": {**manifest.__dict__, "top_k": list(manifest.top_k)}, "manifest_hash": manifest_hash(manifest),
        "git_head": head, "git_dirty": dirty, "test_start": str(TEST_START.date()),
        "blocks": [b[0] for b in VALIDATION_BLOCKS], "counts": counts,
        "pooled": scorecard(df, manifest),
        "by_block": {b: scorecard(g, manifest) for b, g in df.groupby("block", sort=False)},
    }
    if split == "test":
        report["test_read_number"] = read_number
        if read_number > 1:
            report["warning"] = (f"test read #{read_number} for {manifest.experiment_id}: only the first read is "
                                 "confirmatory")
    if daily_close is not None and manifest.hold_sessions:
        report["finite_capital"] = {str(k): finite_capital_curve(df, daily_close, k=int(k),
                                                                 hold_sessions=int(manifest.hold_sessions),
                                                                 cost=manifest.round_trip_cost)
                                    for k in manifest.top_k}
    return report


def write_report(report: dict, out_dir: Path = RESULTS_DIR) -> Path:
    m = report["manifest"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / m["experiment_id"] / f"{report['split']}_{report['manifest_hash']}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--panel", required=True, help="parquet with timestamp, ticker, score, label [+ optional]")
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--open-test", action="store_true")
    ap.add_argument("--reason", default=None, help="why the sealed test is being read (logged)")
    ap.add_argument("--daily-close", default=None, help="parquet: index date, one column per ticker")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = load_manifest(args.manifest)
    panel = pd.read_parquet(args.panel)
    daily = pd.read_parquet(args.daily_close) if args.daily_close else None
    report = evaluate(panel, manifest, split=args.split, open_test=args.open_test,
                      test_read_reason=args.reason, daily_close=daily)
    path = write_report(report)
    p = report["pooled"]
    print(json.dumps({"report": str(path), "counts": report["counts"], "ic": p.get("ic"),
                      "neutral_ic": p.get("neutral_ic"), "perm_delta_ic": p.get("perm_delta_ic")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
