"""
Capstone results lock — regenerate every headline number the paper cites from
FIXED on-disk artifacts (no retraining, no network).

For each model discussed in the paper (multi_ticker_swing, momentum_expansion,
multi_ticker_swing_htf, meta_ranker, spy_intraday baseline) this script:

  1. fingerprints the source artifacts (rows + quick content hash),
  2. recomputes the headline metrics from those artifacts,
  3. prints one metrics table with an explicit split/provenance tag per row
     (in-sample / validation / test / walk-forward-OOF / paper-trading),
  4. optionally writes research/capstone/results_lock.json for the regression
     tests in tests/capstone/.

Provenance tags:
  test(comp)   — competition test split (last 20% by time; winner was PICKED on
                 this split → selection-biased, see leakage_audit.md §0.2)
  test(frozen) — test split evaluated without any selection on it
  wf-oof       — walk-forward out-of-fold, 21-day embargo (clean OOS)
  paper        — Alpaca paper-trading audit logs
  artifact     — stale saved artifact that can no longer be regenerated from
                 trades (locked as-is; see leakage_audit.md §1.4)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/reproduce_results.py                 # all light sections
  PYTHONPATH=. .venv/bin/python scripts/capstone/reproduce_results.py --model meta
  PYTHONPATH=. .venv/bin/python scripts/capstone/reproduce_results.py --write-lock
  PYTHONPATH=. .venv/bin/python scripts/capstone/reproduce_results.py --skip-swing-probs   # skip the 15M-row join

Heavy event-driven backtests are NOT run here (documented in --list-heavy):
they are separate CLIs with their own artifacts and multi-minute runtimes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO / "research" / "capstone" / "results_lock.json"

# ---------------------------------------------------------------------------
# Fixed artifact registry (single source of truth for the paper + tests)
# ---------------------------------------------------------------------------
ARTIFACTS: dict[str, Path] = {
    # multi_ticker_swing (30m primary)
    "swing_eval_metrics":   REPO / "strategies/multi_ticker_swing/models/eval_metrics.json",
    "swing_meta":           REPO / "strategies/multi_ticker_swing/models/meta.json",
    "swing_probs":          REPO / "strategies/multi_ticker_swing/models/p_swing_probs.parquet",
    "swing_matrix":         REPO / "strategies/multi_ticker_swing/data/processed/training_matrix.parquet",
    "swing_bt_grouped":     REPO / "strategies/multi_ticker_swing/backtest/results/sweep_v2/best_v2_grouped.json",
    "swing_bt_summary":     REPO / "strategies/multi_ticker_swing/backtest/results/sweep_v2/sweep_v2_summary.csv",
    "swing_paper_closed":   REPO / "Data/analysis/multi_ticker_swing_live/experiments/multiticker_20260528_20260529_closed_performance_rebuilt.csv",
    # momentum_expansion (4H ranker)
    "mom_eval_metrics":     REPO / "strategies/momentum_expansion/models/expansion_v1/eval_metrics.json",
    "mom_seed_results":     REPO / "strategies/momentum_expansion/models/expansion_v1/seed_results.csv",
    "mom_oof":              REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet",
    # multi_ticker_swing_htf (4H pivot swing)
    "htf_eval_metrics":     REPO / "strategies/multi_ticker_swing_htf/models/eval_metrics.json",
    "htf_seed_results":     REPO / "strategies/multi_ticker_swing_htf/models/seed_results.csv",
    "htf_oof":              REPO / "strategies/multi_ticker_swing_htf/models/oof_preds.parquet",
    # meta_ranker
    "meta_q_eval":          REPO / "signals/meta_context/meta_ranker/models/quality/eval_metrics.json",
    "meta_u_eval":          REPO / "signals/meta_context/meta_ranker/models/upside/eval_metrics.json",
    "meta_q_oof":           REPO / "signals/meta_context/meta_ranker/models/quality/oof_preds.parquet",
    "meta_u_oof":           REPO / "signals/meta_context/meta_ranker/models/upside/oof_preds.parquet",
    # spy_intraday baseline
    "spy_long_meta":        REPO / "Data/models/ga_xgboost/10min/long/swing/meta.json",
    "spy_short_meta":       REPO / "Data/models/ga_xgboost/10min/short/swing/meta.json",
    "spy_long_oos":         REPO / "Data/models/ga_xgboost/10min/long/swing/p_long_oos_manifest.json",
    # benchmark
    "spy_1d_bars":          REPO / "Data/shared/bars/1d/SPY.parquet",
    # val-selected / test-frozen equity-tier backtests (leakage_audit.md patch)
    "mom_family_clean":     REPO / "strategies/momentum_expansion/backtest/results/family_compare_clean/comparison_summary_clean.json",
    "htf_family_clean":     REPO / "strategies/multi_ticker_swing_htf/backtest/results/family_compare_clean/comparison_summary_clean.json",
    "swing_bt_clean":       REPO / "strategies/multi_ticker_swing/backtest/results/sweep_v2_clean/best_v2_clean_summary.json",
}

TOP_K = 10  # matches RANKING_CONFIG.top_n / meta live top-K

# Data/shared/bars/1d/SPY.parquet is a LIVE file the nightly pipeline appends
# to every night (unlike every other artifact here, it is not frozen). The SPY
# forward-return benchmark is pinned to bar history through this date so it
# reproduces indefinitely instead of drifting by a few 4th/5th-decimal points
# each time new bars land. Bump intentionally (with --write-lock) if the
# benchmark window should move; do not let it silently track "today".
SPY_BENCHMARK_CUTOFF = pd.Timestamp("2026-07-14", tz="UTC")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quick_hash(path: Path, chunk_mb: int = 4) -> str:
    """sha256 of (size + first and last `chunk_mb` MB). Full-file hashing of the
    1.2GB matrices is pointlessly slow; head+tail+size catches any regeneration."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    chunk = chunk_mb * 1024 * 1024
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(-chunk, 2)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def _spy_frozen_fingerprint(path: Path) -> dict:
    """spy_1d_bars grows every night; fingerprint only the bars through the
    pinned SPY_BENCHMARK_CUTOFF so the check tracks "did history get revised
    retroactively" rather than "did new bars get appended" (which is expected
    and not a drift bug)."""
    b = pd.read_parquet(path, columns=["timestamp", "close"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    b = b[b["timestamp"] <= SPY_BENCHMARK_CUTOFF].sort_values("timestamp")
    h = hashlib.sha256(b["close"].to_numpy().tobytes()).hexdigest()[:16]
    return {"exists": True, "rows": int(len(b)), "hash": h,
            "frozen_through": str(SPY_BENCHMARK_CUTOFF.date())}


def fingerprint(keys: list[str] | None = None) -> dict[str, dict]:
    out = {}
    for name, path in ARTIFACTS.items():
        if keys and name not in keys:
            continue
        if not path.exists():
            out[name] = {"exists": False}
            continue
        if name == "spy_1d_bars":
            out[name] = _spy_frozen_fingerprint(path)
            continue
        entry: dict = {"exists": True, "bytes": path.stat().st_size, "hash": quick_hash(path)}
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            entry["rows"] = pq.ParquetFile(path).metadata.num_rows
        out[name] = entry
    return out


def row(model: str, metric: str, value, tag: str, source: str, caveat: str = "", n=None) -> dict:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        value = round(value, 6)
    return {"model": model, "metric": metric, "value": value, "n": n,
            "tag": tag, "source": source, "caveat": caveat}


def _read_oof(path: Path) -> pd.DataFrame:
    """Load an OOF parquet and surface (timestamp, ticker) as columns whether the
    file stored them as columns or as a (Multi)Index."""
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        df = df.reset_index()
    # walk-forward fold boundaries can emit duplicate (timestamp,ticker) rows
    return df.drop_duplicates(subset=["timestamp", "ticker"], keep="first")


def _topk_oof_stats(oof: pd.DataFrame, model: str, score_col: str = "score",
                    top_k: int = TOP_K, extra_cols: tuple[str, ...] = ()) -> list[dict]:
    """Per-timestamp top-K signal-quality stats from a walk-forward OOF frame.

    fwd_close_return is a FIXED-horizon (label-window) forward return with
    overlapping windows across bars — signal quality, not a compounded equity
    curve. Tagged wf-oof; the event-driven equity numbers live in the heavy
    backtests (see --list-heavy).
    """
    df = oof.dropna(subset=[score_col, "fwd_close_return"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    rk = df.groupby("timestamp")[score_col].rank(ascending=False, method="first")
    top = df[rk <= top_k]
    rest = df[rk > top_k]
    span = f"{df['timestamp'].min().date()}..{df['timestamp'].max().date()}"
    src = f"oof_preds.parquet ({span})"
    cav = "fixed-horizon fwd returns, overlapping windows"
    rows = [
        row(model, f"oof_top{top_k}_mean_fwd_close_ret", top["fwd_close_return"].mean(), "wf-oof", src, cav, n=len(top)),
        row(model, f"oof_top{top_k}_median_fwd_close_ret", top["fwd_close_return"].median(), "wf-oof", src, cav, n=len(top)),
        row(model, f"oof_top{top_k}_win_rate", (top["fwd_close_return"] > 0).mean(), "wf-oof", src, cav, n=len(top)),
        row(model, "oof_universe_mean_fwd_close_ret", rest["fwd_close_return"].mean(), "wf-oof", src, "baseline: all non-top-K rows", n=len(rest)),
        row(model, f"oof_top{top_k}_minus_universe", top["fwd_close_return"].mean() - rest["fwd_close_return"].mean(), "wf-oof", src, cav),
        row(model, "oof_spearman_score_vs_y", df[score_col].corr(df["y"], method="spearman"), "wf-oof", src, "", n=len(df)),
    ]
    for c in extra_cols:
        if c in top.columns:
            rows.append(row(model, f"oof_top{top_k}_mean_{c}", top[c].mean(), "wf-oof", src, cav, n=len(top)))
    return rows


def _seed_spread(csv_path: Path, metric: str) -> tuple[float, float, float, int]:
    d = pd.read_csv(csv_path)
    v = d[metric].dropna()
    return float(v.min()), float(v.median()), float(v.max()), len(v)


# ---------------------------------------------------------------------------
# multi_ticker_swing
# ---------------------------------------------------------------------------

def swing_model_metrics(recompute_probs: bool = True) -> list[dict]:
    rows: list[dict] = []
    em = json.loads(ARTIFACTS["swing_eval_metrics"].read_text())
    src = "models/eval_metrics.json"
    for split, tag in [("val", "validation"), ("test", "test(frozen)")]:
        for m in ("accuracy", "long_precision", "long_wr", "short_precision", "short_wr"):
            key = f"{split}_{m}"
            if key in em and em[key] is not None:
                rows.append(row("swing", key, em[key], tag, src,
                                n=em.get(f"{split}_{'long' if 'long' in m else 'short'}_n") if m != "accuracy" else None))
    # OOF (train-window sequential folds)
    for m in ("oof_accuracy", "oof_long_wr", "oof_short_wr"):
        if em.get(m) is not None:
            rows.append(row("swing", m, em[m], "oof(train)", src, "sequential folds within train split"))

    if recompute_probs:
        rows += swing_recompute_test_from_probs()
    return rows


def swing_recompute_test_from_probs() -> list[dict]:
    """Recompute test accuracy + predicted-class win rates from the saved
    probability parquet joined to the training-matrix targets. Independent check
    that eval_metrics.json matches its own artifacts (~1-2 min, ~6GB RAM)."""
    probs = pd.read_parquet(ARTIFACTS["swing_probs"],
                            columns=["timestamp", "ticker", "split", "p_long", "p_short", "p_neutral"])
    probs = probs[probs["split"] == "test"]
    tgt = pd.read_parquet(ARTIFACTS["swing_matrix"], columns=["timestamp", "ticker", "target"])
    df = probs.merge(tgt, on=["timestamp", "ticker"], how="inner")
    pred = df[["p_short", "p_neutral", "p_long"]].to_numpy().argmax(axis=1)  # 0=short 1=neutral 2=long
    y = df["target"].to_numpy()
    long_m, short_m = pred == 2, pred == 0
    src = "recomputed: p_swing_probs.parquet x training_matrix.parquet"
    return [
        row("swing", "test_accuracy_recomputed", float((pred == y).mean()), "test(frozen)", src, n=len(df)),
        row("swing", "test_long_wr_recomputed", float((y[long_m] == 2).mean()), "test(frozen)", src, n=int(long_m.sum())),
        row("swing", "test_short_wr_recomputed", float((y[short_m] == 0).mean()), "test(frozen)", src, n=int(short_m.sum())),
    ]


def swing_backtest_lock() -> list[dict]:
    """Lock the saved sweep_v2 artifacts. NOTE (leakage_audit.md §1.4): the best
    combo was selected on the SAME split it reports, per-trade file is gone, and
    the grouped JSON predates the current summary CSV. Locked as-is."""
    rows: list[dict] = []
    grouped = json.loads(ARTIFACTS["swing_bt_grouped"].read_text())
    cav = "policy picked on reporting split; stale artifact (trades parquet missing); PnL-based WR, not directional accuracy"
    for key, label in [("dir:long", "long"), ("dir:short", "short")]:
        if key in grouped:
            g = grouped[key]
            rows.append(row("swing", f"bt_v2_{label}_win_rate", g["win_rate"], "artifact",
                            "sweep_v2/best_v2_grouped.json", cav, n=g["n_trades"]))
    # aggregate from sector groups (this is what the advisor doc's 62.6/60.0 came from)
    sec = {k: v for k, v in grouped.items() if k.startswith("sector:")}
    ln = sum(v["long_n"] for v in sec.values()); sn = sum(v["short_n"] for v in sec.values())
    lw = sum(v["long_wr"] * v["long_n"] for v in sec.values()) / max(ln, 1)
    sw = sum(v["short_wr"] * v["short_n"] for v in sec.values()) / max(sn, 1)
    tot = (lw * ln + sw * sn) / max(ln + sn, 1)
    rows += [
        row("swing", "bt_v2_long_wr_sector_agg", lw, "artifact", "best_v2_grouped.json sector groups", cav, n=ln),
        row("swing", "bt_v2_short_wr_sector_agg", sw, "artifact", "best_v2_grouped.json sector groups", cav, n=sn),
        row("swing", "bt_v2_combined_wr_sector_agg", tot, "artifact", "best_v2_grouped.json sector groups",
            cav + "; advisor doc said 62.3% — correct combined value is this", n=ln + sn),
    ]
    # current sweep summary top row (by Sharpe, test split) for contrast
    summ = pd.read_csv(ARTIFACTS["swing_bt_summary"]).sort_values("sharpe", ascending=False)
    best = summ.iloc[0]
    for m in ("n_trades", "win_rate", "profit_factor", "sharpe", "avg_pnl_pct", "max_dd_pct", "long_wr", "short_wr"):
        rows.append(row("swing", f"bt_v2_current_best_{m}", best[m], "test(comp)",
                        f"sweep_v2_summary.csv best combo {best['combo_name']}",
                        "best-of-180 combos selected on this same split"))
    return rows


def swing_backtest_clean_lock() -> list[dict]:
    """Lock the val-selected/test-frozen sweep_v2_clean patch
    (scripts/capstone/family_backtest_clean.py's swing sibling — see
    strategies/multi_ticker_swing/backtest/sweep_v2_clean.py). Fixes the
    same-split selection bias in swing_backtest_lock() above."""
    if not ARTIFACTS["swing_bt_clean"].exists():
        return []
    d = json.loads(ARTIFACTS["swing_bt_clean"].read_text())
    src = "sweep_v2_clean/best_v2_clean_summary.json"
    cav = f"combo {d['combo']['name']} selected on VAL split, frozen on TEST split"
    fm = d["frozen_test_metrics"]
    rows = [row("swing", f"bt_v2_clean_{k}", fm[k], "test(frozen)", src, cav)
            for k in ("n_trades", "win_rate", "sharpe", "profit_factor", "avg_pnl_pct",
                      "max_dd_pct", "long_wr", "short_wr") if k in fm]
    grp = d.get("frozen_test_grouped_direction", {})
    for key, label in [("dir:long", "long"), ("dir:short", "short")]:
        if key in grp:
            rows.append(row("swing", f"bt_v2_clean_{label}_win_rate", grp[key]["win_rate"],
                            "test(frozen)", src, cav, n=grp[key]["n_trades"]))
    return rows


def swing_paper_trading() -> list[dict]:
    """Reproduce the May 28-29 paper-trading option-return claims from the saved
    per-trade ledger (the advisor doc's +40.6% fresh-call figure)."""
    d = pd.read_csv(ARTIFACTS["swing_paper_closed"])
    src = "multiticker_20260528_20260529_closed_performance_rebuilt.csv"
    cav = "paper trading, 2 sessions, small n"
    rows: list[dict] = []
    fresh_calls = d[(d["is_fresh"]) & (d["direction"] == 1)]
    rows += [
        row("swing", "paper_fresh_call_mean_option_ret_pct", fresh_calls["option_ret_pct"].mean(), "paper", src, cav, n=len(fresh_calls)),
        row("swing", "paper_fresh_call_mean_underlying_ret_pct", fresh_calls["underlying_signed_ret_pct"].mean(), "paper", src, cav, n=len(fresh_calls)),
        row("swing", "paper_all_closed_option_pnl_dollars", d["option_pnl_dollars"].sum(), "paper", src, cav, n=len(d)),
        row("swing", "paper_all_closed_win_rate", (d["option_pnl_dollars"] > 0).mean(), "paper", src, cav, n=len(d)),
    ]
    # hold times (minutes) — reproducible slice of the advisor doc's hold-time table
    ts_open = pd.to_datetime(d["entry_time"], utc=True, format="mixed")
    ts_close = pd.to_datetime(d["closed_ts"], utc=True, format="mixed")
    hold_min = (ts_close - ts_open).dt.total_seconds() / 60.0
    rows += [
        row("swing", "paper_hold_minutes_mean", hold_min.mean(), "paper", src,
            "5/28-5/29 ledger only; advisor doc's 151.2min figure also included 6/1 logs", n=len(d)),
        row("swing", "paper_hold_minutes_median", hold_min.median(), "paper", src, cav, n=len(d)),
    ]
    return rows


# ---------------------------------------------------------------------------
# momentum_expansion / multi_ticker_swing_htf (same artifact shapes)
# ---------------------------------------------------------------------------

def _competition_metrics(model: str, eval_key: str, seed_key: str) -> list[dict]:
    rows: list[dict] = []
    em = json.loads(ARTIFACTS[eval_key].read_text())
    b = em.get("best", {})
    src = f"{Path(ARTIFACTS[eval_key]).parent.name}/eval_metrics.json"
    cav_sel = f"winner picked BY this metric across families x seeds (selection bias, audit §0.2)"
    primary = em.get("primary_metric", "")
    for m in ("test_ndcg_at_10", "test_ndcg_at_20", "test_precision_at_10", "test_precision_at_20",
              "test_spearman", "test_positive_precision", "val_positive_precision", "test_log_loss"):
        if b.get(m) is not None:
            tag = "test(comp)" if m.startswith("test") else "validation"
            rows.append(row(model, f"winner_{m}", b[m], tag, src, cav_sel if m == primary else ""))
    rows.append(row(model, "winner_family_seed", f"{em.get('winner_family')}/{em.get('winner_seed')}",
                    "test(comp)", src, cav_sel))
    # selection-bias context: metric spread across all candidates
    seed_csv = ARTIFACTS.get(seed_key)
    if seed_csv and seed_csv.exists() and primary:
        d = pd.read_csv(seed_csv)
        if primary in d.columns:
            lo, med, hi, n = _seed_spread(seed_csv, primary)
            rows.append(row(model, f"candidates_{primary}_min_median_max",
                            f"{lo:.4f} / {med:.4f} / {hi:.4f}", "test(comp)",
                            seed_csv.name, f"spread over {n} family-seed candidates; winner = max", n=n))
    return rows


def _family_compare_clean_lock(model: str, artifact_key: str) -> list[dict]:
    """Lock the val-selected/test-frozen order-policy equity backtest
    (scripts/capstone/family_backtest_clean.py). Fixes the same-split
    selection bias in run_family_compare.py's comparison_summary.json
    (audit found momentum's test-selected ret/DD=44.6x vs. this clean
    frozen-test ret/DD — selection alone inflated it ~7x)."""
    if not ARTIFACTS[artifact_key].exists():
        return []
    d = json.loads(ARTIFACTS[artifact_key].read_text())
    src = f"{ARTIFACTS[artifact_key].parent.name}/comparison_summary_clean.json"
    dw = d["deployed_winner_frozen_test"]
    cav = (f"deployed winner ({dw['family']}) policy selected on VAL split "
          f"(tp={dw['tp_atr_mult']} sl={dw['sl_atr_mult']} topk={dw['top_k']} hold={dw['max_hold']}), "
          f"frozen on TEST split")
    rows = [row(model, f"clean_deployed_winner_{k}", dw[k], "test(frozen)", src, cav)
            for k in ("trades", "win_rate", "total_return_pct", "max_dd_pct", "ret_over_dd", "profit_factor")]
    return rows


def momentum_metrics() -> list[dict]:
    rows = _competition_metrics("momentum", "mom_eval_metrics", "mom_seed_results")
    oof = _read_oof(ARTIFACTS["mom_oof"])
    rows += _topk_oof_stats(oof, "momentum", extra_cols=("fwd_max_return", "fwd_max_drawdown"))
    rows += _family_compare_clean_lock("momentum", "mom_family_clean")
    return rows


def htf_metrics() -> list[dict]:
    rows = _competition_metrics("htf_swing", "htf_eval_metrics", "htf_seed_results")
    oof = _read_oof(ARTIFACTS["htf_oof"])
    rows += _topk_oof_stats(oof, "htf_swing", extra_cols=("fwd_best_high_return", "fwd_worst_low_return"))
    rows += _family_compare_clean_lock("htf_swing", "htf_family_clean")
    return rows


# ---------------------------------------------------------------------------
# meta_ranker
# ---------------------------------------------------------------------------

def meta_metrics() -> list[dict]:
    rows: list[dict] = []
    rows += _competition_metrics("meta_quality", "meta_q_eval", "__none__")
    rows += _competition_metrics("meta_upside", "meta_u_eval", "__none__")

    q = _read_oof(ARTIFACTS["meta_q_oof"])
    u = _read_oof(ARTIFACTS["meta_u_oof"])
    rows += _topk_oof_stats(q, "meta_quality", extra_cols=("fwd_max_drawdown",))
    rows += _topk_oof_stats(u, "meta_upside", extra_cols=("fwd_max_return",))

    # combo = per-timestamp rank-mean of the two OOF scores (mirrors live s_combo,
    # but built from CLEAN walk-forward OOF scores instead of deployed boosters —
    # this is the leak-free version of the meta headline, audit §4.3)
    m = q[["timestamp", "ticker", "score", "fwd_close_return", "fwd_max_return", "fwd_max_drawdown"]].merge(
        u[["timestamp", "ticker", "score"]], on=["timestamp", "ticker"], suffixes=("_q", "_u"))
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    rq = m.groupby("timestamp")["score_q"].rank(pct=True)
    ru = m.groupby("timestamp")["score_u"].rank(pct=True)
    m["s_combo_oof"] = (rq + ru) / 2.0
    m["y"] = m["fwd_close_return"]  # for the shared helper's Spearman row
    m = m.rename(columns={"s_combo_oof": "score_combo"})
    rows += _topk_oof_stats(m.assign(score=m["score_combo"]), "meta_combo",
                            extra_cols=("fwd_max_return", "fwd_max_drawdown"))
    rows += meta_exit_policy_lock()
    return rows


def meta_exit_policy_lock() -> list[dict]:
    """Event-driven exit-policy comparison (leakage_audit.md §4.3 patch): the
    same top-K-membership simulation as backtest_exits.py, but the top-K
    membership is built from CLEAN walk-forward OOF s_combo scores instead of
    /tmp/meta_scored.parquet (which backtest_exits.py normally reads, produced
    by score.py's DEPLOYED boosters — final-fit models that trained past the
    2025-07-01 holdout start). Reuses backtest_exits.simulate() unmodified;
    only the score source changes. See scripts/capstone/build_meta_scored_from_oof.py."""
    from scripts.capstone.build_meta_scored_from_oof import build_oof_combo_scores
    from signals.meta_context.meta_ranker import backtest_exits as bx

    scored = build_oof_combo_scores().dropna(subset=["s_combo"])
    scored["rk"] = scored.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    member = scored[scored["timestamp"] >= bx.HOLDOUT].copy()
    member["in_top"] = member["rk"] <= bx.TOPK
    member = member[["timestamp", "ticker", "in_top"]]

    src = f"OOF s_combo (models/{{quality,upside}}/oof_preds.parquet), holdout {bx.HOLDOUT.date()}+"
    cav = "clean substitute for backtest_exits.py's deployed-booster scoring (audit §4.3)"
    policies = {
        "current_live_dropout_g0": dict(grace=0),
        "target20_full_exit": dict(target=0.20, scale_frac=1.0, grace=None),
        "scaleout50_at20_horizon25": dict(target=0.20, scale_frac=0.5, horizon=25, grace=None),
    }
    rows: list[dict] = []
    for name, kw in policies.items():
        s = bx.simulate(member, **kw)
        if not s:
            continue
        for metric in ("mean", "median", "win", "ret_std", "avg_hold", "ret_per_bar"):
            rows.append(row("meta_exit_policy", f"{name}_{metric}", s[metric], "wf-oof", src, cav, n=s["n"]))
    return rows


# ---------------------------------------------------------------------------
# spy_intraday baseline
# ---------------------------------------------------------------------------

def spy_metrics() -> list[dict]:
    rows: list[dict] = []
    for side, key in [("long", "spy_long_meta"), ("short", "spy_short_meta")]:
        meta = json.loads(ARTIFACTS[key].read_text())
        src = f"Data/models/ga_xgboost/10min/{side}/swing/meta.json"
        rows.append(row("spy_baseline", f"{side}_ga_best_score", meta.get("best_score"), "validation", src,
                        "GA fitness (neg penalized logloss) on internal train/val split"))
        rows.append(row("spy_baseline", f"{side}_selected_features",
                        f"{meta.get('selected_features')}/{meta.get('n_features')}", "artifact", src))
    oos = json.loads(ARTIFACTS["spy_long_oos"].read_text())
    rows.append(row("spy_baseline", "long_test_rows", oos.get("test_finite_count"), "test(frozen)",
                    "p_long_oos_manifest.json", "fixed on-disk split indices"))
    rows.append(row("spy_baseline", "live_verdict", "directional correctness ~noise live",
                    "live", "research/daily_live_reports/*, advisor doc §SPY",
                    "narrative baseline result; per-session ledgers in daily reports"))
    return rows


# ---------------------------------------------------------------------------
# Benchmark context
# ---------------------------------------------------------------------------

def benchmark_metrics() -> list[dict]:
    b = pd.read_parquet(ARTIFACTS["spy_1d_bars"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    b = b[b["timestamp"] <= SPY_BENCHMARK_CUTOFF].sort_values("timestamp")
    rows: list[dict] = []
    # SPY 25-4H-bar-equivalent (~12.5 trading day) forward return baseline,
    # unconditional over history through SPY_BENCHMARK_CUTOFF (NOT scoped to any
    # single model's OOF/test window — see scripts/capstone/baseline_strategies.py
    # for the per-module, window-matched SPY comparison used in the equity figures).
    for name, days in [("spy_fwd_12d_mean_ret", 12), ("spy_fwd_25d_mean_ret", 25)]:
        fwd = b["close"].shift(-days) / b["close"] - 1.0
        rows.append(row("benchmark", name, fwd.mean(), "reference",
                        "Data/shared/bars/1d/SPY.parquet",
                        f"mean {days}-trading-day forward return, bar history through "
                        f"{SPY_BENCHMARK_CUTOFF:%Y-%m-%d} (frozen cutoff, pinned for reproducibility)",
                        n=int(fwd.notna().sum())))
    return rows


# ---------------------------------------------------------------------------
# Heavy backtests (documented, not run)
# ---------------------------------------------------------------------------

HEAVY = """\
Heavy event-driven backtests (multi-minute; results are committed artifacts
locked into swing/momentum/htf/meta sections above once present). Re-run to
refresh after a model/data change:

1. Swing exit-policy sweep, val-select/test-freeze (audit §1.4 patch):
   PYTHONPATH=. .venv/bin/python -m strategies.multi_ticker_swing.backtest.sweep_v2_clean --top-n 100
   -> strategies/multi_ticker_swing/backtest/results/sweep_v2_clean/  (~30 min)
2. Momentum / HTF order-policy equity backtest, val-select/test-freeze (audit §0.2/§2/§3 patch):
   PYTHONPATH=. .venv/bin/python scripts/capstone/family_backtest_clean.py --strategy all
   -> strategies/{momentum_expansion,multi_ticker_swing_htf}/backtest/results/family_compare_clean/  (~10 min)
3. Meta exit-policy backtest with CLEAN OOF scores (audit §4.3 patch) — locked
   directly in meta_metrics() via meta_exit_policy_lock(); for the full
   backtest_exits.py table (all 11 policies) instead of the 3 locked ones:
   PYTHONPATH=. .venv/bin/python scripts/capstone/build_meta_scored_from_oof.py
   PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/backtest_exits.py

For reference, the ORIGINAL (test-selected / deployed-booster) versions of 1-2
remain on disk at .../family_compare/comparison_summary.json and
.../sweep_v2/best_v2_grouped.json — do not delete; they are the audit's
worked example of the selection-bias magnitude (e.g. momentum ret/DD 44.6x
test-selected vs the clean frozen-test number locked above).
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS = {
    "swing": lambda skip_probs: swing_model_metrics(recompute_probs=not skip_probs)
                                + swing_backtest_lock() + swing_backtest_clean_lock() + swing_paper_trading(),
    "momentum": lambda _s: momentum_metrics(),
    "htf": lambda _s: htf_metrics(),
    "meta": lambda _s: meta_metrics(),
    "spy": lambda _s: spy_metrics(),
    "benchmark": lambda _s: benchmark_metrics(),
}


def run(models: list[str], skip_swing_probs: bool = False) -> list[dict]:
    rows: list[dict] = []
    for name in models:
        print(f"— computing {name} ...", flush=True)
        rows += SECTIONS[name](skip_swing_probs)
    return rows


def print_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    pd.set_option("display.max_rows", None, "display.width", 250, "display.max_colwidth", 60)
    print("\n=== CAPSTONE RESULTS LOCK ===")
    print(df[["model", "metric", "value", "n", "tag", "caveat"]].to_string(index=False))
    print("\nTag legend: test(comp)=winner picked on this split | test(frozen)=untouched test | "
          "wf-oof=walk-forward OOF, 21d embargo | paper=paper trading | artifact=stale saved artifact")


def write_lock(rows: list[dict]) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=REPO).stdout.strip()
    except Exception:
        commit = "unknown"
    lock = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "top_k": TOP_K,
        "artifacts": fingerprint(),
        "metrics": rows,
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(lock, indent=1, default=str))
    print(f"\nlock written -> {LOCK_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all",
                    choices=["all"] + list(SECTIONS), help="section to run")
    ap.add_argument("--skip-swing-probs", action="store_true",
                    help="skip the 15M-row swing probs/matrix join (saves ~2min/6GB)")
    ap.add_argument("--write-lock", action="store_true", help=f"write {LOCK_PATH}")
    ap.add_argument("--list-heavy", action="store_true", help="print heavy backtest commands and exit")
    args = ap.parse_args()

    if args.list_heavy:
        print(HEAVY)
        return

    missing = [k for k, p in ARTIFACTS.items() if not p.exists()]
    if missing:
        print(f"WARNING: missing artifacts (their sections will fail): {missing}")

    models = list(SECTIONS) if args.model == "all" else [args.model]
    rows = run(models, skip_swing_probs=args.skip_swing_probs)
    print_table(rows)
    if args.write_lock:
        write_lock(rows)


if __name__ == "__main__":
    main()
