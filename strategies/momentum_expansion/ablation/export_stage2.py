"""Build the Stage-2 Colab bundle: one matrix, every label arm, noise controls.

Plan: docs/superpowers/plans/2026-09-25-4h-feature-study-plan.md (Stage 2).

No features are rebuilt. The momentum feature file is kept current by the
nightly job for the whole live universe, and the HTF model's features are a
subset of it, so both 4H "slots" train from this one matrix; under the new
targets they differ by label, not by features.

Rows: every 4H bar from START on (both bars of each session), all tickers in the
feature file. Targets come from ``stage2_labels`` (executable entry, split guard).

Arms (each label arm also gets a ``__perm`` twin = labels shuffled within bar):
  L0       the deployed momentum target (25-bar MFE >= 20%)          baseline
  L1_10    forward range, top quintile within its vol decile, 10 sessions
  L2_10 / L2_20 / L2_40   SPY-excess return, top quintile within its vol decile
  PIV      HTF's swing-low zone, on every row (the pivot-detector idea)
  L2_20 with calendar features removed; with the leader/base block added; and
           with the earnings-timing features removed too (they cost 25% of rows)

Horizons 10/20/40: the training-free horizon study found the trend edge per day
flat from ~5 to ~30 sessions (so costs favour longer holds) and a 0.68 rank
overlap between the 10- and 20-session targets; 40 is included so the choice is
measured, not assumed.

Reads production parquets READ-ONLY; writes only under EXPORT_DIR.

Usage:
    .venv/bin/python -m strategies.momentum_expansion.ablation.export_stage2
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from strategies.momentum_expansion.ablation.candidate_features import daily_candidate_features
from strategies.momentum_expansion.ablation.stage2_labels import (
    MIN_NAMES_PER_BAR,
    add_arm_targets,
    flagged_sessions,
    session_dates,
    ticker_forward,
)
from strategies.momentum_expansion.config.momentum_config import MODULE_ROOT
from strategies.momentum_expansion.data.load_bars import load_1d, load_4h

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
FEATURES_4H = MODULE_ROOT / "data" / "processed" / "features_4h.parquet"
MOM_MANIFEST = MODULE_ROOT / "models" / "expansion_v1" / "feature_manifest.json"
EXPORT_DIR = MODULE_ROOT / "data" / "training_export_stage2"
TRAINER = Path(__file__).resolve().parent / "colab" / "stage2_train_colab.py"

START = pd.Timestamp("2020-01-01", tz="UTC")
FINAL_TEST_START = "2026-05-15"      # never seen by any screen; arms are confirmed here once
HORIZONS = (10, 20, 40)
CALENDAR_FEATURES = ["dow", "month", "quarter", "week_of_year", "day_of_month", "dow_sin", "dow_cos",
                     "month_sin", "month_cos", "week_sin", "week_cos", "is_month_start", "is_month_end"]
# Missing on 15-18% of recent rows; with the deployed drop-any-NaN rule they remove a
# quarter of the universe (325 of 2,888 tickers had no complete row in 2025-10..2026-07).
# days_to_earnings beyond the ~2-3 weeks a date is announced ahead is also hindsight.
EARNINGS_FEATURES = ["days_to_earnings", "days_since_earnings", "is_pre_earnings_3d",
                     "is_post_earnings_3d", "earnings_in_fwd_window"]
# Stage-1 screen: these added beyond the deployed keeps (leader/base on L1, upper wick on both);
# 12-1 momentum was the strongest single signal in the long-horizon study.
LEADER_BLOCK = ["d_pullback_126h_atr", "d_base_len_126", "d_run_x_pullback", "d_ret_126", "d_ret_63",
                "d_days_above_ema100", "d_days_above_ema50", "d_range_atr", "d_upper_wick", "d_ret_252_21"]

_CAL: pd.DatetimeIndex | None = None


def _calendar() -> pd.DatetimeIndex:
    global _CAL
    if _CAL is None:
        _CAL = pd.DatetimeIndex(sorted(set(session_dates(load_4h("SPY").index))))
    return _CAL


def embargo_days(horizon_sessions: int) -> int:
    """Calendar days between train end and test start: the label window plus a week."""
    return max(21, math.ceil(horizon_sessions * 7 / 5) + 7)


def _ticker_block(ticker: str) -> pd.DataFrame | None:
    try:
        bars = load_4h(ticker)
    except Exception:
        return None
    if bars is None or len(bars) < 100:
        return None
    try:
        daily = load_1d(ticker)
    except Exception:
        daily = None
    out = ticker_forward(bars, _calendar(), HORIZONS, flagged=flagged_sessions(daily))
    out["px"] = bars["close"].reindex(out.index).astype("float32")
    sess = session_dates(out.index)
    if daily is not None and len(daily) >= 60:
        d = daily.sort_index()
        d = d[~d.index.duplicated(keep="last")]
        cand = daily_candidate_features(d)[LEADER_BLOCK]          # already lagged one session
        cand.index = session_dates(d.index)
        adv = (d["close"] * d["volume"]).rolling(20, min_periods=10).mean().shift(1)
        adv.index = cand.index
        cand = cand[~cand.index.duplicated(keep="last")]
        adv = adv[~adv.index.duplicated(keep="last")]
        for c in LEADER_BLOCK:
            out[c] = cand[c].reindex(sess).to_numpy("float32")
        out["adv20_lag"] = adv.reindex(sess).to_numpy("float32")
    else:
        for c in [*LEADER_BLOCK, "adv20_lag"]:
            out[c] = np.float32(np.nan)
    out = out[out.index >= START]
    if out.empty:
        return None
    out = out.reset_index(names="timestamp")
    out["ticker"] = ticker
    return out


def build_labels(tickers: list[str], *, workers: int = 5) -> pd.DataFrame:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        blocks = [b for b in ex.map(_ticker_block, tickers, chunksize=16) if b is not None]
    lab = pd.concat(blocks, ignore_index=True)
    logger.info("labels: %d rows, %d/%d tickers in %.0fs", len(lab), lab["ticker"].nunique(), len(tickers),
                time.time() - t0)
    spy = ticker_forward(load_4h("SPY"), _calendar(), HORIZONS)[[f"fret_{h}" for h in HORIZONS]]
    spy = spy.rename(columns={f"fret_{h}": f"spy_fret_{h}" for h in HORIZONS}).reset_index(names="timestamp")
    return lab.merge(spy, on="timestamp", how="left")


def feature_sets(base: list[str]) -> dict[str, list[str]]:
    no_cal = [c for c in base if c not in CALENDAR_FEATURES]
    return {"base": list(base), "no_cal": no_cal, "no_cal_leader": no_cal + LEADER_BLOCK,
            "no_cal_noearn": [c for c in no_cal if c not in EARNINGS_FEATURES]}


def arms() -> list[dict]:
    spec = [("L0", "y_L0", 13), ("L1_10", "y_L1_10", 10), ("L2_10", "y_L2_10", 10),
            ("L2_20", "y_L2_20", 20), ("L2_40", "y_L2_40", 40), ("PIV", "y_PIV", 5)]
    out = []
    for name, target, h in spec:
        for suffix in ("", "__perm"):
            out.append({"name": name + suffix, "target": target + suffix, "features": "base",
                        "horizon_sessions": h, "embargo_days": embargo_days(h), "control": bool(suffix)})
    for fs in ("no_cal", "no_cal_leader", "no_cal_noearn"):
        out.append({"name": f"L2_20__{fs}", "target": "y_L2_20", "features": fs,
                    "horizon_sessions": 20, "embargo_days": embargo_days(20), "control": False})
    return out


def run(*, out_dir: Path = EXPORT_DIR, workers: int = 5, limit: int | None = None,
        make_tar: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(MOM_MANIFEST.read_text())["feature_columns"]
    pf = pq.ParquetFile(FEATURES_4H)
    missing = sorted(set(base) - set(pf.schema_arrow.names))
    if missing:
        raise ValueError(f"feature file lacks deployed features: {missing}")

    keys = pd.read_parquet(FEATURES_4H, columns=["daily_atr_pct"]).reset_index()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True)
    keys = keys[keys["timestamp"] >= START]
    tickers = sorted(keys["ticker"].unique())
    if limit:
        tickers = tickers[:limit]
        keys = keys[keys["ticker"].isin(tickers)]
    logger.info("feature rows from %s: %d, tickers %d, through %s", START.date(), len(keys), len(tickers),
                keys["timestamp"].max())

    lab = build_labels(tickers, workers=workers)
    lab = keys.merge(lab, on=["timestamp", "ticker"], how="left")
    joined = lab["px"].notna().mean()
    if joined < 0.98:
        raise ValueError(f"only {joined:.1%} of feature rows matched a 4H bar: timestamp grids differ")
    lab["adv_rank"] = lab.groupby("timestamp")["adv20_lag"].rank(pct=True).astype("float32")
    lab = add_arm_targets(lab, HORIZONS)
    lab = lab.drop(columns=["daily_atr_pct"]).set_index(["timestamp", "ticker"])
    extra_cols = list(lab.columns)

    fsets = feature_sets(base)
    superset = list(dict.fromkeys(base))            # LEADER_BLOCK columns arrive via `lab`
    matrix_path = out_dir / "stage2_matrix.parquet"
    writer, n_rows = None, 0
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=superset + ["timestamp", "ticker"]).to_pandas().reset_index()
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True)
        chunk = chunk[(chunk["timestamp"] >= START) & chunk["ticker"].isin(tickers)]
        if chunk.empty:
            continue
        chunk[superset] = chunk[superset].apply(pd.to_numeric, errors="coerce").astype("float32")
        chunk = chunk.join(lab, on=["timestamp", "ticker"])
        chunk = chunk[["timestamp", "ticker", *superset, *extra_cols]]
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(matrix_path, table.schema, compression="zstd")
        writer.write_table(table)
        n_rows += len(chunk)
        logger.info("row group %d/%d written (%d rows so far)", rg + 1, pf.metadata.num_row_groups, n_rows)
    writer.close()

    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "matrix": matrix_path.name, "rows": n_rows, "tickers": len(tickers),
        "range": [str(keys["timestamp"].min()), str(keys["timestamp"].max())],
        "final_test_start": FINAL_TEST_START,
        "min_names_per_bar": MIN_NAMES_PER_BAR,
        "horizons_sessions": list(HORIZONS),
        "feature_sets": fsets,
        "arms": arms(),
        "walk_forward": {"train_months": 24, "test_months": 6, "inner_val_frac": 0.2},
        "xgb_params": {"n_estimators": 800, "learning_rate": 0.04, "max_depth": 5, "subsample": 0.85,
                       "colsample_bytree": 0.85, "early_stopping_rounds": 60},
        "diagnostic_columns": [c for c in extra_cols if not c.startswith("y_")],
        "notes": "y_* are binary classifier targets; *__perm are within-bar shuffles (noise control). "
                 "Rows with any NaN in the arm's features or target are dropped by the trainer.",
    }
    (out_dir / "stage2_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(TRAINER, out_dir / TRAINER.name)
    logger.info("bundle files in %s: %s (%.2f GB), stage2_manifest.json, %s", out_dir, matrix_path.name,
                matrix_path.stat().st_size / 1e9, TRAINER.name)
    if make_tar:
        # Doubles the disk footprint (the parquet is already compressed). C: filled and
        # WSL crashed on exactly this step on 2026-10-06; the trainer takes loose files.
        bundle = out_dir / "stage2_colab_bundle.tar"
        with tarfile.open(bundle, "w") as tar:
            for name in (matrix_path.name, "stage2_manifest.json", TRAINER.name):
                tar.add(out_dir / name, arcname=name)
        return bundle
    return matrix_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="first N tickers only (smoke build)")
    ap.add_argument("--out-dir", type=Path, default=EXPORT_DIR)
    ap.add_argument("--tar", action="store_true", help="also write one .tar (needs ~3 GB more disk)")
    a = ap.parse_args()
    run(out_dir=a.out_dir, workers=a.workers, limit=a.limit, make_tar=a.tar)


if __name__ == "__main__":
    main()
