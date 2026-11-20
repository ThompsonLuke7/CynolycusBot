"""Build the Stage-3 Colab bundle on POINT-IN-TIME data (pre-registered rule v2).

Pre-registration: research/execution_quality/27_stage3_preregistration.md. Same arms recipe as
Stage 2 (``export_stage2``), but on every common stock that traded since 2016 (delisted names
included), one SIP feed throughout, and only features that exist for dead names too.

Stages (each resumable; research caches are read-only, writes go under STAGE3_DIR / EXPORT_DIR):
  bars    hourly SIP bars (scripts/research_data/fetch_research_bars_1h) -> 4H bars per
          SECURITY, sliced to its life window so a reused ticker's lives stay apart,
          zero-volume bars (Alpaca's padding of dead periods) dropped.
  rows    per security: the live 4H feature function and the Stage-2 forward labels.
          A security that stopped trading exits at its last close (``exit_at_last_bar``).
          Written per year so the next stage reads one cross-section at a time.
  matrix  per year: the per-bar universe (lagged close >= $1, top 3,000 by lagged
          20-session dollar volume), cross-sectional ranks and targets computed over that
          universe, appended to one parquet. Then the manifest and the trainer copy.

The live builder's in-memory combine peaks at 15.7 GiB on 2,886 tickers x 6 years; this
data is ~8,000 securities x 10 years, hence per-security rows and a per-year combine. Both
cross-sectional steps (ranks, targets) are per timestamp, so a per-year pass is exact.

Usage:
    .venv/bin/python -m strategies.momentum_expansion.ablation.build_stage3_pit --stage all
    .venv/bin/python -m strategies.momentum_expansion.ablation.build_stage3_pit --stage all \\
        --limit 300 --work-dir /tmp/s3 --export-dir /tmp/s3/export      # smoke build
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from strategies.momentum_expansion.ablation.export_stage2 import (
    CALENDAR_FEATURES,
    EARNINGS_FEATURES,
    EXPORT_DIR as STAGE2_EXPORT_DIR,
    FINAL_TEST_START,
    HORIZONS,
    TRAINER,
    embargo_days,
)
from strategies.momentum_expansion.ablation.stage2_labels import (
    MIN_NAMES_PER_BAR,
    add_arm_targets,
    flagged_sessions,
    session_dates,
    ticker_forward,
)
from strategies.momentum_expansion.config.momentum_config import (
    CONTEXT_TICKERS,
    MIN_4H_BARS,
    MODULE_ROOT,
    SECTOR_ETFS,
)
from strategies.momentum_expansion.data.bars import resample_1h_to_4h
from strategies.momentum_expansion.features.feature_matrix_4h import (
    _PRECISION_SENSITIVE_COLS,
    _XSEC_RANK_SPECS,
    _add_cross_sectional_features,
    build_ticker_features_4h,
)

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
RESEARCH = REPO / "Data" / "research"
HOURLY_DIR = RESEARCH / "bars_1h_sip_adj"
DAILY_DIR = RESEARCH / "bars_1d_sip_adj_pit"
SECURITIES = RESEARCH / "pit_universe" / "securities.parquet"
STAGE3_DIR = RESEARCH / "stage3_pit"
EXPORT_DIR = MODULE_ROOT / "data" / "training_export_stage3"

RULE = "v2-2026-11-08"
START = pd.Timestamp("2017-01-01", tz="UTC")   # 252 sessions of daily history from 2016-01
UNIVERSE_SIZE = 3000
MIN_LAGGED_CLOSE = 1.0
SEED = 20261108
CONTEXT = sorted(set(CONTEXT_TICKERS) | set(SECTOR_ETFS))
# Exist only for today's survivors: a blank here would mark the dead names.
SURVIVOR_ONLY = ["sector_id", "rs_sector_5", "rs_sector_20", "market_cap_bucket", "asset_type", "is_etf"]
N_PIT_FEATURES = 82                            # fixed by the pre-registration
XSEC_COLS = [*_XSEC_RANK_SPECS.values(), "xsec_near_high_rank"]
TARGETS = ["y_L0", "y_L2_10", "y_L2_20"]
YARDSTICKS = [*(f"L2_{h}" for h in HORIZONS), *(f"fret_{h}" for h in HORIZONS), *(f"fmfe_{h}" for h in HORIZONS),
              *(f"spy_fret_{h}" for h in HORIZONS), *(f"clamped_{h}" for h in HORIZONS)]
NY = "America/New_York"


def pit_features() -> list[str]:
    """The Stage-2 ``no_cal`` set minus earnings timing and survivor-only metadata."""
    no_cal = json.loads((STAGE2_EXPORT_DIR / "stage2_manifest.json").read_text())["feature_sets"]["no_cal"]
    drop = set(CALENDAR_FEATURES) | set(EARNINGS_FEATURES) | set(SURVIVOR_ONLY)
    feats = [c for c in no_cal if c not in drop]
    if len(feats) != N_PIT_FEATURES:
        raise ValueError(f"pit_no_cal has {len(feats)} features, the pre-registration fixed {N_PIT_FEATURES}")
    return feats


def arms() -> list[dict]:
    out = []
    for name, target, h in (("L0", "y_L0", 13), ("L2_10", "y_L2_10", 10), ("L2_20", "y_L2_20", 20)):
        for suffix in ("", "__perm"):
            out.append({"name": f"{name}__pit{suffix}", "target": target + suffix, "features": "pit_no_cal",
                        "horizon_sessions": h, "embargo_days": embargo_days(h), "control": bool(suffix)})
    return out


def securities(limit: int | None = None) -> pd.DataFrame:
    sec = pd.read_parquet(SECURITIES)
    sec = sec[sec["kind"] == "stock"].sort_values("sec_id").reset_index(drop=True)
    return sec.head(limit) if limit else sec


def data_end() -> pd.Timestamp:
    """Snapshot end of the point-in-time daily set: the median last bar of securities still
    trading (the two fetch batches ended 2026-09-24 and 2026-10-02). Rows after it would see
    only part of the universe and stale daily inputs."""
    sec = pd.read_parquet(SECURITIES, columns=["kind", "stopped", "last_bar"])
    return sec.loc[(sec["kind"] == "stock") & ~sec["stopped"].astype(bool), "last_bar"].median()


def valid_prices(bars: pd.DataFrame | None) -> bool:
    """False if any adjusted OHLC price is <= 0. Alpaca's split+dividend adjustment drives a
    few series negative (JTAI hourly: 5,637 of 6,205 bars; LDTC~1 daily: every bar); such a
    series is not a price history, so the security is left out rather than patched."""
    if bars is None:
        return True
    px = bars[[c for c in ("open", "high", "low", "close") if c in bars.columns]].to_numpy(float)
    return bool((px[np.isfinite(px)] > 0).all())


def _session(ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    return session_dates(pd.DatetimeIndex(ts))


def _read_daily(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d.index = pd.DatetimeIndex(pd.to_datetime(d.pop("timestamp"), utc=True))
    d.columns = [c.lower() for c in d.columns]
    d = d.sort_index()
    return d[~d.index.duplicated(keep="last")]


def _read_4h(path: Path) -> pd.DataFrame:
    b = pd.read_parquet(path)
    b.index = pd.DatetimeIndex(b.index, tz="UTC") if b.index.tz is None else b.index
    return b


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------

def life_4h(hourly: pd.DataFrame, first: pd.Timestamp | None, last: pd.Timestamp | None) -> pd.DataFrame:
    """4H bars from one symbol's hourly bars, inside one security's life (session dates inclusive)."""
    h = hourly[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    keep = h["volume"] > 0
    day = _session(h["timestamp"])
    if first is not None:
        keep &= day >= _session([first])[0]
    if last is not None:
        keep &= day <= _session([last])[0]
    return resample_1h_to_4h(h[keep.to_numpy()])


def _bars_job(args: tuple) -> str:
    key, sym, first, last, path_s, force = args
    path = Path(path_s)
    if path.exists() and not force:
        return "cached"
    src = HOURLY_DIR / f"{sym}.parquet"
    if not src.exists():
        return "no_hourly"
    b4 = life_4h(pd.read_parquet(src), first, last)
    if b4.empty:
        return "empty"
    if not valid_prices(b4):
        return "bad_prices"
    b4.to_parquet(path)
    return "written"


def build_bars(sec: pd.DataFrame, work: Path, *, workers: int, force: bool = False) -> dict[str, int]:
    out_dir, ctx_dir = work / "bars_4h", work / "bars_4h" / "_context"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(sym, sym, None, None, str(ctx_dir / f"{sym}.parquet"), force) for sym in CONTEXT]
    # A stopped security ends at its last daily bar (a reused ticker's next life starts later).
    # A live one keeps every hourly bar, so windows near the daily snapshot end still get exits.
    jobs += [(r.sec_id, r.symbol, r.first_bar, r.last_bar if r.stopped else None,
              str(out_dir / f"{r.sec_id}.parquet"), force) for r in sec.itertuples()]
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for status in ex.map(_bars_job, jobs, chunksize=16):
            counts[status] = counts.get(status, 0) + 1
    missing_ctx = [s for s in CONTEXT if not (ctx_dir / f"{s}.parquet").exists()]
    if "SPY" in missing_ctx:
        raise FileNotFoundError("SPY hourly bars are missing; the label calendar and SPY returns need them")
    if missing_ctx:
        logger.warning("context ETFs without bars (their features come out NaN): %s", missing_ctx)
    logger.info("bars: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

_CTX: dict[str, pd.DataFrame] = {}
_CAL: pd.DatetimeIndex | None = None


def _init_worker(work: Path) -> None:
    global _CAL
    ctx_dir = work / "bars_4h" / "_context"
    _CTX.clear()
    for p in ctx_dir.glob("*.parquet"):
        _CTX[p.stem] = _read_4h(p)
    _CAL = pd.DatetimeIndex(sorted(set(_session(_CTX["SPY"].index))))


def security_rows(sec_id: str, stopped: bool, b4: pd.DataFrame, daily: pd.DataFrame | None,
                  ctx: dict[str, pd.DataFrame], cal: pd.DatetimeIndex, feats: list[str]) -> pd.DataFrame | None:
    """One security's features, forward labels and universe inputs, per 4H bar."""
    if len(b4) < MIN_4H_BARS or daily is None:
        return None
    f = build_ticker_features_4h(ticker=sec_id, df_4h=b4, df_1d=daily, ctx_4h=ctx, ticker_meta={})
    if f is None:
        return None
    own = [c for c in feats if c not in XSEC_COLS]
    sources = [c for c in [*_XSEC_RANK_SPECS, "dist_to_52w_high_atr"] if c not in own]
    out = f[own + sources].astype("float32")
    for c in _PRECISION_SENSITIVE_COLS & set(out.columns):    # ranked later: float32 would tie them
        out[c] = f[c].astype("float64")
    fwd = ticker_forward(b4, cal, HORIZONS, flagged=flagged_sessions(daily), exit_at_last_bar=stopped)
    for h in HORIZONS:
        if f"clamped_{h}" not in fwd:
            fwd[f"clamped_{h}"] = np.float32(0.0)
    out = out.join(fwd, how="left")
    out["px"] = b4["close"].reindex(out.index).astype("float32")
    # Universe inputs from the daily bars, lagged one session (known before the session opens).
    dv = (daily["close"] * daily["volume"]).rolling(20, min_periods=10).mean().shift(1)
    lag = pd.DataFrame({"adv20_lag": dv, "close_lag": daily["close"].shift(1)})
    lag.index = _session(lag.index)
    lag = lag[~lag.index.duplicated(keep="last")]
    sess = _session(out.index)
    for c in lag.columns:
        out[c] = lag[c].reindex(sess).to_numpy("float32")
    # The clamped exit session (a stopped security's last session) prices SPY over the same window.
    last_i = int(cal.get_indexer([sess.max()])[0]) if stopped else -1
    out["last_sess"] = np.float32(last_i) if last_i >= 0 else np.float32(np.nan)
    out = out[out.index >= START]
    return out if len(out) else None


def _rows_job(args: tuple) -> tuple[str, str]:
    sec_id, stopped, work_s, feats = args
    work = Path(work_s)
    done = work / "rows" / "_done" / sec_id
    if done.exists():
        return sec_id, "cached"
    b4_path = work / "bars_4h" / f"{sec_id}.parquet"
    if not b4_path.exists():
        return sec_id, "no_bars"
    daily = _read_daily(DAILY_DIR / f"{sec_id}.parquet")
    if not valid_prices(daily):
        done.parent.mkdir(parents=True, exist_ok=True)
        done.touch()
        return sec_id, "bad_prices"
    out = security_rows(sec_id, stopped, _read_4h(b4_path), daily, _CTX, _CAL, feats)
    if out is not None:
        out = out.reset_index(names="timestamp")
        out.insert(1, "ticker", sec_id)
        for year, part in out.groupby(out["timestamp"].dt.year):
            d = work / "rows" / f"year={year}"
            d.mkdir(parents=True, exist_ok=True)
            part.to_parquet(d / f"{sec_id}.parquet", index=False)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.touch()
    return sec_id, "written" if out is not None else "too_short"


def build_rows(sec: pd.DataFrame, work: Path, *, workers: int, force: bool = False) -> dict[str, int]:
    if force:
        shutil.rmtree(work / "rows", ignore_errors=True)
    feats = pit_features()
    jobs = [(r.sec_id, bool(r.stopped), str(work), feats) for r in sec.itertuples()]
    counts: dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(work,)) as ex:
        for i, (_, status) in enumerate(ex.map(_rows_job, jobs, chunksize=8), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 500 == 0:
                logger.info("rows: %d/%d (%.0fs) %s", i, len(jobs), time.time() - t0, counts)
    logger.info("rows: %s in %.0fs", counts, time.time() - t0)
    return counts


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------

def spy_frames(spy4h: pd.DataFrame, cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """SPY forward returns per bar, SPY's executable entry per bar, SPY close per session ordinal."""
    fwd = ticker_forward(spy4h, cal, HORIZONS)[[f"fret_{h}" for h in HORIZONS]]
    fwd = fwd.rename(columns={f"fret_{h}": f"spy_fret_{h}" for h in HORIZONS})
    b = spy4h.sort_index()
    s = pd.Series(np.arange(len(cal)), index=cal).reindex(_session(b.index)).to_numpy(float)
    gap = np.append(s[1:], np.nan) - s
    entry = np.where((gap == 0) | (gap == 1), np.append(b["open"].to_numpy(float)[1:], np.nan), np.nan)
    sess_close = pd.Series(b["close"].to_numpy(float)).groupby(s).last().reindex(range(len(cal))).to_numpy()
    return fwd, pd.Series(entry, index=b.index), sess_close


def select_universe(df: pd.DataFrame, size: int = UNIVERSE_SIZE, min_close: float = MIN_LAGGED_CLOSE) -> pd.DataFrame:
    """Per bar: lagged close >= min_close, then the ``size`` largest lagged dollar volumes."""
    ok = df[(df["close_lag"] >= min_close) & df["adv20_lag"].notna()]
    rank = ok.groupby("timestamp")["adv20_lag"].rank(ascending=False, method="first")
    return ok[rank <= size]


def cross_section(df: pd.DataFrame, spy_fwd: pd.DataFrame, spy_entry: pd.Series, spy_close: np.ndarray,
                  *, seed: int) -> pd.DataFrame:
    """Universe, SPY-relative inputs, ranks and targets for one block of whole timestamps."""
    df = select_universe(df).copy()
    df = df.merge(spy_fwd, left_on="timestamp", right_index=True, how="left")
    entry = spy_entry.reindex(pd.DatetimeIndex(df["timestamp"])).to_numpy()
    last = df["last_sess"].to_numpy()
    ok_last = ~np.isnan(last)
    spy_at_last = np.full(len(df), np.nan)
    spy_at_last[ok_last] = spy_close[last[ok_last].astype(int)]
    for h in HORIZONS:
        clamp = df[f"clamped_{h}"].to_numpy() == 1
        df[f"spy_fret_{h}"] = np.where(clamp, spy_at_last / entry - 1.0, df[f"spy_fret_{h}"].to_numpy())
    df = df.set_index(["timestamp", "ticker"])
    df = _add_cross_sectional_features(df).reset_index()
    df["adv_rank"] = df.groupby("timestamp")["adv20_lag"].rank(pct=True).astype("float32")
    return add_arm_targets(df, HORIZONS, seed=seed)


def build_matrix(work: Path, export_dir: Path, *, n_securities: int) -> Path:
    feats = pit_features()
    spy4h = _read_4h(work / "bars_4h" / "_context" / "SPY.parquet")
    cal = pd.DatetimeIndex(sorted(set(_session(spy4h.index))))
    spy_fwd, spy_entry, spy_close = spy_frames(spy4h, cal)
    end = _session([data_end()])[0]
    keep = ["timestamp", "ticker", *feats, "px", "adv_rank", "daily_atr_pct", "beta_spy_60", *YARDSTICKS,
            *TARGETS, *(f"{t}__perm" for t in TARGETS)]
    keep = list(dict.fromkeys(keep))
    export_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = export_dir / "stage3_matrix.parquet"
    writer, n_rows, tickers, first, last_ts = None, 0, set(), None, None
    years = sorted(int(p.name.split("=")[1]) for p in (work / "rows").glob("year=*"))
    for year in years:
        df = ds.dataset(work / "rows" / f"year={year}", format="parquet").to_table().to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[_session(df["timestamp"]) <= end]
        if df.empty:
            continue
        df = cross_section(df, spy_fwd, spy_entry, spy_close, seed=SEED + year)
        df = df[keep].sort_values(["timestamp", "ticker"])
        num = [c for c in keep if c not in ("timestamp", "ticker")]
        df[num] = df[num].astype("float32")
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(matrix_path, table.schema, compression="zstd")
        writer.write_table(table)
        n_rows += len(df)
        tickers.update(df["ticker"].unique())
        first = first or df["timestamp"].min()
        last_ts = df["timestamp"].max()
        # Illiquid names with no trade in a slot's first hour get a later bar timestamp; such
        # stray bars get no targets and the trainer drops them (as in Stage 2).
        names = df.groupby("timestamp")["ticker"].transform("size")
        full = names >= MIN_NAMES_PER_BAR
        logger.info("matrix %d: %d rows, %d securities, %.0f names per full bar, %.1f%% of rows on stray bars",
                    year, len(df), df["ticker"].nunique(), names[full].groupby(df.loc[full, "timestamp"]).first().median(),
                    100 * (1 - full.mean()))
        del df, table
    if writer is None:
        raise ValueError(f"no rows under {work / 'rows'}")
    writer.close()

    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "prereg": "research/execution_quality/27_stage3_preregistration.md", "rule": RULE,
        "matrix": matrix_path.name, "rows": n_rows, "tickers": len(tickers), "securities_considered": n_securities,
        "range": [str(first), str(last_ts)], "data_end_session": str(end.date()),
        "final_test_start": FINAL_TEST_START, "min_names_per_bar": MIN_NAMES_PER_BAR,
        "horizons_sessions": list(HORIZONS),
        "universe": {"size": UNIVERSE_SIZE, "min_lagged_close": MIN_LAGGED_CLOSE,
                     "rank_by": "20-session mean daily close*volume, lagged one session"},
        "feature_sets": {"pit_no_cal": feats},
        "arms": arms(),
        "walk_forward": {"train_months": 24, "test_months": 6, "inner_val_frac": 0.2},
        "xgb_params": {"n_estimators": 800, "learning_rate": 0.04, "max_depth": 5, "subsample": 0.85,
                       "colsample_bytree": 0.85, "early_stopping_rounds": 60},
        "diagnostic_columns": [c for c in keep if c not in ("timestamp", "ticker") and c not in feats
                               and not c.startswith("y_")],
        "notes": "Point-in-time universe (delisted names included), SIP hourly bars, split+dividend adjusted. "
                 "A stopped security's windows exit at its last close (clamped_h = 1, SPY over the same window). "
                 "y_* are binary targets; *__perm are within-bar shuffles. The trainer drops rows with any NaN "
                 "in the arm's features or target, and bars with fewer than min_names_per_bar names.",
    }
    (export_dir / "stage3_manifest.json").write_text(json.dumps(manifest, indent=2))
    trainer = TRAINER.read_text()
    pin = 'STAGE = "stage2"'
    if trainer.count(pin) != 1:
        raise ValueError(f"trainer must contain exactly one `{pin}` line to pin")
    (export_dir / "stage3_train_colab.py").write_text(trainer.replace(pin, 'STAGE = "stage3"'))
    logger.info("bundle in %s: %s (%.2f GB, %d rows, %d securities), stage3_manifest.json, stage3_train_colab.py",
                export_dir, matrix_path.name, matrix_path.stat().st_size / 1e9, n_rows, len(tickers))
    return matrix_path


def check_matrix(export_dir: Path) -> dict:
    """Sanity summary of a built matrix: breadth, completeness, label rates, delisting rows, price floor."""
    m = json.loads((export_dir / "stage3_manifest.json").read_text())
    feats = m["feature_sets"]["pit_no_cal"]
    cols = ["timestamp", "ticker", *feats, "px", *(f"fret_{h}" for h in HORIZONS[:2]),
            *(f"clamped_{h}" for h in HORIZONS[:2]), *TARGETS, f"{TARGETS[1]}__perm"]
    df = pq.read_table(export_dir / m["matrix"], columns=cols).to_pandas()
    names = df.groupby("timestamp")["ticker"].transform("size")
    full = names >= MIN_NAMES_PER_BAR
    out = {"rows": len(df), "securities": int(df["ticker"].nunique()),
           "range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
           "names_per_full_bar": float(names[full].groupby(df.loc[full, "timestamp"]).first().median()),
           "stray_row_share": float(1 - full.mean()),
           "complete_feature_rows": float(df[feats].notna().all(axis=1).mean()),
           "worst_feature_nan": df[feats].isna().mean().sort_values(ascending=False).head(5).round(4).to_dict(),
           "px_min": float(df["px"].min())}
    for c in [*TARGETS, f"{TARGETS[1]}__perm"]:
        out[f"{c}_labelled"], out[f"{c}_positive"] = float(df[c].notna().mean()), float(df[c].mean())
    for h in HORIZONS[:2]:
        cl = df[f"clamped_{h}"] == 1
        out[f"clamped_{h}_share"] = float(cl.mean())
        out[f"abs_fret_{h}_over_200pct"] = float((df[f"fret_{h}"].abs() > 2).mean())
    problems = [k for k, bad in (("px_min", out["px_min"] <= 0),
                                 ("complete_feature_rows", out["complete_feature_rows"] < 0.9),
                                 ("names_per_full_bar", out["names_per_full_bar"] < 0.8 * UNIVERSE_SIZE),
                                 *((f"{t}_positive", t != "y_L0" and abs(out[f"{t}_positive"] - 0.2) > 0.01)
                                   for t in TARGETS)) if bad]
    out["problems"] = problems
    (export_dir / "stage3_checks.json").write_text(json.dumps(out, indent=2))
    logger.info("checks: %s", json.dumps(out))
    if problems:
        logger.error("matrix checks FAILED: %s", problems)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # build_ticker_features_4h warns per call that the (unused) regime tables are stale for history
    logging.getLogger("strategies.momentum_expansion.features").setLevel(logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["bars", "rows", "matrix", "check", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N securities only (smoke build)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--work-dir", type=Path, default=STAGE3_DIR)
    ap.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    ap.add_argument("--force", action="store_true", help="rebuild bars/rows that already exist")
    a = ap.parse_args()
    sec = securities(a.limit)
    if a.stage in ("bars", "all"):
        build_bars(sec, a.work_dir, workers=a.workers, force=a.force)
    if a.stage in ("rows", "all"):
        build_rows(sec, a.work_dir, workers=a.workers, force=a.force)
    if a.stage in ("matrix", "all"):
        build_matrix(a.work_dir, a.export_dir, n_securities=len(sec))
    if a.stage in ("check", "all"):
        check_matrix(a.export_dir)


if __name__ == "__main__":
    main()
