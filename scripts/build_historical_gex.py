#!/usr/bin/env python
"""Reconstruct a historical dealer-positioning (GEX) surface per (ticker, date)
from expired-contract open interest + cached daily contract bars, and (with
``--validate``) score it against the 16 real Schwab dealer snapshots.

Context: research/options_experiment/04_tail_and_thesis_review.md section 4 --
the real GEX snapshots (Data/dealer_positioning/historical_snapshots/) only
cover 2026-07-02..2026-07-24, zero overlap with the trade history
(2025-05-20..2026-06-04), so the gamma-squeeze mechanism has never been
tested against a single trade. This script builds the reconstruction (via
research/options_lab/gex_reconstruct.py) needed to close that gap, and
separately validates it on the one window where ground truth exists.

Two modes:
  --mode reconstruct : compute reconstructed GEX for --tickers x --dates
      (or --dates-from-real-snapshots) x all 3 oi_source variants, checkpoint
      to --out (atomic, resumable -- rows already present for a
      (symbol, date, oi_source) key are skipped on restart).
  --mode validate     : load the reconstructed output plus the real
      dealer_level_summary.parquet snapshots for the same symbols/dates/scope,
      and report Spearman/Pearson correlation of total_gex/net_gex and
      call_wall/put_wall/gamma_flip hit rates (within 1 strike, within 2%)
      per oi_source variant.

Run (reconstruction over the validation window, ~30 liquid names):
  .venv/bin/python scripts/build_historical_gex.py --mode reconstruct \
      --dates-from-real-snapshots --scope through_month

Run (validation, after reconstruction has produced rows):
  .venv/bin/python scripts/build_historical_gex.py --mode validate \
      --scope through_month
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.options_lab import chain_cache, gex_reconstruct as gr, pricing, surface  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_historical_gex")

BARS_1D_DIR = REPO_ROOT / "Data" / "shared" / "bars" / "1d"
REAL_SNAPSHOT_ROOT = REPO_ROOT / "Data" / "dealer_positioning" / "historical_snapshots"
OUT_DEFAULT = REPO_ROOT / "research" / "options_experiment" / "data" / "gex_reconstructed.parquet"
VALIDATION_REPORT_DEFAULT = REPO_ROOT / "research" / "options_experiment" / "data" / "gex_validation.json"

# Liquid, large-cap sample spanning sectors -- verified present with full
# 13-day through_month-scope coverage in the real snapshot archive (checked
# interactively before writing this script). Not the whole universe: this is
# a validation sample, per the task's own "sample proportionately" guidance.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "NFLX",
    "CRM", "COST", "WMT", "JPM", "BAC", "XOM", "UNH", "AVGO", "ORCL", "ADBE",
    "INTC", "MU", "PLTR", "DIS", "GS", "QCOM",
]

STRIKE_WINDOW_PCT = 0.50
BAR_LOOKBACK_DAYS = 150


def _real_snapshot_dates() -> list[str]:
    dates = []
    for p in sorted(glob.glob(str(REAL_SNAPSHOT_ROOT / "*" / "dealer_level_summary.parquet"))):
        raw = Path(p).parent.name
        dates.append(f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}")
    return dates


def _load_underlying_daily(ticker: str) -> Optional[pd.DataFrame]:
    path = BARS_1D_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
    return df


def _spot_on(underlying: pd.DataFrame, d: date) -> Optional[float]:
    row = underlying.loc[underlying["date"] == d]
    if row.empty:
        return None
    return float(row.iloc[-1]["close"])


def _existing_keys(out_path: Path) -> set[tuple[str, str, str]]:
    if not out_path.exists():
        return set()
    df = pd.read_parquet(out_path, columns=["symbol", "date", "oi_source"])
    return set(zip(df["symbol"], df["date"], df["oi_source"]))


def _flush(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["symbol", "date", "oi_source"], keep="last")
    else:
        combined = new_df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    logger.info("flushed %d new rows -> %s (total now %d)", len(new_df), out_path, len(combined))


def _bars_long(bars_raw: pd.DataFrame) -> pd.DataFrame:
    if bars_raw.empty:
        return pd.DataFrame(columns=["osi_symbol", "date", "volume"])
    out = bars_raw[["osi_symbol", "t", "v"]].copy()
    out["date"] = pd.to_datetime(out["t"], utc=True).dt.date
    out = out.rename(columns={"v": "volume"})
    return out[["osi_symbol", "date", "volume"]]


def _bars_for_surface(bars_raw: pd.DataFrame, asof_d: date) -> pd.DataFrame:
    if bars_raw.empty:
        return pd.DataFrame(columns=["osi_symbol", "ts", "close"])
    out = bars_raw[["osi_symbol", "t", "c"]].copy()
    out["ts"] = pd.to_datetime(out["t"], utc=True)
    out = out.rename(columns={"c": "close"})
    out = out.loc[out["ts"].dt.date <= asof_d]
    return out[["osi_symbol", "ts", "close"]]


def reconstruct_ticker_date(
    ticker: str,
    asof_d: date,
    *,
    contracts_full: pd.DataFrame,
    available_expiries: list[str],
    underlying: pd.DataFrame,
    scope: str,
    env_file: Optional[str],
) -> tuple[list[dict], dict]:
    """Reconstruct all 3 oi_source variants for one (ticker, date). Returns
    (rows, coverage) where coverage records how much of the OI-weighted
    exposure had real terminal OI available -- the load-bearing caveat this
    whole exercise exists to surface, not hide."""
    spot = _spot_on(underlying, asof_d)
    if spot is None:
        return [], {"ticker": ticker, "date": asof_d.isoformat(), "skip_reason": "no_spot"}

    expiries = gr.select_expiration_window(available_expiries, asof_d, scope=scope)
    if not expiries:
        return [], {"ticker": ticker, "date": asof_d.isoformat(), "skip_reason": "no_expiries_in_scope"}

    sub = contracts_full[contracts_full["expiry"].isin(expiries)].copy()
    lo, hi = spot * (1.0 - STRIKE_WINDOW_PCT), spot * (1.0 + STRIKE_WINDOW_PCT)
    sub = sub[(sub["strike"] >= lo) & (sub["strike"] <= hi)]
    if sub.empty:
        return [], {"ticker": ticker, "date": asof_d.isoformat(), "skip_reason": "no_contracts_in_strike_window"}

    max_expiry = max(gr.parse_date(x) for x in expiries)
    bar_start = (asof_d - timedelta(days=BAR_LOOKBACK_DAYS)).isoformat()
    bar_end = max_expiry.isoformat()
    osi_symbols = sub["osi_symbol"].unique().tolist()
    bars_raw = chain_cache.fetch_bars(osi_symbols, "1Day", bar_start, bar_end, env_file=env_file)
    bars_asof = _bars_for_surface(bars_raw, asof_d)
    bars_long = _bars_long(bars_raw)

    iv_frames = []
    for expiry_val, group in sub.groupby("expiry"):
        expiry_d = gr.parse_date(expiry_val)
        T = (expiry_d - asof_d).days / 365.25
        tenor_for_r = max(T, 1.0 / 365.25)
        try:
            r = pricing.risk_free_rate(pd.Timestamp(asof_d), tenor_for_r)
        except ValueError as exc:
            logger.warning("risk_free_rate lookup failed for %s %s: %s", ticker, asof_d, exc)
            continue
        meta = group[["osi_symbol", "ticker", "expiry", "strike", "right"]]
        surf = surface.build_iv_surface(
            bars_asof[bars_asof["osi_symbol"].isin(group["osi_symbol"])],
            meta,
            asof=asof_d,
            spot=spot,
            r=r,
            q=0.0,
            price_field="close",
            american=False,
        )
        if surf.empty:
            continue
        gamma_df = gr.compute_gamma(surf[["strike", "right", "T", "iv"]], spot=spot, r=r, q=0.0)
        surf["gamma"] = gamma_df["gamma"].to_numpy()
        iv_frames.append(surf)

    if not iv_frames:
        return [], {"ticker": ticker, "date": asof_d.isoformat(), "skip_reason": "no_iv_solved"}

    contracts_iv = pd.concat(iv_frames, ignore_index=True)
    contracts_iv = contracts_iv.merge(
        sub[["osi_symbol", "open_interest", "open_interest_date"]], on="osi_symbol", how="left"
    )
    contracts_iv = contracts_iv.rename(columns={"open_interest": "terminal_oi"})

    variants = gr.compute_oi_variants(contracts_iv, bars_long, asof=asof_d)

    n_total = int(len(variants))
    n_with_terminal = int(variants["terminal_oi"].notna().sum())
    coverage = {
        "ticker": ticker,
        "date": asof_d.isoformat(),
        "scope": scope,
        "n_contracts": n_total,
        "n_with_terminal_oi": n_with_terminal,
        "terminal_oi_coverage_pct": (n_with_terminal / n_total) if n_total else 0.0,
        "skip_reason": None,
    }

    rows: list[dict] = []
    for oi_source in gr.OI_SOURCES:
        variant_rows = gr.build_rows_for_variant(variants, oi_source=oi_source, asof=asof_d)
        if variant_rows.empty:
            continue
        snapshot = gr.assemble_snapshot_row(
            variant_rows, symbol=ticker, date_str=asof_d.isoformat(), spot=spot, oi_source=oi_source
        )
        snapshot["scope"] = scope
        rows.append(snapshot)
    return rows, coverage


def run_reconstruct(args: argparse.Namespace) -> None:
    tickers = args.tickers or DEFAULT_TICKERS
    dates = args.dates or (_real_snapshot_dates() if args.dates_from_real_snapshots else [])
    if not dates:
        raise SystemExit("no dates given -- pass --dates or --dates-from-real-snapshots")
    dates_d = sorted({gr.parse_date(d) for d in dates})
    out_path = Path(args.out)
    done_keys = _existing_keys(out_path)
    coverage_rows: list[dict] = []

    expiry_fetch_end = (max(dates_d) + timedelta(days=70)).isoformat()
    expiry_fetch_start = min(dates_d).isoformat()

    buffer: list[dict] = []
    for ti, ticker in enumerate(tickers, start=1):
        underlying = _load_underlying_daily(ticker)
        if underlying is None:
            logger.warning("skip %s: no underlying 1d bars cached", ticker)
            continue
        try:
            contracts_full = chain_cache.discover_contracts_full(
                ticker, expiry_fetch_start, expiry_fetch_end, env_file=args.env_file
            )
        except Exception as exc:  # noqa: BLE001 -- one ticker's API failure must not kill the run
            logger.error("discover_contracts_full failed for %s: %s", ticker, exc)
            continue
        if contracts_full.empty:
            logger.warning("skip %s: no contracts discovered", ticker)
            continue
        available_expiries = sorted(contracts_full["expiry"].unique())

        for d in dates_d:
            keys_needed = [(ticker, d.isoformat(), src) for src in gr.OI_SOURCES]
            if all(k in done_keys for k in keys_needed):
                continue
            try:
                rows, coverage = reconstruct_ticker_date(
                    ticker, d,
                    contracts_full=contracts_full,
                    available_expiries=available_expiries,
                    underlying=underlying,
                    scope=args.scope,
                    env_file=args.env_file,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("reconstruct failed for %s %s: %s", ticker, d, exc)
                continue
            coverage_rows.append(coverage)
            buffer.extend(rows)
            for r_ in rows:
                done_keys.add((r_["symbol"], r_["date"], r_["oi_source"]))
            if len(buffer) >= args.flush_every:
                _flush(buffer, out_path)
                buffer = []
        logger.info("ticker %d/%d (%s) done", ti, len(tickers), ticker)

    _flush(buffer, out_path)
    cov_path = out_path.with_name(out_path.stem + "_coverage.jsonl")
    with cov_path.open("a", encoding="utf-8") as fh:
        for row in coverage_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    logger.info("reconstruction complete: coverage log -> %s", cov_path)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _load_real_snapshots(scope: str) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(REAL_SNAPSHOT_ROOT / "*" / "dealer_level_summary.parquet"))):
        df = pd.read_parquet(p)
        df = df[df["scope"] == scope]
        keep = [c for c in ["symbol", "snapshot_date", "spot", "total_gex", "net_gex",
                             "call_wall", "put_wall", "gamma_flip", "dealer_bias"] if c in df.columns]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"snapshot_date": "date"})
    if "net_gex" not in out.columns:
        out["net_gex"] = out["total_gex"]
    return out


def _hit_rate(real: pd.Series, recon: pd.Series, *, strike_inc: pd.Series, pct_tol: float = 0.02) -> dict:
    valid = real.notna() & recon.notna()
    if valid.sum() == 0:
        return {"n": 0, "hit_rate_1strike": None, "hit_rate_pct": None, "median_abs_pct_error": None}
    diff = (recon[valid] - real[valid]).abs()
    within_strike = (diff <= strike_inc[valid]).mean()
    pct_err = diff / real[valid].abs().replace(0.0, np.nan)
    within_pct = (pct_err <= pct_tol).mean()
    return {
        "n": int(valid.sum()),
        "hit_rate_1strike": float(within_strike),
        "hit_rate_pct": float(within_pct),
        "median_abs_pct_error": float(pct_err.median()) if pct_err.notna().any() else None,
    }


def run_validate(args: argparse.Namespace) -> None:
    recon_path = Path(args.out)
    if not recon_path.exists():
        raise SystemExit(f"no reconstructed output at {recon_path}; run --mode reconstruct first")
    recon = pd.read_parquet(recon_path)
    recon = recon[recon.get("scope", args.scope) == args.scope] if "scope" in recon.columns else recon
    real = _load_real_snapshots(args.scope)
    if real.empty:
        raise SystemExit("no real snapshot rows found for this scope")

    report: dict = {"scope": args.scope, "variants": {}}
    for oi_source in gr.OI_SOURCES:
        sub = recon[recon["oi_source"] == oi_source]
        if sub.empty:
            report["variants"][oi_source] = {"n": 0}
            continue
        merged = sub.merge(real, on=["symbol", "date"], how="inner", suffixes=("_recon", "_real"))
        if merged.empty:
            report["variants"][oi_source] = {"n": 0}
            continue

        # "1 strike" = the reconstructed ladder's own median strike spacing
        # for that (symbol, date) (persisted by assemble_snapshot_row as
        # strike_increment), not a fixed pct-of-spot guess -- a $8 stock's
        # $0.50 strikes and a $600 stock's $10 strikes need different bands.
        # Falls back to 1% of spot only on the rare row with no ladder
        # (e.g. a single-strike reconstruction where spacing is undefined).
        if "strike_increment" in merged.columns:
            fallback = merged["spot_real"] * 0.01
            strike_inc = pd.to_numeric(merged["strike_increment"], errors="coerce").fillna(fallback)
        else:
            strike_inc = merged["spot_real"] * 0.01

        gex_pearson_total = stats.pearsonr(merged["total_gex_recon"], merged["total_gex_real"]) if len(merged) > 1 else (np.nan, np.nan)
        gex_spearman_total = stats.spearmanr(merged["total_gex_recon"], merged["total_gex_real"]) if len(merged) > 1 else (np.nan, np.nan)
        net_pearson = stats.pearsonr(merged["net_gex_recon"], merged["net_gex_real"]) if len(merged) > 1 else (np.nan, np.nan)
        net_spearman = stats.spearmanr(merged["net_gex_recon"], merged["net_gex_real"]) if len(merged) > 1 else (np.nan, np.nan)

        report["variants"][oi_source] = {
            "n": int(len(merged)),
            "n_symbols": int(merged["symbol"].nunique()),
            "n_dates": int(merged["date"].nunique()),
            "total_gex_pearson_r": float(gex_pearson_total[0]),
            "total_gex_pearson_p": float(gex_pearson_total[1]),
            "total_gex_spearman_rho": float(gex_spearman_total[0]),
            "total_gex_spearman_p": float(gex_spearman_total[1]),
            "net_gex_pearson_r": float(net_pearson[0]),
            "net_gex_spearman_rho": float(net_spearman[0]),
            "call_wall": _hit_rate(merged["call_wall_real"], merged["call_wall_recon"], strike_inc=strike_inc),
            "put_wall": _hit_rate(merged["put_wall_real"], merged["put_wall_recon"], strike_inc=strike_inc),
            "gamma_flip": _hit_rate(merged["gamma_flip_real"], merged["gamma_flip_recon"], strike_inc=strike_inc),
        }

    out_path = Path(args.validation_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    logger.info("validation report -> %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["reconstruct", "validate"], required=True)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--dates-from-real-snapshots", action="store_true")
    parser.add_argument("--scope", default="through_month",
                         choices=["next_expiration", "daily_week", "through_month", "two_months"])
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--validation-out", default=str(VALIDATION_REPORT_DEFAULT))
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    if args.mode == "reconstruct":
        run_reconstruct(args)
    else:
        run_validate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
