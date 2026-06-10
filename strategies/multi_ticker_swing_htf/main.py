"""CLI for multi-ticker swing higher-time-frame research."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.data.bars import fetch_context_bars, fetch_universe_bars
from strategies.momentum_expansion.config.momentum_config import RAW_4H_DIR
from strategies.momentum_expansion.features.feature_matrix_4h import build_all_features_4h

from strategies.multi_ticker_swing_htf.config import FEATURES_COMBINED, TRAIN_END, TRAIN_START
from strategies.multi_ticker_swing_htf.labels import build_all_labels_4h, build_training_matrix
from core.shared_universe.universe import build_shared_universe, load_shared_universe, shared_tickers, summarize_universe


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(name)s %(message)s")


def audit_4h_bars(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        path = RAW_4H_DIR / f"{ticker}.parquet"
        row = {"ticker": ticker, "has_4h": path.exists(), "bars_4h": 0, "first_ts": pd.NaT, "last_ts": pd.NaT}
        if path.exists():
            try:
                df = pd.read_parquet(path, columns=["timestamp"])
                ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                row.update({"bars_4h": int(ts.notna().sum()), "first_ts": ts.min(), "last_ts": ts.max()})
            except Exception as exc:
                row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="multi_ticker_swing_htf CLI")
    parser.add_argument("--log", default="INFO")
    parser.add_argument("--build-universe", action="store_true")
    parser.add_argument("--eligible-only", action="store_true", default=True)
    parser.add_argument("--include-ineligible", action="store_true")
    parser.add_argument("--fetch-context", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--audit-bars", action="store_true")
    parser.add_argument("--build-features", action="store_true")
    parser.add_argument("--build-labels", action="store_true")
    parser.add_argument("--build-matrix", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()
    _setup_logging(args.log)

    eligible_only = args.eligible_only and not args.include_ineligible
    universe = build_shared_universe() if args.build_universe else load_shared_universe(eligible_only=False)
    if args.build_universe:
        logging.info("shared universe summary: %s", summarize_universe(universe))

    tickers = [t.upper() for t in args.tickers] if args.tickers else shared_tickers(eligible_only=eligible_only)

    if args.fetch_context:
        fetch_context_bars(force=args.force)
    if args.fetch:
        logging.info("Fetching %d tickers from %s to %s", len(tickers), TRAIN_START, TRAIN_END)
        fetch_universe_bars(tickers=tickers, force=args.force)
    if args.audit_bars:
        audit = audit_4h_bars(tickers)
        out = Path("strategies/multi_ticker_swing_htf/data/processed/4h_bar_audit.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(out, index=False)
        missing = int((~audit["has_4h"]).sum())
        stale = int((pd.to_datetime(audit["last_ts"], utc=True, errors="coerce") < pd.Timestamp(TRAIN_END, tz="UTC") - pd.Timedelta(days=30)).sum())
        logging.info("4H audit saved -> %s; missing=%d stale_30d=%d", out, missing, stale)
    if args.build_features:
        build_all_features_4h(tickers=tickers, out_path=FEATURES_COMBINED, force=args.force)
    if args.build_labels:
        build_all_labels_4h(tickers=tickers, force=args.force)
    if args.build_matrix:
        build_training_matrix(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
