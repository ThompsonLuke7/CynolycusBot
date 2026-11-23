"""Repair hourly bars that the intraday catch-ups cached while still forming.

Until 2026-11-22, ``bars.fetch_one`` cached the hour containing ``end`` (now - 20 min) and
resumed at ``last + 1 min``, so that bar was never revised. In a 12-ticker sample, the
13:00-15:00 ET hours were partial on ~75% of sessions from 2026-06-18 on (median close off
0.135%, volume off far more). The fixed fetcher heals only the latest session; this script
repairs the history.

For each cached ticker it pulls SIP 1H bars for [--since, --end] in one request and replaces
the cached rows in that window with the fresh ones. A cached row the fresh pull lacks is kept
(and counted). Rows outside the window are never touched. --since defaults to 2026-06-15, after
the 1H cache's IEX->SIP switch (~2026-06-10), so the repair never mixes feeds.

Dry run by default: fetch, compare, write a report, change nothing. ``--apply`` first copies
every file it will change to ``Data/shared/bars/_backup_partial_repair_<stamp>/`` and then
rewrites it. It also rebuilds the 4H cache (full resample) for each changed ticker. The nightly
``--build-features --refresh-stale`` then rebuilds features_4h for those tickers (bars newer
than features). The Meta matrix only appends, so its history needs a separate rebuild.

  PYTHONPATH=. .venv/bin/python scripts/repair_partial_intraday_bars.py --sample 150      # dry run
  PYTHONPATH=. .venv/bin/python scripts/repair_partial_intraday_bars.py --apply           # all
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from strategies.momentum_expansion.config.momentum_config import (  # noqa: E402
    BAR_CONFIG, CONTEXT_TICKERS, SECTOR_ETFS,
)
from strategies.momentum_expansion.data import bars  # noqa: E402

BARS_ROOT = REPO / "Data/shared/bars"
LOG_DIR = BARS_ROOT / "_repair_logs"
DEFAULT_SINCE = "2026-06-15"
SIP_DELAY = dt.timedelta(minutes=20)
PARTIAL_LO, PARTIAL_HI = 0.15, 0.98   # cached/fresh volume ratio band called "partial"


def compare(cached: pd.DataFrame, fresh: pd.DataFrame, since: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Row-level comparison of the cached window against a fresh pull. Pure; no I/O."""
    c = cached[(cached["timestamp"] >= since) & (cached["timestamp"] <= end)].set_index("timestamp")
    f = fresh.set_index("timestamp")
    both = c.index.intersection(f.index)
    ratio = (c.loc[both, "volume"] / f.loc[both, "volume"].replace(0, np.nan)).astype(float)
    close_diff = (c.loc[both, "close"] / f.loc[both, "close"] - 1).abs().astype(float)
    changed = (c.loc[both, ["open", "high", "low", "close", "volume"]].to_numpy()
               != f.loc[both, ["open", "high", "low", "close", "volume"]].to_numpy()).any(axis=1)
    return {
        "window_rows_cached": int(len(c)),
        "fresh_rows": int(len(f)),
        "rows_matched": int(len(both)),
        "rows_changed": int(changed.sum()),
        "rows_partial": int(((ratio >= PARTIAL_LO) & (ratio < PARTIAL_HI)).sum()),
        "rows_added": int(len(f.index.difference(c.index))),
        "rows_kept_missing_from_fresh": int(len(c.index.difference(f.index))),
        "median_abs_close_diff_changed": float(close_diff[changed].median()) if changed.any() else 0.0,
    }


def splice(cached: pd.DataFrame, fresh: pd.DataFrame, since: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Cached rows with the window's overlapping/new rows replaced by the fresh pull."""
    in_window = (cached["timestamp"] >= since) & (cached["timestamp"] <= end)
    keep_missing = cached[in_window & ~cached["timestamp"].isin(fresh["timestamp"])]
    out = pd.concat([cached[~in_window], keep_missing, fresh[cached.columns.intersection(fresh.columns)]],
                    ignore_index=True)
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def repair_one(ticker: str, kind: str, *, since: pd.Timestamp, end: pd.Timestamp,
               apply: bool, backup: Path) -> dict:
    path = bars._path_for(ticker, kind)
    row = {"ticker": ticker, "kind": kind}
    try:
        cached = _read(path)
        fresh = bars.fetch_with_retry(
            ticker=ticker, timeframe=BAR_CONFIG["primary_timeframe"], limit=None,
            start=since.isoformat(), end=end.isoformat(), label=f"repair {ticker}/{kind}")
        if fresh is None or fresh.empty:
            return {**row, "status": "no_fresh_bars"}
        fresh = bars._complete_bars_only(fresh, timeframe=BAR_CONFIG["primary_timeframe"], end=end)
        fresh["timestamp"] = pd.to_datetime(fresh["timestamp"], utc=True)
        row.update(compare(cached, fresh, since, end))
        if row["rows_changed"] == 0 and row["rows_added"] == 0:
            return {**row, "status": "clean"}
        if not apply:
            return {**row, "status": "would_repair"}
        dest = backup / kind / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        bars._atomic_to_parquet(splice(cached, fresh, since, end), path, index=False)
        if kind == "1h":
            bars.build_4h_for(ticker, force=True)
        else:   # context: same full resample fetch_context_bars does
            df_4h = bars.resample_1h_to_4h(_read(path))
            if not df_4h.empty:
                bars._atomic_to_parquet(df_4h.reset_index(), bars._path_for(ticker, "4h"), index=False)
        return {**row, "status": "repaired"}
    except Exception as exc:  # noqa: BLE001 - reported per ticker
        return {**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", default=DEFAULT_SINCE)
    ap.add_argument("--end", default=None, help="default: now - 20 min (SIP delay)")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None, help="random sample of cached 1H tickers")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--apply", action="store_true", help="write repairs (default: dry run)")
    args = ap.parse_args(argv)

    since = pd.Timestamp(args.since, tz="UTC")
    end = (pd.Timestamp(args.end) if args.end
           else pd.Timestamp(dt.datetime.now(dt.timezone.utc) - SIP_DELAY))
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    tickers = args.tickers or sorted(p.stem for p in (BARS_ROOT / "1h").glob("*.parquet"))
    if args.sample and not args.tickers:
        tickers = sorted(random.Random(args.seed).sample(tickers, min(args.sample, len(tickers))))
    jobs = [(t, "1h") for t in tickers]
    if not args.no_context and not args.tickers and not args.sample:
        jobs += [(t, "context") for t in list(CONTEXT_TICKERS) + list(SECTOR_ETFS)
                 if bars._path_for(t, "context").exists()]

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = BARS_ROOT / f"_backup_partial_repair_{stamp}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(jobs)} files, window {since} .. {end}")
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(repair_one, t, k, since=since, end=end, apply=args.apply, backup=backup)
                for t, k in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out_csv = LOG_DIR / f"repair_{stamp}{'' if args.apply else '_dryrun'}.csv"
    df.to_csv(out_csv, index=False)
    num = df.select_dtypes("number")
    summary = {
        "mode": "apply" if args.apply else "dry_run", "since": str(since), "end": str(end),
        "files": len(df), "status_counts": df["status"].value_counts().to_dict(),
        "totals": {c: int(num[c].sum()) for c in num.columns if c.startswith("rows") or c.endswith("rows")},
        "share_files_with_partial": float((df.get("rows_partial", pd.Series(dtype=float)) > 0).mean()),
        "backup_dir": str(backup) if args.apply else None, "report": str(out_csv),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (LOG_DIR / f"repair_{stamp}{'' if args.apply else '_dryrun'}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    errors = int((df["status"] == "error").sum())
    return 1 if errors > 0.1 * max(1, len(df)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
