"""Repair the 30m swing raw store (``RAW_30M_DIR``) from a fresh pull under the training rule.

From 2026-04-15 on, live wrote this store in its prefill save-back: partial bars frozen at
mid-session restarts, holes (e.g. 04-30, 10-15), no vwap/trade_count, and no sessions after
the last prefill. Live features read their daily history and 500-bar window from it, so those
defects reach the model (2026-11-23 parity: ``daily_ema_dist_50``, ``distance_to_52w_low``).

For each file of a live-universe ticker it pulls 30m IEX split-adjusted bars (bar starts
09:30-16:00 ET, the ``fetch_data.fetch_ticker`` rule) for [--since, now] and replaces the
window with them. A file with a price-basis change in the window (a split since the original
fetch) is refetched over its whole history instead, so the file stays on one basis, which is
what a research pull made today returns. Cached window rows absent from the fresh pull are
dropped (a fresh pull is the reference); they are counted.

Dry run by default: fetch, compare, report, change nothing. ``--apply`` copies each file it
rewrites to ``<raw>/_backup_30m_repair_<stamp>/`` first. Writes are atomic per file. Live
reads the store only at prefill (server start), so a running server is unaffected until restart.

  .venv/bin/python -m scripts.repair_30m_swing_bars --sample 40     # dry run
  .venv/bin/python -m scripts.repair_30m_swing_bars --apply
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from strategies.multi_ticker_swing.config.pipeline_config import RAW_30M_DIR, TRAIN_END  # noqa: E402

BAR = pd.Timedelta(minutes=30)
NY = "America/New_York"
PRICE_COLS = ["open", "high", "low", "close", "vwap"]
OHLCV = ["open", "high", "low", "close", "volume"]
BASIS_TOL = 0.05        # |old/new - 1| above this in a session = a basis change (smallest split 1.25)
PARTIAL_LO, PARTIAL_HI = 0.15, 0.98    # cached/fresh volume ratio band called "partial"
FULL_START = pd.Timestamp("2021-01-01", tz="UTC")
FETCH_CHUNK = 50


def read_store(path: Path) -> tuple[pd.DataFrame, bool]:
    """(frame on a UTC DatetimeIndex, had_timestamp_column). Two layouts exist in the store."""
    df = pd.read_parquet(path)
    col = "timestamp" in df.columns
    if col:
        df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True), name="timestamp")
    return df.sort_index(), col


def write_store(df: pd.DataFrame, path: Path, *, timestamp_column: bool) -> None:
    out = df.reset_index() if timestamp_column else df
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=not timestamp_column)
    tmp.replace(path)


def complete_only(fresh: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    return fresh[fresh.index + BAR <= now]


def session_factors(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.Series:
    """cached/fresh close per NY session, median over the 10:00-12:30 ET buckets (never the
    restart-partial edge of a session), basis-change sessions only kept as != 1."""
    both = cached.index.intersection(fresh.index)
    ny = both.tz_convert(NY)
    mins = ny.hour * 60 + ny.minute
    morning = both[(mins >= 600) & (mins <= 750)]
    if morning.empty:
        return pd.Series(dtype=float)
    ratio = (cached.loc[morning, "close"] / fresh.loc[morning, "close"]).replace([np.inf, -np.inf], np.nan).dropna()
    f = ratio.groupby(ratio.index.tz_convert(NY).normalize()).median()
    return f.where((f - 1).abs() > BASIS_TOL, 1.0)


def compare(cached: pd.DataFrame, fresh: pd.DataFrame, since: pd.Timestamp) -> dict:
    """Window comparison of the cached file against a fresh pull. Pure."""
    c = cached[cached.index >= since]
    both = c.index.intersection(fresh.index)
    vol_ratio = (c.loc[both, "volume"] / fresh.loc[both, "volume"].replace(0, np.nan)).astype(float)
    changed = (c.loc[both, OHLCV].to_numpy(float) != fresh.loc[both, OHLCV].to_numpy(float)).any(axis=1)
    sess = lambda idx: set(pd.DatetimeIndex(idx).tz_convert(NY).normalize())  # noqa: E731
    f = session_factors(c, fresh)
    shifted = f[(f - 1).abs() > BASIS_TOL]
    close_diff = (c.loc[both, "close"] / fresh.loc[both, "close"] - 1).abs()
    return {
        "window_rows_cached": int(len(c)),
        "fresh_rows": int(len(fresh)),
        "rows_changed": int(changed.sum()),
        "rows_partial": int(((vol_ratio >= PARTIAL_LO) & (vol_ratio < PARTIAL_HI)).sum()),
        "rows_added": int(len(fresh.index.difference(c.index))),
        "rows_dropped": int(len(c.index.difference(fresh.index))),
        "sessions_added": int(len(sess(fresh.index) - sess(c.index))),
        "rows_missing_vwap": int(c["vwap"].isna().sum()) if "vwap" in c else int(len(c)),
        "median_abs_close_diff_changed": float(close_diff[changed].median()) if changed.any() else 0.0,
        "basis_sessions_shifted": int(len(shifted)),
        "basis_factors": ",".join(sorted({f"{x:.4g}" for x in shifted})),
    }


def splice(cached: pd.DataFrame, fresh: pd.DataFrame, since: pd.Timestamp) -> pd.DataFrame:
    """Cached rows before the window, then the fresh pull. Columns follow the cached file."""
    cols = list(cached.columns)
    fresh = fresh.reindex(columns=cols)
    out = pd.concat([cached[cached.index < since], fresh[fresh.index >= since]])
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_fresh(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Training-rule 30m bars per symbol (UTC index), complete bars only."""
    from strategies.multi_ticker_swing.live.feature_builder import fetch_research_bars_multi

    from strategies.momentum_expansion.data.bars import _with_rate_limit_retry

    # The live server and the nightly jobs share the data rate limit (a 429 stopped the
    # first full run at 500/911 files); back off and retry the chunk.
    long = _with_rate_limit_retry(
        lambda: fetch_research_bars_multi(symbols, start.to_pydatetime(), end.to_pydatetime()),
        label=f"30m repair [{symbols[0]}..{symbols[-1]}]", max_attempts=7)
    if long is None or long.empty:
        return {}
    long["timestamp"] = pd.to_datetime(long["timestamp"], utc=True)
    out = {}
    for sym, g in long.groupby("symbol"):
        g = g.set_index("timestamp").sort_index()
        g.index.name = "timestamp"
        out[str(sym).upper()] = complete_only(g, end)
    return out


def live_tickers() -> list[str]:
    from strategies.multi_ticker_swing.live.runner import _CONTEXT_SET
    from strategies.multi_ticker_swing.live.universe import load_universe

    return sorted(set(load_universe()) | set(_CONTEXT_SET))


def main(argv: list[str] | None = None) -> int:
    from core.shared_universe.delisted import is_delisted

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", default=TRAIN_END, help=f"window start (default {TRAIN_END}, first live write)")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    since = pd.Timestamp(args.since, tz="UTC")
    now = pd.Timestamp(dt.datetime.now(dt.timezone.utc))
    tickers = args.tickers or [t for t in live_tickers() if (RAW_30M_DIR / f"{t}.parquet").exists()]
    if args.sample and not args.tickers:
        tickers = sorted(random.Random(args.seed).sample(tickers, min(args.sample, len(tickers))))
    skipped = [t for t in tickers if is_delisted(t)]
    tickers = [t for t in tickers if not is_delisted(t)]

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    backup = RAW_30M_DIR.parent / f"_backup_30m_repair_{stamp}"
    log_dir = RAW_30M_DIR.parent / "_repair_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(tickers)} files, window {since} .. {now}", flush=True)
    t0 = time.time()
    rows, full = [], []
    for i in range(0, len(tickers), FETCH_CHUNK):
        part = tickers[i:i + FETCH_CHUNK]
        fresh = fetch_fresh(part, since, now)
        for t in part:
            rows.append(_one(t, fresh.get(t), since, now, backup, args.apply, full))
        print(f"  {i + len(part)}/{len(tickers)} ({time.time() - t0:.0f}s)", flush=True)
    # Basis changed in the window: refetch the whole history on today's basis.
    for t in full:
        fresh = fetch_fresh([t], FULL_START, now).get(t)
        row = next(r for r in rows if r["ticker"] == t)
        row.update(_full(t, fresh, backup, args.apply))

    rows += [{"ticker": t, "status": "skipped_delisted"} for t in skipped]
    df = pd.DataFrame(rows)
    suffix = "" if args.apply else "_dryrun"
    out_csv = log_dir / f"repair_30m_{stamp}{suffix}.csv"
    df.to_csv(out_csv, index=False)
    num = df.select_dtypes("number")
    summary = {
        "mode": "apply" if args.apply else "dry_run", "since": str(since), "end": str(now),
        "files": len(df), "status_counts": df["status"].value_counts().to_dict(),
        "totals": {c: int(num[c].sum()) for c in num.columns},
        "files_with_partial": int((df.get("rows_partial", pd.Series(dtype=float)) > 0).sum()),
        "files_full_refetch": full, "backup_dir": str(backup) if args.apply else None,
        "report": str(out_csv), "elapsed_s": round(time.time() - t0, 1),
    }
    (log_dir / f"repair_30m_{stamp}{suffix}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 1 if (df["status"] == "error").sum() > 0.05 * max(1, len(df)) else 0


def _one(t: str, fresh: pd.DataFrame | None, since, now, backup: Path, apply: bool, full: list) -> dict:
    path = RAW_30M_DIR / f"{t}.parquet"
    row = {"ticker": t}
    try:
        cached, col = read_store(path)
        row["cached_last"] = str(cached.index.max())
        if fresh is None or fresh.empty:
            return {**row, "status": "no_fresh_bars"}
        row.update(compare(cached, fresh, since))
        if row["basis_sessions_shifted"]:
            full.append(t)
            return {**row, "status": "basis_change"}
        if not (row["rows_changed"] or row["rows_added"] or row["rows_dropped"]):
            return {**row, "status": "clean"}
        if not apply:
            return {**row, "status": "would_repair"}
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / path.name)
        write_store(splice(cached, fresh, since), path, timestamp_column=col)
        return {**row, "status": "repaired"}
    except Exception as exc:  # noqa: BLE001 - reported per ticker
        return {**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _full(t: str, fresh: pd.DataFrame | None, backup: Path, apply: bool) -> dict:
    path = RAW_30M_DIR / f"{t}.parquet"
    try:
        cached, col = read_store(path)
        if fresh is None or fresh.empty:
            return {"status": "basis_change_no_full_pull"}
        fresh = fresh[fresh.index >= cached.index.min()].reindex(columns=list(cached.columns))
        info = {"full_rows_cached": int(len(cached)), "full_rows_fresh": int(len(fresh))}
        if not apply:
            return {**info, "status": "would_refetch_full"}
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / path.name)
        write_store(fresh, path, timestamp_column=col)
        return {**info, "status": "refetched_full"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    raise SystemExit(main())
