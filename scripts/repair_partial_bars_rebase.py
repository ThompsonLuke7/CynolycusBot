"""Put repaired hourly bars back on each file's original price basis.

``repair_partial_intraday_bars.py --apply`` (2026-11-23) spliced a fresh SIP pull for
2026-06-15+ into the shared 1H cache. That pull is split-adjusted as of today, but the cached
rows were adjusted as of the day each was fetched. So for a name with a split or reverse split
after 06-15, the repaired window moved to today's basis while the rows before 06-15 did not.
The jump moved from the split date to the 06-15 boundary (CRWD 686 -> 171, SNDK -87%).

This rebases the fresh rows to the backup's basis, session by session. The factor is
old/new close, median over the 10:00-12:00 ET hours (never partial; 0-1.4% partial in the
audit sample). Prices are multiplied by it, volume is divided by it, and trade_count is left
as is. A session with no morning overlap takes the nearest known factor. Partial bars stay
healed; only the price basis returns to what the file had.

Dry run by default. ``--apply`` copies each file it rewrites to ``<backup>/rebased_from/``
and rebuilds its 4H cache.

  PYTHONPATH=. .venv/bin/python scripts/repair_partial_bars_rebase.py --backup Data/shared/bars/_backup_partial_repair_<stamp>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from strategies.momentum_expansion.data import bars  # noqa: E402

WINDOW_START = pd.Timestamp("2026-06-15", tz="UTC")
PRICE_COLS = ["open", "high", "low", "close", "vwap"]
TOLERANCE = 0.05    # |factor - 1| above this is a basis change. Partial morning bars give
                    # one-session factors of 0.99-1.016; the smallest common split ratio is 1.25.


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def session_factors(old: pd.DataFrame, new: pd.DataFrame, since: pd.Timestamp = WINDOW_START) -> pd.Series:
    """old/new close per NY session in the window (median over 10:00-12:00 ET), indexed by date."""
    o = old[old["timestamp"] >= since].set_index("timestamp")["close"]
    n = new[new["timestamp"] >= since].set_index("timestamp")["close"]
    both = o.index.intersection(n.index)
    ny = both.tz_convert("America/New_York")
    morning = both[(ny.hour >= 10) & (ny.hour <= 12)]
    ratio = (o.loc[morning] / n.loc[morning]).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return pd.Series(dtype=float)
    days = ratio.index.tz_convert("America/New_York").normalize()
    f = ratio.groupby(days).median()
    return f.where((f - 1).abs() > TOLERANCE, 1.0)


def rebase(old: pd.DataFrame, new: pd.DataFrame, since: pd.Timestamp = WINDOW_START) -> tuple[pd.DataFrame, pd.Series]:
    """``new`` with its window rows converted to ``old``'s per-session price basis."""
    f = session_factors(old, new, since)
    out = new.copy()
    win = out["timestamp"] >= since
    days = out.loc[win, "timestamp"].dt.tz_convert("America/New_York").dt.normalize()
    if f.empty:
        return out, f
    per_day = f.reindex(sorted(set(days) | set(f.index))).ffill().bfill()
    factor = days.map(per_day).astype(float).to_numpy()
    for c in PRICE_COLS:
        if c in out.columns:
            out.loc[win, c] = out.loc[win, c].to_numpy(float) * factor
    out.loc[win, "volume"] = out.loc[win, "volume"].to_numpy(float) / factor
    return out, f


def check_one(args: tuple) -> dict:
    ticker, kind, backup_dir, apply = args
    old_p, new_p = Path(backup_dir) / kind / f"{ticker}.parquet", bars._path_for(ticker, kind)
    row = {"ticker": ticker, "kind": kind}
    try:
        old, new = _read(old_p), _read(new_p)
        fixed, f = rebase(old, new)
        shifted = f[(f - 1).abs() > TOLERANCE]
        row.update({"sessions": int(len(f)), "sessions_rebased": int(len(shifted)),
                    "factors": ",".join(sorted({f"{x:.4g}" for x in shifted})) if len(shifted) else ""})
        if shifted.empty:
            return {**row, "status": "basis_unchanged"}
        if not apply:
            return {**row, "status": "would_rebase"}
        keep = Path(backup_dir) / "rebased_from" / kind / f"{ticker}.parquet"
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(new_p, keep)
        bars._atomic_to_parquet(fixed, new_p, index=False)
        if kind == "1h":
            bars.build_4h_for(ticker, force=True)
        else:
            df_4h = bars.resample_1h_to_4h(_read(new_p))
            if not df_4h.empty:
                bars._atomic_to_parquet(df_4h.reset_index(), bars._path_for(ticker, "4h"), index=False)
        return {**row, "status": "rebased"}
    except Exception as exc:  # noqa: BLE001
        return {**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backup", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], help="tickers to leave untouched")
    args = ap.parse_args(argv)
    backup = Path(args.backup)
    skip = {t.upper() for t in args.exclude}
    jobs = [(p.stem, kind, str(backup), args.apply)
            for kind in ("1h", "context") for p in sorted((backup / kind).glob("*.parquet"))
            if p.stem.upper() not in skip]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(check_one, jobs, chunksize=50))
    df = pd.DataFrame(rows)
    out = backup.parent / "_repair_logs" / f"rebase_{backup.name}{'' if args.apply else '_dryrun'}.csv"
    df.to_csv(out, index=False)
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "files": len(df),
                      "status_counts": df["status"].value_counts().to_dict(), "report": str(out)}, indent=2))
    hit = df[df["status"].isin(["would_rebase", "rebased", "error"])]
    print(hit.to_string(index=False))
    return 1 if (df["status"] == "error").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
