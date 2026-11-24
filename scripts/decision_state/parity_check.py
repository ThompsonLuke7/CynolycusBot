"""Research/live feature parity on archived decisions (AGENTS.md "parity" rule).

For each module, join the archived live feature vectors (``f__*``) to the
research file that training data is built from, at the same (bar, ticker), and
compare value by value. Any feature whose mismatch share exceeds
``--max-mismatch`` fails the check (exit 1). A module with no research
reference wired is reported NOT_CHECKED, never as passing.

References:
  momentum_expansion       strategies/momentum_expansion/data/processed/features_4h.parquet
  meta_ranker, multi_ticker_swing_htf
                           signals/meta_context/meta_ranker/meta_ranker_matrix.parquet
  dealer_ranker            NOT_CHECKED: its rankings file is rewritten after the decision
  spy_day_trader           the nightly full-history SPY meta matrix (update_live_meta_base_frame.py
                           --rebuild, pre-stamp); it reads the runtime 1m cache the live loop keeps,
                           so each archived session's cache is first checked against a fresh IEX pull
                           and a mismatch fails the module (reference untrusted)
  multi_ticker_swing_30m   strategies.multi_ticker_swing.research_reference: the training feature
                           code on a FRESH Alpaca pull (never the live deque / RAW_30M_DIR), for a
                           seeded sample of tickers (--sample-30m), keyed by 30m bucket start

    .venv/bin/python -m scripts.decision_state.parity_check --days 7
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.decision_state_archive import FEATURE_PREFIX, archive_root, load_decision_state  # noqa: E402

logger = logging.getLogger("decision_state.parity")
FEATURES_4H = REPO / "strategies/momentum_expansion/data/processed/features_4h.parquet"
META_MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
DEALER_RANKINGS = REPO / "Data/dealer_positioning/rankings"
REPORT_DIR = REPO / "Data/research/decision_state_parity"
NOT_WIRED: dict[str, str] = {}
SPY_MATRIX = REPO / "Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts.parquet"
SPY_1M_CACHE = REPO / "Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet"
SAMPLE_30M = 40     # tickers per run: each needs a ~400-day fresh 30m pull
RTOL, ATOL = 1e-5, 1e-6
# Rows decided before a module's parity fix went live are excluded (and counted):
# they document the old defect, not the current code. momentum: live daily
# features one session stale + xsec ranks over the snapshot only + metadata
# features at 0 (fixed in code 2026-11-19; the intraday context refresh also
# needs a combined-server restart, until which regime_vix_z can still differ).
FIXED_FROM = {"momentum_expansion": pd.Timestamp("2026-11-19 14:00", tz="UTC")}
# Rows written by a process whose code predates the fix commit (archive git_commit, read
# once per process) document the old defect. spy_day_trader: forming 10m bar cached and
# scored one minute in (320dccf7).
FIXED_IN_COMMIT = {"spy_day_trader": "320dccf7"}
# Row-level marker where a fix is visible in the archive itself. 30m swing (2026-11-23): REST
# bar sync, bucket-start stamps, daily-scope history, training static ids and the nightly
# daily context all ship together with the scanner's run_meta "bar_sync" field; it takes
# effect at the next combined-server restart.
PRE_FIX = {"multi_ticker_swing_30m":
           lambda df: ~df["run_meta_json"].fillna("").astype(str).str.contains('"bar_sync"')}
NOT_WIRED["dealer_ranker"] = ("reference invalid: dealer_swing_rankings_<snapshot>.parquet is rewritten after "
                              "the 15:45 decision (11-18 file holds a 17:45 capture); the archive is the record")


@functools.lru_cache(maxsize=256)
def _has_commit(row_commit: str | None, fix: str) -> bool:
    """True when ``row_commit`` contains ``fix`` (is it or a descendant)."""
    if not row_commit:
        return False
    r = subprocess.run(["git", "merge-base", "--is-ancestor", fix, row_commit], cwd=REPO,
                       capture_output=True, timeout=20)
    return r.returncode == 0


def pre_fix_mask(module: str, archived: pd.DataFrame) -> pd.Series | None:
    """Rows decided by code that predates the module's parity fix, or None."""
    masks = []
    if module in FIXED_FROM:
        masks.append(_utc(archived["decision_ts"]) < FIXED_FROM[module])
    if module in PRE_FIX:
        masks.append(PRE_FIX[module](archived))
    if module in FIXED_IN_COMMIT:
        fix = FIXED_IN_COMMIT[module]
        commits = archived.get("git_commit", pd.Series(None, index=archived.index))
        masks.append(~commits.map(lambda c: _has_commit(c if isinstance(c, str) else None, fix)).astype(bool))
    if not masks:
        return None
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


def spy_cache_check(sessions: list, *, cache_path: Path = SPY_1M_CACHE, fetch=None) -> dict:
    """The runtime 1m cache vs a fresh IEX pull, per NY session (the SPY reference's input)."""
    if fetch is None:
        from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

        def fetch(start, end):
            return fetch_intraday(ticker="SPY", start=start, end=end, timeframe="1Min", limit=None,
                                  feed="IEX", save_path="")
    cache = pd.read_parquet(cache_path)
    cache["timestamp"] = pd.to_datetime(cache["timestamp"], utc=True)
    cache = cache.set_index("timestamp").sort_index()
    cols = ["open", "high", "low", "close", "volume"]
    bad = {}
    for day in sessions:
        lo = pd.Timestamp(str(day), tz="America/New_York")
        hi = lo + pd.Timedelta(days=1)
        c = cache.loc[lo.tz_convert("UTC"):hi.tz_convert("UTC") - pd.Timedelta(seconds=1), cols]
        f = fetch(lo.tz_convert("UTC").isoformat(), hi.tz_convert("UTC").isoformat())
        f = f.set_index(pd.to_datetime(f["timestamp"], utc=True))[cols] if f is not None and not f.empty \
            else pd.DataFrame(columns=cols)
        j = c.join(f, rsuffix="_f", how="outer")
        if c.empty:
            # Not in the cache: the nightly rebuild takes this session from a fresh pull
            # (update_live_meta_base_frame.fresh_topup), which needs no vouching.
            continue
        differ = int((~np.isclose(j[cols].to_numpy(float), j[[x + "_f" for x in cols]].to_numpy(float),
                                  rtol=1e-6, equal_nan=False)).any(axis=1).sum())
        if differ:
            bad[str(day)] = {"cache_rows": int(len(c)), "fresh_rows": int(len(f)), "differ": differ}
    return {"sessions": [str(d) for d in sessions], "bad_sessions": bad}


def compare_vectors(archived: pd.DataFrame, reference: pd.DataFrame, *, keys: list[str]) -> tuple[pd.DataFrame, dict]:
    """Per-feature mismatch table for archived ``f__*`` vs ``reference`` columns.

    Values are compared at float32 (what the archive stores); NaN on both sides
    is a match, NaN on one side is a mismatch."""
    feats = [c[len(FEATURE_PREFIX):] for c in archived.columns if c.startswith(FEATURE_PREFIX)]
    shared = [f for f in feats if f in reference.columns]
    ref = reference[keys + shared].drop_duplicates(keys)
    joined = archived[keys + [FEATURE_PREFIX + f for f in shared]].merge(ref, on=keys, how="left",
                                                                         indicator=True)
    found = joined["_merge"] == "both"
    cov = {"archived_rows": int(len(archived)), "matched_rows": int(found.sum()),
           "features_archived": len(feats), "features_compared": len(shared),
           "features_not_in_reference": sorted(set(feats) - set(shared))[:50]}
    rows = []
    j = joined.loc[found]
    for f in shared:
        a = pd.to_numeric(j[FEATURE_PREFIX + f], errors="coerce").to_numpy(dtype=np.float64)
        r = pd.to_numeric(j[f], errors="coerce").to_numpy(dtype=np.float32).astype(np.float64)
        both_nan = np.isnan(a) & np.isnan(r)
        one_nan = np.isnan(a) ^ np.isnan(r)
        close = np.isclose(a, r, rtol=RTOL, atol=ATOL)
        bad = one_nan | (~both_nan & ~one_nan & ~close)
        diff = np.abs(a - r)
        rows.append({"feature": f, "n": int(len(a)), "mismatch": int(bad.sum()),
                     "nan_mismatch": int(one_nan.sum()),
                     "share": float(bad.mean()) if len(a) else 0.0,
                     "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0})
    table = pd.DataFrame(rows, columns=["feature", "n", "mismatch", "nan_mismatch", "share", "max_abs_diff"])
    return table.sort_values("share", ascending=False).reset_index(drop=True), cov


def _utc(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, utc=True)
    return out


def _reference_4h(path: Path, bars: list[pd.Timestamp], tickers: list[str]) -> pd.DataFrame:
    import pyarrow.dataset as ds
    dset = ds.dataset(path, format="parquet")
    ts_type = dset.schema.field("timestamp").type
    import pyarrow as pa
    vals = pa.array([b.to_pydatetime() for b in bars], type=ts_type)
    table = dset.to_table(filter=ds.field("timestamp").isin(vals) & ds.field("ticker").isin(tickers))
    df = table.to_pandas().reset_index()
    df["bar_ts"] = _utc(df["timestamp"])
    return df


def normalize_keys(module: str, archived: pd.DataFrame) -> pd.DataFrame:
    """Map archived bar stamps onto the reference's bar keys.

    30m swing: live stamps a 30m bar with its last 1m bar (15:29, or 15:27 for a name that
    did not print at :29); research bars are stamped at the bucket start (15:00)."""
    if module == "multi_ticker_swing_30m":
        from strategies.multi_ticker_swing.research_reference import bucket_start
        archived = archived.copy()
        archived["archived_bar_ts"] = archived["bar_ts"]
        archived["bar_ts"] = bucket_start(archived["bar_ts"])
    return archived


def sample_tickers(archived: pd.DataFrame, n: int, *, seed: int = 0) -> list[str]:
    """Every selected ticker plus a seeded random fill up to ``n`` (reproducible per seed)."""
    tickers = sorted(archived["ticker"].astype(str).unique())
    if n <= 0 or len(tickers) <= n:
        return tickers
    picked = sorted(archived.loc[archived.get("selected", pd.Series(False, index=archived.index)).fillna(False)
                                 .astype(bool), "ticker"].astype(str).unique())[:n]
    rest = [t for t in tickers if t not in set(picked)]
    rng = np.random.default_rng(seed)
    fill = list(rng.choice(rest, size=min(n - len(picked), len(rest)), replace=False)) if n > len(picked) else []
    return sorted(set(picked) | {str(t) for t in fill})


def reference_for(module: str, archived: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], str]:
    bars = sorted(pd.to_datetime(archived["bar_ts"].unique(), utc=True))
    tickers = sorted(archived["ticker"].astype(str).unique())
    if module == "momentum_expansion":
        return _reference_4h(FEATURES_4H, bars, tickers), ["bar_ts", "ticker"], str(FEATURES_4H)
    if module in ("meta_ranker", "multi_ticker_swing_htf"):
        return _reference_4h(META_MATRIX, bars, tickers), ["bar_ts", "ticker"], str(META_MATRIX)
    if module == "spy_day_trader":
        if not SPY_MATRIX.exists():
            return None, [], f"SPY matrix missing: {SPY_MATRIX}"
        m = pd.read_parquet(SPY_MATRIX)
        m.index = pd.DatetimeIndex(m.index).tz_convert("UTC")
        ref = m.loc[m.index.isin(pd.DatetimeIndex(bars))].copy()
        ref.index.name = "bar_ts"
        ref = ref.reset_index()
        ref["ticker"] = "SPY"
        return ref, ["bar_ts", "ticker"], str(SPY_MATRIX)
    if module == "multi_ticker_swing_30m":
        from strategies.multi_ticker_swing.research_reference import build_reference
        ref, info = build_reference(archived[["bar_ts", "ticker"]])
        if info["failed"]:
            logger.warning("30m reference: %d tickers failed: %s", len(info["failed"]), info["failed"])
        return ref, ["bar_ts", "ticker"], "fresh IEX/split 30m pull -> research builder"
    if module in NOT_WIRED:
        return None, [], NOT_WIRED[module]
    if module == "dealer_ranker":
        snaps = archived.get("x__snapshot_date")
        if snaps is None:
            return None, [], "archive has no snapshot_date"
        frames = []
        for snap in sorted(snaps.dropna().astype(str).unique()):
            p = DEALER_RANKINGS / f"dealer_swing_rankings_{snap.replace('-', '')}.parquet"
            if p.exists():
                f = pd.read_parquet(p)
                f["ticker"] = f["symbol"].astype(str).str.upper()
                f["x__snapshot_date"] = snap
                frames.append(f)
        if not frames:
            return None, [], "no dealer rankings file for the archived snapshot dates"
        return pd.concat(frames, ignore_index=True), ["x__snapshot_date", "ticker"], str(DEALER_RANKINGS)
    return None, [], NOT_WIRED.get(module, "no research reference wired")


def run(*, days: int, root: Path, max_mismatch: float, modules: list[str] | None,
        sample_30m: int = SAMPLE_30M) -> dict:
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    present = sorted(p.name for p in root.glob("*") if p.is_dir()) if root.exists() else []
    report: dict = {"checked_at": datetime.now(timezone.utc).isoformat(), "since": start,
                    "max_mismatch": max_mismatch, "modules": {}}
    failed = False
    for module in modules or present:
        archived = load_decision_state(root, module=module, start=start)
        if archived.empty:
            report["modules"][module] = {"status": "NO_ARCHIVE"}
            continue
        archived["bar_ts"] = _utc(archived["bar_ts"])
        archived["ticker"] = archived["ticker"].astype(str)
        excluded = 0
        pre = pre_fix_mask(module, archived)
        if pre is not None:
            excluded = int(pre.sum())
            archived = archived.loc[~pre]
            if archived.empty:
                report["modules"][module] = {"status": "NO_ARCHIVE", "excluded_pre_fix_rows": excluded}
                continue
        sampled = None
        if module == "multi_ticker_swing_30m":
            keep = sample_tickers(archived, sample_30m, seed=int(start.replace("-", "")))
            sampled = {"tickers": len(keep), "of": int(archived["ticker"].nunique())}
            archived = archived[archived["ticker"].isin(keep)]
        archived = normalize_keys(module, archived)
        input_check = None
        if module == "spy_day_trader":
            sessions = sorted(set(_utc(archived["bar_ts"]).dt.tz_convert("America/New_York").dt.date))
            input_check = spy_cache_check(sessions)
            if input_check["bad_sessions"]:
                failed = True
                report["modules"][module] = {
                    "status": "FAIL", "reason": "reference input (runtime 1m cache) differs from a fresh "
                    "IEX pull; the matrix cannot vouch for these sessions", "input_check": input_check,
                    "rows": int(len(archived)), "excluded_pre_fix_rows": excluded}
                continue
        ref, keys, source = reference_for(module, archived)
        if ref is None:
            report["modules"][module] = {"status": "NOT_CHECKED", "reason": source, "rows": int(len(archived)),
                                         "excluded_pre_fix_rows": excluded}
            continue
        # Each decision bar once: repeated passes on one bar archive the same vectors.
        dedup = archived.sort_values("decision_ts").drop_duplicates(keys, keep="first")
        table, cov = compare_vectors(dedup, ref, keys=keys)
        bad = table[table["share"] > max_mismatch]
        # Per decision slot (NY date + hour): a known gap (the 16:25 4H decision cannot see
        # the 16:00-17:00 bucket research puts in its 14:00 bar) must not hide a new one.
        slot = _utc(dedup["decision_ts"]).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H")
        by_slot = []
        for name, g in dedup.groupby(slot.to_numpy(), sort=True):
            t, c = compare_vectors(g, ref, keys=keys)
            by_slot.append({"slot": name, "rows": c["matched_rows"],
                            "failing": int((t["share"] > max_mismatch).sum()) if c["matched_rows"] else None})
        status = "FAIL" if (len(bad) or cov["matched_rows"] == 0) else "PASS"
        failed |= status == "FAIL"
        report["modules"][module] = {
            "status": status, "reference": source, **cov, "excluded_pre_fix_rows": excluded,
            **({"sampled_tickers": sampled} if sampled else {}),
            **({"input_check": input_check} if input_check else {}),
            "worst": table.head(10).to_dict(orient="records"), "by_slot": by_slot,
            "failing_features": bad["feature"].tolist(),
        }
    passed = any(m.get("status") == "PASS" for m in report["modules"].values())
    report["status"] = "FAIL" if failed else ("PASS" if passed else "NOT_CHECKED")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--module", action="append", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--max-mismatch", type=float, default=0.01,
                    help="Largest tolerated share of rows that disagree, per feature.")
    ap.add_argument("--report-dir", default=str(REPORT_DIR))
    ap.add_argument("--sample-30m", type=int, default=SAMPLE_30M,
                    help="30m swing tickers to rebuild per run (fresh pull each); 0 = all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.root) if args.root else archive_root()
    report = run(days=args.days, root=root, max_mismatch=args.max_mismatch, modules=args.module,
                 sample_30m=args.sample_30m)
    out = Path(args.report_dir) / f"parity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    for mod, r in report["modules"].items():
        extra = r.get("reason") or (f"matched {r.get('matched_rows')}/{r.get('archived_rows')} rows, "
                                    f"{len(r.get('failing_features', []))} failing features")
        if r.get("by_slot"):
            extra += "; by slot " + ", ".join(f"{x['slot'][5:]}h={x['failing']}" for x in r["by_slot"])
        print(f"{r['status']:12s} {mod}: {extra}")
    print(f"parity {report['status']} -> {out}")
    if report["status"] == "FAIL":
        logger.error("PARITY FAIL: live feature vectors disagree with the research files (see %s)", out)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
