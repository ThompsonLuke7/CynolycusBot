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
  dealer_ranker            Data/dealer_positioning/rankings/dealer_swing_rankings_<snapshot>.parquet

    .venv/bin/python -m scripts.decision_state.parity_check --days 7
"""
from __future__ import annotations

import argparse
import json
import logging
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
NOT_WIRED = {
    "spy_day_trader": "no research reference wired (SPY features are rebuilt per bar from the 1m buffer)",
    "multi_ticker_swing_30m": "no research reference wired (live builder vs research matrix not mapped yet)",
}
RTOL, ATOL = 1e-5, 1e-6
# Rows decided before a module's parity fix went live are excluded (and counted):
# they document the old defect, not the current code. momentum: live daily
# features one session stale + xsec ranks over the snapshot only + metadata
# features at 0 (fixed in code 2026-11-19; the intraday context refresh also
# needs a combined-server restart, until which regime_vix_z can still differ).
FIXED_FROM = {"momentum_expansion": pd.Timestamp("2026-11-19 14:00", tz="UTC")}
NOT_WIRED["dealer_ranker"] = ("reference invalid: dealer_swing_rankings_<snapshot>.parquet is rewritten after "
                              "the 15:45 decision (11-18 file holds a 17:45 capture); the archive is the record")


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


def reference_for(module: str, archived: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], str]:
    bars = sorted(pd.to_datetime(archived["bar_ts"].unique(), utc=True))
    tickers = sorted(archived["ticker"].astype(str).unique())
    if module == "momentum_expansion":
        return _reference_4h(FEATURES_4H, bars, tickers), ["bar_ts", "ticker"], str(FEATURES_4H)
    if module in ("meta_ranker", "multi_ticker_swing_htf"):
        return _reference_4h(META_MATRIX, bars, tickers), ["bar_ts", "ticker"], str(META_MATRIX)
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


def run(*, days: int, root: Path, max_mismatch: float, modules: list[str] | None) -> dict:
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
        if module in FIXED_FROM:
            pre = _utc(archived["decision_ts"]) < FIXED_FROM[module]
            excluded = int(pre.sum())
            archived = archived.loc[~pre]
            if archived.empty:
                report["modules"][module] = {"status": "NO_ARCHIVE", "excluded_pre_fix_rows": excluded}
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
        status = "FAIL" if (len(bad) or cov["matched_rows"] == 0) else "PASS"
        failed |= status == "FAIL"
        report["modules"][module] = {
            "status": status, "reference": source, **cov, "excluded_pre_fix_rows": excluded,
            "worst": table.head(10).to_dict(orient="records"),
            "failing_features": bad["feature"].tolist()[:50],
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
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.root) if args.root else archive_root()
    report = run(days=args.days, root=root, max_mismatch=args.max_mismatch, modules=args.module)
    out = Path(args.report_dir) / f"parity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    for mod, r in report["modules"].items():
        extra = r.get("reason") or (f"matched {r.get('matched_rows')}/{r.get('archived_rows')} rows, "
                                    f"{len(r.get('failing_features', []))} failing features")
        print(f"{r['status']:12s} {mod}: {extra}")
    print(f"parity {report['status']} -> {out}")
    if report["status"] == "FAIL":
        logger.error("PARITY FAIL: live feature vectors disagree with the research files (see %s)", out)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
