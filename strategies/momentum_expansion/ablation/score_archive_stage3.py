"""Shadow-score archived live momentum decisions with the Stage 3 pick's final boosters.

Forward (prospective) evidence for the frozen Stage 3 momentum pick (L2_20__pit,
selection 621d0acaa1cd), alongside the sealed test. Reads the live decision-state
archive (``Data/research/decision_state/momentum_expansion/``), builds each row's
82-feature vector from the archived ``f__*`` columns, and writes the mean of the
seeds' P(y=1) to ``Data/research/decision_state_shadow/stage3_L2_20/<ET date>.parquet``
keyed by (run_id, ticker). Nothing trades on these scores.

Rows decided before the live-parity fix (daily features one session stale,
xsec ranks over the snapshot only; fixed 2026-11-19) are kept but flagged
``pre_parity_fix``: they are not valid evidence for the model.

Exits 0 with a note while the boosters have not been delivered yet
(``Data/research/stage3_final/`` from ``stage3_final.tar``).

    .venv/bin/python -m strategies.momentum_expansion.ablation.score_archive_stage3 --days 45
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.decision_state_archive import FEATURE_PREFIX, archive_root, list_partitions

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[3]
BOOSTER_DIR = REPO / "Data" / "research" / "stage3_final"
OUT_DIR = REPO / "Data" / "research" / "decision_state_shadow" / "stage3_L2_20"
MODULE = "momentum_expansion"
# First live momentum pass with the parity fixes (feature_matrix_4h live edge +
# research-universe xsec ranks): the next scheduled run after they were written.
PARITY_FIXED_FROM = pd.Timestamp("2026-11-19 14:00", tz="UTC")


def load_boosters(folder: Path = BOOSTER_DIR):
    """(boosters with their best iteration, feature list), or None if not delivered."""
    meta_path = folder / "stage3_final_meta.json"
    if not meta_path.exists():
        return None
    import xgboost as xgb

    meta = json.loads(meta_path.read_text())
    boosters = []
    for s in meta["seeds"]:
        b = xgb.Booster()
        b.load_model(str(folder / f"stage3_final_{meta['arm']['name']}_s{s['seed']}.json"))
        boosters.append((b, int(s["best_iteration"])))
    return boosters, list(meta["features"]), meta


def shadow_scores(frame: pd.DataFrame, boosters, features: list[str]) -> np.ndarray:
    """Mean P(y=1) over the seeds; NaN where any of the 82 features is missing
    (the walk-forward trained and scored complete rows only)."""
    import xgboost as xgb

    cols = [FEATURE_PREFIX + f for f in features]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"archive lacks {len(missing)} model features, e.g. {missing[:5]}")
    X = frame[cols].to_numpy(dtype=np.float32)
    complete = ~np.isnan(X).any(axis=1)
    out = np.full(len(frame), np.nan)
    if complete.any():
        dm = xgb.DMatrix(X[complete], feature_names=list(features))
        out[complete] = np.mean([b.predict(dm, iteration_range=(0, best + 1)) for b, best in boosters], axis=0)
    return out


def run(*, days: int, root: Path | None = None, booster_dir: Path = BOOSTER_DIR,
        out_dir: Path = OUT_DIR) -> dict:
    loaded = load_boosters(booster_dir)
    if loaded is None:
        return {"status": "no_boosters", "booster_dir": str(booster_dir)}
    boosters, features, meta = loaded
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    by_date: dict[str, list[Path]] = {}
    for p in list_partitions(root, MODULE, start=start):
        by_date.setdefault(p.parent.name, []).append(p)
    keep = ["run_id", "decision_ts", "bar_ts", "ticker", "score", "selected", "g__in_universe"]
    stats = {"status": "ok", "dates": 0, "rows": 0, "scored": 0}
    for date, paths in sorted(by_date.items()):
        out_path = out_dir / f"{date}.parquet"
        if out_path.exists() and out_path.stat().st_mtime > max(p.stat().st_mtime for p in paths):
            continue                                            # nothing new archived for that date
        import pyarrow.parquet as pq

        cols = keep + [FEATURE_PREFIX + f for f in features]
        parts = []
        for p in paths:
            have = set(pq.ParquetFile(p).schema_arrow.names)
            parts.append(pd.read_parquet(p, columns=[c for c in cols if c in have]))
        frame = pd.concat(parts, ignore_index=True)
        frame["stage3_score"] = shadow_scores(frame, boosters, features)
        res = frame[[c for c in keep if c in frame.columns] + ["stage3_score"]].rename(
            columns={"score": "live_score", "selected": "live_selected"})
        res["pre_parity_fix"] = pd.to_datetime(res["decision_ts"], utc=True) < PARITY_FIXED_FROM
        res["booster_selection"] = meta.get("selection")
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f".{out_path.name}.{uuid.uuid4().hex}.tmp"
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
        stats["dates"] += 1
        stats["rows"] += len(res)
        stats["scored"] += int(np.isfinite(res["stage3_score"]).sum())
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = run(days=args.days, root=Path(args.root) if args.root else archive_root())
    print(f"stage3 shadow scores: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
