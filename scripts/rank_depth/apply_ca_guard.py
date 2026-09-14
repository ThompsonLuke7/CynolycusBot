"""Scan the shared daily bar cache with core.corporate_actions and write the flag
table every rank-depth study reads (corporate_action_flags.parquet).

One row per flagged session, with `direction` and `organic`. Research masks the
NON-organic ones (guarded_kvk.py); organic up-moves are real returns and stay in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.corporate_actions import suspect_sessions  # noqa: E402

DATA = REPO / "research/execution_quality/data"
BARS = REPO / "Data/shared/bars/1d"


def main() -> None:
    rows = []
    files = sorted(BARS.glob("*.parquet"))
    for p in files:
        try:
            d = pd.read_parquet(p, columns=["timestamp", "open", "close", "volume"])
        except Exception:
            continue
        f = suspect_sessions(d)
        if f.empty:
            continue
        sess = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
            "America/New_York").dt.normalize().dt.tz_localize(None)
        for r in f.itertuples():
            rows.append({"ticker": p.stem, "date": sess.loc[r.idx],
                         "gap_ratio": r.gap_ratio, "direction": r.direction,
                         "volume_ratio": r.volume_ratio,
                         "dollar_volume_ratio": r.dollar_volume_ratio,
                         "organic": bool(r.organic)})
    flags = pd.DataFrame(rows)
    flags.to_parquet(DATA / "corporate_action_flags.parquet")
    print(f"scanned {len(files)} files -> {len(flags)} flags over "
          f"{flags['ticker'].nunique()} tickers")
    print(flags.groupby(["direction", "organic"]).size().rename("flags").to_string())


if __name__ == "__main__":
    main()
