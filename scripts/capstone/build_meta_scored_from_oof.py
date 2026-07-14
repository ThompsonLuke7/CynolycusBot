"""
Build /tmp/meta_scored.parquet from CLEAN walk-forward OOF scores instead of
the deployed boosters (leakage_audit.md §4.3).

signals/meta_context/meta_ranker/backtest_exits.py reads /tmp/meta_scored.parquet
(a [timestamp, ticker, s_combo] frame) to simulate exit policies over the
"holdout 2025-07-01+" window. The existing way to produce that file is
score.py, which scores with the DEPLOYED boosters — final-fit models whose
training window extends past 2025-07-01, so a meaningful prefix of that
"holdout" is partially in-sample.

This script instead builds s_combo from models/{quality,upside}/oof_preds.parquet
(21-day-embargo walk-forward OOF — the model never saw these rows during
training), using the same per-timestamp rank-mean combo formula as score.py's
score_frame(). Then:

  PYTHONPATH=. .venv/bin/python scripts/capstone/build_meta_scored_from_oof.py
  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/backtest_exits.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
Q_OOF = REPO / "signals/meta_context/meta_ranker/models/quality/oof_preds.parquet"
U_OOF = REPO / "signals/meta_context/meta_ranker/models/upside/oof_preds.parquet"
OUT = Path("/tmp/meta_scored.parquet")


def _read_oof(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        df = df.reset_index()
    return df.drop_duplicates(subset=["timestamp", "ticker"], keep="first")


def build_oof_combo_scores() -> pd.DataFrame:
    """[timestamp, ticker, s_combo] from clean walk-forward OOF quality+upside scores."""
    q = _read_oof(Q_OOF)[["timestamp", "ticker", "score"]].rename(columns={"score": "s_quality"})
    u = _read_oof(U_OOF)[["timestamp", "ticker", "score"]].rename(columns={"score": "s_upside"})
    m = q.merge(u, on=["timestamp", "ticker"], how="inner")
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    ru = m.groupby("timestamp")["s_upside"].rank(pct=True)
    rq = m.groupby("timestamp")["s_quality"].rank(pct=True)
    m["s_combo"] = (ru + rq) / 2.0
    return m


def main() -> None:
    m = build_oof_combo_scores()
    m.to_parquet(OUT, index=False)
    print(f"wrote {len(m):,} rows -> {OUT}  "
          f"range {m['timestamp'].min()} -> {m['timestamp'].max()}  "
          f"rows after 2025-07-01: {(m['timestamp'] >= pd.Timestamp('2025-07-01', tz='UTC')).sum():,}")


if __name__ == "__main__":
    main()
