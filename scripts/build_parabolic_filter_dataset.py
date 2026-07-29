"""Build the parabolic-likelihood training dataset.

Question: which signals go parabolic (large multi-ATR expansion) BEFORE they happen?
That is the missing selector -- Phase 3 showed options dominate the parabolic tail
(93% of tail dollars on 17% of the capital) but lose overall, because ~20% tail hit
rate cannot carry ~22% round-trip spread. A filter that raises the hit rate is the
whole ballgame.

Label:  parabolic = realized_move_atr >= PARABOLIC_ATR  (direction-signed move, in ATR units)
Features: the module's own 4H feature matrix, joined STRICTLY at or before signal time.

Leakage discipline (AGENTS.md):
  * feature bar must be at or before the trade's signal_ts -- merge_asof backward only.
  * no feature computed from the trade's own outcome.
  * `realized_move_atr` is the LABEL and is never a feature.
  * split is by time (walk-forward), never shuffled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPINE = REPO / "research/options_experiment/data/signal_spine.parquet"
OUT = REPO / "research/options_experiment/data/parabolic_dataset.parquet"

FEATURE_MATRIX = {
    "momentum_expansion": REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet",
    "multi_ticker_swing_htf": REPO / "strategies/multi_ticker_swing_htf/data/processed/training_matrix_4h.parquet",
}

PARABOLIC_ATR = 4.0

# Columns that are labels/outcomes in the source training matrices, or otherwise
# leak the future. Excluded from the feature set.
#
# LESSON (2026-07-27): the training matrices have the modules' own LABEL files
# merged in. A first pass scored AUC 0.951 on held-out test data -- not skill,
# but the model reading `expansion_score` / `expansion_target` / `trend_persistence`,
# which are forward-looking by construction. `expansion_target` slipped a
# prefix-based filter because it *ends* with "target". Project memory already
# records `trend_persistence` as a forward label that invalidated an earlier study.
# So: enumerate label columns from the label parquets themselves rather than
# guessing at name patterns.
LEAK_PREFIXES = ("label", "target", "y_", "fwd_", "future_", "ret_fwd", "expansion_",
                 "trend_persistence")
LEAK_SUBSTRINGS = ("_target", "_label", "fwd_", "forward_")
LEAK_EXACT = {
    "timestamp", "ticker", "realized_move_atr", "parabolic",
    "entry_ts", "exit_ts", "signal_ts", "module", "direction",
}

# Every column appearing in a module's LABEL parquet is forbidden as a feature.
LABEL_FILES = (
    REPO / "strategies/momentum_expansion/data/processed/expansion_labels_4h.parquet",
    REPO / "strategies/multi_ticker_swing_htf/data/processed/pivot_swing_labels_4h.parquet",
)


def _label_columns() -> set[str]:
    import pyarrow.parquet as pq
    cols: set[str] = set()
    for p in LABEL_FILES:
        if p.exists():
            cols.update(pq.ParquetFile(p).schema_arrow.names)
    cols -= {"timestamp", "ticker"}  # join keys, not labels
    return cols


def _load_spine() -> pd.DataFrame:
    s = pd.read_parquet(SPINE)
    s = s[s.module.isin(FEATURE_MATRIX)].copy()
    # signed realized move in ATR units -- the label basis
    s["realized_move_atr"] = (
        s.direction * (s.exit_px_underlying - s.entry_px_underlying) / s.atr_at_entry
    )
    s = s[np.isfinite(s.realized_move_atr)]
    s["parabolic"] = (s.realized_move_atr >= PARABOLIC_ATR).astype(int)
    # decision time: prefer signal_ts, fall back to entry_ts
    s["decision_ts"] = s.signal_ts.fillna(s.entry_ts)
    return s


def _feature_cols(df: pd.DataFrame) -> list[str]:
    banned = _label_columns()
    out, dropped = [], []
    for c in df.columns:
        lc = c.lower()
        if (c in LEAK_EXACT or c in banned
                or lc.startswith(LEAK_PREFIXES)
                or any(sub in lc for sub in LEAK_SUBSTRINGS)):
            dropped.append(c)
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        out.append(c)
    if dropped:
        print(f"    excluded {len(dropped)} leaky/label columns: {sorted(dropped)[:8]}"
              f"{' ...' if len(dropped) > 8 else ''}")
    return out


def build(module: str, spine: pd.DataFrame) -> pd.DataFrame:
    path = FEATURE_MATRIX[module]
    trades = spine[spine.module == module].copy()
    if trades.empty:
        return pd.DataFrame()

    feats = pd.read_parquet(path)
    # these matrices are indexed by (timestamp, ticker)
    if "timestamp" not in feats.columns:
        feats = feats.reset_index()
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    fcols = _feature_cols(feats)
    feats = feats[["ticker", "timestamp"] + fcols]

    trades["decision_ts"] = pd.to_datetime(trades["decision_ts"], utc=True)
    trades = trades.sort_values("decision_ts")
    feats = feats.sort_values("timestamp")

    # STRICT point-in-time join: nearest feature bar at or BEFORE the decision.
    merged = pd.merge_asof(
        trades,
        feats,
        left_on="decision_ts",
        right_on="timestamp",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
        tolerance=pd.Timedelta("8h"),   # 4H bars: never reach back more than 2 bars
    )
    before = len(merged)
    merged = merged[merged.timestamp.notna()]
    print(f"  {module}: {before} trades -> {len(merged)} joined "
          f"({len(merged)/max(before,1):.0%}), {len(fcols)} features")

    # hard leakage assertion
    bad = (merged.timestamp > merged.decision_ts).sum()
    if bad:
        raise AssertionError(f"{module}: {bad} rows have a feature bar AFTER the decision")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parabolic-atr", type=float, default=PARABOLIC_ATR)
    args = ap.parse_args()

    globals()["PARABOLIC_ATR"] = args.parabolic_atr
    spine = _load_spine()
    print(f"spine trades (4H modules): {len(spine)}, parabolic base rate "
          f"{spine.parabolic.mean():.1%} at >= {args.parabolic_atr} ATR")

    parts = [build(m, spine) for m in FEATURE_MATRIX]
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT}")
    print(out.groupby("module").agg(n=("parabolic", "size"), rate=("parabolic", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
