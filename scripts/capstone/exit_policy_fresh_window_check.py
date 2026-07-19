"""
Directional confirmation of the segmented-policy study's recommendation on the
genuinely untouched window (2026-05-15 -> present).

IMPORTANT CAVEAT — this is NOT a clean out-of-sample test. The walk-forward
OOF prediction files every prior study (including
exit_policy_segmentation_2026-07-19.md) relied on stop at 2026-05-14, exactly
the frozen-test cutoff; extending them requires a real Colab/GPU retrain,
which is out of scope here. This script instead scores the fresh window with
the DEPLOYED boosters (same classes/artifacts the live system uses:
ExpansionRanker, HTFSwingScorer, meta score.py's score_frame). Those boosters
are periodically retrained and have almost certainly seen some/all of this
window during training (leakage_audit.md Section 4.3) — so this can only be
read as a sanity/direction check ("does anything look obviously reversed?"),
never as validation. Label every result accordingly.

Reuses the exact trade-simulation mechanic from exit_policy_segmentation.py
(all_trades) so results are apples-to-apples with the frozen-test study.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_fresh_window_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/capstone"))
sys.path.insert(0, str(REPO / "signals/meta_context/meta_ranker"))
import exit_policy_segmentation as seg  # noqa: E402
import score as meta_score  # noqa: E402

FRESH_START = pd.Timestamp("2026-05-15", tz="UTC")
FRESH_END = None  # open-ended: use whatever is in the feature files today
TOPK = seg.TOPK
POLICIES = seg.POLICIES
OUT_DIR = seg.OUT_DIR / "fresh_window_directional_check"


def _score_momentum() -> pd.DataFrame:
    import json
    from strategies.momentum_expansion.inference.ranker import ExpansionRanker

    # NOTE: ExpansionRanker() defaults feature_columns to the static
    # FEATURE_COLUMNS_4H superset (108 cols), which does NOT match the
    # installed booster's actual GA-selected feature set (106 cols, per its
    # own feature_manifest.json) -> xgboost rejects the column-count
    # mismatch. Unlike HTFSwingScorer, ExpansionRanker never reads its own
    # manifest. Loading feature_columns from the manifest explicitly here
    # works around it for this backtest; flagged separately as a possible
    # live-path bug, not fixed (out of scope for this check).
    manifest = json.loads(
        (REPO / "strategies/momentum_expansion/models/expansion_v1/feature_manifest.json").read_text())
    feats = pd.read_parquet(REPO / "strategies/momentum_expansion/data/processed/features_4h.parquet")
    feats = feats.reset_index()
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    feats = feats[feats["timestamp"] >= FRESH_START]
    ranker = ExpansionRanker(feature_columns=manifest["feature_columns"])
    score = ranker.score(feats)
    out = feats[["timestamp", "ticker"]].copy()
    out["score"] = score.values
    return out.dropna(subset=["score"])


def _score_htf() -> pd.DataFrame:
    from strategies.multi_ticker_swing_htf.inference.scorer import HTFSwingScorer

    feats = pd.read_parquet(REPO / "strategies/multi_ticker_swing_htf/data/processed/features_4h.parquet")
    feats = feats.reset_index()
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    feats = feats[feats["timestamp"] >= FRESH_START]
    scorer = HTFSwingScorer()
    score = scorer.score(feats)
    out = feats[["timestamp", "ticker"]].copy()
    out["score"] = score.values
    return out.dropna(subset=["score"])


def _score_meta() -> pd.DataFrame:
    df = pd.read_parquet(REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet").reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["timestamp"] >= FRESH_START]
    scored = meta_score.score_frame(df)
    return scored[["timestamp", "ticker", "s_combo"]].rename(columns={"s_combo": "score"}).dropna(subset=["score"])


def to_member(scored: pd.DataFrame, topk: int = TOPK) -> pd.DataFrame:
    scored = scored.copy()
    scored["rk"] = scored.groupby("timestamp")["score"].rank(ascending=False, method="first")
    scored["in_top"] = scored["rk"] <= topk
    return scored[["timestamp", "ticker", "in_top"]], scored[["timestamp", "ticker", "score", "rk"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scorers = {"momentum": _score_momentum, "htf": _score_htf, "meta": _score_meta}

    rows = []
    gate_rows = []
    for module, fn in scorers.items():
        scored = fn()
        n_ts = scored["timestamp"].nunique()
        print(f"\n[{module}] deployed-score fresh window: {scored['timestamp'].min()} -> "
              f"{scored['timestamp'].max()}  ({n_ts} bars, {scored['ticker'].nunique()} tickers)")
        member, full = to_member(scored)
        scored.to_csv(OUT_DIR / f"deployed_scores_{module}.csv", index=False)

        print(f"  -- policy comparison (deployed scores, DIRECTIONAL ONLY) --")
        for pname, cfg in POLICIES.items():
            tr = seg.all_trades(member, **cfg)
            if len(tr) == 0:
                print(f"    {pname:14s} n=0 (window too short for this policy's horizon)")
                continue
            r = tr["ret"].values
            h = tr["bars_held"].values
            row = dict(module=module, policy=pname, n=len(r), mean=r.mean(), median=np.median(r),
                      win=(r > 0).mean(), rpb=(r / np.maximum(h, 1)).mean(), total=r.sum(),
                      hold=h.mean())
            rows.append(row)
            print(f"    {pname:14s} n={row['n']:4d} mean={row['mean']*100:6.2f}% "
                  f"win={row['win']*100:5.1f}% rpb={row['rpb']:.4f} hold={row['hold']:5.1f} "
                  f"total={row['total']:.2f}")

        # 3b re-check: apply the frozen val-selected gate thresholds from the
        # prior study to this fresh window's deployed scores, exit fixed g284
        prior_gates = pd.read_csv(seg.OUT_DIR.parent / "segmentation" / "phase3b_score_gate.csv")
        gate_row = prior_gates[(prior_gates["module"] == module) & (prior_gates["window"] == "val")
                               & (prior_gates["variant"].str.startswith("gate_q"))]
        winner_row = prior_gates[(prior_gates["module"] == module) & (prior_gates["window"] == "test")
                                 & (prior_gates["variant"].str.startswith("gate_q"))]
        if winner_row.empty:
            print(f"  -- no 3b winning gate recorded for {module}, skipping gate re-check --")
            continue
        gate_name = winner_row.iloc[0]["variant"]
        gate_thr = float(winner_row.iloc[0]["threshold"])
        base_g_member, _ = to_member(scored)
        base_tr = seg.all_trades(base_g_member, **POLICIES["g284"])
        gated_scored = full.copy()
        gated_member = base_g_member.copy()
        gated_member["in_top"] = base_g_member["in_top"].values & (gated_scored["score"] >= gate_thr).values
        gated_tr = seg.all_trades(gated_member, **POLICIES["g284"])
        if len(base_tr) == 0 or len(gated_tr) == 0:
            print(f"  -- {module} gate re-check: too few trades in fresh window (base n={len(base_tr)}, "
                  f"gated n={len(gated_tr)}) — underpowered, not reported --")
            continue
        g = dict(module=module, gate=gate_name, threshold=gate_thr,
                base_n=len(base_tr), base_mean=base_tr["ret"].mean(), base_rpb=(base_tr["ret"]/np.maximum(base_tr["bars_held"],1)).mean(),
                gated_n=len(gated_tr), gated_mean=gated_tr["ret"].mean(), gated_rpb=(gated_tr["ret"]/np.maximum(gated_tr["bars_held"],1)).mean())
        gate_rows.append(g)
        print(f"  -- 3b gate re-check ({gate_name}, thr={gate_thr:.4f}, exit=g284, DIRECTIONAL ONLY) --")
        print(f"    base : n={g['base_n']:4d} mean={g['base_mean']*100:6.2f}% rpb={g['base_rpb']:.4f}")
        print(f"    gated: n={g['gated_n']:4d} ({g['gated_n']/g['base_n']:.0%} of base) "
              f"mean={g['gated_mean']*100:6.2f}% rpb={g['gated_rpb']:.4f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "policy_comparison.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUT_DIR / "gate_recheck.csv", index=False)
    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
