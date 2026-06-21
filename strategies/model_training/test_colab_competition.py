"""CPU smoke test for the multi-family / multi-seed competition harness.

Exercises every family (incl. the regression families) on a small synthetic
cross-sectional frame, plus the walk-forward OOF helper. Runs in a few seconds on
CPU with tiny trees — it checks plumbing, not predictive quality.

    .venv/bin/python -m strategies.model_training.test_colab_competition
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.model_training.colab_competition import (
    CompetitionConfig,
    parse_families,
    parse_seeds,
    run_competition,
    walk_forward_oof,
)


def _synthetic_frame(n_days: int = 420, per_day: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B", tz="UTC")
    rows = []
    for d in dates:
        x1 = rng.normal(size=per_day)
        x2 = rng.normal(size=per_day)
        x3 = rng.normal(size=per_day)
        noise = rng.normal(scale=0.5, size=per_day)
        quality = 0.9 * x1 - 0.4 * x2 + 0.2 * x3 + noise  # continuous "trade quality"
        thr = np.quantile(quality, 0.85)                   # top ~15% per day are "good"
        good = (quality >= thr).astype(int)
        rows.append(pd.DataFrame({
            "timestamp": d,
            "ticker": [f"T{i:03d}" for i in range(per_day)],
            "f1": x1, "f2": x2, "f3": x3,
            "trade_quality": quality,
            "is_good": good,
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df = _synthetic_frame()
    feats = ["f1", "f2", "f3"]
    families = parse_families("xgb_regressor,xgb_classifier,xgb_ranker,lgbm_regressor,lgbm_classifier,lgbm_ranker")
    seeds = parse_seeds(None, 2)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = CompetitionConfig(
            task_name="harness_smoke",
            target_column="is_good",
            regression_target_column="trade_quality",
            relevance_column="is_good",
            feature_columns=feats,
            train_frac=0.6,
            val_frac=0.2,
            seeds=seeds,
            families=families,
            output_dir=Path(tmp) / "artifacts",
            rank_group="date",
            top_k=10,
            # tiny trees so the smoke test is fast
            xgb_config={"n_estimators": 40, "max_depth": 3, "early_stopping_rounds": 10},
            lgbm_config={"n_estimators": 40, "num_leaves": 15},
            device="cpu",
        )
        result = run_competition(df, cfg)
        results, summary = result["results"], result["summary"]

        trained = set(results["family"].unique())
        assert trained == set(families), f"missing families: {set(families) - trained}"
        assert any(c.startswith("test_ndcg_at_") for c in results.columns), "no NDCG metric"
        assert "test_spearman" in results.columns, "no Spearman metric"
        assert "noise_flag" in summary.columns, "no noise flag in summary"
        best = result["best"]
        print("best:", best["family"], "seed", int(best["seed"]),
              "ndcg@10", round(float(best.get("test_ndcg_at_10", float("nan"))), 4),
              "spearman", round(float(best.get("test_spearman", float("nan"))), 4))

        # Walk-forward OOF with the winning family/seed.
        oof = walk_forward_oof(
            df, cfg, str(best["family"]), int(best["seed"]),
            train_months=6, embargo_days=10, test_months=2,
            min_train_rows=2000, min_test_rows=200,
            diagnostic_columns=["trade_quality"],
        )
        assert not oof.empty, "walk_forward_oof produced no rows"
        assert {"score", "y"}.issubset(oof.columns), "OOF missing score/y"
        rho = oof["score"].corr(oof["y"], method="spearman")
        print(f"OOF rows={len(oof):,}  OOF Spearman(score,y)={rho:.4f}")
        assert rho > 0.3, f"OOF signal implausibly weak ({rho:.3f}) — plumbing likely broken"

    print("PASS: colab_competition harness smoke test")


if __name__ == "__main__":
    main()
