"""Stage 3 shadow scorer: no-op without boosters, scores archived vectors with them."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core.decision_state_archive import archive_decision_state
from strategies.momentum_expansion.ablation import score_archive_stage3 as sa

FEATS = ["fa", "fb", "fc"]


def _boosters(folder):
    import xgboost as xgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    seeds = []
    for seed in (42, 43):
        m = xgb.XGBClassifier(n_estimators=20, max_depth=2, random_state=seed, verbosity=0)
        m.fit(X, y)
        b = m.get_booster()
        b.feature_names = FEATS
        b.save_model(str(folder / f"stage3_final_L2_20__pit_s{seed}.json"))
        seeds.append({"seed": seed, "best_iteration": 19})
    (folder / "stage3_final_meta.json").write_text(json.dumps(
        {"arm": {"name": "L2_20__pit"}, "features": FEATS, "seeds": seeds, "selection": "621d0acaa1cd"}))


def test_no_boosters_is_a_noop(tmp_path):
    out = sa.run(days=5, booster_dir=tmp_path / "none", out_dir=tmp_path / "shadow")
    assert out["status"] == "no_boosters" and not (tmp_path / "shadow").exists()


def test_scores_archived_rows_and_flags_pre_fix_runs(tmp_path, monkeypatch):
    bdir = tmp_path / "boosters"
    bdir.mkdir()
    _boosters(bdir)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    monkeypatch.setattr(sa, "PARITY_FIXED_FROM", now - pd.Timedelta(minutes=30))
    cand = pd.DataFrame({"ticker": ["A", "B", "C"], "fa": [2.0, -2.0, np.nan], "fb": [0.0, 0.0, 0.0],
                         "fc": [1.0, 1.0, 1.0], "expansion_score": [0.5, 0.4, 0.3]})
    archive_decision_state(module="momentum_expansion", strict=True, candidates=cand,
                           bar_ts=now - pd.Timedelta(hours=1), score_col="expansion_score",
                           feature_cols=FEATS, decision_ts=now)
    stats = sa.run(days=5, booster_dir=bdir, out_dir=tmp_path / "shadow")
    assert stats["status"] == "ok" and stats["rows"] == 3 and stats["scored"] == 2
    res = pd.concat(pd.read_parquet(p) for p in (tmp_path / "shadow").glob("*.parquet")).set_index("ticker")
    assert res.loc["A", "stage3_score"] > res.loc["B", "stage3_score"]      # learned fa > 0
    assert np.isnan(res.loc["C", "stage3_score"])                          # incomplete row: not scored
    assert not res["pre_parity_fix"].any()                                 # decided after the fix
    assert sa.run(days=5, booster_dir=bdir, out_dir=tmp_path / "shadow")["dates"] == 0
