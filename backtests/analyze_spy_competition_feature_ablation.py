from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MATRIX = ROOT / "Data/processed/spy/training_export/spy_daytrader_context_matrix.parquet"
MANIFEST = ROOT / "Data/processed/spy/training_export/spy_daytrader_context_manifest.json"
MODELS = ROOT / "Data/models/ga_xgboost/10min/competition_20260619"
OUT = ROOT / "backtests/20260619_spy_competition/feature_ablation.csv"
TOP_K = 5


def _score(model: xgb.Booster, frame: pd.DataFrame, features: list[str], best_iteration: int) -> np.ndarray:
    matrix = (
        frame.reindex(columns=features)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(np.float32)
    )
    return model.predict(
        xgb.DMatrix(matrix, missing=np.nan),
        iteration_range=(0, best_iteration + 1),
    )


def _evaluate(frame: pd.DataFrame, scores: np.ndarray, *, side: str) -> dict[str, float]:
    work = frame.copy()
    work["score"] = scores
    y = pd.to_numeric(work[f"{side}_swing_label"], errors="coerce").fillna(0).astype(int)
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = float("nan")

    precision = []
    ndcg = []
    selected_returns = []
    ret_col = f"fwd_ret_6_{side}"
    for _, day in work.groupby("session", sort=True):
        if len(day) < 2:
            continue
        top = day.nlargest(min(TOP_K, len(day)), "score")
        precision.append(float(pd.to_numeric(top[f"{side}_swing_label"], errors="coerce").mean()))
        selected_returns.extend(pd.to_numeric(top[ret_col], errors="coerce").dropna().tolist())
        relevance = pd.to_numeric(day[f"{side}_swing_label"], errors="coerce").fillna(0).to_numpy()
        try:
            ndcg.append(float(ndcg_score(relevance.reshape(1, -1), day["score"].to_numpy().reshape(1, -1), k=TOP_K)))
        except ValueError:
            pass
    directional = pd.Series(selected_returns, dtype=float)
    corr = spearmanr(
        scores,
        pd.to_numeric(work[ret_col], errors="coerce"),
        nan_policy="omit",
    ).statistic
    return {
        "auc": auc,
        "ndcg_at_5_binary": float(np.nanmean(ndcg)),
        "precision_at_5": float(np.nanmean(precision)),
        "mean_60m_directional_return": float(directional.mean()),
        "win_rate_60m": float((directional > 0).mean()),
        "score_60m_spearman": float(corr),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    features = list(manifest["feature_columns"])
    group_cols = {
        "options": list(manifest["option_feature_columns"]),
        "bid_ask": list(manifest["bid_ask_feature_columns"]),
        "liquidity": list(manifest["liquidity_feature_columns"]),
        "dealer": list(manifest["dealer_feature_columns"]),
    }
    enriched = sorted({col for cols in group_cols.values() for col in cols})

    frame = pd.read_parquet(MATRIX)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["session"] = local.dt.normalize()
    close = pd.to_numeric(frame["raw_close"], errors="coerce")
    frame["fwd_ret_6_long"] = close.groupby(frame["session"]).shift(-6) / close - 1.0
    frame["fwd_ret_6_short"] = -frame["fwd_ret_6_long"]

    n = len(frame)
    splits = {
        "validation": frame.iloc[int(n * 0.70) : int(n * 0.85)].copy(),
        "test": frame.iloc[int(n * 0.85) :].copy(),
    }
    variants = {
        "full": [],
        "base_only": enriched,
        "no_options": group_cols["options"],
        "no_bid_ask": group_cols["bid_ask"],
        "no_liquidity": group_cols["liquidity"],
        "no_dealer": group_cols["dealer"],
    }

    rows = []
    for side in ("long", "short"):
        side_dir = MODELS / f"{side}_swing_label"
        meta = json.loads((side_dir / "competition_meta.json").read_text())
        model = xgb.Booster()
        model.load_model(side_dir / "winner_model.ubj")
        model.set_param({"device": "cpu"})
        best_iteration = int(meta["best"]["best_iteration"])

        for split_name, split in splits.items():
            option_present = split[group_cols["options"]].notna().any(axis=1)
            subsets = {
                "all": split,
                "options_present": split.loc[option_present],
                "options_missing": split.loc[~option_present],
            }
            for subset_name, subset in subsets.items():
                if len(subset) < 50:
                    continue
                for variant, masked_cols in variants.items():
                    work = subset.copy()
                    existing = [col for col in masked_cols if col in work.columns]
                    if existing:
                        work.loc[:, existing] = np.nan
                    scores = _score(model, work, features, best_iteration)
                    rows.append(
                        {
                            "side": side,
                            "split": split_name,
                            "subset": subset_name,
                            "variant": variant,
                            "rows": len(work),
                            **_evaluate(work, scores, side=side),
                        }
                    )

    result = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
