"""Daily ranking/inference for forward-guidance events."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forward_guidance.config import (
    DASHBOARD_STATE_PATH,
    MODEL_META_PATH,
    PRIMARY_PROBABILITY,
    RANKED_OUTPUT_CSV,
    RANKED_OUTPUT_PARQUET,
    XGB_MODEL_PATH,
    ensure_data_dirs,
)
from forward_guidance.data.ingest_events import load_events_from_csv
from forward_guidance.data.schema import EarningsEvent
from forward_guidance.features.build_matrix import build_feature_row
from forward_guidance.utils.io import read_json, write_dataframe, write_json

logger = logging.getLogger(__name__)


def _load_xgb_model(path: Path | str):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required to score an XGBoost model.") from exc
    model = XGBClassifier()
    model.load_model(str(path))
    return model


def _load_lgb_model(path: Path | str):
    try:
        from lightgbm import Booster
    except ImportError as exc:
        raise ImportError("lightgbm is required to score a LightGBM model.") from exc
    return Booster(model_file=str(path))


def _expected_return(probability: float, meta: dict[str, Any]) -> float:
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    top_mean = metrics.get("top_bucket_mean_return")
    if isinstance(top_mean, (int, float)) and np.isfinite(top_mean):
        return float(probability * top_mean)
    return float("nan")


def score_feature_frame(features: pd.DataFrame, *, model_path: Path | str | None = None, meta_path: Path | str = MODEL_META_PATH) -> pd.DataFrame:
    meta = read_json(meta_path, default={}) or {}
    kind = str(meta.get("model_kind") or "xgboost")
    path = Path(model_path or meta.get("model_path") or XGB_MODEL_PATH)
    cols = list(meta.get("feature_columns") or [])
    if not cols:
        from forward_guidance.features.build_matrix import feature_columns

        cols = feature_columns(features)
    missing = [c for c in cols if c not in features.columns]
    scored = features.copy()
    for col in missing:
        scored[col] = np.nan
    X = scored[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).values
    if kind == "lightgbm":
        model = _load_lgb_model(path)
        proba = model.predict(X)
    else:
        model = _load_xgb_model(path)
        proba = model.predict_proba(X)[:, 1]
    scored[PRIMARY_PROBABILITY] = proba.astype(float)
    scored["expected_return"] = [_expected_return(float(p), meta) for p in scored[PRIMARY_PROBABILITY]]
    scored["confidence"] = (scored[PRIMARY_PROBABILITY] - 0.5).abs().clip(0, 0.5) * 2
    scored["holding_horizon"] = "60d"
    reason_cols = [
        "guidance_strength_score",
        "post_er_gap_pct",
        "post_er_move_pct",
        "guidance_reaction_disagreement_score",
        "technical_stabilization_flag",
    ]
    scored["top_reason_features"] = scored.apply(
        lambda row: ", ".join(
            f"{c}={row[c]:.3f}" for c in reason_cols if c in row and pd.notna(row[c])
        ),
        axis=1,
    )
    return scored.sort_values(PRIMARY_PROBABILITY, ascending=False).reset_index(drop=True)


def score_events(
    events: list[EarningsEvent],
    *,
    generate_embeddings: bool = False,
    generate_finbert: bool = False,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    ensure_data_dirs()
    rows = []
    for event in events:
        row, _labels = build_feature_row(event, generate_embeddings=generate_embeddings, generate_finbert=generate_finbert)
        rows.append(row)
    features = pd.DataFrame(rows)
    if features.empty:
        ranked = pd.DataFrame()
    else:
        ranked = score_feature_frame(features, model_path=model_path)
    write_dataframe(ranked, RANKED_OUTPUT_PARQUET)
    write_dataframe(ranked, RANKED_OUTPUT_CSV)
    write_json(
        {
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "count": int(len(ranked)),
            "output_parquet": str(RANKED_OUTPUT_PARQUET),
            "output_csv": str(RANKED_OUTPUT_CSV),
            "top": ranked.head(25).to_dict(orient="records") if not ranked.empty else [],
        },
        DASHBOARD_STATE_PATH,
    )
    return ranked


def main() -> int:
    parser = argparse.ArgumentParser(description="Score daily forward-guidance earnings opportunities.")
    parser.add_argument("--events-csv", required=True, help="CSV with ticker, earnings_date, report_time, optional sector_etf/cik.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--embeddings", action="store_true", help="Generate sentence-transformer embeddings if dependencies are installed.")
    parser.add_argument("--finbert", action="store_true", help="Generate FinBERT features if dependencies are installed.")
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()), format="%(asctime)s %(levelname)s %(message)s")
    events = load_events_from_csv(args.events_csv)
    ranked = score_events(events, generate_embeddings=args.embeddings, generate_finbert=args.finbert, model_path=args.model_path)
    print(ranked[["ticker", PRIMARY_PROBABILITY, "expected_return", "confidence", "holding_horizon"]].head(25).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
