"""Earnings result and guidance catalyst records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signals.events.forward_guidance.config import FEATURES_PATH, LABELS_PATH


def _read(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p) if p.suffix.lower() != ".csv" else pd.read_csv(p)


def _safe_score(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(np.clip(float(value), -1.0, 1.0))
    except Exception:
        return 0.0


def build_earnings_result_catalysts(
    *,
    features_path: Path | str = FEATURES_PATH,
    labels_path: Path | str = LABELS_PATH,
) -> pd.DataFrame:
    """Convert forward-guidance features into unified post-result catalyst records."""
    features = _read(features_path)
    labels = _read(labels_path)
    if features.empty:
        return pd.DataFrame()
    keys = [c for c in ["event_id", "ticker", "earnings_date", "signal_timestamp"] if c in features.columns and c in labels.columns]
    df = features.merge(labels, on=keys, how="left", suffixes=("", "_label")) if not labels.empty and keys else features.copy()
    rows = []
    for row in df.itertuples(index=False):
        event_id = str(getattr(row, "event_id", ""))
        ticker = str(getattr(row, "ticker", "")).upper().replace("$", "")
        guidance_strength = _safe_score(getattr(row, "guidance_strength_score", 0.0))
        rev_score = _safe_score(getattr(row, "guidance_revenue_raise_cut", 0.0))
        eps_score = _safe_score(getattr(row, "guidance_eps_raise_cut", 0.0))
        uncertainty = float(getattr(row, "uncertainty_language", 0.0) or 0.0)
        confidence = float(getattr(row, "confidence_language", 0.0) or 0.0)
        metric_revenue = getattr(row, "metric_revenue_actual", None)
        metric_eps = getattr(row, "metric_eps_actual", None)
        text_bits = [
            f"{ticker} earnings result",
            f"guidance_strength={guidance_strength:.3f}",
            f"revenue_language={rev_score:.3f}",
            f"eps_language={eps_score:.3f}",
            f"confidence_mentions={confidence:.0f}",
            f"uncertainty_mentions={uncertainty:.0f}",
        ]
        if metric_revenue is not None and not pd.isna(metric_revenue):
            text_bits.append(f"reported_revenue={metric_revenue}")
        if metric_eps is not None and not pd.isna(metric_eps):
            text_bits.append(f"reported_eps={metric_eps}")
        if guidance_strength > 0.25:
            impact_role = "earnings_guidance_raise"
        elif guidance_strength < -0.25:
            impact_role = "earnings_guidance_cut"
        elif uncertainty > confidence and uncertainty >= 2:
            impact_role = "earnings_uncertainty"
        else:
            impact_role = "earnings_result"
        rows.append(
            {
                "catalyst_id": f"earnings_result:{event_id}",
                "record_id": event_id,
                "ticker": ticker,
                "timestamp": pd.to_datetime(getattr(row, "signal_timestamp", None), utc=True, errors="coerce"),
                "catalyst_kind": "earnings_result",
                "event_type": "earnings_result_guidance",
                "headline": f"{ticker} earnings result / guidance",
                "summary": " | ".join(text_bits),
                "source": getattr(row, "source_type", "forward_guidance"),
                "url": getattr(row, "source_url", ""),
                "relation_type": "scheduled_ticker_event",
                "impact_role": impact_role,
                "relation_confidence": 1.0,
                "is_direct_catalyst": 1.0,
                "earnings_guidance_score": guidance_strength,
                "earnings_revenue_language_score": rev_score,
                "earnings_eps_language_score": eps_score,
                "fwd_ret_5d": getattr(row, "fwd_ret_5d", np.nan),
                "max_drawdown": getattr(row, "max_drawdown", np.nan),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["timestamp"]).reset_index(drop=True)
