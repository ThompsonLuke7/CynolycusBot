"""Fixed walk-forward fold spec for the ablation harness (WS-E deliverable 1).

Reuses ``strategies.model_training.colab_competition.date_folds`` verbatim —
the same fold builder the existing momentum/HTF Colab trainers use — rather
than inventing a new split (task instruction: "Use the repo's existing
walk-forward spec ... Do not invent a different split"). Folds are sourced
from ``WALK_FORWARD_CONFIG`` (2.0 train years, 21-day embargo, 6-month test
windows, 50k min train rows), not hand-picked dates.
"""
from __future__ import annotations

import pandas as pd

from strategies.model_training.colab_competition import date_folds
from strategies.momentum_expansion.config.momentum_config import WALK_FORWARD_CONFIG


def build_walk_forward_folds(
    ts: pd.Series,
    *,
    cfg: dict | None = None,
) -> list[dict[str, pd.Timestamp]]:
    """Non-overlapping (train / embargo / test) folds spanning ``ts``'s range.

    ``cfg`` defaults to the repo's ``WALK_FORWARD_CONFIG``; ``train_years`` is
    converted to whole months (``date_folds`` takes ``train_months``) the same
    way the existing Colab trainers do (``round(train_years * 12)``).
    """
    cfg = {**WALK_FORWARD_CONFIG, **(cfg or {})}
    train_months = int(round(float(cfg["train_years"]) * 12))
    return date_folds(
        ts,
        train_months=train_months,
        embargo_days=int(cfg["embargo_days"]),
        test_months=int(cfg["test_months"]),
    )


def fold_label(fold: dict[str, pd.Timestamp]) -> str:
    """Human-readable test-window label, e.g. '2024-01..2024-07'."""
    return f"{fold['test_start'].date()}..{fold['test_end'].date()}"


def train_test_masks(ts: pd.Series, fold: dict[str, pd.Timestamp]) -> tuple[pd.Series, pd.Series]:
    """Boolean train/test masks for one fold. Train excludes the embargo gap
    (``train_end`` already sits ``embargo_days`` before ``test_start`` per
    ``date_folds``) and the two masks never overlap by construction."""
    ts = pd.to_datetime(ts, utc=True)
    train_mask = (ts >= fold["train_start"]) & (ts <= fold["train_end"])
    test_mask = (ts >= fold["test_start"]) & (ts <= fold["test_end"])
    return train_mask, test_mask


def min_train_rows_met(ts: pd.Series, fold: dict[str, pd.Timestamp], *, cfg: dict | None = None) -> bool:
    cfg = {**WALK_FORWARD_CONFIG, **(cfg or {})}
    train_mask, _ = train_test_masks(ts, fold)
    return int(train_mask.sum()) >= int(cfg["min_train_rows"])
