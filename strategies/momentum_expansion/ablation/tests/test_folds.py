"""Tests for strategies/momentum_expansion/ablation/folds.py.

Verifies the fold spec matches WALK_FORWARD_CONFIG verbatim, and the
fold-boundary invariants the task explicitly calls out: embargo respected,
no train/test overlap.
"""
from __future__ import annotations

import pandas as pd

from strategies.momentum_expansion.ablation.folds import (
    build_walk_forward_folds,
    fold_label,
    min_train_rows_met,
    train_test_masks,
)
from strategies.momentum_expansion.config.momentum_config import WALK_FORWARD_CONFIG


def _synthetic_timestamps(start="2020-01-01", end="2026-01-01", freq="4h"):
    return pd.Series(pd.date_range(start, end, freq=freq, tz="UTC"))


class TestFoldSpec:
    def test_uses_walk_forward_config_verbatim(self):
        ts = _synthetic_timestamps()
        folds = build_walk_forward_folds(ts)
        assert len(folds) > 0
        f = folds[0]
        expected_train_span = pd.DateOffset(months=int(round(WALK_FORWARD_CONFIG["train_years"] * 12)))
        # train_end - train_start should equal (train_start + train_span) - train_start
        assert (f["train_start"] + expected_train_span) == f["train_end"]

    def test_embargo_gap_respected(self):
        ts = _synthetic_timestamps()
        folds = build_walk_forward_folds(ts)
        embargo = pd.Timedelta(days=WALK_FORWARD_CONFIG["embargo_days"])
        for f in folds:
            assert f["test_start"] - f["train_end"] == embargo

    def test_no_train_test_overlap_for_any_fold(self):
        ts = _synthetic_timestamps()
        folds = build_walk_forward_folds(ts)
        for f in folds:
            train_mask, test_mask = train_test_masks(ts, f)
            assert not (train_mask & test_mask).any()
            # explicit boundary check too
            assert f["train_end"] < f["test_start"]

    def test_folds_are_chronologically_ordered_and_non_overlapping_test_windows(self):
        ts = _synthetic_timestamps()
        folds = build_walk_forward_folds(ts)
        for a, b in zip(folds, folds[1:]):
            assert a["test_end"] <= b["test_start"]

    def test_test_window_is_six_months(self):
        ts = _synthetic_timestamps()
        folds = build_walk_forward_folds(ts)
        for f in folds:
            assert (f["test_start"] + pd.DateOffset(months=WALK_FORWARD_CONFIG["test_months"])) == f["test_end"]

    def test_custom_cfg_override(self):
        ts = _synthetic_timestamps()
        folds_default = build_walk_forward_folds(ts)
        folds_short = build_walk_forward_folds(ts, cfg={"train_years": 1.0})
        # Shorter train window -> at least as many folds fit in the same span.
        assert len(folds_short) >= len(folds_default)

    def test_fold_label_format(self):
        ts = _synthetic_timestamps()
        folds = build_walk_forward_folds(ts)
        label = fold_label(folds[0])
        assert ".." in label
        assert str(folds[0]["test_start"].date()) in label

    def test_min_train_rows_gate(self):
        ts = _synthetic_timestamps()
        folds = build_walk_forward_folds(ts)
        # every row of ts is 4h apart; a full 2-year train window has far more
        # than min_train_rows(50k) if we imagine 1 row per timestamp per many
        # tickers, but with only ONE series it should be far short.
        assert min_train_rows_met(ts, folds[0], cfg={"min_train_rows": 10}) is True
        assert min_train_rows_met(ts, folds[0], cfg={"min_train_rows": 10**9}) is False

    def test_no_folds_when_history_shorter_than_train_plus_embargo(self):
        ts = _synthetic_timestamps(start="2025-01-01", end="2025-06-01")
        folds = build_walk_forward_folds(ts)
        assert folds == []
