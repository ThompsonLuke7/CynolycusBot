"""Unit tests for the live catalyst signal reader + EV tilt."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.multi_ticker_swing.live.catalyst_signal import (
    CatalystHit,
    LiveCatalystSignal,
    NEUTRAL,
    catalyst_ev_multiplier,
)


def _write_ledger(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_missing_ledger_is_neutral(tmp_path):
    sig = LiveCatalystSignal(tmp_path / "nope.parquet")
    assert sig.for_ticker("AAPL") == NEUTRAL


def test_recency_weighted_score_and_window(tmp_path):
    now = pd.Timestamp.now(tz="UTC")
    p = tmp_path / "live.parquet"
    _write_ledger(p, [
        {"ticker": "aapl", "timestamp": now - pd.Timedelta(minutes=5), "catalyst_score": 0.9,
         "headline": "FDA approval", "catalyst_family": "regulatory"},
        {"ticker": "AAPL", "timestamp": now - pd.Timedelta(minutes=120), "catalyst_score": 0.2,
         "headline": "old note", "catalyst_family": "misc"},
        {"ticker": "AAPL", "timestamp": now - pd.Timedelta(minutes=600), "catalyst_score": 1.0,
         "headline": "stale", "catalyst_family": "misc"},  # outside 180m window
        {"ticker": "TSLA", "timestamp": now, "catalyst_score": 0.5, "headline": "x", "catalyst_family": "y"},
    ])
    sig = LiveCatalystSignal(p, lookback_minutes=180, halflife_minutes=45.0)
    hit = sig.for_ticker("AAPL", now=now)
    assert hit.count == 2                      # stale 600m record excluded
    assert 0.2 < hit.score < 0.9               # weighted between the two in-window
    assert hit.score > 0.55                    # fresh 0.9 dominates via recency weight
    assert hit.top_headline == "FDA approval"  # highest-score record surfaced
    assert sig.for_ticker("NVDA", now=now) == NEUTRAL


def test_tilt_direction_aware():
    bull = CatalystHit(score=0.9, count=1, latest_ts=None, top_headline=None, top_family=None)
    # bullish catalyst boosts longs, dampens shorts
    assert catalyst_ev_multiplier(bull, direction=1, alpha=0.5) > 1.0
    assert catalyst_ev_multiplier(bull, direction=-1, alpha=0.5) < 1.0
    # neutral / absent -> no effect
    assert catalyst_ev_multiplier(NEUTRAL, direction=1, alpha=0.5) == 1.0
    assert catalyst_ev_multiplier(None, direction=1, alpha=0.5) == 1.0
    # baseline score -> no effect
    mid = CatalystHit(score=0.5, count=1, latest_ts=None, top_headline=None, top_family=None)
    assert catalyst_ev_multiplier(mid, direction=1, alpha=0.5) == pytest.approx(1.0)
    # multiplier stays positive even with a strong bearish-vs-long mismatch
    assert catalyst_ev_multiplier(bull, direction=-1, alpha=5.0) >= 0.05
