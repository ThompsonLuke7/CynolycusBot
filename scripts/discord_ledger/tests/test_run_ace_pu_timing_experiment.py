from __future__ import annotations

import pytest

from scripts.discord_ledger.run_ace_pu_timing_experiment import (
    bootstrap_mean_ci,
    group_ranking_metrics,
    join_candidates,
    mde_pct,
)


def _source(*, alert: bool, decision: str) -> dict:
    return {
        "symbol": "SPY", "direction_sign": 1, "decision_at_utc": decision,
        "source_message_ids": ["m1"], "session_date_et": "2026-01-05",
        "directional_return_5m": 0.01, "directional_return_20m": 0.02,
        "directional_vwap_distance_pct": 0.03, "volume_last5_vs_prior20": 1.5,
        "directional_opening_range_high_distance_pct": 0.1,
        "directional_forward_return_20m": 0.01 if alert else 0.0,
        "directional_forward_return_60m": 0.02 if alert else 0.0,
    }


def _context(*, kind: str, decision: str) -> dict:
    return {
        "row_kind": kind, "symbol": "SPY", "direction_sign": 1, "decision_at_utc": decision,
        "source_message_ids": ["m1"], "watchlist_24h": True, "watchlist_7d": True,
        "watchlist_level_7d": False, "watchlist_intention_or_instruction_7d": False,
    }


def test_join_candidates_requires_complete_deterministic_context() -> None:
    alert = _source(alert=True, decision="2026-01-05T15:00:00Z")
    control = _source(alert=False, decision="2026-01-05T15:05:00Z")
    rows = join_candidates([alert], [control], [
        _context(kind="alert", decision=alert["decision_at_utc"]),
        _context(kind="control", decision=control["decision_at_utc"]),
    ])
    assert [row["is_observed_alert"] for row in rows] == [1, 0]
    assert all("session_time_sin" in row for row in rows)
    with pytest.raises(ValueError, match="missing context"):
        join_candidates([alert], [control], [_context(kind="alert", decision=alert["decision_at_utc"])])


def test_group_ranking_and_mde_are_group_level() -> None:
    rows = [
        {"candidate_group_message_ids": ["a"], "is_observed_alert": 1},
        {"candidate_group_message_ids": ["a"], "is_observed_alert": 0},
        {"candidate_group_message_ids": ["b"], "is_observed_alert": 1},
        {"candidate_group_message_ids": ["b"], "is_observed_alert": 0},
    ]
    metrics = group_ranking_metrics(rows, score=[0.9, 0.1, 0.2, 0.8])
    assert metrics["eligible_groups"] == 2
    assert metrics["top_1_rate"] == pytest.approx(0.5)
    assert metrics["mean_rank"] == pytest.approx(1.5)
    assert mde_pct(__import__("numpy").array([0.01, 0.02])) is not None
    assert bootstrap_mean_ci(__import__("numpy").array([0.01, 0.02])) is not None
