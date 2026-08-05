"""Runner-side lineage for governed Meta intents (Task 23, increment 2).

The mapping in ``nervous_system_adapter.py`` is only as good as what the runner
feeds it. These tests pin what the runner alone can supply: the per-ticker
scoring context (including held names that dropped out of the top-K) and the
model/config versions that make a persisted decision reproducible.
"""

from __future__ import annotations

import argparse
from decimal import Decimal

import pandas as pd
import pytest

from core.nervous_system.contracts.enums import InstrumentFamily
from signals.meta_context.meta_ranker.live_runner import (
    MetaRankingResult,
    feature_version,
    intent_config,
    runner_config_version,
    scores_by_ticker,
)


BAR = pd.Timestamp("2026-08-03T20:00:00Z")


def _args(**updates: object) -> argparse.Namespace:
    payload: dict[str, object] = {
        "mode": "equity",
        "matrix": "/somewhere/meta_ranker_matrix.parquet",
        "top_k": 10,
        "liquidity_floor": 0.6,
        "combo_floor": 0.90,
        "quality_floor": 0.4,
        "target_notional": 5000.0,
        "take_profit": 0.30,
        "scale_frac": 0.16,
        "horizon_bars": 53,
        "grace_bars": None,
        "stop_loss": 0.39,
        "trail_stop": None,
        "roll_trading_days": 5,
    }
    payload.update(updates)
    return argparse.Namespace(**payload)


def _result(scored: pd.DataFrame, ranked: pd.DataFrame) -> MetaRankingResult:
    return MetaRankingResult(
        ranked=ranked,
        scored_count=len(scored),
        eligible_count=len(ranked),
        decision_bar=BAR,
        scored=scored,
    )


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring context
# ---------------------------------------------------------------------------


def test_a_held_name_outside_the_top_k_still_has_a_score() -> None:
    """This is the whole reason the scored frame is exposed. Without it, the
    exit of a name that just dropped out of the ranking would be recorded with
    no score at all, even though the bar scored it perfectly well.
    """

    scored = _frame(
        [
            {"ticker": "AMD", "s_combo": 0.97, "s_quality": 0.6, "s_upside": 0.8},
            {"ticker": "MSFT", "s_combo": 0.41, "s_quality": 0.5, "s_upside": 0.3},
        ]
    )
    ranked = scored.iloc[:1]

    out = scores_by_ticker(_result(scored, ranked))

    assert "MSFT" in out
    assert out["MSFT"]["s_combo"] == pytest.approx(0.41)


def test_a_non_finite_score_is_omitted_not_coerced() -> None:
    scored = _frame(
        [{"ticker": "AMD", "s_combo": 0.97, "s_quality": float("nan")}]
    )

    out = scores_by_ticker(_result(scored, scored))

    assert "s_quality" not in out["AMD"]
    assert out["AMD"]["s_combo"] == pytest.approx(0.97)


def test_a_row_without_a_combo_score_is_not_offered_as_scored() -> None:
    scored = _frame([{"ticker": "AMD", "s_quality": 0.6}])

    assert scores_by_ticker(_result(scored, scored)) == {}


def test_tickers_are_canonicalised() -> None:
    scored = _frame([{"ticker": "amd", "s_combo": 0.97}])

    assert "AMD" in scores_by_ticker(_result(scored, scored))


def test_an_empty_scored_frame_yields_no_scores() -> None:
    empty = pd.DataFrame()

    assert scores_by_ticker(_result(empty, empty)) == {}


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def test_the_config_version_is_stable_for_identical_settings() -> None:
    assert runner_config_version(_args()) == runner_config_version(_args())


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_k", 5),
        ("combo_floor", 0.95),
        ("quality_floor", 0.5),
        ("target_notional", 2500.0),
        ("take_profit", 0.25),
        ("scale_frac", 0.20),
        ("horizon_bars", 40),
        ("grace_bars", 3),
        ("stop_loss", 0.25),
        ("trail_stop", 0.15),
        ("liquidity_floor", 0.7),
        ("mode", "options"),
        ("roll_trading_days", 10),
    ],
)
def test_every_decision_relevant_setting_changes_the_config_version(
    field: str, value: object
) -> None:
    """If a setting can change what we trade, it must change the recorded
    version. A silent change would make two different policies indistinguishable
    in the audit trail.
    """

    assert runner_config_version(_args()) != runner_config_version(_args(**{field: value}))


def test_a_non_decision_setting_does_not_change_the_config_version() -> None:
    assert runner_config_version(_args()) == runner_config_version(
        _args(matrix="/elsewhere/other.parquet", submit=True)
    )


def test_the_feature_version_names_the_matrix() -> None:
    assert feature_version("/somewhere/meta_ranker_matrix.parquet") == (
        "meta-matrix@meta_ranker_matrix.parquet"
    )


# ---------------------------------------------------------------------------
# Intent config
# ---------------------------------------------------------------------------


def test_the_intent_config_carries_the_runner_versions() -> None:
    config = intent_config(_args())

    assert config.config_version == runner_config_version(_args())
    assert config.feature_version == "meta-matrix@meta_ranker_matrix.parquet"
    assert config.expected_holding_period == "53x4h"
    assert config.requested_notional == Decimal("5000")


def test_equity_mode_prefers_equity_only() -> None:
    assert intent_config(_args(mode="equity")).instrument_preferences == (
        InstrumentFamily.EQUITY,
    )


def test_options_mode_prefers_options_then_shares() -> None:
    """Order matters: the selector takes the first permitted family that yields
    a candidate, and the existing runner routes to shares only as a fallback.
    """

    assert intent_config(_args(mode="options")).instrument_preferences == (
        InstrumentFamily.SINGLE_OPTION,
        InstrumentFamily.EQUITY,
    )


def test_held_names_are_excluded_from_ranking_entries() -> None:
    """build_trade_intents skips held names; the config is how it learns them."""

    config = intent_config(_args(), held_tickers=frozenset({"amd"}))

    assert config.held_tickers == frozenset({"AMD"})
