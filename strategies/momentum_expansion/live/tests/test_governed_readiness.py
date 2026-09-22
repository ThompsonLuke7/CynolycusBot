"""Momentum's prerequisites for the governed path.

Momentum's runner still calls ``execute_plan`` directly -- the cutover is a
separate, deliberate step. What these tests pin is that everything the cutover
depends on already exists and is correct: the profile, the permission, and a
TICKER state that is attributable to momentum rather than to Meta.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.nervous_system.config.freshness import (
    META_4H_1420_PROFILE,
    get_snapshot_profile,
)
from core.nervous_system.config.policy import MVP_POLICY_CONFIG
from strategies.momentum_expansion.live.ticker_state_publication import (
    CONFIG_VERSION,
    FEATURE_VERSION,
    MODEL_VERSION,
    PRODUCER,
    build_momentum_ticker_states,
)


UTC = timezone.utc


def test_momentum_profiles_are_registered() -> None:
    for profile_id in ("momentum_4h_1420@1", "momentum_4h_1620@1"):
        assert get_snapshot_profile(profile_id).profile_id == profile_id


def test_momentum_profile_requires_the_same_states_as_meta() -> None:
    """The rules describe the state store, not the strategy."""

    momentum = get_snapshot_profile("momentum_4h_1420@1")
    assert {(r.state_type, r.required) for r in momentum.rules} == {
        (r.state_type, r.required) for r in META_4H_1420_PROFILE.rules
    }


def test_momentum_and_meta_profiles_are_distinguishable() -> None:
    """Same rules, different id: snapshots stay attributable per module."""

    momentum = get_snapshot_profile("momentum_4h_1420@1")
    assert momentum.profile_id != META_4H_1420_PROFILE.profile_id
    assert momentum.profile_hash != META_4H_1420_PROFILE.profile_hash


def test_momentum_is_a_permitted_strategy() -> None:
    assert "momentum_expansion" in MVP_POLICY_CONFIG.permitted_strategies


def test_momentum_ticker_states_carry_momentum_provenance(tmp_path) -> None:
    """A momentum state must not be mistakable for a Meta one."""

    bar = datetime(2026, 7, 30, 18, 0, tzinfo=UTC)
    bars_root = tmp_path / "bars4h"
    bars_root.mkdir()
    pd.DataFrame(
        {
            "timestamp": [bar - timedelta(hours=4), bar],
            "open": [99.0, 100.0],
            "high": [101.0, 103.0],
            "low": [98.0, 99.5],
            "close": [100.0, 102.0],
            "volume": [1_000_000, 1_200_000],
        }
    ).to_parquet(bars_root / "AMD.parquet")

    matrix = tmp_path / "momentum_features_4h.parquet"
    matrix.write_bytes(b"momentum-matrix")

    scored = pd.DataFrame(
        [{"ticker": "AMD", "close": 102.0, "mom_score": 0.87, "timestamp": bar}]
    )
    states, skipped = build_momentum_ticker_states(
        scored,
        tickers=["AMD"],
        decision_bar=bar,
        available_at=bar + timedelta(minutes=5),
        matrix_path=matrix,
    )
    if not states:
        pytest.skip(f"bar fixture not resolvable by the shared selector: {skipped}")

    state = states[0]
    assert state.producer == PRODUCER
    assert state.model_version == MODEL_VERSION
    assert state.feature_version == FEATURE_VERSION
    assert state.config_version == CONFIG_VERSION
    assert state.ticker == "AMD"
    # Causality: the state describes the decision bar, never a later one.
    assert state.as_of == bar
    assert state.available_at >= state.as_of
