"""Trailing-correlation peer groups: causality, screening, and shape.

The screens here are not hygiene. Each one corresponds to a specific way this
repo has previously reached a wrong answer:

* the causal cut -> look-ahead;
* the corporate-action blanking -> WOLF/SBET split prints that inflated a
  retracted study 3-6x;
* the liquidity floor -> 87% of the leader/follower events were in names under
  $1M/day and those names carried the entire measured effect.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from signals.peer_structure import correlation_groups as cg
from signals.peer_structure.correlation_groups import (
    PeerGroupConfig,
    build_peer_groups,
    load_screened_returns,
)


SESSIONS = 90


def write_bars(
    root,
    ticker: str,
    *,
    returns: np.ndarray,
    price: float = 100.0,
    volume: float = 2_000_000.0,
    split_at: int | None = None,
) -> None:
    index = pd.bdate_range("2026-01-05", periods=len(returns))
    close = price * np.cumprod(1.0 + returns)
    if split_at is not None:
        # A 1-for-5 reverse split in an UNADJUSTED cache: price jumps 5x with
        # no economic move. This is the shape that must not reach a correlation.
        close[split_at:] *= 5.0
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(index, utc=True),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(returns), volume),
        }
    )
    frame.to_parquet(root / f"{ticker}.parquet")


@pytest.fixture
def bars(tmp_path, monkeypatch):
    root = tmp_path / "1d"
    root.mkdir()
    monkeypatch.setattr(cg, "BARS_1D", root)
    return root


def correlated_block(rng, n: int, sessions: int, *, rho: float) -> np.ndarray:
    common = rng.normal(0.0, 0.01, sessions)
    return np.array(
        [rho * common + (1 - rho) * rng.normal(0.0, 0.01, sessions) for _ in range(n)]
    )


def test_bars_after_the_as_of_session_are_never_read(bars) -> None:
    rng = np.random.default_rng(1)
    write_bars(bars, "AAA", returns=rng.normal(0.0, 0.01, SESSIONS))
    config = PeerGroupConfig(min_observations=10, lookback_sessions=30)

    index = pd.bdate_range("2026-01-05", periods=SESSIONS)
    cutoff = index[50]
    frame = load_screened_returns(["AAA"], as_of_session=cutoff, config=config)
    assert not frame.empty
    assert frame.index.max() <= cutoff


def test_illiquid_names_are_screened_out(bars) -> None:
    rng = np.random.default_rng(2)
    write_bars(bars, "LIQ", returns=rng.normal(0.0, 0.01, SESSIONS), volume=2_000_000)
    # $100 x 1,000 shares = $100k/day, far under the $10M floor.
    write_bars(bars, "THIN", returns=rng.normal(0.0, 0.01, SESSIONS), volume=1_000)
    config = PeerGroupConfig(min_observations=10, lookback_sessions=60)

    frame = load_screened_returns(
        ["LIQ", "THIN"], as_of_session=pd.Timestamp("2026-05-01"), config=config
    )
    assert "LIQ" in frame.columns
    assert "THIN" not in frame.columns


def test_penny_names_are_screened_out(bars) -> None:
    rng = np.random.default_rng(3)
    write_bars(
        bars, "PENNY", returns=rng.normal(0.0, 0.01, SESSIONS), price=2.0, volume=50_000_000
    )
    config = PeerGroupConfig(min_observations=10, lookback_sessions=60)
    frame = load_screened_returns(
        ["PENNY"], as_of_session=pd.Timestamp("2026-05-01"), config=config
    )
    assert frame.empty or "PENNY" not in frame.columns


def test_a_split_print_is_blanked_not_correlated(bars) -> None:
    """Left in, one 5x gap dominates every correlation the name appears in."""

    rng = np.random.default_rng(4)
    write_bars(bars, "SPLIT", returns=rng.normal(0.0, 0.01, SESSIONS), split_at=40)
    config = PeerGroupConfig(min_observations=10, lookback_sessions=80)
    frame = load_screened_returns(
        ["SPLIT"], as_of_session=pd.Timestamp("2026-05-01"), config=config
    )
    series = frame["SPLIT"].dropna()
    # The 400% bar must not survive into the return series.
    assert series.max() < 0.5, f"corporate-action gap reached the returns: {series.max()}"


def test_groups_recover_a_planted_correlation_block(bars) -> None:
    rng = np.random.default_rng(5)
    block_a = correlated_block(rng, 6, SESSIONS, rho=0.9)
    block_b = correlated_block(rng, 6, SESSIONS, rho=0.9)
    for i, series in enumerate(block_a):
        write_bars(bars, f"A{i}", returns=series)
    for i, series in enumerate(block_b):
        write_bars(bars, f"B{i}", returns=series)

    config = PeerGroupConfig(
        min_observations=10, lookback_sessions=80, n_groups=2, min_members=3
    )
    tickers = [f"A{i}" for i in range(6)] + [f"B{i}" for i in range(6)]
    frame = load_screened_returns(
        tickers, as_of_session=pd.Timestamp("2026-05-01"), config=config
    )
    groups = build_peer_groups(
        frame,
        as_of_session=pd.Timestamp("2026-05-01"),
        config=config,
        universe_version="test@1",
    )
    assert groups, "no groups built from a clean two-block panel"
    # Each planted block should land predominantly in one group.
    for group in groups:
        prefixes = {member[0] for member in group.members}
        assert len(prefixes) == 1, f"{group.group_id} mixed blocks: {group.members}"


def test_group_states_are_causal_and_stable(bars) -> None:
    rng = np.random.default_rng(6)
    for i in range(8):
        write_bars(bars, f"T{i}", returns=rng.normal(0.0, 0.01, SESSIONS))
    config = PeerGroupConfig(
        min_observations=10, lookback_sessions=80, n_groups=2, min_members=3
    )
    as_of = pd.Timestamp("2026-05-01")
    frame = load_screened_returns([f"T{i}" for i in range(8)], as_of_session=as_of, config=config)
    groups = build_peer_groups(
        frame, as_of_session=as_of, config=config, universe_version="test@1"
    )
    if not groups:
        pytest.skip("random panel produced no group above min_members")

    for group in groups:
        assert group.as_of <= group.available_at
        assert group.valid_until > group.available_at
        assert group.source_window_end <= group.as_of
        assert tuple(sorted(group.members)) == group.members
        assert group.method == "trailing_correlation_kmeans"
        assert group.lookback_sessions == config.lookback_sessions

    # Deterministic: the same panel produces the same ids and membership.
    again = build_peer_groups(
        frame, as_of_session=as_of, config=config, universe_version="test@1"
    )
    assert [g.group_id for g in again] == [g.group_id for g in groups]
    assert [g.members for g in again] == [g.members for g in groups]


def test_groups_expire(bars) -> None:
    """A stale grouping silently applied to a new regime is the failure mode."""

    rng = np.random.default_rng(7)
    for i in range(8):
        write_bars(bars, f"E{i}", returns=rng.normal(0.0, 0.01, SESSIONS))
    config = PeerGroupConfig(
        min_observations=10,
        lookback_sessions=80,
        n_groups=2,
        min_members=3,
        valid_for=timedelta(days=3),
    )
    as_of = pd.Timestamp("2026-05-01")
    frame = load_screened_returns([f"E{i}" for i in range(8)], as_of_session=as_of, config=config)
    groups = build_peer_groups(
        frame, as_of_session=as_of, config=config, universe_version="test@1"
    )
    if not groups:
        pytest.skip("random panel produced no group above min_members")
    assert groups[0].valid_until - groups[0].available_at == timedelta(days=3)
