"""Normalization tests for the Phase-0 options-experiment signal spine.

Covers scripts/build_options_experiment_spine.py: schema completeness,
tz-awareness of the three timestamp columns, the {-1, +1} direction domain,
exit-not-before-entry, and the provenance domain. See
docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md
(Phase 0) for the schema this guards.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_options_experiment_spine import (
    SPINE_COLUMNS,
    VALID_CADENCE,
    VALID_PROVENANCE,
    build_spine,
    validate_spine,
)


@pytest.fixture(scope="module")
def spine() -> pd.DataFrame:
    """Build the real spine once and share it across tests in this module.

    This is an integration check: it reads the actual source files under
    backtests/, Data/, and strategies/multi_ticker_swing/backtest/, so it
    also acts as a regression guard on those sources' schemas.
    """
    return build_spine()


def _minimal_valid_row(**overrides) -> dict:
    row = {
        "module": "momentum_expansion",
        "ticker": "AAPL",
        "signal_ts": pd.Timestamp("2025-06-01 14:00", tz="UTC"),
        "entry_ts": pd.Timestamp("2025-06-01 14:00", tz="UTC"),
        "exit_ts": pd.Timestamp("2025-06-02 14:00", tz="UTC"),
        "direction": 1,
        "entry_px_underlying": 100.0,
        "exit_px_underlying": 105.0,
        "exit_reason": "tp",
        "bars_held": 6,
        "atr_at_entry": 2.0,
        "tp_price": 106.0,
        "sl_price": 96.0,
        "score": 0.5,
        "cadence": "4h",
        "source_file": "synthetic/for_test.parquet",
        "provenance": "backtest_frozen_test",
    }
    row.update(overrides)
    return row


def _frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).reindex(columns=SPINE_COLUMNS)


# ---------------------------------------------------------------------------
# Real-spine integration checks
# ---------------------------------------------------------------------------

def test_spine_is_nonempty_and_covers_expected_modules(spine: pd.DataFrame) -> None:
    assert len(spine) > 0
    expected = {
        "momentum_expansion", "multi_ticker_swing_htf", "meta_ranker",
        "dealer_ranker", "multi_ticker_swing",
    }
    assert expected <= set(spine["module"].unique())


def test_spine_schema_completeness(spine: pd.DataFrame) -> None:
    assert list(spine.columns) == SPINE_COLUMNS


def test_spine_timestamps_are_tz_aware(spine: pd.DataFrame) -> None:
    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        non_null = spine[c].dropna()
        assert isinstance(non_null.dtype, pd.DatetimeTZDtype), f"{c} is not tz-aware"
        assert str(non_null.dtype.tz) == "UTC", f"{c} is not UTC"


def test_spine_direction_domain(spine: pd.DataFrame) -> None:
    observed = set(spine["direction"].dropna().unique().tolist())
    assert observed <= {1, -1}


def test_spine_no_exit_before_entry(spine: pd.DataFrame) -> None:
    both = spine["entry_ts"].notna() & spine["exit_ts"].notna()
    assert (spine.loc[both, "exit_ts"] >= spine.loc[both, "entry_ts"]).all()


def test_spine_provenance_domain(spine: pd.DataFrame) -> None:
    assert set(spine["provenance"].unique().tolist()) <= VALID_PROVENANCE


def test_spine_cadence_domain(spine: pd.DataFrame) -> None:
    assert set(spine["cadence"].unique().tolist()) <= VALID_CADENCE


def test_spine_passes_its_own_validator(spine: pd.DataFrame) -> None:
    validate_spine(spine)  # must not raise


def test_ev_experiments_subset_rows_are_not_double_counted(spine: pd.DataFrame) -> None:
    """htf_final/momentum_final are verified subsets of source 1 (see script notes) and
    must not appear as extra rows: total htf/momentum row counts should match the raw
    equal_notional_trades.parquet split, not that split plus the ev_experiments subset."""
    n_htf = int((spine["module"] == "multi_ticker_swing_htf").sum())
    n_mom = int((spine["module"] == "momentum_expansion").sum())
    assert n_htf == 23173
    assert n_mom == 3876


# ---------------------------------------------------------------------------
# Unit checks on validate_spine() against synthetic, deliberately-broken frames
# ---------------------------------------------------------------------------

def test_validate_spine_accepts_minimal_valid_frame() -> None:
    df = _frame(_minimal_valid_row())
    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    df["direction"] = df["direction"].astype("Int64")
    validate_spine(df)  # must not raise


def test_validate_spine_rejects_bad_direction() -> None:
    df = _frame(_minimal_valid_row(direction=2))
    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    df["direction"] = df["direction"].astype("Int64")
    with pytest.raises(AssertionError, match="direction"):
        validate_spine(df)


def test_validate_spine_rejects_exit_before_entry() -> None:
    df = _frame(_minimal_valid_row(
        entry_ts=pd.Timestamp("2025-06-02 14:00", tz="UTC"),
        exit_ts=pd.Timestamp("2025-06-01 14:00", tz="UTC"),
    ))
    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    df["direction"] = df["direction"].astype("Int64")
    with pytest.raises(AssertionError, match="exit_ts < entry_ts"):
        validate_spine(df)


def test_validate_spine_allows_equal_entry_and_exit_ts() -> None:
    """dealer_ranker's horizon_sessions=1 rows legitimately have entry_ts == exit_ts
    (documented session-label limitation) -- the validator must accept equality."""
    ts = pd.Timestamp("2026-07-06 04:00", tz="UTC")
    df = _frame(_minimal_valid_row(entry_ts=ts, exit_ts=ts))
    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    df["direction"] = df["direction"].astype("Int64")
    validate_spine(df)  # must not raise


def test_validate_spine_rejects_bad_provenance() -> None:
    df = _frame(_minimal_valid_row(provenance="backtest_live_oops"))
    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    df["direction"] = df["direction"].astype("Int64")
    with pytest.raises(AssertionError, match="provenance"):
        validate_spine(df)


def test_validate_spine_rejects_bad_cadence() -> None:
    df = _frame(_minimal_valid_row(cadence="weekly"))
    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    df["direction"] = df["direction"].astype("Int64")
    with pytest.raises(AssertionError, match="cadence"):
        validate_spine(df)


def test_validate_spine_rejects_naive_timestamps() -> None:
    df = _frame(_minimal_valid_row())
    df["entry_ts"] = pd.to_datetime(df["entry_ts"]).dt.tz_localize(None)  # strip tz
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df["direction"] = df["direction"].astype("Int64")
    with pytest.raises(AssertionError, match="tz-aware"):
        validate_spine(df)


def test_validate_spine_rejects_missing_column() -> None:
    df = _frame(_minimal_valid_row()).drop(columns=["score"])
    with pytest.raises(AssertionError, match="missing required columns"):
        validate_spine(df)
