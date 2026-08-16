"""Resumability of the 4H feature build.

Guards the 2026-07-30 failure: the nightly job rebuilt all ~2,900 tickers under
``--force``, so two consecutive kills inside this stage each restarted from zero
and the 4H entry gate stayed shut for a third session.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from strategies.momentum_expansion.features import feature_matrix_4h as fm


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    feats = tmp_path / "features_4h"
    bars = tmp_path / "bars_4h"
    feats.mkdir()
    bars.mkdir()
    monkeypatch.setattr(fm, "PROCESSED_FEAT_DIR", feats)
    monkeypatch.setattr(fm, "RAW_4H_DIR", bars)
    return feats, bars


def _touch(path, mtime: float):
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))


def test_features_newer_than_bars_are_reused(dirs):
    feats, bars = dirs
    _touch(bars / "AAPL.parquet", 1000)
    _touch(feats / "AAPL_features.parquet", 2000)

    assert fm._per_ticker_is_current(feats / "AAPL_features.parquet", "AAPL")


def test_features_older_than_bars_are_rebuilt(dirs):
    """The catch-up appended new bars; those features no longer reflect them."""
    feats, bars = dirs
    _touch(feats / "SPY_features.parquet", 1000)
    _touch(bars / "SPY.parquet", 2000)

    assert not fm._per_ticker_is_current(feats / "SPY_features.parquet", "SPY")


def test_equal_mtimes_count_as_current(dirs):
    feats, bars = dirs
    _touch(bars / "MSFT.parquet", 1500)
    _touch(feats / "MSFT_features.parquet", 1500)

    assert fm._per_ticker_is_current(feats / "MSFT_features.parquet", "MSFT")


def test_missing_features_are_not_current(dirs):
    feats, bars = dirs
    _touch(bars / "NVDA.parquet", 1000)

    assert not fm._per_ticker_is_current(feats / "NVDA_features.parquet", "NVDA")


def test_ticker_without_bars_keeps_whatever_features_exist(dirs):
    """A delisted name has nothing to become stale against."""
    feats, _bars = dirs
    _touch(feats / "DEAD_features.parquet", 1000)

    assert fm._per_ticker_is_current(feats / "DEAD_features.parquet", "DEAD")


def test_a_killed_run_resumes_instead_of_restarting(dirs, monkeypatch):
    """The real scenario: 3 of 4 tickers built, then the process was killed."""
    feats, bars = dirs
    for ticker in ("AAA", "BBB", "CCC", "DDD"):
        _touch(bars / f"{ticker}.parquet", 1000)
    # The interrupted run got three of them done.
    for ticker in ("AAA", "BBB", "CCC"):
        frame = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.DatetimeIndex(["2026-07-29", "2026-07-30"], tz="UTC", name="timestamp"),
        )
        frame.to_parquet(feats / f"{ticker}_features.parquet")
        os.utime(feats / f"{ticker}_features.parquet", (2000, 2000))

    rebuilt: list[str] = []

    def fake_build(*, ticker, **_kwargs):
        rebuilt.append(ticker)
        return pd.DataFrame(
            {"close": [3.0]},
            index=pd.DatetimeIndex(["2026-07-30"], tz="UTC", name="timestamp"),
        )

    monkeypatch.setattr(fm, "build_ticker_features_4h", fake_build)
    monkeypatch.setattr(fm, "load_4h", lambda t: pd.DataFrame({"close": [1.0]}))
    monkeypatch.setattr(fm, "load_1d", lambda t: None)
    monkeypatch.setattr(fm, "_load_context_4h", lambda: None)
    monkeypatch.setattr(fm, "load_candidate_metadata", lambda: pd.DataFrame())
    monkeypatch.setattr(fm, "_build_metadata_encodings", lambda meta: ({}, {}, {}))
    monkeypatch.setattr(fm, "_metadata_for_ticker", lambda **_k: {})
    monkeypatch.setattr(fm, "_add_cross_sectional_features", lambda df: df)
    monkeypatch.setattr(fm, "_add_earnings_features_4h", lambda df: df)

    fm.build_all_features_4h(
        tickers=["AAA", "BBB", "CCC", "DDD"],
        out_path=feats.parent / "combined.parquet",
        refresh_stale=True,
    )

    assert rebuilt == ["DDD"], "only the unfinished ticker should be rebuilt"


def test_force_still_rebuilds_everything(dirs, monkeypatch):
    """--force must stay usable after feature-code changes, which mtimes miss."""
    feats, bars = dirs
    for ticker in ("AAA", "BBB"):
        _touch(bars / f"{ticker}.parquet", 1000)
        _touch(feats / f"{ticker}_features.parquet", 2000)

    rebuilt: list[str] = []

    def fake_build(*, ticker, **_kwargs):
        rebuilt.append(ticker)
        return pd.DataFrame(
            {"close": [3.0]},
            index=pd.DatetimeIndex(["2026-07-30"], tz="UTC", name="timestamp"),
        )

    monkeypatch.setattr(fm, "build_ticker_features_4h", fake_build)
    monkeypatch.setattr(fm, "load_4h", lambda t: pd.DataFrame({"close": [1.0]}))
    monkeypatch.setattr(fm, "load_1d", lambda t: None)
    monkeypatch.setattr(fm, "_load_context_4h", lambda: None)
    monkeypatch.setattr(fm, "load_candidate_metadata", lambda: pd.DataFrame())
    monkeypatch.setattr(fm, "_build_metadata_encodings", lambda meta: ({}, {}, {}))
    monkeypatch.setattr(fm, "_metadata_for_ticker", lambda **_k: {})
    monkeypatch.setattr(fm, "_add_cross_sectional_features", lambda df: df)
    monkeypatch.setattr(fm, "_add_earnings_features_4h", lambda df: df)

    fm.build_all_features_4h(
        tickers=["AAA", "BBB"],
        out_path=feats.parent / "combined.parquet",
        force=True,
    )

    assert sorted(rebuilt) == ["AAA", "BBB"]
