"""Per-ticker readiness fallback.

The behaviour under test is the one that cost three consecutive sessions
(2026-07-28 .. 07-30): the shared 4H bar catch-up succeeded and every tradeable
ticker had current data, but a later stage of the same script was killed, no
global stamp was written, and every 4H entry was blocked anyway.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.live_readiness import (
    filter_entry_orders_for_readiness,
    ticker_data_status,
    underlying_for_symbol,
)

NOW = datetime(2026, 7, 30, 18, 20, tzinfo=timezone.utc)  # Thu 14:20 ET, a 4H run


def current_bar() -> datetime:
    """A bar timestamp that is current whenever the suite happens to run.

    `ticker_data_status` judges freshness against the last COMPLETED trading
    session, not an hours tolerance, and `filter_entry_orders_for_readiness`
    has no `now` hook to pin it through — so a hard-coded "fresh" date silently
    becomes stale as real time moves past it. Four tests here were written with
    bars dated 2026-07-30 and started failing once that stopped being the prior
    session. Tests that mean "this ticker's data is current" must say so
    relative to now; tests that mean "stale" keep their fixed past dates.
    """
    return datetime.now(timezone.utc)


@pytest.fixture()
def bars(tmp_path):
    """A bar cache directory plus a helper to plant a ticker's last bar."""
    directory = tmp_path / "4h"
    directory.mkdir()

    def write(ticker: str, last_bar: datetime, rows: int = 5):
        index = pd.date_range(end=last_bar, periods=rows, freq="4h", tz="UTC")
        frame = pd.DataFrame({"close": range(rows)}, index=index)
        frame.index.name = "timestamp"
        frame.to_parquet(directory / f"{ticker}.parquet")

    return directory, write


@pytest.fixture()
def stale_stamp(monkeypatch, tmp_path):
    """A global stamp that exists but predates the last completed session."""
    monkeypatch.delenv("CYNOLYCUS_READINESS_REQUIRED", raising=False)
    stamp = tmp_path / "latest_success.json"
    stamp.write_text(json.dumps({
        "job": "nightly_data_readiness",
        "status": "success",
        "completed_at_utc": datetime(2026, 7, 28, 2, 5, tzinfo=timezone.utc).isoformat(),
    }))
    monkeypatch.setattr("core.live_readiness.DEFAULT_READINESS_PATH", stamp)
    return stamp


# --------------------------------------------------------------------------
# underlying_for_symbol
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("ZS260821C00150000", "ZS"),        # the real 2026-07-30 swing entry
        ("CALM260821P00090000", "CALM"),
        ("SPY260730P00736000", "SPY"),
        ("AAPL", "AAPL"),                   # equity order
        ("aapl", "AAPL"),
    ],
)
def test_underlying_for_symbol(symbol, expected):
    assert underlying_for_symbol(symbol) == expected


# --------------------------------------------------------------------------
# ticker_data_status
# --------------------------------------------------------------------------

def test_ticker_with_current_bars_is_ready(bars):
    directory, write = bars
    write("ULCC", datetime(2026, 7, 30, 14, 0, tzinfo=timezone.utc))

    ok, reason = ticker_data_status("ULCC", bars_dir=directory, now=NOW)

    assert ok, reason


def test_ticker_whose_bars_stop_before_the_last_session_is_blocked(bars):
    """MRO's real cache ends 2024-11-21; it must never authorize an entry."""
    directory, write = bars
    write("MRO", datetime(2024, 11, 21, 19, 0, tzinfo=timezone.utc))

    ok, reason = ticker_data_status("MRO", bars_dir=directory, now=NOW)

    assert not ok
    assert "before the last completed session" in reason


def test_bars_covering_only_the_prior_session_are_enough(bars):
    """Pre-open: today has no bars yet, and that must not block the open."""
    directory, write = bars
    write("AAPL", datetime(2026, 7, 29, 19, 0, tzinfo=timezone.utc))

    ok, reason = ticker_data_status(
        "AAPL", bars_dir=directory, now=datetime(2026, 7, 30, 13, 35, tzinfo=timezone.utc)
    )

    assert ok, reason


def test_missing_empty_and_unreadable_files_fail_closed(bars, tmp_path):
    directory, _ = bars
    (directory / "EMPTY.parquet").write_bytes(b"")
    pd.DataFrame({"close": []}, index=pd.DatetimeIndex([], tz="UTC")).to_parquet(
        directory / "NOROWS.parquet"
    )

    for ticker in ("NEVERSEEN", "EMPTY", "NOROWS", ""):
        ok, _reason = ticker_data_status(ticker, bars_dir=directory, now=NOW)
        assert not ok, f"{ticker!r} must fail closed"


def test_status_reflects_a_refreshed_file_not_a_cached_answer(bars):
    """A live process must not answer from a cache older than the data on disk."""
    directory, write = bars
    write("NVDA", datetime(2024, 1, 2, 19, 0, tzinfo=timezone.utc))
    assert not ticker_data_status("NVDA", bars_dir=directory, now=NOW)[0]

    write("NVDA", datetime(2026, 7, 30, 14, 0, tzinfo=timezone.utc))

    ok, reason = ticker_data_status("NVDA", bars_dir=directory, now=NOW)
    assert ok, f"catch-up refreshed the file but the gate still said: {reason}"


# --------------------------------------------------------------------------
# filter_entry_orders_for_readiness
# --------------------------------------------------------------------------

def test_stale_stamp_but_current_bars_now_authorizes_entries(bars, stale_stamp):
    """The 2026-07-30 case: stage 1 finished, stages 4-5 were killed."""
    directory, write = bars
    for ticker in ("ULCC", "XRAY", "FRSH"):
        write(ticker, current_bar())
    plan = [
        ("ULCC", "buy", 100, "entry", "equity"),
        ("XRAY", "buy", 100, "entry", "equity"),
        ("FRSH", "buy", 100, "entry", "equity"),
    ]

    kept, skipped, reason = filter_entry_orders_for_readiness(
        plan, bars_dir=directory, max_age_hours=999
    )

    assert kept == plan
    assert skipped == set()
    assert "authorized 3/3" in reason


def test_only_the_ticker_with_stale_data_is_blocked(bars, stale_stamp):
    directory, write = bars
    write("ULCC", current_bar())
    write("MRO", datetime(2024, 11, 21, 19, 0, tzinfo=timezone.utc))
    plan = [
        ("ULCC", "buy", 100, "entry", "equity"),
        ("MRO", "buy", 100, "entry", "equity"),
    ]
    managed = {
        "ULCC": {"route": "equity", "symbol": "ULCC", "runs_held": 0},
        "MRO": {"route": "equity", "symbol": "MRO", "runs_held": 0},
    }

    kept, skipped, reason = filter_entry_orders_for_readiness(
        plan, new_managed=managed, bars_dir=directory, max_age_hours=999
    )

    assert kept == [("ULCC", "buy", 100, "entry", "equity")]
    assert skipped == {"MRO"}
    assert "authorized 1/2" in reason
    assert "MRO" in managed_removed(managed), "the blocked entry must leave managed state"


def managed_removed(managed: dict) -> set[str]:
    """Tickers no longer present after the gate pruned them."""
    return {"MRO", "ULCC"} - set(managed)


def test_sells_are_never_blocked_by_stale_data(bars, stale_stamp):
    """Risk-reducing exits must survive a gate that blocks entries."""
    directory, write = bars
    write("MRO", datetime(2024, 11, 21, 19, 0, tzinfo=timezone.utc))
    plan = [
        ("MRO", "sell", 100, "exit", "equity"),
        ("MRO", "buy", 100, "entry", "equity"),
    ]

    kept, skipped, _reason = filter_entry_orders_for_readiness(
        plan, bars_dir=directory, max_age_hours=999
    )

    assert ("MRO", "sell", 100, "exit", "equity") in kept
    assert skipped == {"MRO"}


def test_option_orders_resolve_to_their_underlying(bars, stale_stamp):
    directory, write = bars
    write("ZS", current_bar())
    plan = [("ZS260821C00150000", "buy", 7, "entry", "option")]

    kept, skipped, _reason = filter_entry_orders_for_readiness(
        plan, bars_dir=directory, max_age_hours=999
    )

    assert kept == plan, "the underlying's bars are current, so the call is tradeable"
    assert skipped == set()


def test_explicit_symbol_tickers_win_over_occ_parsing(bars, stale_stamp):
    """Callers that know the ticker should not depend on symbol-shape inference."""
    directory, write = bars
    write("BRK.B", current_bar())
    plan = [("BRKB260821C00150000", "buy", 1, "entry", "option")]

    kept, skipped, _ = filter_entry_orders_for_readiness(
        plan,
        symbol_tickers={"BRKB260821C00150000": "BRK.B"},
        bars_dir=directory,
        max_age_hours=999,
    )

    assert kept == plan
    assert skipped == set()


def test_per_ticker_fallback_can_be_switched_off_for_matrix_backed_modules(bars, stale_stamp):
    """Meta Ranker scores a prebuilt matrix; fresh bars say nothing about it."""
    directory, write = bars
    write("ULCC", datetime(2026, 7, 30, 14, 0, tzinfo=timezone.utc))
    plan = [("ULCC", "buy", 100, "entry", "equity")]

    kept, skipped, reason = filter_entry_orders_for_readiness(
        plan, bars_dir=directory, per_ticker_fallback=False, max_age_hours=999
    )

    assert kept == []
    assert skipped == {"ULCC"}
    assert "predates latest completed trading session" in reason


def test_fresh_stamp_short_circuits_without_touching_bars(stale_stamp, tmp_path, monkeypatch):
    """The fast path must not depend on the bar cache existing at all."""
    monkeypatch.setattr(
        "core.live_readiness.DEFAULT_READINESS_PATH", tmp_path / "fresh.json"
    )
    (tmp_path / "fresh.json").write_text(json.dumps({
        "job": "nightly_data_readiness",
        "status": "success",
        "completed_at_utc": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    }))
    plan = [("ANYTHING", "buy", 100, "entry", "equity")]

    kept, skipped, _reason = filter_entry_orders_for_readiness(
        plan, bars_dir=tmp_path / "does-not-exist"
    )

    assert kept == plan
    assert skipped == set()
