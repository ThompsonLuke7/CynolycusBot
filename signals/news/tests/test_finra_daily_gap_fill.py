"""The nightly FINRA stage must fill days it missed, not only "yesterday".

Seven sessions between 2026-08-18 and 2026-10-08 had no daily short-volume file:
the stage fetched one day per run and never looked back, so a missed or late
run, or a swallowed network error, left a permanent hole.
"""

from __future__ import annotations

import pandas as pd
import pytest

from signals.news.sources import fill_missing_finra_daily_files

pytestmark = pytest.mark.safe


def _frame(day: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [day.normalize()],
            "ticker": ["AAA"],
            "short_volume": [1.0],
            "short_exempt_volume": [0.0],
            "total_volume": [2.0],
            "market": ["Q"],
        }
    )


class _Fetch:
    def __init__(self, *, empty: set[str] = frozenset(), misdated: set[str] = frozenset()) -> None:
        self.requested: list[str] = []
        self._empty = set(empty)
        self._misdated = set(misdated)

    def __call__(self, day: pd.Timestamp) -> pd.DataFrame:
        key = day.strftime("%Y%m%d")
        self.requested.append(key)
        if key in self._empty:
            return _frame(day).iloc[0:0]
        if key in self._misdated:
            return _frame(day - pd.Timedelta(days=1))
        return _frame(day)


def test_fills_every_missing_session_and_leaves_existing_files_alone(tmp_path):
    # Thu 10-08 back five sessions: 10-08, 10-07, 10-06, 10-05, 10-02 (weekend skipped).
    for key in ("20261002", "20261006", "20261007"):
        (tmp_path / f"{key}.parquet").write_bytes(b"existing")
    fetch = _Fetch()

    result = fill_missing_finra_daily_files(tmp_path, end="2026-10-08", lookback_sessions=5, fetch=fetch)

    assert result == {"written": ["20261005", "20261008"], "empty": []}
    assert fetch.requested == ["20261005", "20261008"]
    assert (tmp_path / "20261002.parquet").read_bytes() == b"existing"
    written = pd.read_parquet(tmp_path / "20261005.parquet")
    assert written["date"].dt.strftime("%Y%m%d").tolist() == ["20261005"]
    assert not list(tmp_path.glob("*.tmp"))


def test_non_trading_days_are_never_requested(tmp_path):
    # Tue 2026-09-08 back three sessions crosses Labor Day (Mon 09-07) and a weekend.
    fetch = _Fetch()

    result = fill_missing_finra_daily_files(tmp_path, end="2026-09-08", lookback_sessions=3, fetch=fetch)

    assert result["written"] == ["20260903", "20260904", "20260908"]
    assert fetch.requested == ["20260903", "20260904", "20260908"]


def test_window_ending_on_a_weekend_rolls_back_to_the_last_session(tmp_path):
    fetch = _Fetch()

    result = fill_missing_finra_daily_files(tmp_path, end=pd.Timestamp("2026-10-11", tz="UTC"), lookback_sessions=1, fetch=fetch)

    assert result["written"] == ["20261009"]


def test_empty_day_is_reported_not_written_so_the_next_run_retries_it(tmp_path):
    fetch = _Fetch(empty={"20261007"})

    result = fill_missing_finra_daily_files(tmp_path, end="2026-10-08", lookback_sessions=2, fetch=fetch)

    assert result == {"written": ["20261008"], "empty": ["20261007"]}
    assert not (tmp_path / "20261007.parquet").exists()

    retry = fill_missing_finra_daily_files(tmp_path, end="2026-10-08", lookback_sessions=2, fetch=_Fetch())
    assert retry == {"written": ["20261007"], "empty": []}


def test_misdated_file_fails_fast_instead_of_being_stored(tmp_path):
    fetch = _Fetch(misdated={"20261008"})

    with pytest.raises(ValueError, match="20261008"):
        fill_missing_finra_daily_files(tmp_path, end="2026-10-08", lookback_sessions=1, fetch=fetch)

    assert not list(tmp_path.iterdir())


def test_rejects_a_non_positive_window(tmp_path):
    with pytest.raises(ValueError, match="lookback_sessions"):
        fill_missing_finra_daily_files(tmp_path, end="2026-10-08", lookback_sessions=0, fetch=_Fetch())
