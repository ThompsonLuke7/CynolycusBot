"""Earnings features are resolved on deduplicated (date, ticker) pairs.

Guards the 2026-07-31 failure: the combined 4H matrix is ~8.5 GB in RAM, and the
earnings step used to run over every row — ``combined.reset_index()`` here plus
``add_earnings_features``'s own ``out = df.copy()`` — so the tail of the build
held four whole-frame copies, exhausted all 24 GB of swap and livelocked the VM,
leaving the readiness stamp stale and the 4H entry gate shut.

The optimisation is only safe because these five features are a pure function of
(calendar date, ticker): a 4H matrix carries ~6 bars per ticker-day, so the
deduplicated result must broadcast back identically. These tests pin that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.events.earnings_calendar import add_earnings_features
from strategies.momentum_expansion.features import feature_matrix_4h as fm

ADDED = [
    "days_to_earnings",
    "days_since_earnings",
    "is_pre_earnings_3d",
    "is_post_earnings_3d",
    "earnings_in_fwd_window",
]


def _matrix(tickers=("AAA", "BBB"), days=6, bars_per_day=6) -> pd.DataFrame:
    """A (timestamp, ticker) matrix with several 4H bars per calendar day."""
    stamps = []
    for d in pd.date_range("2026-03-02", periods=days, freq="B", tz="UTC"):
        stamps.extend(d + pd.to_timedelta(np.arange(bars_per_day) * 4 + 10, unit="h"))
    idx = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(stamps), list(tickers)], names=["timestamp", "ticker"]
    )
    rng = np.random.default_rng(0)
    return pd.DataFrame({"close": rng.normal(size=len(idx))}, index=idx).sort_index()


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2026-03-05", "2026-06-04", "2026-03-03"]),
        }
    )


def _row_wise(combined: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """The pre-optimisation path: resolve earnings for every row individually."""
    flat = combined.reset_index()
    flat["__date"] = (
        pd.to_datetime(flat["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    )
    flat = add_earnings_features(
        flat,
        date_col="__date",
        ticker_col="ticker",
        fwd_window_days=fm._EARNINGS_FWD_WINDOW_DAYS,
        calendar=calendar,
    )
    return flat.drop(columns="__date").set_index(["timestamp", "ticker"]).sort_index()


def test_dedup_broadcast_matches_row_wise(monkeypatch):
    calendar = _calendar()
    monkeypatch.setattr(
        "signals.events.earnings_calendar.load_earnings_calendar", lambda: calendar
    )
    combined = _matrix()

    expected = _row_wise(combined, calendar)
    actual = fm._add_earnings_features_4h(combined.copy())

    assert actual.index.equals(expected.index)
    for col in ADDED:
        pd.testing.assert_series_equal(actual[col], expected[col], check_names=False)


def test_every_bar_in_a_day_gets_that_day_value(monkeypatch):
    """The broadcast must not smear one bar's value across a ticker or a week."""
    calendar = _calendar()
    monkeypatch.setattr(
        "signals.events.earnings_calendar.load_earnings_calendar", lambda: calendar
    )
    out = fm._add_earnings_features_4h(_matrix().copy())

    dates = out.index.get_level_values("timestamp").normalize().tz_convert(None)
    grouped = out.groupby([dates, out.index.get_level_values("ticker")])

    # Constant within a (day, ticker) cell ...
    assert (grouped["days_to_earnings"].nunique(dropna=False) == 1).all()
    # ... and genuinely varying across days, not one value smeared everywhere.
    aaa = out.xs("AAA", level="ticker")["days_to_earnings"]
    assert aaa.groupby(aaa.index.normalize()).first().nunique() > 1

    # AAA reports 2026-03-05: the 03-04 bars are 1 day out, inside the 3-day flag.
    day_before = aaa[aaa.index.normalize() == pd.Timestamp("2026-03-04", tz="UTC")]
    assert (day_before == 1.0).all()
    assert (
        out.xs("AAA", level="ticker")["is_pre_earnings_3d"][
            day_before.index
        ] == 1.0
    ).all()


def test_original_columns_and_row_count_are_untouched(monkeypatch):
    monkeypatch.setattr(
        "signals.events.earnings_calendar.load_earnings_calendar", _calendar
    )
    combined = _matrix()
    before_close = combined["close"].copy()

    out = fm._add_earnings_features_4h(combined.copy())

    assert len(out) == len(combined)
    assert list(out.columns) == ["close"] + ADDED
    pd.testing.assert_series_equal(out["close"], before_close)


def test_empty_calendar_yields_nan_columns_not_a_crash(monkeypatch):
    empty = pd.DataFrame({"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})
    monkeypatch.setattr(
        "signals.events.earnings_calendar.load_earnings_calendar", lambda: empty
    )
    out = fm._add_earnings_features_4h(_matrix().copy())

    assert len(out) == len(_matrix())
    for col in ADDED:
        assert out[col].isna().all()
