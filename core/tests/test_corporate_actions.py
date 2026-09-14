"""Real corporate actions from the cache, plus the moves the guard must leave alone."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.corporate_actions import (
    ENTRY_LOOKBACK_SESSIONS,
    ORGANIC_MIN_VOLUME_RATIO,
    SUSPECT_RATIO,
    mask_windows,
    recent_corporate_action,
    suspect_sessions,
)


def _frame(closes, opens=None, volumes=None):
    n = len(closes)
    return pd.DataFrame({
        "open": opens if opens is not None else closes,
        "close": closes,
        "volume": volumes if volumes is not None else [1_000_000] * n,
    })


def _dated(frame, start="2026-06-01"):
    """Same bars on a business-day DatetimeIndex, as the 4H loader returns them."""
    idx = pd.bdate_range(start, periods=len(frame), tz="America/New_York") + pd.Timedelta(hours=10)
    return frame.set_index(idx.tz_convert("UTC"))


# --- suspect_sessions ------------------------------------------------------------

def test_flags_the_wolf_recapitalisation():
    """Real bars: WOLF 2025-09 Chapter 11 emergence. Price x15, volume /13."""
    closes = [1.475, 1.88, 2.345, 1.20, 21.51, 28.95]
    opens = [1.66, 1.67, 2.03, 1.545, 17.88, 28.96]
    vols = [482_267, 650_127, 1_731_644, 2_303_766, 180_927, 898_532]
    flags = suspect_sessions(_frame(closes, opens, vols))
    assert len(flags) == 1
    f = flags.iloc[0]
    assert f["idx"] == 4
    assert f["direction"] == "up"
    assert f["gap_ratio"] == pytest.approx(17.88 / 1.20, rel=1e-3)
    assert not f["organic"]


def test_flags_the_tenx_forward_split():
    """Real bars: TENX 10:1 forward split, 2026-08-10. The first version of the
    guard thresholded the arithmetic gap at +300%, which a down-gap can never
    reach -- it could not see this at all."""
    closes = [13.82, 13.05, 13.31, 13.44, 1.375, 1.62]
    opens = [14.51, 13.53, 12.99, 13.50, 1.79, 1.435]
    vols = [2_233_439, 2_240_262, 1_179_111, 2_062_600, 65_496_704, 32_899_842]
    flags = suspect_sessions(_frame(closes, opens, vols))
    assert len(flags) == 1
    f = flags.iloc[0]
    assert f["idx"] == 4
    assert f["direction"] == "down"
    # Volume x30 -- exactly what a real collapse looks like too. Down-gaps are
    # never classified organic: nothing local can tell the two apart.
    assert f["volume_ratio"] > ORGANIC_MIN_VOLUME_RATIO
    assert not f["organic"]


def test_a_real_explosive_up_move_is_flagged_but_organic():
    """4.4x on 30x volume: real participation. Flagged, and marked organic so
    neither research nor live screening removes it."""
    closes = [10.0] * 20 + [45.0]
    opens = [10.0] * 20 + [44.0]
    vols = [1_000_000] * 20 + [30_000_000]
    flags = suspect_sessions(_frame(closes, opens, vols))
    assert len(flags) == 1
    assert bool(flags.iloc[0]["organic"]) is True


def test_missing_volume_is_never_organic():
    frame = _frame([10.0, 10.0, 50.0], [10.0, 10.0, 50.0]).drop(columns="volume")
    flags = suspect_sessions(frame)
    assert len(flags) == 1 and not flags.iloc[0]["organic"]


def test_does_not_fire_on_hard_but_ordinary_moves():
    """+120% and -60% are brutal and real; both sit inside the 4x band."""
    assert suspect_sessions(_frame([10.0] * 20 + [22.0], [10.0] * 20 + [22.0])).empty
    assert suspect_sessions(_frame([10.0] * 20 + [4.0], [10.0] * 20 + [4.0])).empty


def test_does_not_fire_on_ordinary_bars():
    rng = np.random.default_rng(0)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 200)))
    assert suspect_sessions(_frame(closes)).empty


def test_threshold_is_symmetric_in_ratio():
    assert SUSPECT_RATIO == 4.0
    up = _frame([10.0, 10.0, 40.0], [10.0, 10.0, 40.0])
    down = _frame([10.0, 10.0, 2.5], [10.0, 10.0, 2.5])
    assert len(suspect_sessions(up)) == 1
    assert len(suspect_sessions(down)) == 1


# --- mask_windows ----------------------------------------------------------------

def test_window_mask_covers_every_entry_whose_hold_spans_the_gap():
    df = _frame([1.0] * 10 + [30.0] * 5)
    bad = mask_windows(df, hold=5)
    assert bad.iloc[6:11].all()
    assert not bad.iloc[:6].any()
    assert not bad.iloc[11:].any()


def test_window_mask_keeps_organic_moves_by_default():
    closes = [10.0] * 20 + [45.0] * 5
    vols = [1_000_000] * 20 + [30_000_000] * 5
    df = _frame(closes, volumes=vols)
    assert not mask_windows(df, hold=5).any()
    assert mask_windows(df, hold=5, include_organic=True).any()


def test_empty_and_missing_columns_are_safe():
    assert suspect_sessions(pd.DataFrame()).empty
    assert suspect_sessions(pd.DataFrame({"close": [1.0, 2.0]})).empty
    assert not mask_windows(pd.DataFrame({"open": [], "close": []}), hold=5).any()


# --- recent_corporate_action (the live entry screen) ------------------------------

def _reverse_split_at(n_before, n_after):
    closes = [2.0] * n_before + [30.0] * n_after
    vols = [1_000_000] * n_before + [80_000] * n_after
    return _dated(_frame(closes, volumes=vols))


def test_a_recent_reverse_split_is_reported():
    df = _reverse_split_at(40, 5)
    hit = recent_corporate_action(df)
    assert hit is not None
    assert hit["direction"] == "up"
    assert hit["sessions_ago"] == 4
    assert hit["lookback_sessions"] == ENTRY_LOOKBACK_SESSIONS


def test_a_split_outside_the_lookback_is_clear():
    df = _reverse_split_at(40, ENTRY_LOOKBACK_SESSIONS + 5)
    assert recent_corporate_action(df) is None


def test_no_look_ahead_a_future_split_cannot_veto_today():
    """A backtest asking 'was this clear on day 30?' must not see day 40."""
    df = _reverse_split_at(40, 5)
    as_of = df.index[30]
    assert recent_corporate_action(df, as_of=as_of) is None
    assert recent_corporate_action(df, as_of=df.index[-1]) is not None


def test_an_organic_recent_move_does_not_block_entry():
    closes = [10.0] * 40 + [45.0] * 3
    vols = [1_000_000] * 40 + [30_000_000] * 3
    assert recent_corporate_action(_dated(_frame(closes, volumes=vols))) is None


def test_lookback_counts_sessions_not_rows():
    """Two 4H bars a session: 5 sessions ago is still 5, not 10."""
    daily = _reverse_split_at(40, 5)
    rows = []
    for ts, r in daily.iterrows():
        rows.append((ts, r["open"], r["close"], r["volume"] / 2))
        rows.append((ts + pd.Timedelta(hours=4), r["close"], r["close"], r["volume"] / 2))
    four_h = pd.DataFrame(rows, columns=["ts", "open", "close", "volume"]).set_index("ts")
    hit = recent_corporate_action(four_h)
    assert hit is not None and hit["sessions_ago"] == 4


def test_no_bars_means_no_opinion():
    assert recent_corporate_action(None) is None
    assert recent_corporate_action(pd.DataFrame()) is None
