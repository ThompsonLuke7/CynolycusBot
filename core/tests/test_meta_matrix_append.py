"""Appending to the cached 10m meta matrix must not restart `day_id`.

`day_id` is `factorize(session dates)`: it counts sessions from the start of
whatever window produced it. The live append path and the nightly refresher both
recompute a short warm-up tail, so the tail's `day_id` starts at 0 while the
cached rows carry their original numbering. Merging the two without renumbering
produces ... 1446, 1447, 0, 1, 2 — and `day_id` is a live model input, listed in
the entry (97 features), exit (104) and swing-setup selected features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.API.Alpaca_API.inference.live_inference import continue_origin_relative_columns

_ET = "America/New_York"


def _frame(day_ids, start="2026-09-01 09:30", sessions_per_day=2):
    """One row per bar, `sessions_per_day` bars per session."""
    stamps, values = [], []
    day = pd.Timestamp(start, tz=_ET)
    for offset, value in enumerate(day_ids):
        base = day.normalize() + pd.Timedelta(days=offset) + pd.Timedelta(hours=9, minutes=30)
        for bar in range(sessions_per_day):
            stamps.append(base + pd.Timedelta(minutes=10 * bar))
            values.append(float(value))
    return pd.DataFrame({"day_id": values, "close": np.arange(len(values), dtype=float)},
                        index=pd.DatetimeIndex(stamps))


def test_a_recomputed_tail_continues_the_cached_numbering():
    cached = _frame([1445, 1446, 1447])
    appended = _frame([0, 1], start="2026-09-04 09:30")  # tail restarted at zero

    out = continue_origin_relative_columns(cached, appended)

    assert out["day_id"].tolist() == [1448.0, 1448.0, 1449.0, 1449.0]
    assert out["close"].tolist() == appended["close"].tolist(), "only day_id is renumbered"


def test_a_tail_continuing_the_same_session_keeps_that_session_id():
    """A mid-session restart appends more bars of the session already cached."""
    cached = _frame([1445, 1446])
    same_day = cached.index[-1].normalize() + pd.Timedelta(hours=10)
    appended = pd.DataFrame({"day_id": [0.0, 0.0], "close": [9.0, 10.0]},
                            index=pd.DatetimeIndex([same_day, same_day + pd.Timedelta(minutes=10)]))

    out = continue_origin_relative_columns(cached, appended)

    assert out["day_id"].tolist() == [1446.0, 1446.0]


def test_frames_without_day_id_or_without_rows_pass_through():
    cached = _frame([1445])
    appended = _frame([0], start="2026-09-04 09:30")

    assert continue_origin_relative_columns(cached.drop(columns=["day_id"]), appended) is appended
    assert continue_origin_relative_columns(cached, appended.drop(columns=["day_id"])).equals(
        appended.drop(columns=["day_id"]))
    assert continue_origin_relative_columns(cached.iloc[:0], appended) is appended
    assert continue_origin_relative_columns(cached, appended.iloc[:0]).empty


def test_an_unusable_cached_day_id_leaves_the_tail_alone():
    cached = _frame([1445]).assign(day_id=np.nan)
    appended = _frame([0], start="2026-09-04 09:30")

    assert continue_origin_relative_columns(cached, appended) is appended
