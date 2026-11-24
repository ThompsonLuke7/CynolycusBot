"""Alignment audit: each drift class fires on the defect it was written for, and not on
features that legitimately sit still (calendar fields, one-value-per-session context)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.alignment_audit import drift_flags, profile_frame

pytestmark = pytest.mark.safe

TICKERS = [f"T{i:02d}" for i in range(30)]


def _training(n_sessions: int = 260, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_sessions)
    rows = []
    for d in days:
        ts = pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=14)
        vix = rng.normal()
        for i, t in enumerate(TICKERS):
            rows.append({"timestamp": ts.tz_convert("UTC"), "ticker": t,
                         "ret": rng.normal(0, 0.02),             # cross-sectional, daily
                         "cap_bucket": i % 4,                    # static per name, varies across names
                         "days_to_earn": float(rng.integers(0, 60)),
                         "month": float(d.month),               # calendar: unchanged most days
                         "vix_z": vix})                          # one value per session
    return pd.DataFrame(rows)


def _live(day: str, *, seed: int, **overrides) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bar = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=14)
    out = pd.DataFrame({"bar_ts": bar.tz_convert("UTC"), "ticker": TICKERS,
                        "f__ret": rng.normal(0, 0.02, len(TICKERS)),
                        "f__cap_bucket": [i % 4 for i in range(len(TICKERS))],
                        "f__days_to_earn": rng.integers(0, 60, len(TICKERS)).astype(float),
                        "f__month": float(pd.Timestamp(day).month),
                        "f__vix_z": float(rng.normal())})
    for k, v in overrides.items():
        out[f"f__{k}"] = v
    return out


@pytest.fixture(scope="module")
def profile():
    tr = _training()
    feats = ["ret", "cap_bucket", "days_to_earn", "month", "vix_z"]
    return {"features": profile_frame(tr, feats, ts="timestamp", ticker="ticker")}


def _flagged(flags):
    return {k: sorted(x["feature"] for x in v) for k, v in flags.items() if v}


def test_aligned_session_raises_nothing(profile):
    prev, live = _live("2025-09-02", seed=1), _live("2025-09-03", seed=2)
    assert _flagged(drift_flags(live, prev, profile, multi_name=True)) == {}


def test_each_defect_class_fires(profile):
    prev = _live("2025-09-02", seed=1)
    live = _live("2025-09-03", seed=2,
                 cap_bucket=0,                 # 30m/Meta: market_cap_bucket 0 for every name
                 days_to_earn=np.nan)          # earnings loader read snapshots only
    live["f__ret"] = prev["f__ret"].to_numpy()  # daily context frozen at an old session
    live["f__vix_z"] = 33.0                     # macro count 33/day vs training <= 1
    got = _flagged(drift_flags(live, prev, profile, multi_name=True))
    assert got == {"null_jump": ["days_to_earn"], "stuck": ["cap_bucket"],
                   "frozen": ["ret"], "out_of_range": ["vix_z"]}


def test_calendar_and_session_constants_are_not_frozen_or_stuck(profile):
    # month equal to yesterday's, and vix_z one value across all names, are normal.
    prev, live = _live("2025-09-02", seed=1), _live("2025-09-03", seed=2)
    got = _flagged(drift_flags(live, prev, profile, multi_name=True))
    assert "month" not in sum(got.values(), []) and "vix_z" not in sum(got.values(), [])


def test_profile_unchanged_rate_uses_consecutive_sessions_only():
    tr = _training(n_sessions=40)
    # keep every 5th session only: no consecutive pairs remain
    sess = tr["timestamp"].dt.tz_convert("America/New_York").dt.normalize()
    all_sessions = sorted(sess.unique())
    sparse = tr[sess.isin(all_sessions[::5])]
    prof = profile_frame(sparse, ["ret"], ts="timestamp", ticker="ticker", sessions=all_sessions)
    assert prof["ret"]["unchanged"] is None
