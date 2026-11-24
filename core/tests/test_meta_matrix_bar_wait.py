"""Meta waits for the matrix to hold the newest completed 4H bar, as HTF does.

11-23 and 11-24: Meta's 14:20 pass read the matrix before the refresher could append
the 10:00 ET bar (~14:30-14:37 under the complete-bar gate) and aborted as stale;
HTF, which waits, traded that bar at 14:37.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from core.live_readiness import wait_for_matrix_bar
from signals.meta_context.meta_ranker import live_runner as meta

ET = ZoneInfo("America/New_York")


def _matrix(path, stamps, n=300):
    rows = [(pd.Timestamp(s).tz_convert("UTC"), f"T{i}") for s in stamps for i in range(n)]
    pd.DataFrame(rows, columns=["timestamp", "ticker"]).assign(s_combo=0.5).to_parquet(path, index=False)


def test_meta_waits_for_the_10am_bar_instead_of_aborting(tmp_path):
    path = tmp_path / "m.parquet"
    prev = pd.Timestamp("2026-11-23 14:00", tz=ET)
    _matrix(path, [prev])
    clock = {"t": datetime(2026, 11, 24, 14, 20, tzinfo=ET)}
    appended_at = datetime(2026, 11, 24, 14, 37, tzinfo=ET)

    def sleep(s):
        clock["t"] += timedelta(seconds=s)
        if clock["t"] >= appended_at:
            _matrix(path, [prev, pd.Timestamp("2026-11-24 10:00", tz=ET)])

    ok, why = wait_for_matrix_bar(path, timeout_s=1500, latest_full_bar=meta._latest_full_bar,
                                  now_fn=lambda: clock["t"], sleep_fn=sleep)
    assert ok, why
    assert clock["t"] >= appended_at
    # 900 s (HTF's wait) from 14:20 gives up at 14:35, before the bar lands.
    _matrix(path, [prev])
    clock["t"] = datetime(2026, 11, 24, 14, 20, tzinfo=ET)
    ok, _ = wait_for_matrix_bar(path, timeout_s=900, latest_full_bar=meta._latest_full_bar,
                                now_fn=lambda: clock["t"], sleep_fn=sleep)
    assert not ok


def test_a_thin_partial_append_does_not_count_as_the_bar(tmp_path):
    path = tmp_path / "m.parquet"
    prev = pd.Timestamp("2026-11-23 14:00", tz=ET)
    rows = [(prev.tz_convert("UTC"), f"T{i}") for i in range(300)]
    rows += [(pd.Timestamp("2026-11-24 10:00", tz=ET).tz_convert("UTC"), "CUB")]  # one stray ticker
    pd.DataFrame(rows, columns=["timestamp", "ticker"]).to_parquet(path, index=False)
    clock = {"t": datetime(2026, 11, 24, 14, 20, tzinfo=ET).astimezone(timezone.utc)}
    ok, why = wait_for_matrix_bar(path, timeout_s=60, latest_full_bar=meta._latest_full_bar,
                                  now_fn=lambda: clock["t"],
                                  sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + timedelta(seconds=s)))
    assert not ok and "after 60s" in why


def test_the_runner_exposes_the_wait():
    import inspect

    src = inspect.getsource(meta.main)
    assert "--bar-wait-seconds" in src and "wait_for_matrix_bar(" in src
    from signals.meta_context.meta_ranker import run_4h_loop
    assert "default=1500.0" in src and 1500 + 180 < run_4h_loop.RUNNER_TIMEOUT_SEC, \
        "the wait plus a normal run must fit inside the loop's runner timeout"
