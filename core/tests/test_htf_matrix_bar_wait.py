"""HTF waits for the matrix to hold the newest completed 4H bar (2026-11-22)."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from strategies.multi_ticker_swing_htf.live import runner as htf

ET = ZoneInfo("America/New_York")


def _matrix(path, stamps, n=300):
    rows = [(pd.Timestamp(s).tz_convert("UTC"), f"T{i}") for s in stamps for i in range(n)]
    pd.DataFrame(rows, columns=["timestamp", "ticker"]).assign(htf_score=0.5).to_parquet(path, index=False)


def test_waits_for_todays_first_bar_then_passes(tmp_path):
    path = tmp_path / "m.parquet"
    prev_b = pd.Timestamp("2026-11-19 14:00", tz=ET)
    _matrix(path, [prev_b])
    clock = {"t": datetime(2026, 11, 20, 14, 25, tzinfo=ET)}

    def sleep(s):
        clock["t"] += timedelta(seconds=s)
        _matrix(path, [prev_b, pd.Timestamp("2026-11-20 10:00", tz=ET)])   # refresher appends it

    ok, why = htf.wait_for_matrix_bar(path, timeout_s=900, now_fn=lambda: clock["t"], sleep_fn=sleep)
    assert ok and "waited 20s" in why


def test_times_out_and_leaves_the_decision_to_the_staleness_guard(tmp_path):
    path = tmp_path / "m.parquet"
    _matrix(path, [pd.Timestamp("2026-11-19 14:00", tz=ET)])
    clock = {"t": datetime(2026, 11, 20, 14, 25, tzinfo=ET).astimezone(timezone.utc)}
    ok, why = htf.wait_for_matrix_bar(
        path, timeout_s=60, now_fn=lambda: clock["t"],
        sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + timedelta(seconds=s)))
    assert not ok and "after 60s" in why
