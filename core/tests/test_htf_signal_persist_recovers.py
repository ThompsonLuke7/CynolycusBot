"""A corrupt HTF signal history must not block every future write.

`_persist_scan` reads the existing parquet to append to it. When that file
is truncated — a crash mid-write, before the write-then-rename guard existed —
the read raises, the whole method fails, and the dashboard logs
"HTF signal persist failed: Parquet magic bytes not found in footer" forever.
The live file was in exactly that state from 2026-07-21 to 2026-08-25: five
weeks of live HTF picks were never recorded, and the only symptom was a
repeating WARNING. See research/daily_live_reports/2026-08-25.md.
"""
from __future__ import annotations

import pandas as pd
import pytest

from UI import htf_dashboard


@pytest.fixture
def signals_log(tmp_path, monkeypatch):
    path = tmp_path / "htf_signals.parquet"
    monkeypatch.setattr(htf_dashboard, "SIGNALS_LOG", path)
    return path


def _Persister():
    """The app, without an HTTP server or a scan around it."""

    return htf_dashboard.HTFDashboardApp(persist=True)


def _out(bar: str, ticker: str = "INDP"):
    return {"bar": bar, "picks": [{"rank": 1, "ticker": ticker, "score": 0.64}]}


def test_a_readable_history_is_appended_to(signals_log):
    _Persister()._persist_scan(_out("2026-08-25 14:00:00+00:00"))
    _Persister()._persist_scan(_out("2026-08-25 18:00:00+00:00", "KSS"))

    saved = pd.read_parquet(signals_log)
    assert sorted(saved["ticker"]) == ["INDP", "KSS"]


def test_a_corrupt_history_is_quarantined_and_the_new_signal_is_saved(signals_log):
    signals_log.write_bytes(b"not a parquet file at all")

    _Persister()._persist_scan(_out("2026-08-25 14:00:00+00:00"))

    saved = pd.read_parquet(signals_log)
    assert list(saved["ticker"]) == ["INDP"]
    quarantined = list(signals_log.parent.glob("htf_signals.parquet.corrupt.*"))
    assert len(quarantined) == 1
    # Never deleted: it is the only copy of whatever survived.
    assert quarantined[0].read_bytes() == b"not a parquet file at all"


def test_recovery_is_not_a_silent_wipe_on_every_write(signals_log):
    """Quarantine once, then behave normally. A guard that re-quarantined would
    throw away each day's signals as fast as it wrote them."""

    signals_log.write_bytes(b"corrupt")
    _Persister()._persist_scan(_out("2026-08-25 14:00:00+00:00"))
    _Persister()._persist_scan(_out("2026-08-25 18:00:00+00:00", "KSS"))

    saved = pd.read_parquet(signals_log)
    assert sorted(saved["ticker"]) == ["INDP", "KSS"]
    assert len(list(signals_log.parent.glob("htf_signals.parquet.corrupt.*"))) == 1
