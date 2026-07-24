"""HTF dashboard's signal-log persist must never leave a half-written parquet
file visible to a concurrent reader.

2026-07-21 live audit: `logger.warning("HTF signal persist failed: %s", exc)`
fired 3x with "Parquet magic bytes not found in footer" -- a torn read caused
by `_persist_scan` writing directly to SIGNALS_LOG (`ThreadingHTTPServer`
spawns a thread per request, so two overlapping dashboard polls could have one
thread's `read_parquet()` catch another thread's `to_parquet()` mid-write).
Fixed with the same write-to-temp-then-atomic-rename pattern already used by
`core/broker_equity_snapshot.py` and `strategies/intraday_structure/state_store.py`.
"""
from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

import UI.htf_dashboard as htf_dashboard
from UI.htf_dashboard import HTFDashboardApp


@pytest.fixture(autouse=True)
def _signals_log(tmp_path, monkeypatch):
    path = tmp_path / "htf_signals.parquet"
    monkeypatch.setattr(htf_dashboard, "SIGNALS_LOG", path)
    return path


def _out(bar: str, tickers: list[str]) -> dict:
    return {"bar": bar, "picks": [{"rank": i + 1, "ticker": t, "score": 0.5} for i, t in enumerate(tickers)]}


def test_persist_scan_writes_no_stray_temp_files(_signals_log):
    app = HTFDashboardApp()
    app._persist_scan(_out("2026-07-21T18:00:00+00:00", ["AAA", "BBB"]))
    assert _signals_log.exists()
    assert list(_signals_log.parent.glob("*.tmp")) == []
    assert set(pd.read_parquet(_signals_log)["ticker"]) == {"AAA", "BBB"}


def test_persist_scan_replaces_same_bar_and_appends_new_bar(_signals_log):
    app = HTFDashboardApp()
    app._persist_scan(_out("2026-07-21T14:00:00+00:00", ["AAA"]))
    app._persist_scan(_out("2026-07-21T18:00:00+00:00", ["BBB"]))
    df = pd.read_parquet(_signals_log)
    assert sorted(df["bar"].unique()) == ["2026-07-21T14:00:00+00:00", "2026-07-21T18:00:00+00:00"]
    assert set(df["ticker"]) == {"AAA", "BBB"}


def test_concurrent_persist_scan_never_corrupts_the_log_for_a_reader(_signals_log):
    app = HTFDashboardApp()
    app._persist_scan(_out("2026-07-21T14:00:00+00:00", ["SEED"]))  # file exists before the race starts

    stop = threading.Event()
    read_errors: list[Exception] = []

    def _reader() -> None:
        while not stop.is_set():
            try:
                pd.read_parquet(_signals_log)
            except Exception as exc:  # noqa: BLE001
                read_errors.append(exc)

    def _writer(n: int) -> None:
        for i in range(20):
            app._persist_scan(_out(f"2026-07-21T18:{n:02d}:{i:02d}+00:00", [f"T{n}{i}"]))

    reader_threads = [threading.Thread(target=_reader) for _ in range(4)]
    for t in reader_threads:
        t.start()
    writer_threads = [threading.Thread(target=_writer, args=(n,)) for n in range(4)]
    for t in writer_threads:
        t.start()
    for t in writer_threads:
        t.join()
    stop.set()
    for t in reader_threads:
        t.join()

    assert read_errors == [], f"reader saw a torn/corrupt parquet file: {read_errors}"
