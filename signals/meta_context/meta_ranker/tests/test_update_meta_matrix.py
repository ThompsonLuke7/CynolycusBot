from __future__ import annotations

import pandas as pd
import pytest

import signals.meta_context.meta_ranker.update_meta_matrix as updater


pytestmark = pytest.mark.safe


def test_latest_reference_bar_timestamp_uses_newest_reference(monkeypatch):
    def fake_read(path):
        ticker = path.stem
        stamp = "2026-07-16 18:00:00+00:00" if ticker == "SPY" else "2026-07-16 17:00:00+00:00"
        return pd.DataFrame({"timestamp": [pd.Timestamp(stamp)]})

    monkeypatch.setattr(updater, "_read_bars", fake_read)

    assert updater._latest_reference_bar_timestamp() == pd.Timestamp("2026-07-16 18:00:00+00:00")
