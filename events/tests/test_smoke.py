"""Fixture smoke tests for scheduled event context."""

from __future__ import annotations

import pandas as pd

from events.features import build_event_features
from events.schema import events_from_frame


def test_treasury_auctions_are_excluded() -> None:
    raw = pd.DataFrame(
        [
            {"event_type": "CPI", "timestamp": "2026-06-10 08:30", "title": "CPI"},
            {"event_type": "treasury_auction", "timestamp": "2026-06-10 13:00", "title": "Auction"},
        ]
    )
    events = events_from_frame(raw)
    assert events["event_type"].tolist() == ["cpi"]


def test_scheduled_features_are_point_in_time(tmp_path) -> None:
    macro = events_from_frame(
        pd.DataFrame(
            [
                {"event_type": "CPI", "timestamp": "2026-06-10 08:30", "title": "CPI"},
                {"event_type": "FOMC", "timestamp": "2026-06-11 14:00", "title": "FOMC"},
                {"event_type": "OPEX", "timestamp": "2026-06-19 16:00", "title": "OPEX"},
            ]
        )
    )
    earnings = pd.DataFrame(
        [{"event_type": "earnings", "timestamp": pd.Timestamp("2026-06-12", tz="UTC"), "ticker": "RKLB"}]
    )
    base = pd.DataFrame([{"timestamp": pd.Timestamp("2026-06-10 12:00", tz="UTC"), "ticker": "RKLB"}])
    features = build_event_features(base, macro, earnings, output_path=tmp_path / "features.parquet")
    row = features.iloc[0]
    assert row["hours_to_cpi"] >= 0
    assert row["hours_to_fomc"] > row["hours_to_cpi"]
    assert row["macro_event_next_24h"] == 1.0
    assert row["earnings_next_7d"] == 1.0


def run_all() -> None:
    from tempfile import TemporaryDirectory
    from pathlib import Path

    test_treasury_auctions_are_excluded()
    with TemporaryDirectory() as d:
        test_scheduled_features_are_point_in_time(Path(d))
    print("events smoke tests passed")


if __name__ == "__main__":
    run_all()

