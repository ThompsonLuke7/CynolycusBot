from __future__ import annotations

import pandas as pd

from scripts.discord_ledger.fetch_underlying_proxy import snapshot


def test_snapshot_uses_completed_bar_and_maps_spx_to_spy() -> None:
    bars = pd.DataFrame({"symbol": ["SPY", "SPY"],
                         "timestamp": pd.to_datetime(["2026-01-05T15:00:00Z", "2026-01-05T15:01:00Z"], utc=True),
                         "close": [600.0, 601.0], "volume": [1, 1]})
    target = {"target_id": "t", "trade_id": "trade", "role": "entry", "symbol": "SPX",
              "timestamp_utc": "2026-01-05T15:01:30Z", "source_message_ids": ["m"]}
    result = snapshot(target, bars, "bars.parquet")
    assert result["status"] == "ok"
    assert result["price"] == 600.0
    assert result["proxy_symbol"] == "SPY"
    assert result["proxy_mapping"] == "index_to_etf"


def test_snapshot_does_not_use_opening_minute_before_complete() -> None:
    bars = pd.DataFrame({"symbol": ["SPY"], "timestamp": pd.to_datetime(["2026-01-05T14:30:00Z"], utc=True),
                         "close": [600.0], "volume": [1]})
    target = {"target_id": "t", "trade_id": "trade", "role": "entry", "symbol": "SPY",
              "timestamp_utc": "2026-01-05T14:30:30Z", "source_message_ids": ["m"]}
    assert snapshot(target, bars, "bars.parquet")["status"] == "unavailable_outside_regular_session"
