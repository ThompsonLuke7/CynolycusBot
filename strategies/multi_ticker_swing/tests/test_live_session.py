from __future__ import annotations

from datetime import datetime
import unittest
from zoneinfo import ZoneInfo

from strategies.multi_ticker_swing.live.session import (
    confirmation_breakout,
    entry_bucket,
    is_regular_trading_time,
    should_check_confirmation,
    should_scan_after_30m_close,
)


ET = ZoneInfo("America/New_York")


class LiveSessionTests(unittest.TestCase):
    def test_market_time_gates(self) -> None:
        self.assertFalse(is_regular_trading_time(datetime(2026, 6, 5, 9, 29, tzinfo=ET)))
        self.assertTrue(is_regular_trading_time(datetime(2026, 6, 5, 9, 30, tzinfo=ET)))
        self.assertFalse(is_regular_trading_time(datetime(2026, 6, 5, 16, 0, tzinfo=ET)))
        self.assertTrue(should_check_confirmation(datetime(2026, 6, 5, 10, 0, tzinfo=ET)))
        self.assertFalse(should_check_confirmation(datetime(2026, 6, 5, 15, 56, tzinfo=ET)))
        self.assertTrue(should_scan_after_30m_close(datetime(2026, 6, 5, 15, 30, tzinfo=ET)))
        self.assertFalse(should_scan_after_30m_close(datetime(2026, 6, 5, 15, 55, tzinfo=ET)))

    def test_entry_bucket_rounds_to_half_hour(self) -> None:
        self.assertEqual(entry_bucket(datetime(2026, 6, 5, 10, 29, tzinfo=ET)), "10:00")
        self.assertEqual(entry_bucket(datetime(2026, 6, 5, 10, 30, tzinfo=ET)), "10:30")

    def test_confirmation_breakout_requires_body_close(self) -> None:
        self.assertTrue(
            confirmation_breakout(
                direction=1,
                ref_high=100,
                ref_low=95,
                bar={"open": 99, "high": 101, "low": 98, "close": 100.5},
            )
        )
        self.assertFalse(
            confirmation_breakout(
                direction=1,
                ref_high=100,
                ref_low=95,
                bar={"open": 100.8, "high": 101, "low": 98, "close": 100.5},
            )
        )
        self.assertTrue(
            confirmation_breakout(
                direction=-1,
                ref_high=100,
                ref_low=95,
                bar={"open": 96, "high": 97, "low": 94, "close": 94.5},
            )
        )


if __name__ == "__main__":
    unittest.main()
