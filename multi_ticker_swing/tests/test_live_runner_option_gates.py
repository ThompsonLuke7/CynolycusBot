from __future__ import annotations

from datetime import datetime
import unittest
from zoneinfo import ZoneInfo

from multi_ticker_swing.live.runner import _entry_contract_ref_date, _entry_quote_spread_ok


ET = ZoneInfo("America/New_York")


class LiveRunnerOptionGateTests(unittest.TestCase):
    def test_friday_after_one_skips_monday_expiry_start(self) -> None:
        self.assertEqual(
            _entry_contract_ref_date(datetime(2026, 6, 5, 12, 59, tzinfo=ET)).isoformat(),
            "2026-06-05",
        )
        self.assertEqual(
            _entry_contract_ref_date(datetime(2026, 6, 5, 13, 0, tzinfo=ET)).isoformat(),
            "2026-06-09",
        )
        self.assertEqual(
            _entry_contract_ref_date(datetime(2026, 6, 5, 15, 30, tzinfo=ET)).isoformat(),
            "2026-06-09",
        )

    def test_non_friday_after_one_skips_same_day_only(self) -> None:
        self.assertEqual(
            _entry_contract_ref_date(datetime(2026, 6, 8, 13, 0, tzinfo=ET)).isoformat(),
            "2026-06-09",
        )

    def test_entry_spread_must_be_strictly_below_eighteen_percent(self) -> None:
        self.assertEqual(_entry_quote_spread_ok({"spread_pct_mid": 0.179})[0], True)
        self.assertEqual(_entry_quote_spread_ok({"spread_pct_mid": 0.18})[:2], (False, "entry_spread_too_wide"))
        self.assertEqual(_entry_quote_spread_ok({"spread_pct_mid": None})[:2], (False, "entry_spread_missing"))


if __name__ == "__main__":
    unittest.main()
