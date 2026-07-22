from __future__ import annotations

import unittest

from strategies.multi_ticker_swing.live.runner import (
    DEFAULT_QTY,
    TARGET_NOTIONAL_USD,
    _entry_contracts_for_quote,
)


class EntrySizingTests(unittest.TestCase):
    def test_sizes_off_mid_premium(self) -> None:
        # $2.00 mid -> $200/contract -> round(5000/200) = 25 contracts
        self.assertEqual(_entry_contracts_for_quote({"mid": 2.0, "ask": 2.1}), 25)

    def test_falls_back_to_ask_when_mid_missing(self) -> None:
        self.assertEqual(_entry_contracts_for_quote({"ask": 1.0}), 50)

    def test_falls_back_to_default_qty_without_a_quote(self) -> None:
        self.assertEqual(_entry_contracts_for_quote({}), DEFAULT_QTY)
        self.assertEqual(_entry_contracts_for_quote(None), DEFAULT_QTY)

    def test_floors_at_one_contract_for_expensive_premium(self) -> None:
        self.assertEqual(_entry_contracts_for_quote({"mid": 500.0}), 1)

    def test_target_notional_matches_shared_engine_default(self) -> None:
        self.assertEqual(TARGET_NOTIONAL_USD, 5000.0)


if __name__ == "__main__":
    unittest.main()
