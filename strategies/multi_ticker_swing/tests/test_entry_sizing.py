from __future__ import annotations

import unittest

from strategies.multi_ticker_swing.live.runner import (
    DEFAULT_QTY,
    TARGET_NOTIONAL_USD,
    _entry_contracts_for_quote,
    _owned_qty_from_fill,
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


class OwnedQtyFromFillTests(unittest.TestCase):
    """Attribution on a shared account.

    Alpaca nets option positions by symbol, so when a sibling module is long the
    same contract the broker reports the combined size and cannot say which part
    is ours. Our own entry order's `filled_qty` is the only attribution, and it
    is what the exit is sized from.
    """

    def test_a_full_fill_owns_the_requested_size(self) -> None:
        self.assertEqual(
            _owned_qty_from_fill({"filled_qty": "10"}, requested=10), 10)

    def test_a_partial_fill_owns_only_what_filled(self) -> None:
        self.assertEqual(
            _owned_qty_from_fill({"filled_qty": "4"}, requested=10), 4)

    def test_a_fill_larger_than_requested_is_capped(self) -> None:
        """A netted read must never inflate the claim above what we asked for."""
        self.assertEqual(
            _owned_qty_from_fill({"filled_qty": "25"}, requested=10), 10)

    def test_a_missing_fill_field_falls_back_to_requested(self) -> None:
        self.assertEqual(_owned_qty_from_fill({}, requested=10), 10)
        self.assertEqual(_owned_qty_from_fill(None, requested=10), 10)

    def test_a_zero_fill_falls_back_rather_than_claiming_nothing(self) -> None:
        self.assertEqual(_owned_qty_from_fill({"filled_qty": "0"}, requested=10), 10)
