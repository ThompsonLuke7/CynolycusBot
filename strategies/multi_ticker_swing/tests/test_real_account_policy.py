from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from zoneinfo import ZoneInfo

from strategies.multi_ticker_swing.live.real_account_policy import RealAccountBookkeeper, RealAccountPolicyConfig


ET = ZoneInfo("America/New_York")


def _signal(direction: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        ticker="AMD",
        direction=direction,
        p_dir=0.7,
        ev_score=0.02,
        entry_threshold=0.55,
        atr=2.0,
        ref_high=100.0,
        ref_low=95.0,
    )


def _option_meta() -> dict:
    return {
        "underlying_price_at_selection": 100.0,
        "dte": 2,
        "selected_abs_delta": 0.45,
        "greeks": {"theta": -0.08},
    }


def _quote_meta() -> dict:
    return {"ask": 2.0, "mid": 1.95, "spread_pct_mid": 0.05}


class RealAccountPolicyTests(unittest.TestCase):
    def test_calls_only_policy_rejects_short_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            policy = RealAccountBookkeeper(
                RealAccountPolicyConfig(enabled=True, calls_only=True, state_path=Path(tmp) / "book.json")
            )

            decision = policy.evaluate_entry(
                signal=_signal(direction=-1),
                option_symbol="AMD260619P00100000",
                option_meta=_option_meta(),
                quote_meta=_quote_meta(),
                limit_prices=[2.01],
                open_positions_count=0,
                account={"equity": 1000, "cash": 1000, "buying_power": 1000},
                now=datetime(2026, 6, 5, 11, 0, tzinfo=ET),
            )

            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "real_policy_calls_only")

    def test_premium_and_open_position_limits_are_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            policy = RealAccountBookkeeper(
                RealAccountPolicyConfig(
                    enabled=True,
                    max_open_positions=1,
                    max_premium_per_trade=350,
                    max_open_premium=350,
                    state_path=Path(tmp) / "book.json",
                )
            )

            too_many_positions = policy.evaluate_entry(
                signal=_signal(),
                option_symbol="AMD260619C00100000",
                option_meta=_option_meta(),
                quote_meta=_quote_meta(),
                limit_prices=[2.0],
                open_positions_count=1,
                account={"equity": 1000, "cash": 1000, "buying_power": 1000},
                now=datetime(2026, 6, 5, 11, 0, tzinfo=ET),
            )
            self.assertEqual(too_many_positions.reason, "real_policy_max_open_positions")

            policy.record_entry(
                ticker="AMD",
                option_symbol="AMD260619C00100000",
                qty=1,
                premium_at_risk=300.0,
                reason="test",
                now=datetime(2026, 6, 5, 11, 0, tzinfo=ET),
            )
            over_budget = policy.evaluate_entry(
                signal=_signal(),
                option_symbol="NVDA260619C00100000",
                option_meta=_option_meta(),
                quote_meta=_quote_meta(),
                limit_prices=[2.0],
                open_positions_count=0,
                account={"equity": 1000, "cash": 1000, "buying_power": 1000},
                now=datetime(2026, 6, 5, 11, 5, tzinfo=ET),
            )
            self.assertEqual(over_budget.reason, "real_policy_premium_budget_exceeded")

    def test_clean_call_is_allowed_and_close_releases_premium(self) -> None:
        with TemporaryDirectory() as tmp:
            policy = RealAccountBookkeeper(
                RealAccountPolicyConfig(enabled=True, state_path=Path(tmp) / "book.json")
            )

            decision = policy.evaluate_entry(
                signal=_signal(),
                option_symbol="AMD260619C00100000",
                option_meta=_option_meta(),
                quote_meta=_quote_meta(),
                limit_prices=[2.0],
                open_positions_count=0,
                account={"equity": 1000, "cash": 1000, "buying_power": 1000},
                entry_quality={"body_frac": 0.5},
                now=datetime(2026, 6, 5, 11, 0, tzinfo=ET),
            )
            self.assertTrue(decision.allowed)
            self.assertEqual(decision.qty, 1)

            policy.record_entry(
                ticker="AMD",
                option_symbol="AMD260619C00100000",
                qty=decision.qty,
                premium_at_risk=decision.premium_at_risk,
                reason=decision.reason,
                now=datetime(2026, 6, 5, 11, 0, tzinfo=ET),
            )
            self.assertEqual(policy.snapshot()["state"]["open_premium"], decision.premium_at_risk)

            policy.mark_position_closed(
                ticker="AMD",
                option_symbol="AMD260619C00100000",
                entry_premium=2.0,
                qty=1,
                now=datetime(2026, 6, 5, 12, 0, tzinfo=ET),
            )
            self.assertEqual(policy.snapshot()["state"]["open_premium"], 0.0)


if __name__ == "__main__":
    unittest.main()
