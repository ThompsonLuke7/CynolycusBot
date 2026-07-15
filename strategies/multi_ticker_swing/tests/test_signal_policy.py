from __future__ import annotations

import unittest
from types import SimpleNamespace

from strategies.multi_ticker_swing.live.signal_policy import (
    CalibrationBucket,
    EVCalibrationTable,
    SignalPolicyConfig,
    SignalPolicyLayer,
    score_bucket_for,
)


def _signal(**overrides):
    data = {
        "ticker": "AMD",
        "direction": 1,
        "p_dir": 0.72,
        "ev_score": 0.018,
        "atr": 2.0,
        "features": {
            "qqq_ret_16": 0.012,
            "rel_str_qqq_4": 1.5,
            "rel_str_spy_16": 2.0,
            "stock_beta_bucket": 1.0,
            "beta_like_spy_64": 1.0,
            "range_pos_20": 0.50,
            "daily_range_pos_20": 0.50,
            "zscore_close_64": 0.0,
        },
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class SignalPolicyTests(unittest.TestCase):
    def test_score_bucket_uses_stable_5pct_bins(self) -> None:
        self.assertEqual(score_bucket_for(0.72), "p_dir_0.70_0.75")
        self.assertEqual(score_bucket_for(1.0), "p_dir_0.95_1.00")

    def test_defensive_high_beta_long_blocks(self) -> None:
        policy = SignalPolicyLayer(
            SignalPolicyConfig(enabled=True),
            calibration=EVCalibrationTable(),
        )
        sig = _signal(
            features={
                "qqq_ret_16": -0.012,
                "rel_str_qqq_4": -1.0,
                "rel_str_spy_16": -6.0,
                "stock_beta_bucket": 3.0,
                "beta_like_spy_64": 1.5,
                "daily_range_pos_20": 0.50,
                "range_pos_20": 0.50,
                "zscore_close_64": 0.0,
            }
        )

        decision = policy.evaluate_signal(sig)

        self.assertEqual(decision.action, "BLOCK")
        self.assertIn("high_beta_long_in_defensive_regime", decision.reasons)
        self.assertEqual(decision.recommended_qty, 0)

    def test_bad_calibration_bucket_blocks_when_sample_is_large(self) -> None:
        calibration = EVCalibrationTable(
            {
                "multi_ticker_swing|long|aggressive|p_dir_0.70_0.75": CalibrationBucket(
                    key="multi_ticker_swing|long|aggressive|p_dir_0.70_0.75",
                    count=30,
                    win_rate=0.35,
                    avg_return=-0.01,
                )
            }
        )
        policy = SignalPolicyLayer(
            SignalPolicyConfig(enabled=True, min_bucket_trades=20),
            calibration=calibration,
        )

        decision = policy.evaluate_signal(_signal())

        self.assertEqual(decision.action, "BLOCK")
        self.assertIn("ev_bucket_low_win_rate", decision.reasons)
        self.assertEqual(decision.calibration["count"], 30)

    def test_entry_context_routes_wide_spread_options_to_skip(self) -> None:
        policy = SignalPolicyLayer(
            SignalPolicyConfig(enabled=True, max_entry_spread_pct_mid=0.18),
            calibration=EVCalibrationTable(),
        )
        sig = _signal()
        decision = policy.evaluate_signal(sig)

        enriched = policy.with_entry_context(
            decision,
            signal=sig,
            option_meta={
                "strike": 100.0,
                "underlying_price_at_selection": 100.0,
                "dte": 7,
            },
            quote_meta={"ask": 2.0, "mid": 1.5, "spread_pct_mid": 0.50},
        )

        self.assertEqual(enriched.action, "BLOCK")
        self.assertEqual(enriched.option_translation["route"], "skip_options")
        self.assertEqual(enriched.recommended_qty, 0)


if __name__ == "__main__":
    unittest.main()
