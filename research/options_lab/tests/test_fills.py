"""Tests for research/options_lab/fills.py.

Covers: estimate_spread's None-propagation (unknown liquidity input stays
unknown, never coerced to a guessed number) and domain validation, the clip
range, apply_fill's three assumptions and buy/sell sign convention,
commission_cost, and structure_commission's per-leg summation.
"""
from __future__ import annotations

import math

import pytest

from research.options_lab import fills


# --------------------------------------------------------------------------
# estimate_spread
# --------------------------------------------------------------------------

def test_estimate_spread_none_when_any_input_unknown():
    assert fills.estimate_spread(None, 5, 100, 50, 1_000_000) is None
    assert fills.estimate_spread(0.02, None, 100, 50, 1_000_000) is None
    assert fills.estimate_spread(0.02, 5, None, 50, 1_000_000) is None
    assert fills.estimate_spread(0.02, 5, 100, None, 1_000_000) is None
    assert fills.estimate_spread(0.02, 5, 100, 50, None) is None


def test_estimate_spread_rejects_negative_known_inputs():
    with pytest.raises(ValueError):
        fills.estimate_spread(-0.01, 5, 100, 50, 1_000_000)
    with pytest.raises(ValueError):
        fills.estimate_spread(0.02, -1, 100, 50, 1_000_000)
    with pytest.raises(ValueError):
        fills.estimate_spread(0.02, 5, -1, 50, 1_000_000)
    with pytest.raises(ValueError):
        fills.estimate_spread(0.02, 5, 100, -1, 1_000_000)
    with pytest.raises(ValueError):
        fills.estimate_spread(0.02, 5, 100, 50, -1)


def test_estimate_spread_rejects_non_finite():
    with pytest.raises(ValueError):
        fills.estimate_spread(float("nan"), 5, 100, 50, 1_000_000)
    with pytest.raises(ValueError):
        fills.estimate_spread(0.02, float("inf"), 100, 50, 1_000_000)


def test_estimate_spread_returns_positive_float_within_clip_range():
    calib = fills.DEFAULT_CALIBRATION
    result = fills.estimate_spread(0.02, 5, 1000, 500, 5_000_000)
    assert result is not None
    assert calib.min_spread_pct <= result <= calib.max_spread_pct


def test_estimate_spread_clips_extreme_inputs_to_bounds():
    calib = fills.DEFAULT_CALIBRATION
    # Deep OTM, illiquid, thin underlying -> pushes toward the cap.
    high = fills.estimate_spread(5.0, 0, 0, 0, 0, calibration=calib)
    assert high == pytest.approx(calib.max_spread_pct) or high <= calib.max_spread_pct
    assert high >= calib.min_spread_pct


def test_estimate_spread_custom_calibration_used():
    calib = fills.SpreadCalibration(
        intercept=math.log(0.10), moneyness_coef=0.0, log1p_dte_coef=0.0,
        log1p_oi_coef=0.0, log1p_volume_coef=0.0, log1p_adv_coef=0.0,
    )
    result = fills.estimate_spread(0.0, 0, 0, 0, 0, calibration=calib)
    assert result == pytest.approx(0.10, abs=1e-9)


def test_estimate_spread_moneyness_increases_spread_at_default_calibration():
    # Default calibration's moneyness_coef is positive (further OTM/ITM ->
    # wider spread), matching the economic prior and the G1a finding that
    # far-from-ATM contracts are thinner/harder to price.
    near = fills.estimate_spread(0.01, 5, 1000, 500, 5_000_000)
    far = fills.estimate_spread(0.30, 5, 1000, 500, 5_000_000)
    assert far >= near


# --------------------------------------------------------------------------
# apply_fill
# --------------------------------------------------------------------------

def test_apply_fill_optimistic_is_exact_mid():
    assert fills.apply_fill(1.00, "buy", 0.20, "optimistic") == pytest.approx(1.00)
    assert fills.apply_fill(1.00, "sell", 0.20, "optimistic") == pytest.approx(1.00)


def test_apply_fill_buy_pays_above_mid():
    px = fills.apply_fill(1.00, "buy", 0.20, "calibrated")
    assert px > 1.00
    assert px == pytest.approx(1.00 * (1 + 0.5 * 0.20))


def test_apply_fill_sell_receives_below_mid():
    px = fills.apply_fill(1.00, "sell", 0.20, "calibrated")
    assert px < 1.00
    assert px == pytest.approx(1.00 * (1 - 0.5 * 0.20))


def test_apply_fill_pessimistic_crosses_full_spread():
    buy_px = fills.apply_fill(2.00, "buy", 0.10, "pessimistic")
    sell_px = fills.apply_fill(2.00, "sell", 0.10, "pessimistic")
    assert buy_px == pytest.approx(2.00 * 1.10)
    assert sell_px == pytest.approx(2.00 * 0.90)


def test_apply_fill_pessimistic_worse_than_calibrated_worse_than_optimistic_for_buy():
    opt = fills.apply_fill(1.00, "buy", 0.20, "optimistic")
    cal = fills.apply_fill(1.00, "buy", 0.20, "calibrated")
    pes = fills.apply_fill(1.00, "buy", 0.20, "pessimistic")
    assert opt < cal < pes


def test_apply_fill_zero_spread_is_mid_regardless_of_assumption():
    for assumption in ("optimistic", "calibrated", "pessimistic"):
        assert fills.apply_fill(3.50, "buy", 0.0, assumption) == pytest.approx(3.50)
        assert fills.apply_fill(3.50, "sell", 0.0, assumption) == pytest.approx(3.50)


def test_apply_fill_rejects_bad_inputs():
    with pytest.raises(ValueError):
        fills.apply_fill(-1.0, "buy", 0.1, "optimistic")
    with pytest.raises(ValueError):
        fills.apply_fill(1.0, "hold", 0.1, "optimistic")
    with pytest.raises(ValueError):
        fills.apply_fill(1.0, "buy", -0.1, "optimistic")
    with pytest.raises(ValueError):
        fills.apply_fill(1.0, "buy", 0.1, "best_case")


# --------------------------------------------------------------------------
# commission_cost / structure_commission
# --------------------------------------------------------------------------

def test_commission_cost_default_rate():
    assert fills.commission_cost(1) == pytest.approx(0.65)
    assert fills.commission_cost(3) == pytest.approx(1.95)


def test_commission_cost_custom_rate():
    assert fills.commission_cost(2, rate=1.00) == pytest.approx(2.00)


def test_commission_cost_zero_contracts_is_zero():
    assert fills.commission_cost(0) == 0.0


def test_commission_cost_rejects_negative():
    with pytest.raises(ValueError):
        fills.commission_cost(-1)
    with pytest.raises(ValueError):
        fills.commission_cost(1, rate=-0.1)


def test_structure_commission_sums_per_leg():
    # 1-lot vertical spread: two legs, one contract each.
    assert fills.structure_commission([1, 1]) == pytest.approx(1.30)
    # 4-leg iron condor at 2 contracts each.
    assert fills.structure_commission([2, 2, 2, 2], rate=0.65) == pytest.approx(5.20)


def test_structure_commission_matches_n_single_leg_calls():
    legs = [1, 2, 3]
    total = fills.structure_commission(legs)
    manual = sum(fills.commission_cost(c) for c in legs)
    assert total == pytest.approx(manual)


def test_structure_commission_rejects_empty_legs():
    with pytest.raises(ValueError):
        fills.structure_commission([])
