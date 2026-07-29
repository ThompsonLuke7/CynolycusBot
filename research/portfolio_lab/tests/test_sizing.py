"""Tests for research/portfolio_lab/sizing.py.

Covers: per-position/sector/theme cap enforcement, correlation-aware
down-weighting of near-duplicate positions, volatility targeting hitting its
annualized target within tolerance, liquidity (ADV participation) capping,
and whole-share rounding (including the floor-at-0 behavior that
deliberately differs from the live baseline's floor-at-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.portfolio_lab import sizing


def _cfg(**overrides) -> sizing.SizingConfig:
    base = dict(
        capital=100_000.0, target_concurrent=10, per_position_cap_pct=0.12,
        per_sector_cap_pct=0.30, per_theme_cap_pct=0.25,
        target_portfolio_vol_annual=0.15, max_vol_scale=3.0,
        adv_participation_cap_pct=0.05, corr_penalty_strength=1.0,
    )
    base.update(overrides)
    return sizing.SizingConfig(**base)


# --------------------------------------------------------------------------
# whole_shares
# --------------------------------------------------------------------------

def test_whole_shares_rounds_to_nearest():
    assert sizing.whole_shares(1000.0, 33.0) == round(1000.0 / 33.0)


def test_whole_shares_floors_at_zero_not_one():
    # A tiny dollar amount at a high price should round to 0, unlike the live
    # baseline's shares_for_notional which floors at 1.
    assert sizing.whole_shares(5.0, 500.0) == 0


def test_whole_shares_handles_bad_price():
    assert sizing.whole_shares(1000.0, 0.0) == 0
    assert sizing.whole_shares(1000.0, -5.0) == 0
    assert sizing.whole_shares(0.0, 50.0) == 0


# --------------------------------------------------------------------------
# cap_headroom (per-position handled directly in size_candidate; this covers
# sector/theme group caps)
# --------------------------------------------------------------------------

def test_cap_headroom_unlimited_when_group_unknown():
    book = {"AAPL": sizing.OpenPosition("AAPL", 5000, 50, sector="tech")}
    room = sizing.cap_headroom(book, attr="sector", group=None, cap_pct=0.3, capital=100_000)
    assert room == float("inf")


def test_cap_headroom_shrinks_as_group_fills():
    book = {
        "AAPL": sizing.OpenPosition("AAPL", 20_000, 100, sector="tech"),
        "MSFT": sizing.OpenPosition("MSFT", 10_000, 40, sector="tech"),
    }
    room = sizing.cap_headroom(book, attr="sector", group="tech", cap_pct=0.30, capital=100_000)
    assert room == pytest.approx(30_000 - 30_000)  # cap $30k, already used $30k -> 0
    room2 = sizing.cap_headroom(book, attr="sector", group="energy", cap_pct=0.30, capital=100_000)
    assert room2 == pytest.approx(30_000)  # untouched group -> full cap


# --------------------------------------------------------------------------
# correlation_penalty
# --------------------------------------------------------------------------

def test_correlation_penalty_downweights_near_duplicates():
    tickers = ["A", "B", "C"]
    corr = pd.DataFrame(
        [[1.0, 0.95, 0.05],
         [0.95, 1.0, 0.05],
         [0.05, 0.05, 1.0]],
        index=tickers, columns=tickers,
    )
    book_dup = {"B": sizing.OpenPosition("B", 5000, 50)}       # A is a near-duplicate of B
    book_indep = {"C": sizing.OpenPosition("C", 5000, 50)}     # A is ~uncorrelated with C

    penalty_dup = sizing.correlation_penalty("A", book_dup, corr, strength=1.0)
    penalty_indep = sizing.correlation_penalty("A", book_indep, corr, strength=1.0)

    assert penalty_dup < penalty_indep
    assert 0.0 < penalty_dup < 1.0
    assert penalty_indep == pytest.approx(1.0 / (1.0 + 0.05), rel=1e-6)


def test_correlation_penalty_no_op_on_empty_book_or_zero_strength():
    corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    assert sizing.correlation_penalty("A", {}, corr, strength=1.0) == 1.0
    book = {"B": sizing.OpenPosition("B", 1000, 10)}
    assert sizing.correlation_penalty("A", book, corr, strength=0.0) == 1.0
    assert sizing.correlation_penalty("A", book, None, strength=1.0) == 1.0


# --------------------------------------------------------------------------
# portfolio_vol_scale (volatility targeting)
# --------------------------------------------------------------------------

def test_vol_targeting_hits_target_on_diagonal_cov():
    # Independent names, known annualized vols -> portfolio vol is closed-form.
    tickers = ["A", "B"]
    weights = {"A": 0.10, "B": 0.10}
    vols = np.array([0.40, 0.20])  # 40% and 20% annualized vol
    cov = pd.DataFrame(np.diag(vols ** 2), index=tickers, columns=tickers)

    w = np.array([weights[t] for t in tickers])
    realized_vol = float(np.sqrt(w @ cov.values @ w))

    target = 0.15
    scale = sizing.portfolio_vol_scale(weights, cov, target_vol=target, max_scale=10.0)
    scaled_vol = realized_vol * scale
    assert scaled_vol == pytest.approx(target, rel=1e-6)


def test_vol_targeting_scale_is_clamped():
    tickers = ["A"]
    weights = {"A": 0.01}
    cov = pd.DataFrame([[0.0001]], index=tickers, columns=tickers)  # tiny vol -> huge scale-up needed
    scale = sizing.portfolio_vol_scale(weights, cov, target_vol=0.15, max_scale=2.0)
    assert scale <= 2.0


def test_vol_targeting_falls_back_to_one_without_cov():
    assert sizing.portfolio_vol_scale({"A": 0.1}, None, target_vol=0.15, max_scale=3.0) == 1.0
    assert sizing.portfolio_vol_scale({}, pd.DataFrame(), target_vol=0.15, max_scale=3.0) == 1.0


# --------------------------------------------------------------------------
# size_candidate (full pipeline)
# --------------------------------------------------------------------------

def test_size_candidate_respects_position_cap():
    cfg = _cfg(per_position_cap_pct=0.05)  # $5,000 cap on $100k capital
    result = sizing.size_candidate(
        "AAPL", price=100.0, cfg=cfg, book={}, base_dollar_override=50_000.0,
    )
    assert result.dollar_size <= 5_000.0 + 100.0  # within one share of the cap


def test_size_candidate_respects_sector_cap():
    cfg = _cfg(per_sector_cap_pct=0.10)  # $10k sector cap
    book = {"MSFT": sizing.OpenPosition("MSFT", 9_500, 95, sector="tech")}
    result = sizing.size_candidate(
        "AAPL", price=100.0, cfg=cfg, book=book, sector="tech",
        base_dollar_override=20_000.0,
    )
    assert result.dollar_size <= 500.0 + 100.0  # only ~$500 of sector headroom left


def test_size_candidate_respects_theme_cap():
    cfg = _cfg(per_theme_cap_pct=0.08)
    book = {"MSFT": sizing.OpenPosition("MSFT", 7_500, 75, theme="ai_infra")}
    result = sizing.size_candidate(
        "NVDA", price=50.0, cfg=cfg, book=book, theme="ai_infra",
        base_dollar_override=20_000.0,
    )
    assert result.dollar_size <= 500.0 + 50.0


def test_size_candidate_respects_liquidity_cap():
    cfg = _cfg(adv_participation_cap_pct=0.01)
    result = sizing.size_candidate(
        "SMALLCAP", price=10.0, cfg=cfg, book={}, adv_dollars=100_000.0,
        base_dollar_override=50_000.0,
    )
    assert result.dollar_size <= 1_000.0 + 10.0  # 1% of $100k ADV


def test_size_candidate_zero_when_group_cap_exhausted():
    cfg = _cfg(per_sector_cap_pct=0.10)
    book = {"MSFT": sizing.OpenPosition("MSFT", 10_000, 100, sector="tech")}  # cap fully used
    result = sizing.size_candidate(
        "AAPL", price=100.0, cfg=cfg, book=book, sector="tech",
        base_dollar_override=20_000.0,
    )
    assert result.shares == 0
    assert result.dollar_size == 0.0


def test_size_candidate_diagnostics_trace_each_stage():
    cfg = _cfg()
    result = sizing.size_candidate("AAPL", price=100.0, cfg=cfg, book={}, base_dollar_override=9_000.0)
    for key in ("base_dollar", "after_corr_penalty", "after_position_cap",
                "after_sector_cap", "after_theme_cap", "shares", "final_dollar"):
        assert key in result.diagnostics
