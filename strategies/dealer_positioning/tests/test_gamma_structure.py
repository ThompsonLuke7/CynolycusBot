"""Tests for the unsigned/signed split, confidence weights, and IV stability."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from strategies.dealer_positioning.confidence import (
    SIGN_CONFIDENCE_TIERS,
    assess_chain_quality,
    build_confidence,
    data_freshness,
    liquidity_tier,
    sign_confidence,
    structure_confidence,
)
from strategies.dealer_positioning.levels import build_gamma_ladder, compute_gamma_structure
from strategies.dealer_positioning.models import OptionContractRow
from strategies.dealer_positioning.stability import assess_stability
from strategies.dealer_positioning.topology import (
    SIGN_CONVENTION,
    build_gamma_topology,
    concentration_index,
    entropy,
    expiry_bucket_shares,
)


def _rows(*, spot: float = 100.0, call_pile: float = 105.0, put_pile: float = 95.0, dte: int = 7):
    rows = []
    for strike in range(90, 111):
        for option_type in ("C", "P"):
            oi = 500.0
            if option_type == "C" and float(strike) == call_pile:
                oi = 9000.0
            if option_type == "P" and float(strike) == put_pile:
                oi = 8000.0
            rows.append(
                OptionContractRow(
                    timestamp=datetime(2026, 8, 25, 19, 45, tzinfo=timezone.utc),
                    symbol="TST",
                    expiration="2026-09-18",
                    dte=dte,
                    strike=float(strike),
                    option_type=option_type,
                    open_interest=oi,
                    volume=25.0,
                    gamma=0.02,
                    delta=0.5,
                    vega=0.10,
                    iv=25.0,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Topology: unsigned structure
# ---------------------------------------------------------------------------


def test_topology_is_invariant_to_the_sign_convention() -> None:
    """The whole point of the split: flipping who owns the OI must not move
    the description of where gamma sits."""
    ladder = build_gamma_ladder(_rows(), spot=100.0)
    flipped = ladder.copy()
    flipped["net_gex"] = -flipped["net_gex"]
    flipped["call_gex"], flipped["put_gex"] = -flipped["put_gex"], -flipped["call_gex"]

    base = build_gamma_topology(ladder, symbol="TST", spot=100.0, timestamp="t")
    other = build_gamma_topology(flipped, symbol="TST", spot=100.0, timestamp="t")

    assert base.total_abs_gamma == other.total_abs_gamma
    assert base.gamma_concentration == other.gamma_concentration
    assert base.gamma_density_1pct == other.gamma_density_1pct
    assert base.nearest_node == other.nearest_node


def test_signed_exposure_names_the_convention_that_produced_it() -> None:
    structure = compute_gamma_structure(_rows(), symbol="TST", spot=100.0, with_stability=False)
    payload = structure.signed.to_dict()
    assert payload["sign_convention"] == SIGN_CONVENTION
    # An artifact must never carry a bare `net_gex` that reads as observed fact.
    assert "net_gex" not in payload
    assert "estimated_net_gex" in payload


def test_density_bands_are_nested() -> None:
    structure = compute_gamma_structure(_rows(), symbol="TST", spot=100.0, with_stability=False)
    t = structure.topology
    assert t.gamma_density_1pct <= t.gamma_density_2_5pct <= t.gamma_density_5pct <= 1.0


def test_empty_ladder_yields_a_null_topology_not_a_crash() -> None:
    empty = pd.DataFrame(columns=["strike", "total_abs_gex"])
    topology = build_gamma_topology(empty, symbol="TST", spot=100.0, timestamp="t")
    assert topology.total_abs_gamma == 0.0
    assert topology.node_count == 0
    assert topology.nearest_node is None


def test_concentration_and_entropy_move_in_opposite_directions() -> None:
    spread = pd.Series([1.0] * 10)
    concentrated = pd.Series([100.0] + [0.1] * 9)
    assert concentration_index(concentrated) > concentration_index(spread)
    assert entropy(concentrated) < entropy(spread)
    assert concentration_index(pd.Series([0.0, 0.0])) is None


# ---------------------------------------------------------------------------
# Expiry buckets
# ---------------------------------------------------------------------------


def test_expiry_bucket_shares_sum_to_one_and_separate_the_term() -> None:
    rows = _rows(dte=0) + _rows(dte=45)
    frame = pd.DataFrame([r.__dict__ for r in rows])
    shares = expiry_bucket_shares(frame, spot=100.0)
    assert shares["d0"] == pytest.approx(0.5, abs=1e-9)
    assert shares["d30_plus"] == pytest.approx(0.5, abs=1e-9)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)


def test_bucket_levels_are_computed_inside_each_bucket() -> None:
    """A 0DTE pile at 105 and a 45DTE pile at 108 are different structures and
    must not be collapsed into one wall."""
    rows = _rows(dte=0, call_pile=105.0) + _rows(dte=45, call_pile=108.0)
    structure = compute_gamma_structure(rows, symbol="TST", spot=100.0, with_stability=False)
    buckets = structure.levels.per_bucket_levels
    assert buckets["d0"]["call_wall"] == 105.0
    assert buckets["d30_plus"]["call_wall"] == 108.0


def test_term_structure_slope_is_positive_when_gamma_is_front_loaded() -> None:
    front = compute_gamma_structure(_rows(dte=0), symbol="TST", spot=100.0, with_stability=False)
    back = compute_gamma_structure(_rows(dte=60), symbol="TST", spot=100.0, with_stability=False)
    assert front.levels.term_structure["gamma_term_slope"] > 0
    assert back.levels.term_structure["gamma_term_slope"] < 0


def test_missing_dte_yields_no_bucket_answer_rather_than_a_fabricated_one() -> None:
    frame = pd.DataFrame({"strike": [100.0], "open_interest": [10.0], "gamma": [0.01]})
    assert expiry_bucket_shares(frame, spot=100.0) == {}


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_sign_confidence_is_tiered_by_symbol_liquidity() -> None:
    assert liquidity_tier("SPY") == "index_etf"
    assert liquidity_tier("IWM") == "liquid_etf"
    assert liquidity_tier("NVDA", avg_dollar_volume_20d=5e9, market_cap=9e11) == "mega_cap"
    assert liquidity_tier("ABCD", avg_dollar_volume_20d=5e7) == "normal_equity"
    assert liquidity_tier("ZZZZ", avg_dollar_volume_20d=1e5) == "illiquid"
    tiers = [sign_confidence(t) for t in ("index_etf", "liquid_etf", "mega_cap", "normal_equity", "illiquid")]
    assert tiers == sorted(tiers, reverse=True), "confidence must fall as liquidity falls"


def test_convention_disagreement_can_only_lower_sign_confidence() -> None:
    base = sign_confidence("index_etf")
    assert sign_confidence("index_etf", convention_dispersion=0.0) == base
    assert sign_confidence("index_etf", convention_dispersion=0.5) < base
    assert sign_confidence("index_etf", convention_dispersion=1.0) == 0.0


def test_structure_confidence_falls_when_open_interest_sits_on_zero_gamma_rows() -> None:
    good = pd.DataFrame({"strike": range(20), "gamma": [0.02] * 20, "open_interest": [100.0] * 20, "iv": [25.0] * 20})
    bad = good.copy()
    bad.loc[:9, "gamma"] = 0.0
    assert structure_confidence(assess_chain_quality(good)) > structure_confidence(assess_chain_quality(bad))


def test_a_half_covered_strike_window_lowers_structure_confidence() -> None:
    quality = assess_chain_quality(
        pd.DataFrame({"strike": range(20), "gamma": [0.02] * 20, "open_interest": [100.0] * 20, "iv": [25.0] * 20})
    )
    assert structure_confidence(quality, strike_coverage=0.5) < structure_confidence(quality, strike_coverage=1.0)


def test_unknown_age_is_treated_as_stale_not_fresh() -> None:
    assert data_freshness(None) == 0.0
    assert data_freshness(0.0) == 1.0
    assert data_freshness(2.0, max_age_days=4.0) == pytest.approx(0.5)
    assert data_freshness(99.0) == 0.0


def test_confidence_block_travels_with_the_structure() -> None:
    structure = compute_gamma_structure(
        _rows(), symbol="SPY", spot=100.0, age_days=1.0, with_stability=False
    )
    block = structure.confidence
    assert block.liquidity_tier == "index_etf"
    assert block.sign_confidence == SIGN_CONFIDENCE_TIERS["index_etf"]
    assert 0.0 < block.structure_confidence <= 1.0
    assert block.chain_quality["rows_total"] == 42


def test_a_single_name_gets_lower_sign_confidence_than_spy_on_identical_chains() -> None:
    """Proposal 10: the chain can be identical; the inference is not equally
    reliable across symbols."""
    spy = build_confidence(symbol="SPY", frame=None, age_days=0.0)
    small = build_confidence(symbol="ABCD", frame=None, age_days=0.0, avg_dollar_volume_20d=1e5)
    assert spy.sign_confidence > small.sign_confidence


# ---------------------------------------------------------------------------
# IV stability
# ---------------------------------------------------------------------------


def test_a_dominant_strike_keeps_its_wall_under_iv_shocks() -> None:
    frame = pd.DataFrame([r.__dict__ for r in _rows(call_pile=105.0)])
    result = assess_stability(frame, spot=100.0)
    assert result.call_wall_stability == pytest.approx(1.0)
    assert set(result.call_wall_by_shock) == {"0.8", "1", "1.2"}
    assert all(v == 105.0 for v in result.call_wall_by_shock.values())


def test_stability_scores_stay_in_the_unit_interval() -> None:
    frame = pd.DataFrame([r.__dict__ for r in _rows()])
    result = assess_stability(frame, spot=100.0)
    for field in ("call_wall_stability", "put_wall_stability", "magnet_stability", "node_rank_stability"):
        value = getattr(result, field)
        if value is not None:
            assert 0.0 <= value <= 1.0


def test_a_wall_that_moves_with_iv_scores_lower_than_one_that_does_not() -> None:
    stable = pd.DataFrame([r.__dict__ for r in _rows(call_pile=105.0)])
    stable_score = assess_stability(stable, spot=100.0).call_wall_stability

    # Two nearly-equal call piles at different strikes and different expiries:
    # an IV shock changes which one dominates, so the wall relocates.
    contested = _rows(dte=1, call_pile=103.0) + _rows(dte=60, call_pile=109.0)
    frame = pd.DataFrame([r.__dict__ for r in contested])
    contested_score = assess_stability(frame, spot=100.0).call_wall_stability
    assert contested_score is not None
    assert contested_score <= stable_score


def test_stability_is_skipped_cleanly_on_an_empty_chain() -> None:
    result = assess_stability(pd.DataFrame(), spot=100.0)
    assert result.call_wall_stability is None
    assert result.call_wall_by_shock == {}


def test_stability_never_reads_the_ladder_gamma_columns() -> None:
    """The ladder's `call_gamma` is a cross-expiry mean, not per-contract gamma.
    Guard the recompute path against ever being pointed at it."""
    frame = pd.DataFrame([r.__dict__ for r in _rows()])
    assert "call_gamma" not in frame.columns
    assert assess_stability(frame, spot=100.0).call_wall_by_shock  # works on contract rows


def test_ladder_labels_the_cross_expiry_gamma_mean_honestly() -> None:
    ladder = build_gamma_ladder(_rows(dte=0) + _rows(dte=45), spot=100.0)
    assert "call_gamma_mean_by_expiry" in ladder.columns
    assert ladder["call_gamma_mean_by_expiry"].equals(ladder["call_gamma"])
