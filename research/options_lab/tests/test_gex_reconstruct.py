"""Tests for research/options_lab/gex_reconstruct.py.

All synthetic, no network. Covers: expiration-window scope selection,
the three OI-variant reconstructions (terminal_oi / volume_accumulated /
volume_proxy) against hand-computed expectations, gamma computed from a
known BSM implied vol, and the full snapshot assembly (total_gex/net_gex,
call_wall/put_wall, gamma_flip, dealer_bias) against a synthetic chain
whose correct answer is known by construction.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research.options_lab import gex_reconstruct as gr
from research.options_lab import pricing


# --------------------------------------------------------------------------
# select_expiration_window
# --------------------------------------------------------------------------


def test_select_expiration_window_next_expiration():
    expiries = ["2026-08-07", "2026-08-14", "2026-08-21", "2026-09-18"]
    out = gr.select_expiration_window(expiries, "2026-08-01", scope="next_expiration")
    assert out == ["2026-08-07"]


def test_select_expiration_window_daily_week_prefers_non_monthly_friday():
    # 2026-08-21 is the monthly (3rd Friday); 2026-08-07/14 are weeklies.
    expiries = ["2026-08-07", "2026-08-14", "2026-08-21"]
    out = gr.select_expiration_window(expiries, "2026-08-01", scope="daily_week")
    assert out == ["2026-08-07"]


def test_select_expiration_window_through_month_stops_at_monthly():
    expiries = ["2026-08-07", "2026-08-14", "2026-08-21", "2026-09-18"]
    out = gr.select_expiration_window(expiries, "2026-08-01", scope="through_month")
    assert out == ["2026-08-07", "2026-08-14", "2026-08-21"]


def test_select_expiration_window_two_months_uses_calendar_cutoff():
    expiries = ["2026-08-07", "2026-09-18", "2026-10-16", "2026-11-20"]
    out = gr.select_expiration_window(expiries, "2026-08-01", scope="two_months")
    # cutoff = 2026-10-01; 2026-10-16 excluded, 2026-11-20 excluded
    assert out == ["2026-08-07", "2026-09-18"]


def test_select_expiration_window_excludes_past_dates():
    expiries = ["2026-07-01", "2026-08-07"]
    out = gr.select_expiration_window(expiries, "2026-08-01", scope="next_expiration")
    assert out == ["2026-08-07"]


def test_select_expiration_window_empty_when_no_future_expiries():
    assert gr.select_expiration_window(["2026-01-01"], "2026-08-01", scope="next_expiration") == []


def test_select_expiration_window_rejects_unknown_scope():
    with pytest.raises(ValueError):
        gr.select_expiration_window(["2026-08-07"], "2026-08-01", scope="bogus")


# --------------------------------------------------------------------------
# compute_oi_variants
# --------------------------------------------------------------------------


def _volume_row(osi, d, v):
    return {"osi_symbol": osi, "date": d, "volume": v}


def test_compute_oi_variants_terminal_oi_passthrough():
    contracts = pd.DataFrame(
        [{"osi_symbol": "A250C", "expiry": "2026-08-21", "terminal_oi": 500.0}]
    )
    vol = pd.DataFrame([_volume_row("A250C", "2026-08-01", 10.0)])
    out = gr.compute_oi_variants(contracts, vol, asof="2026-08-01")
    assert out.loc[0, "oi_terminal_oi"] == 500.0


def test_compute_oi_variants_volume_accumulated_subtracts_future_volume():
    # terminal OI = 500. Volume after asof (exclusive) through expiry sums
    # to 50+30 = 80 -> reconstructed OI at asof = 500 - 80 = 420.
    contracts = pd.DataFrame(
        [{"osi_symbol": "A250C", "expiry": "2026-08-10", "terminal_oi": 500.0}]
    )
    vol = pd.DataFrame(
        [
            _volume_row("A250C", "2026-08-01", 999.0),  # on/before asof: irrelevant to accumulated calc
            _volume_row("A250C", "2026-08-05", 50.0),   # after asof, before expiry
            _volume_row("A250C", "2026-08-08", 30.0),   # after asof, before expiry
        ]
    )
    out = gr.compute_oi_variants(contracts, vol, asof="2026-08-01")
    assert out.loc[0, "volume_after_date"] == pytest.approx(50.0 + 30.0)
    assert out.loc[0, "oi_volume_accumulated"] == pytest.approx(500.0 - (50.0 + 30.0))


def test_compute_oi_variants_volume_accumulated_clips_at_zero():
    contracts = pd.DataFrame(
        [{"osi_symbol": "A250C", "expiry": "2026-08-10", "terminal_oi": 10.0}]
    )
    vol = pd.DataFrame([_volume_row("A250C", "2026-08-05", 1000.0)])
    out = gr.compute_oi_variants(contracts, vol, asof="2026-08-01")
    assert out.loc[0, "oi_volume_accumulated"] == 0.0


def test_compute_oi_variants_volume_accumulated_nan_when_terminal_missing():
    contracts = pd.DataFrame(
        [{"osi_symbol": "A250C", "expiry": "2026-08-10", "terminal_oi": None}]
    )
    vol = pd.DataFrame([_volume_row("A250C", "2026-08-05", 5.0)])
    out = gr.compute_oi_variants(contracts, vol, asof="2026-08-01")
    assert pd.isna(out.loc[0, "oi_volume_accumulated"])


def test_compute_oi_variants_volume_proxy_is_cumulative_to_date():
    contracts = pd.DataFrame(
        [{"osi_symbol": "A250C", "expiry": "2026-08-10", "terminal_oi": None}]
    )
    vol = pd.DataFrame(
        [
            _volume_row("A250C", "2026-07-28", 10.0),
            _volume_row("A250C", "2026-08-01", 5.0),
            _volume_row("A250C", "2026-08-05", 999.0),  # after asof: excluded from proxy
        ]
    )
    out = gr.compute_oi_variants(contracts, vol, asof="2026-08-01")
    assert out.loc[0, "oi_volume_proxy"] == pytest.approx(15.0)
    assert out.loc[0, "day_volume"] == pytest.approx(5.0)


def test_compute_oi_variants_no_volume_rows_defaults_to_zero():
    contracts = pd.DataFrame(
        [{"osi_symbol": "A250C", "expiry": "2026-08-10", "terminal_oi": 100.0}]
    )
    empty_vol = pd.DataFrame(columns=["osi_symbol", "date", "volume"])
    out = gr.compute_oi_variants(contracts, empty_vol, asof="2026-08-01")
    assert out.loc[0, "day_volume"] == 0.0
    assert out.loc[0, "oi_volume_proxy"] == 0.0
    assert out.loc[0, "oi_volume_accumulated"] == 100.0


def test_compute_oi_variants_requires_columns():
    with pytest.raises(ValueError):
        gr.compute_oi_variants(pd.DataFrame({"osi_symbol": ["A"]}), pd.DataFrame({"osi_symbol": [], "date": [], "volume": []}), asof="2026-08-01")


# --------------------------------------------------------------------------
# compute_gamma
# --------------------------------------------------------------------------


def test_compute_gamma_matches_bsm_greeks_directly():
    spot, K, T, r, q, sigma = 100.0, 100.0, 30 / 365.25, 0.04, 0.0, 0.30
    contracts_iv = pd.DataFrame([{"strike": K, "right": "C", "T": T, "iv": sigma}])
    out = gr.compute_gamma(contracts_iv, spot=spot, r=r, q=q)
    expected = pricing.bsm_greeks(spot, K, T, r, q, sigma, "C").gamma
    assert out.loc[0, "gamma"] == pytest.approx(expected)


def test_compute_gamma_zero_for_missing_iv_or_expired():
    contracts_iv = pd.DataFrame(
        [
            {"strike": 100.0, "right": "C", "T": 30 / 365.25, "iv": None},
            {"strike": 100.0, "right": "P", "T": 0.0, "iv": 0.3},
        ]
    )
    out = gr.compute_gamma(contracts_iv, spot=100.0, r=0.04, q=0.0)
    assert (out["gamma"] == 0.0).all()


def test_compute_gamma_requires_columns():
    with pytest.raises(ValueError):
        gr.compute_gamma(pd.DataFrame({"strike": [100.0]}), spot=100.0, r=0.04)


# --------------------------------------------------------------------------
# assemble_snapshot_row -- known-by-construction synthetic chain
# --------------------------------------------------------------------------


def _known_chain():
    """Three strikes around spot=100. Gamma chosen as simple round numbers
    so call_gex/put_gex/net_gex/dealer_bias/gamma_flip can be hand-verified.

    call_gex = call_oi * gamma * 100 * spot
    put_gex  = -put_oi * gamma * 100 * spot   (dealer short calls/long puts
               convention from strategies/dealer_positioning/levels.py)
    """
    spot = 100.0
    rows = pd.DataFrame(
        [
            # strike 90 (below spot): only a put -> put_gex negative
            {"strike": 90.0, "option_type": "P", "open_interest": 200.0, "volume": 20.0, "gamma": 0.02},
            # strike 100 (at spot): call + put
            {"strike": 100.0, "option_type": "C", "open_interest": 100.0, "volume": 10.0, "gamma": 0.05},
            {"strike": 100.0, "option_type": "P", "open_interest": 50.0, "volume": 5.0, "gamma": 0.05},
            # strike 110 (above spot): only a call -> largest positive call_gex -> call wall
            {"strike": 110.0, "option_type": "C", "open_interest": 300.0, "volume": 30.0, "gamma": 0.03},
        ]
    )
    return spot, rows


def test_assemble_snapshot_row_total_gex_matches_hand_calculation():
    spot, rows = _known_chain()
    call_gex_90 = 0.0
    put_gex_90 = -200.0 * 0.02 * 100.0 * spot
    call_gex_100 = 100.0 * 0.05 * 100.0 * spot
    put_gex_100 = -50.0 * 0.05 * 100.0 * spot
    call_gex_110 = 300.0 * 0.03 * 100.0 * spot
    expected_total = call_gex_90 + put_gex_90 + call_gex_100 + put_gex_100 + call_gex_110

    out = gr.assemble_snapshot_row(rows, symbol="TEST", date_str="2026-08-01", spot=spot, oi_source="terminal_oi")
    assert out["total_gex"] == pytest.approx(expected_total)
    assert out["net_gex"] == pytest.approx(expected_total)


def test_assemble_snapshot_row_call_wall_and_put_wall():
    spot, rows = _known_chain()
    out = gr.assemble_snapshot_row(rows, symbol="TEST", date_str="2026-08-01", spot=spot, oi_source="terminal_oi")
    # Only strike > spot with a call is 110 -> call wall.
    assert out["call_wall"] == pytest.approx(110.0)
    # Only strike < spot with a put is 90 -> put wall.
    assert out["put_wall"] == pytest.approx(90.0)


def test_assemble_snapshot_row_pct_to_levels():
    spot, rows = _known_chain()
    out = gr.assemble_snapshot_row(rows, symbol="TEST", date_str="2026-08-01", spot=spot, oi_source="terminal_oi")
    assert out["pct_to_call_wall"] == pytest.approx((110.0 - spot) / spot)
    if out["gamma_flip"] is not None:
        assert out["pct_to_gamma_flip"] == pytest.approx((out["gamma_flip"] - spot) / spot)


def test_assemble_snapshot_row_dealer_bias_sign():
    # Heavy positive gex above spot (call wall at 110), smaller negative
    # gex below (put at 90) -> positive net_gex clipped-sum above should
    # dominate -> dealer_bias > 0.
    spot, rows = _known_chain()
    out = gr.assemble_snapshot_row(rows, symbol="TEST", date_str="2026-08-01", spot=spot, oi_source="terminal_oi")
    assert out["dealer_bias"] > 0.0


def test_assemble_snapshot_row_total_oi_and_volume_default_sums():
    spot, rows = _known_chain()
    out = gr.assemble_snapshot_row(rows, symbol="TEST", date_str="2026-08-01", spot=spot, oi_source="terminal_oi")
    assert out["total_oi"] == pytest.approx(200.0 + 100.0 + 50.0 + 300.0)
    assert out["total_volume"] == pytest.approx(20.0 + 10.0 + 5.0 + 30.0)


def test_assemble_snapshot_row_oi_source_recorded_and_validated():
    spot, rows = _known_chain()
    out = gr.assemble_snapshot_row(rows, symbol="TEST", date_str="2026-08-01", spot=spot, oi_source="volume_proxy")
    assert out["oi_source"] == "volume_proxy"
    with pytest.raises(ValueError):
        gr.assemble_snapshot_row(rows, symbol="TEST", date_str="2026-08-01", spot=spot, oi_source="bogus")


def test_assemble_snapshot_row_gamma_flip_between_signed_strikes():
    # net_gex at 90 is negative (put only), at 100 is (call+put) mixed sign
    # depending on magnitudes, at 110 positive (call only) -- gamma_flip
    # should land strictly between the sign change nearest spot.
    spot, rows = _known_chain()
    out = gr.assemble_snapshot_row(rows, symbol="TEST", date_str="2026-08-01", spot=spot, oi_source="terminal_oi")
    flip = out["gamma_flip"]
    assert flip is not None
    assert 90.0 <= flip <= 110.0


# --------------------------------------------------------------------------
# build_rows_for_variant
# --------------------------------------------------------------------------


def test_build_rows_for_variant_drops_unknown_and_zero_oi():
    contracts = pd.DataFrame(
        [
            {"osi_symbol": "A", "strike": 100.0, "right": "C", "gamma": 0.05,
             "oi_terminal_oi": 50.0, "oi_volume_accumulated": np.nan, "oi_volume_proxy": 0.0, "day_volume": 5.0},
            {"osi_symbol": "B", "strike": 110.0, "right": "P", "gamma": 0.02,
             "oi_terminal_oi": np.nan, "oi_volume_accumulated": np.nan, "oi_volume_proxy": 10.0, "day_volume": 1.0},
        ]
    )
    terminal_rows = gr.build_rows_for_variant(contracts, oi_source="terminal_oi", asof="2026-08-01")
    assert len(terminal_rows) == 1
    assert terminal_rows.iloc[0]["strike"] == 100.0
    assert terminal_rows.iloc[0]["option_type"] == "C"

    proxy_rows = gr.build_rows_for_variant(contracts, oi_source="volume_proxy", asof="2026-08-01")
    assert len(proxy_rows) == 1
    assert proxy_rows.iloc[0]["strike"] == 110.0

    accumulated_rows = gr.build_rows_for_variant(contracts, oi_source="volume_accumulated", asof="2026-08-01")
    assert accumulated_rows.empty


def test_build_rows_for_variant_rejects_unknown_source():
    contracts = pd.DataFrame(
        [{"osi_symbol": "A", "strike": 100.0, "right": "C", "gamma": 0.05, "day_volume": 1.0}]
    )
    with pytest.raises(ValueError):
        gr.build_rows_for_variant(contracts, oi_source="bogus", asof="2026-08-01")


# --------------------------------------------------------------------------
# strike_increment
# --------------------------------------------------------------------------


def test_strike_increment_median_gap():
    assert gr.strike_increment([90.0, 95.0, 100.0, 110.0]) == pytest.approx(5.0)


def test_strike_increment_none_for_single_strike():
    assert gr.strike_increment([100.0]) is None


def test_strike_increment_from_dataframe():
    df = pd.DataFrame({"strike": [10.0, 12.5, 15.0]})
    assert gr.strike_increment(df) == pytest.approx(2.5)
