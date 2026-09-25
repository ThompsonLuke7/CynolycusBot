"""Decision-time implied vol must reach the on-disk audit.

The whole point of this capture is that it is unrecoverable after the fact:
historical option marks are stale trade prints, not marks (see
research/options_experiment/10_RETRACTION_option_pnl_invalid.md), so an IV
series can only be built forward from the moment of the decision. A helper that
computes IV correctly but drops it before the ledger is written would be
indistinguishable from not having built this at all — so these tests assert on
the ORDER DICT and the AUDIT RECORD, not on the parser in isolation.
"""
from __future__ import annotations

from core.live_signal_audit import build_option_order_audit
from signals.meta_context.meta_ranker.options_exec import _snapshot_vol_surface


class TestSnapshotVolSurface:
    def test_reads_camelcase_iv_and_greeks(self):
        iv, greeks = _snapshot_vol_surface(
            {"impliedVolatility": 0.62, "greeks": {"delta": 0.47, "theta": -0.18, "vega": 0.09}}
        )
        assert iv == 0.62
        assert greeks == {"delta": 0.47, "theta": -0.18, "vega": 0.09}

    def test_reads_snake_case_iv(self):
        iv, _ = _snapshot_vol_surface({"implied_volatility": 0.35})
        assert iv == 0.35

    def test_missing_iv_is_none_not_zero(self):
        """A missing IV means UNKNOWN. Zero would read as 'implies no movement'."""
        iv, greeks = _snapshot_vol_surface({"greeks": {"delta": 0.5}})
        assert iv is None
        assert greeks == {"delta": 0.5}

    def test_nonpositive_and_nonfinite_iv_rejected(self):
        for bad in (0, -0.4, float("nan"), float("inf"), "abc", None):
            iv, _ = _snapshot_vol_surface({"impliedVolatility": bad})
            assert iv is None, f"{bad!r} should not parse as an IV"

    def test_unparseable_greek_is_dropped_not_fatal(self):
        iv, greeks = _snapshot_vol_surface(
            {"impliedVolatility": 0.5, "greeks": {"delta": 0.4, "gamma": "n/a", "vega": None}}
        )
        assert iv == 0.5
        assert greeks == {"delta": 0.4}

    def test_empty_snapshot_is_survivable(self):
        assert _snapshot_vol_surface({}) == (None, {})


class TestOrderAuditCarriesVolSurface:
    def test_iv_and_greeks_land_in_the_audit_record(self):
        rec = build_option_order_audit(
            signal_audit={"side": "long"},
            option_symbol="NBIS260918C00250000",
            route="call_option",
            side="buy",
            qty=4,
            underlying_price=243.66,
            strike=250.0,
            premium=13.86,
            delta=0.47,
            dte=17,
            expiration="2026-09-18",
            iv=0.615,
            greeks={"delta": 0.47, "theta": -0.21, "vega": 0.11},
        )
        assert rec["schema_version"] == "order_audit_v2"
        assert rec["iv"] == 0.615
        assert rec["greeks"]["theta"] == -0.21

    def test_absent_iv_is_recorded_as_unknown(self):
        """v1 rows predate the capture; readers must not read absence as zero."""
        rec = build_option_order_audit(
            signal_audit=None, option_symbol="X260918C00010000", route="call_option",
            side="buy", qty=1,
        )
        assert rec["iv"] is None
        assert rec["greeks"] is None

    def test_non_finite_iv_does_not_poison_the_record(self):
        rec = build_option_order_audit(
            signal_audit=None, option_symbol="X260918C00010000", route="call_option",
            side="buy", qty=1, iv=float("nan"), greeks={"delta": float("inf")},
        )
        assert rec["iv"] is None
        assert rec["greeks"] == {"delta": None}
