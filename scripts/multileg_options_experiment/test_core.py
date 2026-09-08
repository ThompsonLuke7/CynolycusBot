from __future__ import annotations

import pandas as pd

from scripts.multileg_options_experiment.core import (
    LegSpec,
    build_structures,
    capital_required,
    structure_return,
    validate_leg_bars,
)


def contract(symbol: str, strike: float, right: str, expiry: str = "2026-10-16") -> dict:
    return {"symbol": symbol, "strike_price": str(strike), "type": right, "expiration_date": expiry}


def test_build_structures_does_not_collapse_duplicate_strikes() -> None:
    chain = [
        contract("C100", 100, "call"), contract("P100", 100, "put"),
        contract("C105", 105, "call"), contract("C110", 110, "call"),
        contract("C115", 115, "call"), contract("P95", 95, "put"),
        contract("P90", 90, "put"),
    ]
    got = build_structures("XYZ", 100.0, "C105", chain)
    assert set(("long_shares", "long_call", "bull_call_debit", "iron_condor", "iron_butterfly")) <= set(got)
    assert [x.quantity for x in got["call_butterfly"]] == [1.0, -2.0, 1.0]


def test_leg_validation_enforces_derivative_direction() -> None:
    dates = pd.date_range("2026-01-02 16:00", periods=8, freq="B", tz="UTC")
    underlying = pd.DataFrame({"close": range(100, 108)}, index=dates.date.astype(str))
    bars = [{"t": d.isoformat(), "c": float(2 + i)} for i, d in enumerate(dates)]
    assert validate_leg_bars(bars, underlying, "C")[0]
    ok, stats = validate_leg_bars(bars, underlying, "P")
    assert not ok and stats["reason"] == "corr_wrong_or_low"


def test_vertical_capital_and_per_leg_cost_are_exact() -> None:
    legs = (
        LegSpec("C100", "C", 100.0, 1.0, "2026-10-16"),
        LegSpec("C110", "C", 110.0, -1.0, "2026-10-16"),
    )
    entry = {"C100": 5.0, "C110": 2.0}
    capital, source = capital_required(legs, entry, 100.0)
    assert capital == 300.0
    assert source == "exact_expiry_bound"
    ret, detail = structure_return(legs, entry, {"C100": 8.0, "C110": 3.0}, 100.0, 105.0, 0.08)
    assert detail["gross_pnl"] == 200.0
    assert detail["cost"] == 34.6
    assert round(ret, 6) == round(165.4 / 300.0, 6)


def test_unbounded_ratio_spread_is_rejected() -> None:
    legs = (
        LegSpec("C100", "C", 100.0, 1.0, "2026-10-16"),
        LegSpec("C110", "C", 110.0, -2.0, "2026-10-16"),
    )
    assert capital_required(legs, {"C100": 5.0, "C110": 2.0}, 100.0)[0] is None
