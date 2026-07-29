"""Regression tests pinning the credit-leg duration fix (see the "Credit
leg" section of daily_regime.py's module docstring).

HYG/LQD is duration-contaminated: LQD's long duration (~8-9y) makes the
ratio track the rate cycle as much as credit spreads, and it INVERTED sign
during the real 2022 drawdown (HYG/LQD printed risk-ON while SPY fell 25%).
credit_risk_z, risk_appetite_z's credit leg, and liquidity_stress_z's credit
term must all be driven by the duration-matched HYG/IEI ratio instead.

Two independent checks:
  1. A synthetic "rate shock" fixture reproducing the real mechanism (a
     long-duration instrument crushed by a rate move, contaminating a
     LQD-style ratio, while a duration-matched instrument isolates the
     credit-spread effect) — hermetic, no network, always runs.
  2. The real cached 2022 history (HYG/LQD/IEI), gated behind a skip if that
     cache is absent in the current environment — this is the literal,
     numbers-verified regression the fix was made for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.market_regime.config import CREDIT_DENOMINATOR
from signals.market_regime.daily_regime import build_daily_regime
from signals.market_regime.tests.conftest import (
    build_full_universe_bars,
    make_bars,
    make_loader,
    random_walk_close,
    session_calendar,
)

WINDOW = 20
MIN_PERIODS = 20
STRESS_START = 150
N_SESSIONS = 220


def _build_rate_shock_fixture():
    """A pre-stress random-walk regime, then a 'rate shock' window where a
    long-duration instrument (LQD-analog) is crushed far more than a
    duration-matched instrument (IEI-analog) or HYG — mirroring 2022's
    actual mechanism (duration, not credit spread, dominated HYG/LQD)."""
    dates, bars = build_full_universe_bars(start="2021-01-04", n=N_SESSIONS, seed_base=500)

    rng = np.random.default_rng(501)
    n_stress = N_SESSIONS - STRESS_START

    # Pure "rate shock" daily drift applied to every rate-sensitive instrument,
    # scaled by an approximate duration multiplier (LQD ~8y, IEI ~4.5y,
    # HYG ~3.5y) — this alone should make HYG "outperform" LQD (duration
    # illusion of credit health) while roughly tracking IEI.
    rate_shock_daily = -0.0025
    lqd_duration_mult = 8.0
    iei_duration_mult = 4.5
    hyg_duration_mult = 3.5
    # HYG also carries a genuine incremental credit-spread-widening effect
    # that duration alone does not explain — this is the "real" signal a
    # credit-risk feature should pick up. Sized so HYG underperforms the
    # duration-matched IEI (correct risk-off) while still outperforming the
    # long-duration LQD (preserving the real-world HYG/LQD contamination):
    # duration-only gap vs IEI is (4.5-3.5)*rate_shock = -0.0025/day, and vs
    # LQD is (8.0-3.5)*rate_shock = -0.01125/day, so -0.004 sits between them.
    hyg_credit_spread_daily = -0.004

    def _apply_shock(bars_dict, ticker, duration_mult, extra_daily=0.0):
        close = bars_dict[ticker]["close"].to_numpy().copy()
        shocked = close.copy()
        level = close[STRESS_START - 1]
        for i in range(n_stress):
            level = level * (1 + rate_shock_daily * duration_mult + extra_daily)
            shocked[STRESS_START + i] = level
        bars_dict[ticker] = make_bars(dates=dates, close=shocked, volume=bars_dict[ticker]["volume"].to_numpy())

    _apply_shock(bars, "LQD", lqd_duration_mult)
    _apply_shock(bars, CREDIT_DENOMINATOR, iei_duration_mult)
    _apply_shock(bars, "HYG", hyg_duration_mult, extra_daily=hyg_credit_spread_daily)

    return dates, bars


def test_synthetic_rate_shock_credit_leg_reads_risk_off_via_iei_not_lqd():
    """During a rate-shock scenario that inverts HYG/LQD (duration illusion),
    the production credit leg (HYG/IEI) must correctly read risk-off
    (negative), while the plan's literal HYG/LQD ratio inverts positive —
    exactly the real 2022 mechanism. If the denominator is ever swapped back
    to LQD, credit_risk_z will start behaving like the `hyg_lqd_reference`
    series below and this assertion will fail."""
    dates, bars = _build_rate_shock_fixture()
    regime = build_daily_regime(
        loader=make_loader(bars),
        zscore_window=WINDOW,
        zscore_min_periods=MIN_PERIODS,
        component_window_short=5,
        component_window_long=8,
        excess_return_window=3,
    )

    stress_rows = regime.iloc[STRESS_START + 10:]  # skip the first few sessions of the shock to let it build up
    assert stress_rows["credit_risk_z"].notna().mean() > 0.9

    # Production credit_risk_z (HYG/IEI) must read risk-off during the shock.
    assert stress_rows["credit_risk_z"].mean() < -0.3

    # The diagnostic HYG/LQD column must show the SAME sign inversion that
    # the real 2022 data showed: it goes positive even though this is a
    # stress scenario, because it is dominated by LQD's larger duration.
    assert stress_rows["credit_risk_hyg_lqd_z"].mean() > 0.3

    # And risk_appetite_z's credit leg specifically must track the corrected
    # (negative) direction, not the diagnostic column's inverted one.
    assert stress_rows["risk_appetite_hyg_iei_z"].mean() < -0.3


@pytest.mark.parametrize("real_start,real_end,label", [
    ("2022-01-01", "2022-10-15", "2022 drawdown"),
])
def test_real_2022_drawdown_credit_leg_is_risk_off(real_start, real_end, label):
    """The literal regression this fix was made for: over the real 2022
    drawdown (SPY -25.4%), the duration-controlled credit_risk_z must be
    negative on average — not the duration-contaminated HYG/LQD reading of
    ~+1.5 sigma "risk-on" that the plan's literal formula produced.

    Skipped (not failed) if the real daily-bar cache for HYG/LQD/IEI isn't
    present in this environment — this test intentionally reads the actual
    on-disk cache (no network) rather than a synthetic fixture, since the
    thing being pinned is a real historical fact about real market data.
    """
    from strategies.momentum_expansion.data.load_bars import load_1d

    try:
        for t in ("HYG", "LQD", CREDIT_DENOMINATOR, "SPY"):
            load_1d(t)
    except FileNotFoundError:
        pytest.skip("real HYG/LQD/IEI/SPY daily-bar cache not present in this environment")

    regime = build_daily_regime()
    regime["date"] = pd.to_datetime(regime["date"])
    window = regime[(regime["date"] >= real_start) & (regime["date"] <= real_end)]
    if window["credit_risk_z"].notna().sum() < 20:
        pytest.skip(f"insufficient real post-warmup coverage over {label} in this cache")

    mean_z = window["credit_risk_z"].mean()
    mean_z_diag = window["credit_risk_hyg_lqd_z"].mean()

    assert mean_z < 0, (
        f"credit_risk_z (HYG/{CREDIT_DENOMINATOR}) over {label} averaged {mean_z:.2f} "
        f"(expected negative/risk-off) — the duration fix may have regressed."
    )
    # The diagnostic (duration-contaminated) column should still show the
    # historically-verified inversion, confirming this is the right window.
    assert mean_z_diag > 0, (
        f"credit_risk_hyg_lqd_z over {label} averaged {mean_z_diag:.2f}; expected the "
        f"known-inverted (~+1.5) duration-contaminated reading — check the window/cache."
    )
