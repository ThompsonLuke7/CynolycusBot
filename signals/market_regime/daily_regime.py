"""Canonical daily market-regime table.

Builds the composites specified in
docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md §4 (WS-A):

  risk_appetite_z      = mean of trailing-252 z of log(XLY/XLP), log(IWM/SPY),
                          log(HYG/IEI), log(RSP/SPY)
  liquidity_stress_z   = mean of z(Amihud_20) - z(DollarVolume_20)
                          - z(log(HYG/IEI)) + z(RV20_SPY)
                          (rising value = MORE stress; note the sign flips
                          on the dollar-volume and credit legs)
  credit_risk_z        = trailing z of log(HYG/IEI)  (duration-controlled;
                          see "Credit leg" below — rising = credit
                          conditions IMPROVING, not negated)
  breadth_z            = trailing z of the fraction of the 11 sector ETFs
                          above their own 20d and 50d SMA
  sector_dispersion_z  = trailing z of the cross-sectional std of sector
                          21d excess return vs SPY
  spy_rv20_z           = trailing z of SPY's 20d realized vol
  spy_trend_state      = sign(EMA20(SPY) - EMA50(SPY))

Every composite carries its raw/leg components plus a companion
``*_n_components`` count and ``*_stale_days`` age (time-correctness contract,
plan §3). All rolling statistics are trailing-only; z-scores are NaN before
their explicit ``min_periods``, never zero-filled.

Credit leg: HYG/LQD is a duration-contaminated credit proxy, NOT a clean
credit-quality signal — do not "simplify" this back to the plan's literal
LQD denominator.
    HYG (high yield corporates) has ~3.5y effective duration; LQD
    (investment grade corporates) has ~8-9y duration. Their ratio therefore
    moves with the yield curve as much as with credit spreads. This was
    verified against the real cached history (trailing-252 z of the log
    ratio, same formula as production):

        2022 drawdown (SPY -25.4%, 2022-01-01..2022-10-15): log(HYG/LQD)
        z-mean = +1.53 (reads "credit is 1.5 sigma HEALTHY" during a bear
        market this table should be flagging as stressed) vs. log(HYG/SHY)
        z-mean = -2.18 and log(HYG/IEI) z-mean = -1.35 (both correctly
        risk-off).

    The 2022 selloff was a rate-shock (duration) event, not primarily a
    credit-spread-widening event: LQD's long duration got crushed by rising
    rates far more than HYG's, so HYG/LQD rose even as credit conditions
    were genuinely deteriorating. IEI (3-7y treasuries, ~4.5y duration — the
    closest duration match to HYG among cached ETFs) is used as the
    denominator instead, so the ratio isolates credit spread rather than
    duration. This affects three places: the standalone ``credit_risk_z``,
    one of the four ``risk_appetite_z`` legs, and the ``liquidity_stress_z``
    credit term — all three now use HYG/IEI. The original HYG/LQD z-score is
    still exposed, but only as the diagnostic ``credit_risk_hyg_lqd_z``
    column, which feeds none of the three composites.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategies.momentum_expansion.data.load_bars import load_1d

from .config import (
    COMPONENT_WINDOW_LONG,
    COMPONENT_WINDOW_SHORT,
    CREDIT_DENOMINATOR,
    EXCESS_RETURN_WINDOW,
    SECTOR_ETFS_LIST,
    SETTLE_MARGIN_MINUTES,
    ZSCORE_MIN_PERIODS,
    ZSCORE_WINDOW,
)
from .timeutil import align_with_staleness, available_at_for_dates, session_date_index, trailing_zscore

logger = logging.getLogger(__name__)


def _load_ohlcv(ticker: str, *, loader=load_1d) -> pd.DataFrame:
    """Session-date-indexed close/volume for one ticker, from cached daily bars."""
    df = loader(ticker)
    dates = session_date_index(df)
    out = pd.DataFrame(
        {"close": df["close"].to_numpy(), "volume": df["volume"].to_numpy()},
        index=dates,
    )
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    ratio = a / b
    return np.log(ratio.where(ratio > 0))


def _log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    ratio = close / close.shift(periods)
    return np.log(ratio.where(ratio > 0))


def _max_stale(*stale_series: pd.Series) -> pd.Series:
    return pd.concat(stale_series, axis=1).max(axis=1, skipna=True)


def build_daily_regime(
    *,
    loader=load_1d,
    zscore_window: int = ZSCORE_WINDOW,
    zscore_min_periods: int = ZSCORE_MIN_PERIODS,
    component_window_short: int = COMPONENT_WINDOW_SHORT,
    component_window_long: int = COMPONENT_WINDOW_LONG,
    excess_return_window: int = EXCESS_RETURN_WINDOW,
    settle_margin_minutes: int = SETTLE_MARGIN_MINUTES,
    sector_etfs: list[str] = SECTOR_ETFS_LIST,
) -> pd.DataFrame:
    """Build the canonical daily market-regime table (one row per SPY session).

    ``loader`` defaults to ``strategies.momentum_expansion.data.load_bars.load_1d``
    and is injectable so tests can supply synthetic fixtures with no network
    or on-disk dependency.
    """
    spy_ohlcv = _load_ohlcv("SPY", loader=loader)
    calendar = spy_ohlcv.index  # SPY trades essentially every session -> canonical calendar.

    def _aligned_close(ticker: str) -> tuple[pd.Series, pd.Series]:
        raw = _load_ohlcv(ticker, loader=loader)["close"]
        return align_with_staleness(raw, calendar)

    spy_close, spy_stale = align_with_staleness(spy_ohlcv["close"], calendar)
    spy_vol, _spy_vol_stale = align_with_staleness(spy_ohlcv["volume"], calendar)
    iwm, iwm_stale = _aligned_close("IWM")
    hyg, hyg_stale = _aligned_close("HYG")
    # Duration-matched credit denominator (see module docstring: HYG/LQD is a
    # duration-contaminated proxy that inverted sign in 2022). LQD is still
    # loaded, but ONLY to expose the original ratio as a diagnostic column —
    # it must not feed credit_risk_z, risk_appetite_z, or liquidity_stress_z.
    credit_denom, credit_denom_stale = _aligned_close(CREDIT_DENOMINATOR)
    lqd, lqd_stale = _aligned_close("LQD")
    rsp, rsp_stale = _aligned_close("RSP")
    xly, xly_stale = _aligned_close("XLY")
    xlp, xlp_stale = _aligned_close("XLP")

    sector_closes: dict[str, pd.Series] = {}
    sector_stales: dict[str, pd.Series] = {}
    for etf in sector_etfs:
        c, s = _aligned_close(etf)
        sector_closes[etf] = c
        sector_stales[etf] = s

    out = pd.DataFrame(index=calendar)
    out.index.name = "date"

    # ------------------------------------------------------------------
    # risk_appetite_z
    # ------------------------------------------------------------------
    z_xly_xlp = trailing_zscore(_log_ratio(xly, xlp), window=zscore_window, min_periods=zscore_min_periods)
    z_iwm_spy = trailing_zscore(_log_ratio(iwm, spy_close), window=zscore_window, min_periods=zscore_min_periods)
    # Duration-controlled credit ratio (HYG / IEI) — this is the one that
    # feeds credit_risk_z, risk_appetite_z, and liquidity_stress_z.
    credit_ratio = _log_ratio(hyg, credit_denom)
    z_credit = trailing_zscore(credit_ratio, window=zscore_window, min_periods=zscore_min_periods)
    # Diagnostic only: the plan's literal HYG/LQD ratio, duration-contaminated
    # (see module docstring). Exposed for comparison; feeds no composite.
    credit_ratio_hyg_lqd = _log_ratio(hyg, lqd)
    z_hyg_lqd_diag = trailing_zscore(credit_ratio_hyg_lqd, window=zscore_window, min_periods=zscore_min_periods)
    z_rsp_spy = trailing_zscore(_log_ratio(rsp, spy_close), window=zscore_window, min_periods=zscore_min_periods)

    out["risk_appetite_xly_xlp_z"] = z_xly_xlp
    out["risk_appetite_iwm_spy_z"] = z_iwm_spy
    out["risk_appetite_hyg_iei_z"] = z_credit
    out["risk_appetite_rsp_spy_z"] = z_rsp_spy

    risk_components = pd.concat(
        [z_xly_xlp.rename("a"), z_iwm_spy.rename("b"), z_credit.rename("c"), z_rsp_spy.rename("d")],
        axis=1,
    )
    out["risk_appetite_z"] = risk_components.mean(axis=1, skipna=True).where(
        risk_components.notna().any(axis=1)
    )
    out["risk_appetite_z_n_components"] = risk_components.notna().sum(axis=1)
    out["risk_appetite_z_stale_days"] = _max_stale(
        xly_stale, xlp_stale, iwm_stale, spy_stale, hyg_stale, credit_denom_stale, rsp_stale
    )

    # ------------------------------------------------------------------
    # liquidity_stress_z, spy_rv20_z, credit_risk_z (share SPY/HYG/IEI legs)
    # ------------------------------------------------------------------
    spy_dollar_vol = spy_close * spy_vol
    spy_ret1 = _log_return(spy_close)
    amihud_raw = spy_ret1.abs() / spy_dollar_vol.replace(0, np.nan)
    amihud_20 = amihud_raw.rolling(component_window_short, min_periods=component_window_short).mean()
    dollar_vol_20 = spy_dollar_vol.rolling(component_window_short, min_periods=component_window_short).mean()
    rv20_spy = spy_ret1.rolling(component_window_short, min_periods=component_window_short).std()

    z_amihud = trailing_zscore(amihud_20, window=zscore_window, min_periods=zscore_min_periods)
    z_dollar_vol = trailing_zscore(dollar_vol_20, window=zscore_window, min_periods=zscore_min_periods)
    z_rv20_spy = trailing_zscore(rv20_spy, window=zscore_window, min_periods=zscore_min_periods)

    out["liquidity_stress_amihud_z"] = z_amihud
    out["liquidity_stress_dollar_vol_z"] = z_dollar_vol
    out["liquidity_stress_credit_z"] = -z_credit
    out["liquidity_stress_rv20_z"] = z_rv20_spy

    liquidity_components = pd.concat(
        [z_amihud.rename("a"), (-z_dollar_vol).rename("b"), (-z_credit).rename("c"), z_rv20_spy.rename("d")],
        axis=1,
    )
    out["liquidity_stress_z"] = liquidity_components.mean(axis=1, skipna=True).where(
        liquidity_components.notna().any(axis=1)
    )
    out["liquidity_stress_z_n_components"] = liquidity_components.notna().sum(axis=1)
    out["liquidity_stress_z_stale_days"] = _max_stale(spy_stale, hyg_stale, credit_denom_stale)

    out["spy_rv20_raw"] = rv20_spy
    out["spy_rv20_z"] = z_rv20_spy
    out["spy_rv20_z_n_components"] = z_rv20_spy.notna().astype(int)
    out["spy_rv20_z_stale_days"] = spy_stale

    out["credit_risk_ratio"] = credit_ratio
    out["credit_risk_z"] = z_credit
    out["credit_risk_z_n_components"] = z_credit.notna().astype(int)
    out["credit_risk_z_stale_days"] = _max_stale(hyg_stale, credit_denom_stale)
    # Diagnostic-only: original (duration-contaminated) HYG/LQD z. Not used by
    # any composite above — see module docstring.
    out["credit_risk_hyg_lqd_z"] = z_hyg_lqd_diag

    # ------------------------------------------------------------------
    # breadth_z
    # ------------------------------------------------------------------
    above20_cols: dict[str, pd.Series] = {}
    above50_cols: dict[str, pd.Series] = {}
    for etf, close in sector_closes.items():
        sma20 = close.rolling(component_window_short, min_periods=component_window_short).mean()
        sma50 = close.rolling(component_window_long, min_periods=component_window_long).mean()
        above20_cols[etf] = (close > sma20).astype(float).where(close.notna() & sma20.notna())
        above50_cols[etf] = (close > sma50).astype(float).where(close.notna() & sma50.notna())
    above20_df = pd.DataFrame(above20_cols)
    above50_df = pd.DataFrame(above50_cols)
    breadth_raw = pd.concat([above20_df, above50_df], axis=1).mean(axis=1, skipna=True)
    breadth_valid_etfs = (above20_df.notna() & above50_df.notna()).sum(axis=1)
    breadth_raw = breadth_raw.where(breadth_valid_etfs > 0)

    out["breadth_raw"] = breadth_raw
    out["breadth_z"] = trailing_zscore(breadth_raw, window=zscore_window, min_periods=zscore_min_periods)
    out["breadth_z_n_components"] = breadth_valid_etfs
    out["breadth_z_stale_days"] = _max_stale(*sector_stales.values())

    # ------------------------------------------------------------------
    # sector_dispersion_z
    # ------------------------------------------------------------------
    spy_ret_excess = spy_close.pct_change(excess_return_window)
    excess21_cols: dict[str, pd.Series] = {}
    for etf, close in sector_closes.items():
        excess21_cols[etf] = close.pct_change(excess_return_window) - spy_ret_excess
    excess21_df = pd.DataFrame(excess21_cols)
    dispersion_n_components = excess21_df.notna().sum(axis=1)
    dispersion_raw = excess21_df.std(axis=1, ddof=1, skipna=True).where(dispersion_n_components >= 2)

    out["sector_dispersion_raw"] = dispersion_raw
    out["sector_dispersion_z"] = trailing_zscore(dispersion_raw, window=zscore_window, min_periods=zscore_min_periods)
    out["sector_dispersion_z_n_components"] = dispersion_n_components
    out["sector_dispersion_z_stale_days"] = _max_stale(spy_stale, *sector_stales.values())

    # ------------------------------------------------------------------
    # spy_trend_state
    # ------------------------------------------------------------------
    ema20 = spy_close.ewm(span=20, adjust=False).mean()
    ema50 = spy_close.ewm(span=50, adjust=False).mean()
    warm = spy_close.rolling(component_window_long, min_periods=component_window_long).count() >= component_window_long
    spy_trend_state = np.sign(ema20 - ema50).where(warm)
    out["spy_trend_state"] = spy_trend_state
    out["spy_trend_state_n_components"] = spy_trend_state.notna().astype(int)
    out["spy_trend_state_stale_days"] = spy_stale

    # ------------------------------------------------------------------
    # Assemble with the time-correctness contract columns first.
    # ------------------------------------------------------------------
    out = out.reset_index()
    out.insert(1, "available_at", available_at_for_dates(calendar, settle_margin_minutes=settle_margin_minutes))

    ordered = [
        "date", "available_at",
        "risk_appetite_xly_xlp_z", "risk_appetite_iwm_spy_z",
        "risk_appetite_hyg_iei_z", "risk_appetite_rsp_spy_z",
        "risk_appetite_z", "risk_appetite_z_n_components", "risk_appetite_z_stale_days",
        "liquidity_stress_amihud_z", "liquidity_stress_dollar_vol_z",
        "liquidity_stress_credit_z", "liquidity_stress_rv20_z",
        "liquidity_stress_z", "liquidity_stress_z_n_components", "liquidity_stress_z_stale_days",
        "credit_risk_ratio", "credit_risk_z", "credit_risk_z_n_components", "credit_risk_z_stale_days",
        "credit_risk_hyg_lqd_z",
        "breadth_raw", "breadth_z", "breadth_z_n_components", "breadth_z_stale_days",
        "sector_dispersion_raw", "sector_dispersion_z",
        "sector_dispersion_z_n_components", "sector_dispersion_z_stale_days",
        "spy_rv20_raw", "spy_rv20_z", "spy_rv20_z_n_components", "spy_rv20_z_stale_days",
        "spy_trend_state", "spy_trend_state_n_components", "spy_trend_state_stale_days",
    ]
    out = out[ordered]
    return out

