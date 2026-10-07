"""Stage-2 label construction: executable entries, guards, and the control labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.ablation import stage2_labels as L


def _bars(n_sessions: int = 60, seed: int = 0, start: str = "2024-01-02") -> pd.DataFrame:
    """Two 4H bars per session at 15:00 and 19:00 UTC (10:00 / 14:00 ET in winter)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_sessions)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in (15, 19)], tz="UTC")
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    o = c * np.exp(rng.normal(0, 0.003, len(idx)))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996,
                         "close": c, "volume": 1e6}, index=idx)


def _cal(b: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(L.session_dates(b.index))))


def test_entry_is_next_bar_open_and_exit_is_session_close():
    b = _bars(); cal = _cal(b); h = 5
    out = L.ticker_forward(b, cal, (h,))
    i_am, i_pm = 20, 21                      # session 10: morning bar, afternoon bar
    # afternoon decision: enter next session's open, exit close of session 10+5 (its last bar = row 31)
    assert out["fret_5"].iloc[i_pm] == pytest.approx(b["close"].iloc[31] / b["open"].iloc[22] - 1, rel=1e-6)
    # morning decision: enter the afternoon bar's open the same day, same exit
    assert out["fret_5"].iloc[i_am] == pytest.approx(b["close"].iloc[31] / b["open"].iloc[21] - 1, rel=1e-6)
    # MFE spans the entry bar through the exit session
    assert out["fmfe_5"].iloc[i_pm] == pytest.approx(b["high"].iloc[22:32].max() / b["open"].iloc[22] - 1, rel=1e-6)
    assert out["fmfe_5"].iloc[i_am] == pytest.approx(b["high"].iloc[21:32].max() / b["open"].iloc[21] - 1, rel=1e-6)
    # the last h sessions have no exit yet
    assert out["fret_5"].iloc[-2 * h:].isna().all() and out["fret_5"].iloc[: -2 * h - 2].notna().all()


def test_labels_do_not_change_when_the_past_is_rewritten():
    b = _bars(); cal = _cal(b)
    base = L.ticker_forward(b, cal, (5,))
    past = b.copy(); past.iloc[:20, :4] *= 3.0          # rewrite everything before row 20
    again = L.ticker_forward(past, cal, (5,))
    pd.testing.assert_series_equal(base["fret_5"].iloc[20:], again["fret_5"].iloc[20:])


def test_no_fill_is_assumed_across_a_missing_session():
    b = _bars(); cal = _cal(b)
    holed = b.drop(b.index[[22, 23]])                    # the ticker has no bars in session 11
    out = L.ticker_forward(holed, cal, (5,))
    assert np.isnan(out["fret_5"].iloc[21])              # session-10 afternoon: next bar is 2 sessions away
    assert not np.isnan(out["fret_5"].iloc[20])          # session-10 morning still enters that afternoon
    # a decision whose EXIT session is the missing one has no exit price
    assert np.isnan(out["fret_5"].iloc[13])              # session 6 afternoon -> exit session 11


def test_split_gap_voids_windows_that_contain_it_but_not_entries_after_it():
    b = _bars(); cal = _cal(b); h = 5
    b.iloc[24:, :4] /= 10                                # 10:1 split at the open of session 12
    flagged = {cal[12]}
    out = L.ticker_forward(b, cal, (h,), flagged=flagged)
    # session-11 afternoon enters at the open of session 12, after the gap: valid
    assert out["fret_5"].iloc[23] == pytest.approx(b["close"].iloc[2 * 16 + 1] / b["open"].iloc[24] - 1, rel=1e-6)
    # session-11 morning enters that afternoon, before the gap: void; so is session 10
    assert np.isnan(out["fret_5"].iloc[22]) and np.isnan(out["fret_5"].iloc[21])
    # the deployed-label input measured from the session-11 close also spans the gap
    assert np.isnan(out["l0_mfe"].iloc[23])
    assert not np.isnan(out["l0_mfe"].iloc[24])


def test_l0_matches_the_deployed_label_formula():
    b = _bars(120); cal = _cal(b)
    out = L.ticker_forward(b, cal, (5,))
    w = L.L0_BARS
    expect = b["high"].shift(-1).rolling(w, min_periods=w).max().shift(-(w - 1)) / b["close"] - 1
    np.testing.assert_allclose(out["l0_mfe"].to_numpy(), expect.to_numpy(), rtol=1e-5, equal_nan=True)


def test_pivot_target_is_defined_on_every_row_and_blank_where_unknowable():
    b = _bars(120); cal = _cal(b)
    piv = L.ticker_forward(b, cal, (5,))["piv_long"]
    assert piv.iloc[-5:].isna().all() and piv.iloc[:-5].notna().all()
    assert set(piv.dropna().unique()) == {0.0, 1.0} and 0.02 < piv.mean() < 0.6


def _panel(n_bars: int = 60, n_names: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_bars):
        vol = rng.uniform(0.01, 0.10, n_names)
        ret = vol * rng.normal(0, 1, n_names) + 0.6 * vol          # a volatility premium
        rows.append(pd.DataFrame({
            "timestamp": pd.Timestamp("2024-01-02", tz="UTC") + pd.Timedelta(hours=12 * t),
            "daily_atr_pct": vol, "fret_10": ret, "fmfe_10": vol * rng.uniform(1, 3, n_names),
            "spy_fret_10": 0.01, "l0_mfe": vol * rng.uniform(1, 4, n_names),
            "piv_long": (rng.random(n_names) < 0.2).astype(float),
        }))
    df = pd.concat(rows, ignore_index=True)
    df.loc[df.sample(frac=0.03, random_state=1).index, "fret_10"] = np.nan
    return df


def test_vol_neutral_targets_do_not_reward_volatility():
    df = L.add_arm_targets(_panel(), (10,))
    by_bar = lambda col: df.groupby("timestamp").apply(
        lambda g: g[col].corr(g["daily_atr_pct"], method="spearman"), include_groups=False).mean()
    assert by_bar("y_L0") > 0.3                       # the deployed target is a vol proxy
    assert abs(by_bar("y_L1_10")) < 0.05
    assert abs(by_bar("y_L2_10")) < 0.05              # bar-wide top quintile scored 0.27 here
    assert df["y_L2_10"].mean() == pytest.approx(0.20, abs=0.01)
    # every volatility decile supplies the same share of positives
    dec = df.groupby("timestamp")["daily_atr_pct"].rank(pct=True).mul(10).apply(np.ceil)
    share = df.groupby(dec)["y_L2_10"].mean()
    assert share.max() - share.min() < 0.02
    assert df.loc[df["fret_10"].isna(), ["L2_10", "y_L2_10"]].isna().all().all()


def test_permuted_labels_keep_each_bars_label_counts_and_carry_no_signal():
    df = L.add_arm_targets(_panel(), (10,))
    for col in ("y_L0", "y_L2_10", "y_PIV"):
        perm = df[f"{col}__perm"]
        assert (perm.isna() == df[col].isna()).all()                                  # same rows usable
        pd.testing.assert_series_equal(df.groupby("timestamp")[col].sum(),
                                       df.groupby("timestamp")[f"{col}__perm"].sum(), check_names=False)
        assert abs(df[col].corr(perm)) < 0.02
    # the shuffled deployed label no longer tracks volatility
    assert abs(df["y_L0__perm"].corr(df["daily_atr_pct"])) < 0.02


def test_zero_price_bar_yields_no_label_not_an_infinite_return():
    b = _bars(); cal = _cal(b)
    b.iloc[22, b.columns.get_loc("open")] = 0.0          # bad print at the entry bar of row 21
    b.iloc[40, b.columns.get_loc("close")] = 0.0         # bad print at a decision bar
    out = L.ticker_forward(b, cal, (5,))
    assert np.isfinite(out.to_numpy()[~np.isnan(out.to_numpy())]).all()
    assert np.isnan(out["fret_5"].iloc[21]) and np.isnan(out["l0_mfe"].iloc[40])


def test_stray_thin_bar_gets_no_cross_sectional_target():
    # 2026-10-06: 1.15% of matrix rows sit on timestamps with a handful of names
    # (late first prints). One name per vol decile would always rank "top".
    df = _panel(n_bars=4)
    thin = df[df["timestamp"] == df["timestamp"].iloc[0]].head(5).copy()
    thin["timestamp"] = thin["timestamp"] + pd.Timedelta(hours=1)
    out = L.add_arm_targets(pd.concat([df, thin], ignore_index=True), (10,))
    stray = out["timestamp"] == thin["timestamp"].iloc[0]
    assert out.loc[stray, ["y_L1_10", "y_L2_10", "L2_10"]].isna().all().all()
    assert out.loc[stray, ["y_L0", "y_PIV"]].notna().all().all()        # per-ticker labels are unaffected
    assert out.loc[~stray, "y_L2_10"].mean() == pytest.approx(0.20, abs=0.01)
