"""Stage-2 training targets on every 4H bar (plan Stage 2/3).

All forward quantities are measured from an EXECUTABLE entry: the open of the
next 4H bar after the decision bar closes (same session for the morning bar,
next session for the afternoon bar). The exit is the close of the session
``h`` market sessions after the decision session.

Per ticker (``ticker_forward``):
  fret_h   close(session s+h) / entry - 1
  fmfe_h   max high from the entry bar through session s+h, / entry - 1
  l0_mfe   the DEPLOYED momentum label input: max high of the next 25 bars /
           decision-bar close - 1 (kept identical so the baseline arm is the
           live model's own target)
  piv_long HTF's swing-low zone (fractal pivot, 3 bars each side), defined on
           EVERY row: 1 inside a long zone, 0 elsewhere. The deployed HTF model
           was trained and scored only on rows inside zones, which selects on
           the future; as a label on all rows it is legitimate.

Price semantics of this cache (checked 2026-10-06): the 4H bars are resampled
from hourly bars that start at 10:00 ET, so "next bar open" is the 10:00 ET
price for an afternoon decision (median 0.65% from the 09:30 auction print) and
the 14:00 ET price for a morning one. The last bar's close is the official close
on most days and an after-hours print up to 17:00 ET when that hour has a bar
(differs by >0.1% on 12% of ticker-days, p99 2.9%). Both are prices the bot can
see and trade; the difference from daily-bar prices is noise, not look-ahead.

Guards: no entry is assumed across a gap in the ticker's own bars (next bar must
be in the same or the next market session); a window containing a non-organic
corporate-action gap is void; the exit session must exist for the ticker.

Cross-section (``add_arm_targets``): SPY-excess, volatility-bucketed L1/L2 and
the binary top-quintile targets the classifier trains on, plus within-bar
permuted copies for the noise-trained control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.corporate_actions import suspect_sessions
from strategies.multi_ticker_swing_htf.config import PIVOT_LABEL_CONFIG
from strategies.multi_ticker_swing_htf.labels import _fractal_pivots, _shift_core, _zone

L0_BARS = 25          # LABEL_CONFIG["forward_window_4h_bars"]
L0_THRESHOLD = 0.20   # strong_setup_threshold in the deployed momentum export
TOP_QUANTILE = 0.20   # positive class for the vol-neutral arms
MIN_NAMES_PER_BAR = 200  # below this a timestamp is not a cross-section (stray late-open bars)


def session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """New York session date (tz-naive midnight) of each bar."""
    return index.tz_convert("America/New_York").normalize().tz_localize(None)


def flagged_sessions(daily: pd.DataFrame | None) -> set[pd.Timestamp]:
    """Session dates whose open is a non-organic corporate-action gap."""
    if daily is None or len(daily) < 2:
        return set()
    d = daily.sort_index()
    flags = suspect_sessions(d.reset_index(drop=True))
    if flags.empty:
        return set()
    flags = flags[~flags["organic"].astype(bool)]
    dates = session_dates(d.index)
    return {dates[int(i)] for i in flags["idx"]}


def ticker_forward(
    bars: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizons: tuple[int, ...],
    *,
    flagged: set[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Forward labels for one ticker's 4H bars (UTC DatetimeIndex, oldest first).

    ``calendar`` is the ordered list of market session dates (from SPY).
    """
    b = bars.sort_index()
    b = b[~b.index.duplicated(keep="last")]
    n = len(b)
    out = pd.DataFrame(index=b.index)
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    s = pos.reindex(session_dates(b.index)).to_numpy(float)          # market-session ordinal per bar
    # A non-positive print is a bad bar, not a price: treat it as missing.
    o, hi, c = (np.where(b[k].to_numpy(float) > 0, b[k].to_numpy(float), np.nan)
                for k in ("open", "high", "close"))

    nxt_s = np.append(s[1:], np.nan)
    entry = np.append(o[1:], np.nan)
    gap = nxt_s - s
    entry = np.where((gap == 0) | (gap == 1), entry, np.nan)         # no fill across missing sessions
    entry_s = np.where(np.isnan(entry), np.nan, nxt_s)

    # Per market session (NaN where the ticker has no bar that day)
    valid = ~np.isnan(s)
    sess = pd.DataFrame({"s": s[valid].astype(int), "high": hi[valid], "close": c[valid]})
    by = sess.groupby("s").agg(high=("high", "max"), close=("close", "last")).reindex(range(len(calendar)))
    sess_high, sess_close = by["high"].to_numpy(), by["close"].to_numpy()
    flag = np.zeros(len(calendar) + 1)
    if flagged:
        idx = pos.reindex(sorted(flagged)).dropna().astype(int).to_numpy()
        flag[idx + 1] = 1.0
    cum_flag = np.cumsum(flag)                                       # cum_flag[k+1] = flags in sessions <= k

    # High of later bars in the decision session (the afternoon bar, for a morning decision)
    same_sess_next_high = np.where(gap == 0, np.append(hi[1:], np.nan), np.nan)

    def _at(arr: np.ndarray, k: np.ndarray) -> np.ndarray:
        ok = ~np.isnan(k) & (k >= 0) & (k < len(arr))
        res = np.full(len(k), np.nan)
        res[ok] = arr[k[ok].astype(int)]
        return res

    for h in horizons:
        exit_s = s + h
        exit_close = _at(sess_close, exit_s)
        # max session high over sessions s+1 .. s+h (market calendar; missing sessions skipped)
        roll = pd.Series(sess_high).rolling(h, min_periods=1).max().to_numpy()
        fwd_high = np.fmax(_at(roll, exit_s), same_sess_next_high)
        # non-organic gap strictly after the entry session, up to the exit session
        void = (_at(cum_flag, exit_s + 1) - _at(cum_flag, entry_s + 1)) > 0
        ok = ~np.isnan(entry) & ~np.isnan(exit_close) & ~void
        out[f"fret_{h}"] = np.where(ok, exit_close / entry - 1.0, np.nan)
        out[f"fmfe_{h}"] = np.where(ok, fwd_high / entry - 1.0, np.nan)

    # Deployed label input: 25-bar max high from the decision close (bar counts, as in expansion_labels)
    h_s = pd.Series(hi, index=b.index)
    fwd_high_25 = h_s.shift(-1).rolling(L0_BARS, min_periods=L0_BARS).max().shift(-(L0_BARS - 1)).to_numpy()
    end_s = np.append(s[L0_BARS:], np.full(min(L0_BARS, n), np.nan))[:n]
    void25 = (_at(cum_flag, end_s + 1) - _at(cum_flag, s + 1)) > 0     # any gap after the decision session
    out["l0_mfe"] = np.where(void25 | np.isnan(end_s), np.nan, fwd_high_25 / c - 1.0)

    # HTF swing-low zone on all rows
    cfg = PIVOT_LABEL_CONFIG
    left, right = int(cfg["pivot_left_bars"]), int(cfg["pivot_right_bars"])
    shift, win = int(cfg["label_shift_bars"]), int(cfg["positive_window_bars"])
    if n > left + right + 10:
        lo_piv, hi_piv = _fractal_pivots(b.rename(columns=str.lower), left, right)
        long_zone = _zone(_shift_core(lo_piv, shift), win)
        short_zone = _zone(_shift_core(hi_piv, shift), win)
        piv = (long_zone & ~short_zone).astype(float).to_numpy()
        piv[max(0, n - (right + shift + win)):] = np.nan               # pivots there are not yet knowable
        out["piv_long"] = piv
    else:
        out["piv_long"] = np.nan
    return out.astype("float32")


def _vol_bucket(vol: pd.Series, ts: pd.Series, n: int) -> pd.Series:
    pct = vol.groupby(ts).rank(pct=True, method="first")
    return np.minimum(np.ceil(pct * n) - 1, n - 1)


def add_arm_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    vol_col: str = "daily_atr_pct",
    seed: int = 20261005,
    min_names_per_bar: int = MIN_NAMES_PER_BAR,
) -> pd.DataFrame:
    """Add the cross-sectional targets in place and return ``df``.

    Needs ``timestamp``, ``vol_col``, ``l0_mfe``, ``piv_long`` and, per horizon,
    ``fret_h`` / ``fmfe_h`` / ``spy_fret_h``. Adds per horizon:
      L2_h     fret - SPY, minus the mean of same-bar vol-decile peers (continuous)
      y_L2_h   1 if the SPY-excess return is in the top quintile of its
               same-bar vol decile (every vol bucket supplies 20% positives)
      y_L1_h   1 if fmfe_h is in the top quintile of its same-bar vol DECILE
               (quintile buckets left a 0.06 rank correlation with volatility)
    and ``y_L0`` (l0_mfe >= 20%), ``y_PIV`` (piv_long). Every ``y_*`` gets a
    ``*__perm`` twin: the same labels shuffled within the bar (row-aligned NaNs
    kept), for the noise-trained control.
    """
    ts = df["timestamp"]
    q10 = _vol_bucket(df[vol_col], ts, 10)
    # No decile ranks on a stray bar: one name per bucket would always rank top.
    thin = df.groupby(ts)[vol_col].transform("size") < min_names_per_bar
    no_vol = df[vol_col].isna() | thin

    df["y_L0"] = np.where(df["l0_mfe"].isna(), np.nan, (df["l0_mfe"] >= L0_THRESHOLD).astype(float))
    df["y_PIV"] = df["piv_long"]
    for h in horizons:
        xret = df[f"fret_{h}"] - df[f"spy_fret_{h}"]
        l2 = (xret - xret.groupby([ts, q10]).transform("mean")).where(~no_vol)
        df[f"L2_{h}"] = l2.astype("float32")
        # Rank WITHIN the vol decile. Ranking L2 across the whole bar is not
        # vol-neutral: volatile names have wider L2 and fill both tails, so a
        # bar-wide top quintile correlated +0.27 with volatility in testing.
        r2 = xret.where(~no_vol).groupby([ts, q10]).rank(pct=True)
        df[f"y_L2_{h}"] = np.where(l2.isna(), np.nan, (r2 > 1 - TOP_QUANTILE).astype(float))
        mfe = df[f"fmfe_{h}"].where(~no_vol)
        r1 = mfe.groupby([ts, q10]).rank(pct=True)
        df[f"y_L1_{h}"] = np.where(mfe.isna(), np.nan, (r1 > 1 - TOP_QUANTILE).astype(float))

    rng = np.random.default_rng(seed)
    bar = pd.factorize(ts)[0]
    for col in [c for c in df.columns if c.startswith("y_") and not c.endswith("__perm")]:
        ok = df[col].notna().to_numpy()
        rows = np.flatnonzero(ok)
        g = bar[rows]
        # within each bar, hand the labels (in one random order) to the rows (in another)
        src = rows[np.lexsort((rng.random(len(rows)), g))]
        dst = rows[np.lexsort((rng.random(len(rows)), g))]
        perm = np.full(len(df), np.nan, dtype="float32")
        perm[dst] = df[col].to_numpy()[src]
        df[f"{col}__perm"] = perm
    for col in [c for c in df.columns if c.startswith("y_")]:
        df[col] = df[col].astype("float32")
    return df
