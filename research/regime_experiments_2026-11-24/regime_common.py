"""Shared definitions for the regime experiments (E1-E3). See PLAN.md.

Everything here is trailing-only and known at the close of the row's session (FRED inputs carry
their availability lag from E0). Forward targets are the ONLY forward-looking columns and are
named ``fwd_*``.

Long-history vote (LH vote), mirroring the live 7-vote's structure (signals/market_regime/
regime_vote.py) with the inputs that exist for 40 years:
  fast_down    target index close < EMA20 and EMA20 < its value 5 sessions ago   (live: QQQ)
  below_sma50  target index close < SMA50                                         (live: QQQ)
  spx_trend    S&P EMA20 < EMA50                                                  (live: SPY)
  rv           S&P RV20 (log returns) trailing-252 z > 1                          (live: SPY)
  vix          VIX trailing-252 z > 1                     (1991+, replaces breadth)
  curve        10y - 3m bill < 0 (Yahoo ^TNX - ^IRX)      (1962+, replaces risk appetite)
  credit       Moody's Baa-10y spread trailing-252 z > 0  (1987+, replaces HYG/IEI)
The first five rules and their round thresholds are the live ones; none is tuned.

State. bear = bearish votes, avail = votes with inputs. The five price/vol/curve votes are
required (else NaN). The live 0-7 mapping is applied to the count scaled to 7 votes,
floor(7 * bear / avail + 0.5): with 7 inputs that is the live count exactly.
Yields, the dollar and WTI are carried forward at most 5 sessions (bond/NYMEX holidays and half days
while stocks trade; WTI: 15 such sessions 2000-2016); nothing else is filled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO / "Data" / "research" / "regime_40y"
OUT = Path(__file__).resolve().parent / "out"

VOTES = ("fast_down", "below_sma50", "spx_trend", "rv", "vix", "curve", "credit")
REQUIRED_VOTES = ("fast_down", "below_sma50", "spx_trend", "rv", "curve")
STATES = ("RISK_ON", "NEUTRAL", "DETERIORATING", "RISK_OFF", "CRISIS")
LIVE_SIZE = {"RISK_ON": 1.0, "NEUTRAL": 1.0, "DETERIORATING": 0.8, "RISK_OFF": 0.5, "CRISIS": 0.25}
CARRY_LIMIT = 5
H = 20  # forward horizon, sessions


def latest_panel() -> tuple[pd.DataFrame, str]:
    paths = sorted(PANEL_DIR.glob("panel_*.parquet"))
    if not paths:
        raise FileNotFoundError("no E0 panel; run e0_fetch_long_history.py")
    p = pd.read_parquet(paths[-1])
    for col in ("ust10y", "tbill3m", "dxy", "wti"):
        p[col] = p[col].ffill(limit=CARRY_LIMIT)
    return p, paths[-1].name


def zscore(s: pd.Series, window: int = 252) -> pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    return (s - m) / sd.replace(0, np.nan)


def rv20(close: pd.Series) -> pd.Series:
    return np.log(close).diff().rolling(20, min_periods=20).std()


def _flag(cond: pd.Series, valid: pd.Series) -> pd.Series:
    return cond.astype(float).where(valid)


def lh_votes(p: pd.DataFrame, target: str) -> pd.DataFrame:
    """Bearish flags (1/0, NaN when the input is missing) for one target index column."""
    c = p[target]
    ema20 = c.ewm(span=20, adjust=False).mean()
    warm = c.rolling(50, min_periods=50).count() >= 50
    sma50 = c.rolling(50, min_periods=50).mean()
    spx = p["spx"]
    s20, s50 = spx.ewm(span=20, adjust=False).mean(), spx.ewm(span=50, adjust=False).mean()
    swarm = spx.rolling(50, min_periods=50).count() >= 50
    rvz = zscore(rv20(spx))
    vixz = zscore(p["vix"])
    curve = p["ust10y"] - p["tbill3m"]
    baaz = zscore(p["baa_10y_spread"]) if "baa_10y_spread" in p else pd.Series(np.nan, index=p.index)
    v = pd.DataFrame(index=p.index)
    v["fast_down"] = _flag((c < ema20) & (ema20 < ema20.shift(5)), warm & c.notna())
    v["below_sma50"] = _flag(c < sma50, warm & sma50.notna())
    v["spx_trend"] = _flag(s20 < s50, swarm)
    v["rv"] = _flag(rvz > 1, rvz.notna())
    v["vix"] = _flag(vixz > 1, vixz.notna())
    v["curve"] = _flag(curve < 0, curve.notna())
    v["credit"] = _flag(baaz > 0, baaz.notna())
    return v


def lh_state(votes: pd.DataFrame) -> pd.DataFrame:
    bear = votes[list(VOTES)].sum(axis=1, min_count=1)
    avail = votes[list(VOTES)].notna().sum(axis=1)
    ok = votes[list(REQUIRED_VOTES)].notna().all(axis=1)
    scaled = np.floor(7 * bear / avail.replace(0, np.nan) + 0.5).where(ok)
    state = pd.Series(pd.NA, index=votes.index, dtype="object")
    for lo, hi, name in ((0, 1, "RISK_ON"), (2, 3, "NEUTRAL"), (4, 4, "DETERIORATING"),
                         (5, 6, "RISK_OFF"), (7, 7, "CRISIS")):
        state[(scaled >= lo) & (scaled <= hi)] = name
    return pd.DataFrame({"bear_votes": bear.where(ok), "avail_votes": avail.where(ok),
                         "vote7": scaled, "state": state})


def forward_targets(close: pd.Series, h: int = H) -> pd.DataFrame:
    """From the close of t: path max drawdown over C[t..t+h], realised vol of t+1..t+h
    (annualised), and the h-session return. NaN where the window is incomplete."""
    c = close.to_numpy(dtype=float)
    n = len(c)
    mdd = np.full(n, np.nan)
    vol = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    lr = np.diff(np.log(c), prepend=np.nan)
    for i in range(n - h):
        w = c[i:i + h + 1]
        if np.isnan(w).any():
            continue
        mdd[i] = (w / np.maximum.accumulate(w) - 1).min()
        vol[i] = lr[i + 1:i + h + 1].std(ddof=1) * np.sqrt(252)
        ret[i] = w[-1] / w[0] - 1
    return pd.DataFrame({"fwd_mdd20": mdd, "fwd_rv20": vol, "fwd_ret20": ret}, index=close.index)


def decade_of(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((idx.year // 10) * 10, index=idx).astype(str) + "s"


def eta2(codes: np.ndarray, y: np.ndarray) -> float:
    """Share of y's variance explained by the categorical codes (0..k-1)."""
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k)
    sums = np.bincount(codes, weights=y, minlength=k)
    keep = cnt > 0
    means = sums[keep] / cnt[keep]
    ss_between = (cnt[keep] * (means - y.mean()) ** 2).sum()
    ss_total = ((y - y.mean()) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def circular_shift_null(codes: np.ndarray, y: np.ndarray, stat, draws: int, rng,
                        min_shift: int) -> np.ndarray:
    n = len(codes)
    if n <= 2 * min_shift:
        raise ValueError(f"series of {n} too short for min shift {min_shift}")
    shifts = rng.integers(min_shift, n - min_shift, size=draws)
    return np.array([stat(np.roll(codes, s), y) for s in shifts])


def block_bootstrap_mean(y: np.ndarray, mask: np.ndarray, draws: int, rng, block: int = H) -> tuple[float, float]:
    """95% CI of mean(y[mask]) under a circular block bootstrap of (y, mask) pairs."""
    n = len(y)
    nb = int(np.ceil(n / block))
    out = np.empty(draws)
    for d in range(draws):
        starts = rng.integers(0, n, size=nb)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        m = mask[idx]
        out[d] = y[idx][m].mean() if m.any() else np.nan
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def build_frame(target: str) -> tuple[pd.DataFrame, str]:
    """Panel rows with LH votes, state and forward targets for one target index."""
    p, name = latest_panel()
    v = lh_votes(p, target)
    st = lh_state(v)
    f = forward_targets(p[target].dropna()).reindex(p.index)
    frame = pd.concat([p[[target, "spx", "vix", "tbill3m"]], v, st, f], axis=1)
    frame["trail_rv20"] = rv20(p[target]) * np.sqrt(252)
    return frame, name
