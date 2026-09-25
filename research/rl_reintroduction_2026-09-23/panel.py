"""Shared substrate for the RL-reintroduction stages.

One module so every stage reads the SAME rows, the SAME risk unit and the SAME
train/val/test boundary. Three jobs:

1. `load()`      -- the versioned decision panel, corporate-action guard applied.
2. `risk_units()`-- ATR-denominated R columns. The panel stores fractional returns;
                    every reward and every metric in this series is in R, because
                    `momentum_expansion`'s label composite is a percentile RANK and
                    therefore cannot express the right tail we are studying
                    (see 00_preregistration.md C5).
3. `split()`     -- the HOUSE split, reused from `scripts/horizon_thesis/run_horizon_grid.py`
                    (bar-fraction 0.60 / 0.78 with an EMBARGO_BARS gap each side) so
                    numbers here are comparable to the horizon-grid and depth-null studies
                    rather than sitting on a private convention.

ATR is not in the decision panel, so it is rebuilt here from the daily cache with the
same Wilder EWM used by `research/regime_coverage_2026-09-21/recall_precision.py`, taken
at the session STRICTLY BEFORE the entry session (the last close a decision could have
seen). The daily cache is UNADJUSTED and survivor-shaped; the CA guard handles the first
and `survivorship_note()` states the second.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/horizon_thesis"))

from build_decision_panel import HOLDS, load_panel  # noqa: E402
from run_horizon_grid import EMBARGO_BARS  # noqa: E402

BARS_1D = REPO / "Data/shared/bars/1d"
HERE = Path(__file__).resolve().parent
ATR_CACHE = HERE / "data" / "atr_by_session.parquet"

# The risk unit. K_RISK * ATR% is "1R". 2.0 is the width the live 4H modules size
# against (momentum_config stop family); it is a CONSTANT here, never fitted.
K_RISK = 2.0
ATR_WINDOW = 14
# Below this, ATR% is numerically unstable and 1R becomes a rounding error.
MIN_ATR_PCT = 0.002


# --------------------------------------------------------------------------- ATR

def _ticker_atr(ticker: str) -> pd.DataFrame | None:
    path = BARS_1D / f"{ticker}.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["timestamp", "high", "low", "close"])
    if len(d) < ATR_WINDOW * 3:
        return None
    d["session_date"] = (pd.to_datetime(d["timestamp"], utc=True)
                         .dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None))
    d = d.sort_values("session_date").reset_index(drop=True)
    h, lo, c = d["high"], d["low"], d["close"]
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_WINDOW, adjust=False).mean()
    # shift(1): the ATR a decision at the PRIOR close could have known.
    return pd.DataFrame({"ticker": ticker,
                         "session_date": d["session_date"],
                         "atr_pct": (atr / c.replace(0, np.nan)).shift(1).to_numpy()})


def build_atr_cache(tickers: list[str], *, verbose: bool = True) -> pd.DataFrame:
    ATR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, t in enumerate(sorted(set(tickers)), 1):
        f = _ticker_atr(t)
        if f is not None:
            frames.append(f)
        if verbose and i % 250 == 0:
            print(f"  atr {i}/{len(set(tickers))}", flush=True)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["atr_pct"])
    out["atr_pct"] = out["atr_pct"].astype("float32")
    out.to_parquet(ATR_CACHE, index=False)
    return out


def risk_units(panel: pd.DataFrame, *, holds: list[int] | None = None) -> pd.DataFrame:
    """Adds `atr_pct` and, per hold, `r_{h}` / `rmfe_{h}` / `rmae_{h}` in R units."""
    holds = holds or list(HOLDS)
    if not ATR_CACHE.exists():
        build_atr_cache(panel["ticker"].unique().tolist())
    atr = pd.read_parquet(ATR_CACHE)
    p = panel.merge(atr, left_on=["ticker", "entry_session"],
                    right_on=["ticker", "session_date"], how="left").drop(columns=["session_date"])
    unit = (K_RISK * p["atr_pct"]).where(p["atr_pct"] >= MIN_ATR_PCT)
    for h in holds:
        p[f"r_{h}"] = p[f"fwdret_{h}"] / unit
        p[f"rmfe_{h}"] = p[f"mfe_{h}"] / unit
        p[f"rmae_{h}"] = p[f"mae_{h}"] / unit
    return p


# ------------------------------------------------------------------------- split

def split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """House split on DECISION BARS, embargoed. Same fractions as run_horizon_grid."""
    uniq = np.array(sorted(panel["timestamp"].unique()))
    n = len(uniq)
    i_tr, i_va = int(n * 0.60), int(n * 0.78)
    tr_end = uniq[i_tr]
    va_start = uniq[min(i_tr + EMBARGO_BARS, n - 1)]
    va_end = uniq[i_va]
    te_start = uniq[min(i_va + EMBARGO_BARS, n - 1)]
    ts = panel["timestamp"].to_numpy()
    bounds = dict(n_bars=n, embargo_bars=EMBARGO_BARS,
                  train_end=str(pd.Timestamp(tr_end).date()),
                  val=(str(pd.Timestamp(va_start).date()), str(pd.Timestamp(va_end).date())),
                  test=(str(pd.Timestamp(te_start).date()), str(pd.Timestamp(uniq[-1]).date())))
    return (panel[ts <= tr_end], panel[(ts >= va_start) & (ts <= va_end)],
            panel[ts >= te_start], bounds)


# ------------------------------------------------------------------- disclosures

def survivorship_note() -> dict:
    """What the daily cache cannot see. Printed by every stage that reports a tail metric."""
    files = sorted(BARS_1D.glob("*.parquet"))
    last = []
    for p in files:
        try:
            d = pd.read_parquet(p, columns=["timestamp"])
        except Exception:
            continue
        if len(d):
            last.append(pd.to_datetime(d["timestamp"].iloc[-1], utc=True))
    s = pd.Series(last)
    if s.empty:
        return {"tickers": 0}
    cutoff = s.max() - pd.Timedelta(days=30)
    return {"tickers": int(len(s)),
            "last_bar_max": str(s.max().date()),
            "stale_gt_30d": int((s < cutoff).sum()),
            "stale_share": float((s < cutoff).mean())}


def load(*, require: list[str] | None = None, drop_flagged: bool = True,
         with_risk: bool = True) -> pd.DataFrame:
    """`require` is applied AFTER the R columns are derived, so callers may require them.

    load_panel's own `require` runs before risk_units exists, so passing an r_* column
    straight through would KeyError.
    """
    p = load_panel(drop_flagged=drop_flagged)
    if with_risk:
        p = risk_units(p)
    return p.dropna(subset=require) if require else p
