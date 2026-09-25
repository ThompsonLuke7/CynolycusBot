"""Tests for the shared RL-reintroduction substrate.

These guard the three things every later stage silently depends on. A look-ahead bug in
the risk unit, or an un-embargoed split, would invalidate Stages 1-6 at once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import panel as P  # noqa: E402


def _bars(n: int = 60, start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, tz="America/New_York")
    close = np.linspace(100.0, 160.0, n)
    return pd.DataFrame({"timestamp": idx.tz_convert("UTC"),
                         "high": close + 2.0, "low": close - 2.0, "close": close})


def test_atr_is_shifted_so_it_cannot_see_its_own_session(tmp_path, monkeypatch):
    """atr_pct on session t must be computable from data up to t-1's close only."""
    d = _bars()
    monkeypatch.setattr(P, "BARS_1D", tmp_path)
    d.to_parquet(tmp_path / "FAKE.parquet", index=False)
    f = P._ticker_atr("FAKE")
    assert f is not None
    # first row has no prior session, so ATR is undefined there
    assert pd.isna(f["atr_pct"].iloc[0])
    # a single session's bar cannot move its OWN atr_pct: blow up the last bar's range and
    # the last row's atr_pct must be unchanged.
    d2 = d.copy()
    d2.loc[d2.index[-1], "high"] = 10_000.0
    d2.to_parquet(tmp_path / "FAKE2.parquet", index=False)
    g = P._ticker_atr("FAKE2")
    assert g["atr_pct"].iloc[-1] == pytest.approx(f["atr_pct"].iloc[-1], rel=1e-12)


def test_atr_rejects_too_short_history(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BARS_1D", tmp_path)
    _bars(n=5).to_parquet(tmp_path / "SHORT.parquet", index=False)
    assert P._ticker_atr("SHORT") is None
    assert P._ticker_atr("ABSENT") is None


def test_risk_units_scale_and_floor(tmp_path, monkeypatch):
    """r = fractional return / (K_RISK * atr_pct), and a sub-floor ATR yields NaN, never
    a division that inflates R to nonsense."""
    monkeypatch.setattr(P, "ATR_CACHE", tmp_path / "atr.parquet")
    atr = pd.DataFrame({"ticker": ["A", "B"],
                        "session_date": pd.to_datetime(["2024-03-01", "2024-03-01"]),
                        "atr_pct": [0.04, P.MIN_ATR_PCT / 2.0]})
    atr.to_parquet(P.ATR_CACHE, index=False)
    pan = pd.DataFrame({"ticker": ["A", "B"],
                        "entry_session": pd.to_datetime(["2024-03-01", "2024-03-01"]),
                        "fwdret_10": [0.08, 0.08], "mfe_10": [0.12, 0.12],
                        "mae_10": [-0.04, -0.04]})
    out = P.risk_units(pan, holds=[10])
    # A: 0.08 / (2.0 * 0.04) = 1.0R
    assert out.loc[out.ticker == "A", "r_10"].iloc[0] == pytest.approx(1.0)
    assert out.loc[out.ticker == "A", "rmfe_10"].iloc[0] == pytest.approx(1.5)
    assert out.loc[out.ticker == "A", "rmae_10"].iloc[0] == pytest.approx(-0.5)
    # B sits under the floor: R must be NaN, not a huge number
    assert pd.isna(out.loc[out.ticker == "B", "r_10"].iloc[0])


def test_risk_units_leaves_unmatched_rows_null_rather_than_guessing(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "ATR_CACHE", tmp_path / "atr.parquet")
    pd.DataFrame({"ticker": ["A"], "session_date": pd.to_datetime(["2024-03-01"]),
                  "atr_pct": [0.04]}).to_parquet(P.ATR_CACHE, index=False)
    pan = pd.DataFrame({"ticker": ["A", "Z"],
                        "entry_session": pd.to_datetime(["2024-03-01", "2024-03-01"]),
                        "fwdret_10": [0.08, 0.08], "mfe_10": [0.1, 0.1],
                        "mae_10": [0.0, 0.0]})
    out = P.risk_units(pan, holds=[10])
    assert pd.isna(out.loc[out.ticker == "Z", "atr_pct"].iloc[0])
    assert pd.isna(out.loc[out.ticker == "Z", "r_10"].iloc[0])


def test_split_is_ordered_embargoed_and_covers_no_overlap():
    ts = pd.date_range("2022-01-03", periods=600, freq="B", tz="UTC")
    pan = pd.DataFrame({"timestamp": np.repeat(ts, 3),
                        "ticker": list("abc") * len(ts)})
    tr, va, te, b = P.split(pan)
    assert b["n_bars"] == len(ts)
    assert b["embargo_bars"] == P.EMBARGO_BARS
    tr_max, va_min = tr["timestamp"].max(), va["timestamp"].min()
    va_max, te_min = va["timestamp"].max(), te["timestamp"].min()
    # strictly ordered
    assert tr_max < va_min < va_max < te_min
    # the embargo gap is at least EMBARGO_BARS distinct bars wide on BOTH seams
    gap1 = ((ts > tr_max) & (ts < va_min)).sum()
    gap2 = ((ts > va_max) & (ts < te_min)).sum()
    assert gap1 >= P.EMBARGO_BARS - 1
    assert gap2 >= P.EMBARGO_BARS - 1
    # no row appears in two slices
    assert len(tr) + len(va) + len(te) <= len(pan)


def test_split_respects_bar_fractions():
    ts = pd.date_range("2022-01-03", periods=1000, freq="B", tz="UTC")
    pan = pd.DataFrame({"timestamp": ts, "ticker": ["a"] * len(ts)})
    tr, va, te, b = P.split(pan)
    # train is the first 60% of BARS, not of rows
    assert tr["timestamp"].max() == ts[600]
