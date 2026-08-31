"""
Regression tests for strategies/momentum_expansion/features/live_feature_panel_4h.py.

This is the shared builder that replaced two independently-duplicated live
feature-build loops (signals/meta_context/meta_ranker/update_meta_matrix.py and
momentum's live/runner.py::evaluate_now) whose divergence let both deployed
boosters score live with missing manifest columns, silently, indefinitely (see
LIVING_SUMMARY.md 2026-07-19). These tests assert the contract that fix
depends on: the panel this builder returns always carries every xsec_*/
earnings column, and the missing-column check actually catches a gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.config.momentum_config import MIN_4H_BARS
from strategies.momentum_expansion.features.live_feature_panel_4h import (
    assert_manifest_coverage,
    build_live_feature_panel_4h,
    clear_feature_panel_memo,
)

N_BARS = MIN_4H_BARS + 20


@pytest.fixture(autouse=True)
def _isolate_memo():
    """The per-ticker memo is module state; do not let it leak between tests.

    It is keyed on bar CONTENT, so two tests that build the same synthetic
    frames for the same ticker name would otherwise share entries -- fine in
    production, silently misleading in a test that expects a rebuild.
    """
    clear_feature_panel_memo()
    yield
    clear_feature_panel_memo()


def _synthetic_4h_bars(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=N_BARS, freq="4h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, N_BARS))
    close = np.clip(close, 5, None)
    high = close + rng.uniform(0, 1, N_BARS)
    low = close - rng.uniform(0, 1, N_BARS)
    open_ = close + rng.normal(0, 0.3, N_BARS)
    volume = rng.uniform(1e5, 1e6, N_BARS)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _make_loaders(tickers: list[str], *, missing: set[str] = frozenset()):
    bars = {t: _synthetic_4h_bars(seed=i) for i, t in enumerate(tickers)}

    def load_4h_bars(ticker: str) -> pd.DataFrame:
        if ticker in missing:
            raise FileNotFoundError(ticker)
        return bars[ticker]

    def load_1d_bars(ticker: str):
        return None

    return load_4h_bars, load_1d_bars


XSEC_COLS = ["xsec_ret_5_rank", "xsec_ret_20_rank", "xsec_rs_spy_20_rank",
             "xsec_rvol_20_rank", "xsec_dollar_vol_surge_20_rank",
             "xsec_atr_expand_rank", "xsec_atr_pct_rank", "xsec_near_high_rank"]
EARNINGS_COLS = ["days_to_earnings", "days_since_earnings", "is_pre_earnings_3d",
                 "is_post_earnings_3d", "earnings_in_fwd_window"]


def test_panel_always_has_xsec_and_earnings_columns_tail_mode():
    tickers = ["AAA", "BBB", "CCC"]
    load_4h, load_1d = _make_loaders(tickers)
    result = build_live_feature_panel_4h(
        tickers=tickers, load_4h_bars=load_4h, load_1d_bars=load_1d, ctx_4h={}, tail=1)

    assert result.tickers_requested == 3
    assert result.tickers_built == 3
    assert result.tickers_skipped == []
    assert not result.panel.empty
    for col in XSEC_COLS + EARNINGS_COLS:
        assert col in result.panel.columns, f"missing {col}"
    # tail=1 -> exactly one row per ticker
    assert result.panel.reset_index().groupby("ticker").size().eq(1).all()


def test_panel_always_has_xsec_and_earnings_columns_since_mode():
    tickers = ["AAA", "BBB"]
    load_4h, load_1d = _make_loaders(tickers)
    bars = load_4h("AAA")
    since = bars.index[-5]  # keep only the last few bars
    result = build_live_feature_panel_4h(
        tickers=tickers, load_4h_bars=load_4h, load_1d_bars=load_1d, ctx_4h={}, since=since)

    assert not result.panel.empty
    for col in XSEC_COLS + EARNINGS_COLS:
        assert col in result.panel.columns
    ts = result.panel.index.get_level_values("timestamp")
    assert (ts > since).all()


def test_missing_ticker_is_skipped_not_silently_dropped():
    tickers = ["AAA", "BBB", "GHOST"]
    load_4h, load_1d = _make_loaders(tickers, missing={"GHOST"})
    result = build_live_feature_panel_4h(
        tickers=tickers, load_4h_bars=load_4h, load_1d_bars=load_1d, ctx_4h={}, tail=1)

    assert result.tickers_requested == 3
    assert result.tickers_built == 2
    assert result.tickers_skipped == ["GHOST"]
    assert "GHOST" not in result.panel.reset_index()["ticker"].tolist()


def test_all_tickers_missing_returns_empty_panel_not_an_exception():
    load_4h, load_1d = _make_loaders([], missing=set())

    def always_raises(ticker: str) -> pd.DataFrame:
        raise FileNotFoundError(ticker)

    result = build_live_feature_panel_4h(
        tickers=["X", "Y"], load_4h_bars=always_raises, load_1d_bars=load_1d, ctx_4h={}, tail=1)
    assert result.panel.empty
    assert result.tickers_skipped == ["X", "Y"]
    assert result.tickers_built == 0


class _FakeScorer:
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns


def test_assert_manifest_coverage_reports_missing_columns():
    panel = pd.DataFrame({"a": [1.0], "b": [2.0]})
    scorer = _FakeScorer(["a", "b", "c", "d"])
    missing = assert_manifest_coverage(scorer=scorer, panel=panel, label="test")
    assert missing == ["c", "d"]


def test_assert_manifest_coverage_no_op_when_complete():
    panel = pd.DataFrame({"a": [1.0], "b": [2.0]})
    scorer = _FakeScorer(["a", "b"])
    assert assert_manifest_coverage(scorer=scorer, panel=panel, label="test") == []


def test_assert_manifest_coverage_raises_when_asked():
    panel = pd.DataFrame({"a": [1.0]})
    scorer = _FakeScorer(["a", "b"])
    with pytest.raises(ValueError, match="b"):
        assert_manifest_coverage(scorer=scorer, panel=panel, label="test", raise_on_missing=True)


def test_real_deployed_scorers_have_zero_missing_columns_against_a_built_panel():
    """End-to-end contract check with the REAL manifests: a panel built by this
    module must satisfy both deployed boosters' actual feature_columns, not
    just a fake scorer's. This is the exact regression the 2026-07-19 audit
    would have caught immediately had it existed then."""
    from strategies.momentum_expansion.inference.ranker import ExpansionRanker
    from strategies.multi_ticker_swing_htf.inference.scorer import HTFSwingScorer

    tickers = ["AAA", "BBB", "CCC"]
    load_4h, load_1d = _make_loaders(tickers)
    result = build_live_feature_panel_4h(
        tickers=tickers, load_4h_bars=load_4h, load_1d_bars=load_1d, ctx_4h={}, tail=1)

    assert assert_manifest_coverage(scorer=ExpansionRanker(), panel=result.panel, label="momentum") == []
    assert assert_manifest_coverage(scorer=HTFSwingScorer(), panel=result.panel, label="htf") == []
