"""Tests for strategies/momentum_expansion/ablation/metrics.py (synthetic only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.ablation import metrics as M


def _perfect_rank_frame(n_groups: int = 5, n_per_group: int = 20, seed: int = 0) -> pd.DataFrame:
    """score == label exactly within each group -> IC should be ~1.0."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        vals = rng.normal(0, 1, n_per_group)
        for i, v in enumerate(vals):
            rows.append({"timestamp": g, "ticker": f"T{i}", "score": v, "label": v})
    return pd.DataFrame(rows)


def _inverted_rank_frame(n_groups: int = 5, n_per_group: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        vals = rng.normal(0, 1, n_per_group)
        for i, v in enumerate(vals):
            rows.append({"timestamp": g, "ticker": f"T{i}", "score": v, "label": -v})
    return pd.DataFrame(rows)


def _random_rank_frame(n_groups: int = 30, n_per_group: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        score = rng.normal(0, 1, n_per_group)
        label = rng.normal(0, 1, n_per_group)
        for i in range(n_per_group):
            rows.append({"timestamp": g, "ticker": f"T{i}", "score": score[i], "label": label[i]})
    return pd.DataFrame(rows)


class TestRankIC:
    def test_perfect_rank_ic_is_one(self):
        df = _perfect_rank_frame()
        ic = M.cross_sectional_rank_ic(df, score_col="score", label_col="label")
        assert (ic.dropna() > 0.999).all()
        assert M.mean_rank_ic(ic) == pytest.approx(1.0, abs=1e-6)

    def test_inverted_rank_ic_is_negative_one(self):
        df = _inverted_rank_frame()
        ic = M.cross_sectional_rank_ic(df, score_col="score", label_col="label")
        assert (ic.dropna() < -0.999).all()

    def test_random_rank_ic_is_near_zero_on_average(self):
        df = _random_rank_frame()
        ic = M.cross_sectional_rank_ic(df, score_col="score", label_col="label")
        assert abs(M.mean_rank_ic(ic)) < 0.25

    def test_constant_score_yields_nan_ic(self):
        """A market-wide constant feature (same value for every ticker in the
        group) has zero cross-sectional variance -> IC is NaN, not 0.0 —
        this is the structural-zero hazard the taxonomy note warns about."""
        df = _perfect_rank_frame().copy()
        df["score"] = 5.0  # constant within (and across) every group
        ic = M.cross_sectional_rank_ic(df, score_col="score", label_col="label")
        assert ic.isna().all()

    def test_small_group_below_min_n_is_nan(self):
        df = pd.DataFrame({"timestamp": [1, 1, 1], "score": [1, 2, 3], "label": [1, 2, 3]})
        ic = M.cross_sectional_rank_ic(df, score_col="score", label_col="label", min_group_n=5)
        assert ic.isna().all()

    def test_icir_matches_manual_computation(self):
        ic = pd.Series([0.1, 0.2, 0.3, np.nan, 0.4])
        expected = ic.dropna().mean() / ic.dropna().std(ddof=1)
        assert M.rank_ic_icir(ic) == pytest.approx(expected)


class TestNDCG:
    def test_perfect_ranking_gives_ndcg_one(self):
        score = pd.Series([5, 4, 3, 2, 1])
        label = pd.Series([5, 4, 3, 2, 1])
        assert M.ndcg_at_k_group(score, label, k=3) == pytest.approx(1.0)

    def test_worst_ranking_is_below_one(self):
        score = pd.Series([1, 2, 3, 4, 5])  # exactly inverted vs label
        label = pd.Series([5, 4, 3, 2, 1])
        ndcg = M.ndcg_at_k_group(score, label, k=5)
        assert ndcg < 1.0

    def test_ndcg_handles_negative_labels(self):
        score = pd.Series([3, 1, 2])
        label = pd.Series([-0.1, -0.5, 0.2])  # forward returns can be negative
        ndcg = M.ndcg_at_k_group(score, label, k=3)
        assert not np.isnan(ndcg)


class TestTopN:
    def test_top_n_per_group_selects_correct_rows(self):
        df = pd.DataFrame({
            "timestamp": [1, 1, 1, 2, 2, 2],
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "score": [0.1, 0.9, 0.5, 0.2, 0.8, 0.3],
        })
        top1 = M.top_n_per_group(df, score_col="score", n=1)
        assert set(top1["ticker"]) == {"B", "E"}

    def test_top_n_forward_metrics_mean_and_win_rate(self):
        df = pd.DataFrame({
            "timestamp": [1, 1, 1, 1],
            "ticker": ["A", "B", "C", "D"],
            "score": [0.9, 0.8, 0.1, 0.05],
            "fwd": [0.10, -0.05, 0.01, 0.02],
        })
        out = M.top_n_forward_metrics(df, score_col="score", label_col="fwd", n=2)
        assert out["n_picks"] == 2
        assert out["mean_label"] == pytest.approx((0.10 - 0.05) / 2)
        assert out["win_rate"] == pytest.approx(0.5)

    def test_turnover_zero_for_identical_picks_every_bar(self):
        df = pd.DataFrame({
            "timestamp": [1, 1, 2, 2, 3, 3],
            "ticker": ["A", "B", "A", "B", "A", "B"],
            "score": [1, 0.9, 1, 0.9, 1, 0.9],
        })
        assert M.turnover(df, score_col="score", n=2) == pytest.approx(0.0)

    def test_turnover_one_for_fully_disjoint_picks(self):
        df = pd.DataFrame({
            "timestamp": [1, 1, 2, 2],
            "ticker": ["A", "B", "C", "D"],
            "score": [1, 0.9, 1, 0.9],
        })
        assert M.turnover(df, score_col="score", n=2) == pytest.approx(1.0)


class TestRiskMetrics:
    def test_sharpe_of_constant_positive_return_is_high(self):
        r = pd.Series([0.01] * 20)
        # zero variance -> nan by construction (guarded), use tiny noise instead
        r = r + np.linspace(0, 1e-6, 20)
        sharpe = M.sharpe_ratio(r, periods_per_year=252)
        assert sharpe > 0

    def test_sharpe_nan_for_zero_variance(self):
        r = pd.Series([0.01] * 20)
        assert np.isnan(M.sharpe_ratio(r, periods_per_year=252))

    def test_max_drawdown_matches_manual_calc(self):
        r = pd.Series([0.10, -0.20, 0.05])
        equity = (1 + r).cumprod()
        peak = equity.cummax()
        expected = float(((peak - equity) / peak).max())
        assert M.max_drawdown(r) == pytest.approx(expected)

    def test_max_drawdown_zero_for_monotonic_gains(self):
        r = pd.Series([0.01, 0.02, 0.03])
        assert M.max_drawdown(r) == pytest.approx(0.0)
