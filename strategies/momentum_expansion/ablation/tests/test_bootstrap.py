"""Tests for strategies/momentum_expansion/ablation/bootstrap.py (synthetic only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.ablation.bootstrap import bh_fdr, week_block_bootstrap_ci, week_of


class TestWeekOf:
    def test_same_iso_week_maps_to_same_key(self):
        ts = pd.Series(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"], utc=True))  # Mon/Tue/Wed same ISO week
        w = week_of(ts)
        assert w.nunique() == 1

    def test_different_weeks_map_to_different_keys(self):
        ts = pd.Series(pd.to_datetime(["2026-01-05", "2026-01-13"], utc=True))
        w = week_of(ts)
        assert w.nunique() == 2


class TestWeekBlockBootstrapCI:
    def test_ci_excludes_zero_for_clearly_positive_signal(self):
        rng = np.random.default_rng(0)
        n_weeks = 30
        rows = []
        for wk in range(n_weeks):
            for _ in range(10):
                rows.append({"value": rng.normal(1.0, 0.2), "week": f"W{wk}"})
        df = pd.DataFrame(rows)
        out = week_block_bootstrap_ci(df["value"], df["week"], n_boot=500, seed=1)
        assert out["point"] == pytest.approx(1.0, abs=0.1)
        assert out["excludes_zero"] is True
        assert out["ci_lo"] > 0

    def test_ci_does_not_exclude_zero_for_pure_noise(self):
        rng = np.random.default_rng(0)
        n_weeks = 30
        rows = []
        for wk in range(n_weeks):
            for _ in range(10):
                rows.append({"value": rng.normal(0.0, 1.0), "week": f"W{wk}"})
        df = pd.DataFrame(rows)
        out = week_block_bootstrap_ci(df["value"], df["week"], n_boot=500, seed=1)
        assert out["excludes_zero"] is False

    def test_single_week_returns_point_estimate_and_nan_ci(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "week": ["W1", "W1", "W1"]})
        out = week_block_bootstrap_ci(df["value"], df["week"], n_boot=100)
        assert out["n_weeks"] == 1
        assert out["point"] == pytest.approx(2.0)
        assert np.isnan(out["ci_lo"])
        assert np.isnan(out["ci_hi"])
        assert out["excludes_zero"] is False

    def test_empty_input_returns_all_nan(self):
        out = week_block_bootstrap_ci(pd.Series(dtype=float), pd.Series(dtype=object))
        assert out["n_weeks"] == 0
        assert np.isnan(out["point"])

    def test_block_resampling_respects_within_week_autocorrelation(self):
        """A per-row bootstrap on this data would show a tight CI around 0
        because most rows are iid noise; the true signal lives entirely in
        one anomalous week. Week-block resampling should let that week's
        block occasionally dominate a draw, widening the CI relative to a
        naive row-level bootstrap — i.e. it must not be equivalent to it."""
        rng = np.random.default_rng(3)
        rows = []
        for wk in range(20):
            val = 5.0 if wk == 0 else rng.normal(0, 0.1)
            for _ in range(50):
                rows.append({"value": val, "week": f"W{wk}"})
        df = pd.DataFrame(rows)
        week_boot = week_block_bootstrap_ci(df["value"], df["week"], n_boot=1000, seed=5)
        # Naive row-level bootstrap CI (ignores week blocks) for comparison.
        rng2 = np.random.default_rng(5)
        row_vals = df["value"].to_numpy()
        row_boot_means = [rng2.choice(row_vals, size=len(row_vals), replace=True).mean() for _ in range(1000)]
        row_width = np.quantile(row_boot_means, 0.95) - np.quantile(row_boot_means, 0.05)
        week_width = week_boot["ci_hi"] - week_boot["ci_lo"]
        assert week_width > row_width


class TestBHFDR:
    def test_all_significant_p_values_survive(self):
        p = pd.Series([0.001, 0.002, 0.003, 0.004])
        q = bh_fdr(p)
        assert (q <= 0.10).all()

    def test_all_null_p_values_do_not_survive(self):
        p = pd.Series(np.random.default_rng(0).uniform(0.3, 1.0, 20))
        q = bh_fdr(p)
        assert (q > 0.10).all()

    def test_nan_passthrough(self):
        p = pd.Series([0.01, np.nan, 0.5])
        q = bh_fdr(p)
        assert np.isnan(q.iloc[1])

    def test_reused_not_reimplemented(self):
        from scripts.confluence_discovery.search import bh_fdr as original_bh_fdr
        assert bh_fdr is original_bh_fdr
