"""Tests for strategies/momentum_expansion/ablation/export_colab_ablation.py.

Uses small synthetic parquet fixtures (never the real multi-GB production
matrix) written to a tmp_path, and asserts the export never touches the real
production TRAINING_MATRIX path.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.ablation import export_colab_ablation as E
from strategies.momentum_expansion.ablation.feature_blocks import ARMS
from strategies.momentum_expansion.features import feature_matrix_4h as fm4h


def _tiny_production_matrix(tmp_path, n_bars=20, tickers=("AAPL", "XOM")) -> "pd.io.parquet":
    rng = np.random.default_rng(0)
    ts = pd.date_range("2021-06-01", periods=n_bars, freq="4h", tz="UTC")
    rows = []
    for t in ts:
        for tk in tickers:
            row = {"timestamp": t, "ticker": tk}
            for col in fm4h.FEATURE_COLUMNS_4H:
                row[col] = rng.normal(0, 1)
            for col in ["fwd_max_return", "fwd_max_alpha", "fwd_atr_adj_return",
                        "fwd_max_drawdown", "fwd_close_return", "trend_persistence",
                        "expansion_score", "expansion_target", "expansion_survival_score"]:
                row[col] = rng.uniform(0, 1)
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["timestamp", "ticker"])
    path = tmp_path / "tiny_training_matrix.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def patched_tables(monkeypatch):
    from signals.market_regime.timeutil import available_at_for_dates

    calendar = pd.bdate_range("2021-01-04", periods=300)
    available_at = available_at_for_dates(pd.DatetimeIndex(calendar), settle_margin_minutes=30)
    rng = np.random.default_rng(1)
    daily_regime = pd.DataFrame({
        "date": calendar, "available_at": available_at,
        "risk_appetite_z": rng.normal(0, 1, len(calendar)),
        "liquidity_stress_z": rng.normal(0, 1, len(calendar)),
        "credit_risk_z": rng.normal(0, 1, len(calendar)),
        "sector_dispersion_raw": rng.uniform(0, 0.05, len(calendar)),
        "breadth_z": rng.normal(0, 1, len(calendar)),
    })
    frames = []
    for etf in ["XLK", "XLE"]:
        frames.append(pd.DataFrame({
            "date": calendar, "sector_etf": etf, "available_at": available_at,
            "excess_21d": rng.normal(0, 0.05, len(calendar)), "excess_63d": rng.normal(0, 0.05, len(calendar)),
            "rank_21d": rng.uniform(0, 1, len(calendar)), "rank_63d": rng.uniform(0, 1, len(calendar)),
            "rs_accel": rng.normal(0, 0.02, len(calendar)),
            "above_20d": rng.integers(0, 2, len(calendar)).astype(float),
            "above_50d": rng.integers(0, 2, len(calendar)).astype(float),
        }))
    sector_state = pd.concat(frames, ignore_index=True)
    monkeypatch.setattr(fm4h, "_load_daily_regime_table", lambda: daily_regime)
    monkeypatch.setattr(fm4h, "_load_sector_state_table", lambda: sector_state)


class TestBuildAblationTrainingMatrix:
    def test_never_touches_production_matrix_path(self, tmp_path, patched_tables):
        matrix_path = _tiny_production_matrix(tmp_path)
        before_mtime = matrix_path.stat().st_mtime
        out_path = tmp_path / "ablation_matrix.parquet"
        E.build_ablation_training_matrix(matrix_path=matrix_path, out_path=out_path, force=True)
        assert matrix_path.stat().st_mtime == before_mtime
        assert out_path.exists()
        assert out_path != matrix_path

    def test_augmented_matrix_has_regime_columns_production_lacks(self, tmp_path, patched_tables):
        matrix_path = _tiny_production_matrix(tmp_path)
        out_path = tmp_path / "ablation_matrix.parquet"
        augmented = E.build_ablation_training_matrix(matrix_path=matrix_path, out_path=out_path, force=True)
        production = pd.read_parquet(matrix_path)
        for col in fm4h.REGIME_FEATURE_COLUMNS_4H:
            assert col not in production.columns
            assert col in augmented.columns

    def test_missing_matrix_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            E.build_ablation_training_matrix(matrix_path=tmp_path / "nope.parquet", out_path=tmp_path / "out.parquet")

    def test_reuses_cached_output_without_force(self, tmp_path, patched_tables):
        matrix_path = _tiny_production_matrix(tmp_path)
        out_path = tmp_path / "ablation_matrix.parquet"
        first = E.build_ablation_training_matrix(matrix_path=matrix_path, out_path=out_path, force=True)
        out_path_mtime = out_path.stat().st_mtime
        second = E.build_ablation_training_matrix(matrix_path=matrix_path, out_path=out_path, force=False)
        assert out_path.stat().st_mtime == out_path_mtime
        assert len(first) == len(second)


class TestExportAblationBundle:
    def test_bundle_contains_manifest_and_matrix_and_harness(self, tmp_path, patched_tables):
        matrix_path = _tiny_production_matrix(tmp_path)
        out_dir = tmp_path / "export"
        bundle_path = E.export_ablation_bundle(matrix_path=matrix_path, out_dir=out_dir, force=True)
        assert bundle_path.exists()

        manifest = json.loads((out_dir / "ablation_feature_manifest.json").read_text())
        assert set(manifest["arms"].keys()) == set(ARMS.keys())
        assert manifest["banned_target_column"] == "trend_persistence"
        assert (out_dir / "training_matrix_4h_with_regime.parquet").exists()
        assert (out_dir / "colab_competition.py").exists()

        import tarfile
        with tarfile.open(bundle_path, "r:gz") as tar:
            names = tar.getnames()
        assert "ablation_feature_manifest.json" in names
        assert "training_matrix_4h_with_regime.parquet" in names
        assert "colab_competition.py" in names

    def test_arm_feature_lists_are_filtered_to_present_columns(self, tmp_path, patched_tables):
        matrix_path = _tiny_production_matrix(tmp_path)
        out_dir = tmp_path / "export"
        E.export_ablation_bundle(matrix_path=matrix_path, out_dir=out_dir, force=True)
        manifest = json.loads((out_dir / "ablation_feature_manifest.json").read_text())
        augmented = pd.read_parquet(out_dir / "training_matrix_4h_with_regime.parquet")
        for arm, cols in manifest["arms"].items():
            assert set(cols).issubset(set(augmented.columns))
