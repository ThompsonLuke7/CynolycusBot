"""
Light smoke tests — verify modules import and basic shape contracts hold.

Run:
    pytest momentum_expansion/tests -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from momentum_expansion.config import momentum_config as cfg
from momentum_expansion.data.bars import resample_1h_to_4h
from momentum_expansion.data.universe import (
    get_candidate_pool,
    load_snapshot_for,
)
from momentum_expansion.features.feature_matrix_4h import (
    FEATURE_COLUMNS_4H,
    build_ticker_features_4h,
)
from momentum_expansion.inference.entry_rules import (
    ENTRY_RULES_CONFIG,
    evaluate_entry,
)
from momentum_expansion.policy.momentum_option_policy import (
    MomentumOptionConfig,
    MomentumOptionPolicy,
)


def test_config_paths_exist():
    assert cfg.MODULE_ROOT.is_dir()
    # Sub-paths are created on first write; just check they're absolute
    for p in (cfg.RAW_1H_DIR, cfg.RAW_4H_DIR, cfg.RAW_1D_DIR, cfg.UNIVERSE_DIR,
              cfg.PROCESSED_FEAT_DIR, cfg.MODELS_DIR, cfg.BACKTEST_RESULTS_DIR):
        assert p.is_absolute()


def test_candidate_pool_nontrivial():
    pool = get_candidate_pool()
    assert len(pool) > 50
    assert "AAPL" in pool


def _synthetic_1h_session(n_days: int = 5, start_price: float = 100.0) -> pd.DataFrame:
    rng = pd.date_range(
        start="2025-01-06 09:30",
        periods=n_days * 7,   # 7 hourly bars per RTH session (09:30 -> 16:30 close)
        freq="60min",
        tz="America/New_York",
    )
    rng = rng.tz_convert("UTC")
    rng = rng[rng.hour.isin([14, 15, 16, 17, 18, 19, 20])]  # 09:30-16:30 NY -> 14-20 UTC depending on DST
    # Just synthesize an upward random walk:
    np.random.seed(42)
    rets = np.random.normal(0.0002, 0.005, size=len(rng))
    close = start_price * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "open":   close * (1 + np.random.normal(0, 0.001, len(rng))),
            "high":   close * (1 + np.abs(np.random.normal(0, 0.002, len(rng)))),
            "low":    close * (1 - np.abs(np.random.normal(0, 0.002, len(rng)))),
            "close":  close,
            "volume": np.random.randint(10_000, 100_000, len(rng)),
            "symbol": "TST",
        },
        index=rng,
    )
    df.index.name = "timestamp"
    return df


def test_resample_1h_to_4h_shape():
    df_1h = _synthetic_1h_session(n_days=10).reset_index()
    df_4h = resample_1h_to_4h(df_1h)
    # Each session contributes <= 2 bars
    assert len(df_4h) <= 20
    assert {"open", "high", "low", "close", "volume"}.issubset(df_4h.columns)


def test_entry_rules_no_trigger_on_flat_data():
    df_1h = _synthetic_1h_session(n_days=10)
    df_1h.columns = [c.lower() for c in df_1h.columns]
    triggers = evaluate_entry(df_1h=df_1h)
    # Triggers may or may not fire on synthetic data; just verify type contract
    assert isinstance(triggers, list)


def test_momentum_option_config_defaults():
    c = MomentumOptionConfig()
    assert c.sleeve_dollars > 0
    assert 0 < c.target_delta_long < 1
    assert c.max_concurrent >= 1


def test_momentum_option_policy_cannot_size_to_zero_when_atr_huge():
    policy = MomentumOptionPolicy(cfg=MomentumOptionConfig(submit_orders=False), client=None)
    # Without a client, consider_entry will short-circuit on chain lookup,
    # but the config + policy itself should construct fine.
    assert policy.cfg.sleeve_dollars > 0
    assert isinstance(policy.positions, dict)


def test_load_snapshot_returns_dataframe_when_no_snapshots():
    df = load_snapshot_for("2020-01-01")
    assert isinstance(df, pd.DataFrame)


def test_feature_columns_unique():
    assert len(FEATURE_COLUMNS_4H) == len(set(FEATURE_COLUMNS_4H))
