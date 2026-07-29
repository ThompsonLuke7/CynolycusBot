"""Tests for strategies/momentum_expansion/ablation/feature_blocks.py."""
from __future__ import annotations

from strategies.momentum_expansion.ablation.feature_blocks import (
    ARM_ADDED_COLUMNS,
    ARMS,
    get_arm_columns,
)
from strategies.momentum_expansion.features.feature_matrix_4h import (
    FEATURE_COLUMNS_4H,
    REGIME_FEATURE_COLUMNS_4H,
)


class TestArms:
    def test_five_arms_defined(self):
        assert set(ARMS.keys()) == {"baseline", "+risk", "+liquidity", "+sector", "+all"}

    def test_baseline_equals_feature_columns_4h(self):
        assert ARMS["baseline"] == FEATURE_COLUMNS_4H

    def test_every_non_baseline_arm_is_a_superset_of_baseline(self):
        for name, cols in ARMS.items():
            if name == "baseline":
                continue
            assert set(FEATURE_COLUMNS_4H).issubset(set(cols))

    def test_added_columns_only_come_from_regime_feature_columns_4h(self):
        for name, added in ARM_ADDED_COLUMNS.items():
            assert set(added).issubset(set(REGIME_FEATURE_COLUMNS_4H) | {"regime_available", "regime_stale_days"})

    def test_all_arm_covers_every_regime_feature_column(self):
        assert set(REGIME_FEATURE_COLUMNS_4H).issubset(set(ARMS["+all"]))

    def test_arms_contain_no_duplicate_columns(self):
        for name, cols in ARMS.items():
            assert len(cols) == len(set(cols)), f"{name} has duplicate columns"

    def test_risk_liquidity_sector_are_disjoint_additions(self):
        risk = set(ARM_ADDED_COLUMNS["+risk"]) - {"regime_available", "regime_stale_days"}
        liq = set(ARM_ADDED_COLUMNS["+liquidity"]) - {"regime_available", "regime_stale_days"}
        sect = set(ARM_ADDED_COLUMNS["+sector"]) - {"regime_available", "regime_stale_days"}
        assert risk.isdisjoint(liq)
        assert risk.isdisjoint(sect)
        assert liq.isdisjoint(sect)

    def test_risk_liquidity_sector_additions_union_to_all_minus_shared(self):
        union = (
            set(ARM_ADDED_COLUMNS["+risk"])
            | set(ARM_ADDED_COLUMNS["+liquidity"])
            | set(ARM_ADDED_COLUMNS["+sector"])
        )
        assert union == set(ARM_ADDED_COLUMNS["+all"])

    def test_get_arm_columns_returns_copy(self):
        cols = get_arm_columns("+all")
        cols.append("mutated")
        assert "mutated" not in ARMS["+all"]

    def test_get_arm_columns_unknown_name_raises(self):
        import pytest
        with pytest.raises(KeyError):
            get_arm_columns("+not_a_real_arm")
