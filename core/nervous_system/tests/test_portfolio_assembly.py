"""Runtime assembly of the sector map from empirical assignments.

The curated map covers 95/2,903 live names (3.3%), under which the sector arm
of the concentration gate is inert: ~97% of positions bucket as UNALLOCATED and
UNALLOCATED deliberately never vetoes.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from core.nervous_system.config.portfolio import MVP_PORTFOLIO_CONFIG
from core.nervous_system.config.portfolio_assembly import (
    load_sector_map,
    portfolio_config_with_sectors,
)


def write_assignments(path, rows) -> None:
    pd.DataFrame(rows).to_parquet(path)


def test_a_missing_file_degrades_to_the_curated_map(tmp_path) -> None:
    """A missing research artifact must not fail a trading process."""

    missing = tmp_path / "nope.parquet"
    assert load_sector_map(missing) == {}
    config = portfolio_config_with_sectors(path=missing)
    assert config is MVP_PORTFOLIO_CONFIG


def test_weak_correlations_are_left_unallocated(tmp_path) -> None:
    """A weak correlation is not an assignment; UNALLOCATED says so."""

    path = tmp_path / "s.parquet"
    write_assignments(path, [
        {"ticker": "STRONG", "sector_etf": "XLK", "correlation": 0.62, "curated": False},
        {"ticker": "WEAK", "sector_etf": "XLK", "correlation": 0.11, "curated": False},
    ])
    mapping = load_sector_map(path, min_correlation=0.30)
    assert mapping == {"STRONG": "XLK"}


def test_a_curated_name_survives_a_weak_resolver_correlation(tmp_path) -> None:
    """Correlation describes the resolver's confidence, not the hand map's."""

    path = tmp_path / "s.parquet"
    write_assignments(path, [
        {"ticker": "AMD", "sector_etf": "XLK", "correlation": 0.05, "curated": True},
    ])
    assert load_sector_map(path, min_correlation=0.30) == {"AMD": "XLK"}


def test_the_curated_map_wins_on_conflict(tmp_path) -> None:
    """The resolver agreed with the hand map on only 77.9% of curated names."""

    path = tmp_path / "s.parquet"
    write_assignments(path, [
        {"ticker": "AMD", "sector_etf": "XLE", "correlation": 0.90, "curated": False},
        {"ticker": "NEWNAME", "sector_etf": "XLV", "correlation": 0.55, "curated": False},
    ])
    config = portfolio_config_with_sectors(path=path)
    # MVP_PORTFOLIO_CONFIG maps AMD to XLK; the empirical XLE must not win.
    assert config.sector_for("AMD") == "XLK"
    assert config.sector_for("NEWNAME") == "XLV"


def test_widening_changes_the_config_version(tmp_path) -> None:
    """Two runs with different coverage are not the same configuration."""

    path = tmp_path / "s.parquet"
    write_assignments(path, [
        {"ticker": "NEWNAME", "sector_etf": "XLV", "correlation": 0.55, "curated": False},
    ])
    config = portfolio_config_with_sectors(path=path)
    assert config.config_version != MVP_PORTFOLIO_CONFIG.config_version
    assert "sectors@" in config.config_version


def test_missing_columns_fail_loudly(tmp_path) -> None:
    path = tmp_path / "s.parquet"
    write_assignments(path, [{"ticker": "AMD"}])
    with pytest.raises(ValueError, match="missing columns"):
        load_sector_map(path)


def test_a_widened_map_makes_the_sector_bucket_real(tmp_path) -> None:
    """The point of all of it: sector exposure stops being UNALLOCATED."""

    from uuid import NAMESPACE_URL, uuid5

    from core.nervous_system.portfolio.exposure import UNALLOCATED, calculate_exposure
    from core.nervous_system.tests.test_portfolio_exposure import (
        build_snapshot, equity_position, portfolio_state,
    )

    path = tmp_path / "s.parquet"
    write_assignments(path, [
        {"ticker": "ZZTOP", "sector_etf": "XLK", "correlation": 0.71, "curated": False},
    ])
    portfolio = portfolio_state(
        positions=(equity_position(symbol="ZZTOP", quantity=100, market_value=10_000.0),)
    )
    snapshot = build_snapshot()

    before = calculate_exposure(
        portfolio, snapshot, config=MVP_PORTFOLIO_CONFIG,
        report_id=uuid5(NAMESPACE_URL, "assembly-test/before"),
    )
    after = calculate_exposure(
        portfolio, snapshot, config=portfolio_config_with_sectors(path=path),
        report_id=uuid5(NAMESPACE_URL, "assembly-test/after"),
    )
    assert before.sector_notional.get(UNALLOCATED) == Decimal("10000.00")
    assert after.sector_notional.get("XLK") == Decimal("10000.00")
    assert UNALLOCATED not in after.sector_notional
