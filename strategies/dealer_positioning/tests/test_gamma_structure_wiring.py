"""Tests for the three subsystems this work gave a consumer to.

* the level-dynamics join (features that existed but nothing read),
* dealer-state publication (an adapter that existed but nothing called),
* intraday level dynamics (an archive that existed but nothing derived).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from strategies.dealer_positioning.level_dynamics_feed import (
    DYNAMICS_FEATURE_COLUMNS,
    join_level_dynamics,
)
from strategies.dealer_positioning.scripts.build_intraday_level_dynamics import (
    compute_intraday_dynamics,
    _snapshot_features,
)
from strategies.dealer_positioning.state_publication import (
    IMBALANCE_DEADBAND,
    build_dealer_states,
    dealer_regime_from_structure,
    select_scope_row,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Level-dynamics join
# ---------------------------------------------------------------------------


def _dynamics_frame(dates, *, symbol="SPY", scope="through_month", stability=5.0):
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(dates),
            "scope": [scope] * len(dates),
            "snapshot_date": pd.to_datetime(dates),
            "wall_change_1d": [1.5] * len(dates),
            "level_stability_days": [stability] * len(dates),
            "atr_14d": [2.0] * len(dates),
        }
    )


def _snapshot_frame(dates, *, symbol="SPY", scope="through_month"):
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(dates),
            "scope": [scope] * len(dates),
            "snapshot_date": pd.to_datetime(dates),
        }
    )


def test_dynamics_join_attaches_the_matching_day(tmp_path: Path) -> None:
    path = tmp_path / "dyn.parquet"
    _dynamics_frame(["2026-08-20"]).to_parquet(path, index=False)
    out = join_level_dynamics(_snapshot_frame(["2026-08-20"]), path=path)
    assert out.loc[0, "wall_change_1d"] == 1.5
    assert out.loc[0, "dynamics_days_since_refresh"] == 0


def test_dynamics_join_never_reads_a_later_row(tmp_path: Path) -> None:
    """A dynamics row dated after the snapshot describes structure the snapshot's
    own decision time could not have seen."""
    path = tmp_path / "dyn.parquet"
    _dynamics_frame(["2026-08-25"]).to_parquet(path, index=False)
    out = join_level_dynamics(_snapshot_frame(["2026-08-20"]), path=path)
    assert pd.isna(out.loc[0, "wall_change_1d"])


def test_dynamics_carry_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "dyn.parquet"
    _dynamics_frame(["2026-08-10"]).to_parquet(path, index=False)
    inside = join_level_dynamics(_snapshot_frame(["2026-08-12"]), path=path, max_carry_days=3)
    outside = join_level_dynamics(_snapshot_frame(["2026-08-20"]), path=path, max_carry_days=3)
    assert inside.loc[0, "wall_change_1d"] == 1.5
    assert inside.loc[0, "dynamics_days_since_refresh"] == 2
    assert pd.isna(outside.loc[0, "wall_change_1d"]), "stale dynamics must go null, not carry forever"
    assert pd.isna(outside.loc[0, "dynamics_days_since_refresh"])


def test_dynamics_join_fails_soft_when_the_artifact_is_missing(tmp_path: Path) -> None:
    out = join_level_dynamics(_snapshot_frame(["2026-08-20"]), path=tmp_path / "absent.parquet")
    assert len(out) == 1
    for col in DYNAMICS_FEATURE_COLUMNS:
        assert col in out.columns and pd.isna(out.loc[0, col])


def test_dynamics_join_matches_on_scope_not_just_symbol(tmp_path: Path) -> None:
    path = tmp_path / "dyn.parquet"
    _dynamics_frame(["2026-08-20"], scope="daily_week").to_parquet(path, index=False)
    out = join_level_dynamics(_snapshot_frame(["2026-08-20"], scope="through_month"), path=path)
    assert pd.isna(out.loc[0, "wall_change_1d"])


# ---------------------------------------------------------------------------
# Dealer-state publication
# ---------------------------------------------------------------------------


def _summary_row(**overrides):
    captured = datetime(2026, 8, 25, 19, 45, tzinfo=UTC)
    row = {
        "symbol": "SPY",
        "ticker": "SPY",
        "scope": "through_month",
        "captured_at": captured,
        "snapshot_date": captured.date().isoformat(),
        "spot": 600.0,
        "row_count": 40,
        "dealer_imbalance": 0.4,
        "pinning_score": 2.0,
        "call_wall": 610.0,
        "put_wall": 590.0,
    }
    row.update(overrides)
    return row


def test_regime_mapping_is_volatility_not_direction() -> None:
    """The thesis is that gamma informs dispersion, not direction. The mapping
    must be structurally incapable of asserting a direction."""
    directional = {"UPSIDE_ACCELERATION", "DOWNSIDE_ACCELERATION"}
    for imbalance in (-0.9, -0.2, 0.0, 0.2, 0.9):
        regime = dealer_regime_from_structure(
            _summary_row(dealer_imbalance=imbalance), structure=1.0
        )
        assert regime not in directional


def test_negative_imbalance_maps_to_short_gamma() -> None:
    assert dealer_regime_from_structure(_summary_row(dealer_imbalance=-0.5), structure=1.0) == "SHORT_GAMMA"


def test_pinned_price_maps_to_pinning() -> None:
    pinned = _summary_row(dealer_imbalance=0.5, pinning_score=0.1)
    loose = _summary_row(dealer_imbalance=0.5, pinning_score=3.0)
    assert dealer_regime_from_structure(pinned, structure=1.0) == "PINNING"
    assert dealer_regime_from_structure(loose, structure=1.0) == "POSITIVE_GAMMA"


def test_a_small_imbalance_is_not_read_as_a_regime() -> None:
    tiny = _summary_row(dealer_imbalance=IMBALANCE_DEADBAND / 2)
    assert dealer_regime_from_structure(tiny, structure=1.0) == "NEUTRAL_GAMMA"


def test_low_structure_confidence_asserts_no_regime() -> None:
    """Falling back to UNKNOWN matters: UNKNOWN sizes at 0.75 while
    NEUTRAL_GAMMA sizes at 1.0, so a weak read must not raise size."""
    assert dealer_regime_from_structure(_summary_row(), structure=0.1) is None


def test_published_states_carry_per_row_capture_times(tmp_path: Path) -> None:
    early = datetime(2026, 8, 25, 19, 40, tzinfo=UTC)
    late = datetime(2026, 8, 25, 19, 50, tzinfo=UTC)
    frame = pd.DataFrame(
        [
            _summary_row(symbol="SPY", ticker="SPY", captured_at=early),
            _summary_row(symbol="QQQ", ticker="QQQ", captured_at=late),
        ]
    )
    artifact = tmp_path / "dealer_level_summary.parquet"
    artifact.write_bytes(b"fixture")
    states, skipped = build_dealer_states(frame, captured_at=late, snapshot_path=artifact)
    assert not skipped
    assert {s.ticker for s in states} == {"SPY", "QQQ"}
    assert {s.available_at for s in states} == {early, late}


def test_publication_hashes_the_artifact_it_read(tmp_path: Path) -> None:
    artifact = tmp_path / "dealer_level_summary.parquet"
    artifact.write_bytes(b"fixture-bytes")
    states, _ = build_dealer_states(
        pd.DataFrame([_summary_row()]),
        captured_at=datetime(2026, 8, 25, 19, 45, tzinfo=UTC),
        snapshot_path=artifact,
    )
    lineage = json.loads(states[0].lineage_ids[0])
    assert lineage["record_locator"] == artifact.name
    assert lineage["source_id"] == "dealer-capture"
    # The hash must be of the bytes actually read, not of the path.
    assert lineage["content_hash"] == hashlib.sha256(b"fixture-bytes").hexdigest()


def test_scope_precedence_is_deterministic() -> None:
    rows = pd.DataFrame(
        [_summary_row(scope="two_months"), _summary_row(scope="through_month"), _summary_row(scope="daily_week")]
    )
    assert select_scope_row(rows)["scope"] == "through_month"
    assert select_scope_row(rows.iloc[[0, 2]])["scope"] == "daily_week"
    assert select_scope_row(rows.iloc[0:0]) is None


def test_empty_summary_reports_why_rather_than_silently_publishing_nothing() -> None:
    states, skipped = build_dealer_states(
        pd.DataFrame(), captured_at=datetime(2026, 8, 25, tzinfo=UTC), snapshot_path=Path("x.parquet")
    )
    assert states == []
    assert skipped == {"empty_summary": 1}


# ---------------------------------------------------------------------------
# Intraday level dynamics
# ---------------------------------------------------------------------------


def _series(times, *, call_walls, symbol="SPY"):
    return pd.DataFrame(
        {
            "captured_at": pd.to_datetime(times, utc=True),
            "symbol": [symbol] * len(times),
            "spot": [600.0] * len(times),
            "estimated_net_gex": [1.0e9] * len(times),
            "total_abs_gamma": [2.0e9] * len(times),
            "dealer_imbalance": [0.5] * len(times),
            "gamma_density_1pct": [0.2] * len(times),
            "gamma_concentration": [0.1] * len(times),
            "atm_iv": [20.0] * len(times),
            "call_wall": call_walls,
            "put_wall": [590.0] * len(times),
            "nearest_magnet": [600.0] * len(times),
        }
    )


def test_intraday_delta_measures_wall_migration() -> None:
    times = [f"2026-08-25T14:{m:02d}:00Z" for m in range(0, 31, 5)]
    walls = [610.0, 611.0, 612.0, 613.0, 614.0, 615.0, 616.0]
    out = compute_intraday_dynamics(_series(times, call_walls=walls), horizons=(5, 15))
    assert out["delta_call_wall_5m"].dropna().iloc[-1] == pytest.approx(1.0)
    assert out["delta_call_wall_15m"].dropna().iloc[-1] == pytest.approx(3.0)


def test_a_polling_gap_yields_a_null_delta_not_a_fabricated_move() -> None:
    """The runner polls every 60s. A 40-minute hole is an outage, and a delta
    across it would report structure moving when really nobody was looking."""
    times = ["2026-08-25T14:00:00Z", "2026-08-25T14:40:00Z", "2026-08-25T14:45:00Z"]
    out = compute_intraday_dynamics(_series(times, call_walls=[610.0, 640.0, 641.0]), horizons=(5,))
    assert pd.isna(out.loc[1, "delta_call_wall_5m"])
    assert out.loc[2, "delta_call_wall_5m"] == pytest.approx(1.0)


def test_deltas_never_cross_a_session_boundary() -> None:
    times = ["2026-08-24T19:59:00Z", "2026-08-25T13:31:00Z", "2026-08-25T13:36:00Z"]
    out = compute_intraday_dynamics(_series(times, call_walls=[610.0, 630.0, 631.0]), horizons=(5,))
    assert pd.isna(out.loc[1, "delta_call_wall_5m"]), "overnight is not an intraday move"


def test_deltas_are_backward_looking_only() -> None:
    times = [f"2026-08-25T14:{m:02d}:00Z" for m in range(0, 16, 5)]
    out = compute_intraday_dynamics(_series(times, call_walls=[610.0, 611.0, 612.0, 613.0]), horizons=(5,))
    assert pd.isna(out.loc[0, "delta_call_wall_5m"]), "the first snapshot has no past to compare to"


def test_snapshot_features_locate_walls_on_the_correct_side_of_spot() -> None:
    ladder = pd.DataFrame(
        {
            "strike": [590.0, 600.0, 610.0],
            "spot": [600.0] * 3,
            "call_gex": [10.0, 20.0, 900.0],
            "put_gex": [-800.0, -20.0, -10.0],
            "net_gex": [-790.0, 0.0, 890.0],
            "total_abs_gex": [810.0, 40.0, 910.0],
            "call_iv": [21.0] * 3,
            "put_iv": [21.0] * 3,
        }
    )
    features = _snapshot_features(ladder)
    assert features["call_wall"] == 610.0
    assert features["put_wall"] == 590.0
    assert features["atm_iv"] == pytest.approx(21.0)


def test_empty_ladder_produces_no_snapshot_row() -> None:
    assert _snapshot_features(pd.DataFrame()) == {}
