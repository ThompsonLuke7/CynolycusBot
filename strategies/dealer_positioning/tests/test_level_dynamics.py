from __future__ import annotations

import pandas as pd
import pytest

from strategies.dealer_positioning.scripts.build_level_dynamics import (
    STALE_GAP_DAYS,
    _normalize,
    compute_strike_dynamics,
    compute_summary_dynamics,
)

pytestmark = pytest.mark.safe


def _summary_row(
    date: str,
    *,
    captured_hour: str = "19:45",
    call_wall: float,
    put_wall: float,
    gamma_flip: float,
    gex_concentration_index: float,
    total_option_oi: float,
    total_option_volume: float,
    spot: float = 100.0,
    nearest_magnet: float = 105.0,
    symbol: str = "TST",
    scope: str = "daily_week",
) -> dict:
    return {
        "captured_at": f"{date}T{captured_hour}:00Z",
        "snapshot_date": date,
        "symbol": symbol,
        "scope": scope,
        "spot": spot,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "nearest_magnet": nearest_magnet,
        "gex_concentration_index": gex_concentration_index,
        "total_option_oi": total_option_oi,
        "total_option_volume": total_option_volume,
    }


def _ladder_row(
    date: str,
    strike: float,
    *,
    call_oi: float,
    put_oi: float,
    call_volume: float = 5.0,
    put_volume: float = 3.0,
    call_iv: float = 0.30,
    put_iv: float = 0.35,
    call_delta: float = 0.25,
    put_delta: float = -0.25,
    spot: float = 100.0,
    symbol: str = "TST",
    scope: str = "daily_week",
) -> dict:
    return {
        "captured_at": f"{date}T19:45:00Z",
        "snapshot_date": date,
        "symbol": symbol,
        "scope": scope,
        "spot": spot,
        "strike": strike,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "call_volume": call_volume,
        "put_volume": put_volume,
        "call_iv": call_iv,
        "put_iv": put_iv,
        "call_delta": call_delta,
        "put_delta": put_delta,
    }


def _build_history():
    """Three snapshots for TST with a real calendar gap between the 2nd and
    3rd (07-03 -> 07-08, i.e. a missed capture window), mirroring the real
    dataset's gaps around the July 4th holiday week."""
    summary = pd.DataFrame(
        [
            _summary_row(
                "2026-07-02", call_wall=110.0, put_wall=90.0, gamma_flip=100.0,
                gex_concentration_index=0.50, total_option_oi=1000.0, total_option_volume=200.0,
            ),
            _summary_row(
                "2026-07-03", call_wall=110.0, put_wall=90.0, gamma_flip=100.5,
                gex_concentration_index=0.51, total_option_oi=1100.0, total_option_volume=210.0, spot=101.0,
            ),
            _summary_row(
                "2026-07-08", call_wall=112.0, put_wall=90.0, gamma_flip=101.0,
                gex_concentration_index=0.52, total_option_oi=1200.0, total_option_volume=220.0, spot=102.0,
            ),
        ]
    )
    ladder = pd.DataFrame(
        [
            _ladder_row("2026-07-02", 90.0, call_oi=100, put_oi=200, spot=100.0),
            _ladder_row("2026-07-02", 110.0, call_oi=300, put_oi=50, spot=100.0),
            _ladder_row("2026-07-02", 112.0, call_oi=40, put_oi=10, spot=100.0),
            _ladder_row("2026-07-03", 90.0, call_oi=105, put_oi=202, spot=101.0),
            _ladder_row("2026-07-03", 110.0, call_oi=310, put_oi=50, spot=101.0),
            _ladder_row("2026-07-03", 112.0, call_oi=40, put_oi=10, spot=101.0),
            _ladder_row("2026-07-08", 90.0, call_oi=110, put_oi=204, spot=102.0),
            _ladder_row("2026-07-08", 110.0, call_oi=320, put_oi=50, spot=102.0),
            _ladder_row("2026-07-08", 112.0, call_oi=40, put_oi=10, spot=102.0),
        ]
    )
    return _normalize(summary), _normalize(ladder)


# Deterministic ATR lookup so distance_to_*_wall_atr tests don't depend on
# real cached daily bars.
def _atr_lookup(summary: pd.DataFrame) -> dict:
    return {(row.symbol, row.snapshot_date): 2.0 for row in summary.itertuples()}


def test_wall_change_and_gap_days_use_actual_prior_snapshot_not_fixed_offset():
    summary, ladder = _build_history()
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary))
    out = out.sort_values("snapshot_date").reset_index(drop=True)

    # Day 1: no prior snapshot at all -> NaN, not zero.
    assert pd.isna(out.loc[0, "wall_change_1d"])
    assert bool(out.loc[0, "has_prior_snapshot_1d"]) is False

    # Day 2 vs day 1: call_wall unchanged (110->110), put_wall unchanged (90->90) -> mean(|0|,|0|) = 0.
    assert out.loc[1, "wall_change_1d"] == pytest.approx(0.0)
    assert out.loc[1, "wall_change_1d_gap_days"] == 1

    # Day 3 vs day 2: call_wall moved 110->112 (=2), put_wall unchanged (=0) -> mean = 1.0.
    assert out.loc[2, "wall_change_1d"] == pytest.approx(1.0)
    # The real gap is 5 calendar days (07-03 -> 07-08), not 1 -- must be exposed, not hidden.
    assert out.loc[2, "wall_change_1d_gap_days"] == 5


def test_wall_change_3d_compares_against_third_prior_available_snapshot():
    dates = ["2026-07-08", "2026-07-09", "2026-07-13", "2026-07-14", "2026-07-15"]
    call_walls = [100.0, 100.0, 101.0, 103.0, 106.0]
    rows = [
        _summary_row(
            d, call_wall=cw, put_wall=90.0, gamma_flip=100.0, gex_concentration_index=0.5,
            total_option_oi=1000.0, total_option_volume=200.0,
        )
        for d, cw in zip(dates, call_walls)
    ]
    summary = _normalize(pd.DataFrame(rows))
    ladder = _normalize(pd.DataFrame([_ladder_row(dates[0], 90.0, call_oi=1, put_oi=1)]))
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary))
    out = out.sort_values("snapshot_date").reset_index(drop=True)

    # First 3 rows have fewer than 3 prior snapshots -> NaN wall_change_3d.
    assert out["wall_change_3d"].isna().tolist()[:3] == [True, True, True]
    # Row index 3 (2026-07-14): 3rd-prior available snapshot is index 0 (2026-07-08), call_wall 100->103 = 3,
    # put_wall unchanged (90->90) = 0 -> mean = 1.5. Gap is the real 6 calendar days, not a fixed "3".
    assert out.loc[3, "wall_change_3d"] == pytest.approx(1.5)
    assert out.loc[3, "wall_change_3d_gap_days"] == 6
    assert bool(out.loc[3, "third_gap_stale"]) is True  # 6 > STALE_GAP_DAYS(4)
    # Row index 4 (2026-07-15): 3rd-prior is index 1 (2026-07-09), call_wall 100->106 = 6, put unchanged = 0 -> mean 3.0.
    assert out.loc[4, "wall_change_3d"] == pytest.approx(3.0)
    assert out.loc[4, "wall_change_3d_gap_days"] == 6


def test_prior_gap_stale_flag_fires_only_past_the_threshold():
    summary, ladder = _build_history()
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary)).sort_values("snapshot_date")
    gaps = out["prior_gap_days"].tolist()
    stale = out["prior_gap_stale"].tolist()
    # day1: no prior -> NaN gap -> stale True (conservative, distinguishable via has_prior_snapshot_1d)
    assert pd.isna(gaps[0]) and stale[0] is True
    # day2: 1-day gap, under STALE_GAP_DAYS -> not stale
    assert gaps[1] == 1 and stale[1] is False
    # day3: 5-day gap, over STALE_GAP_DAYS(=4) -> stale
    assert gaps[2] == 5 and stale[2] is True
    assert STALE_GAP_DAYS == 4


def test_gex_concentration_change_and_gamma_flip_velocity():
    summary, ladder = _build_history()
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary)).sort_values("snapshot_date").reset_index(drop=True)

    assert pd.isna(out.loc[0, "gex_concentration_change"])
    assert out.loc[1, "gex_concentration_change"] == pytest.approx(0.01)
    assert out.loc[2, "gex_concentration_change"] == pytest.approx(0.01)

    # day2: gamma_flip 100.0 -> 100.5 over 1 day => velocity 0.5/day
    assert out.loc[1, "gamma_flip_velocity"] == pytest.approx(0.5)
    # day3: gamma_flip 100.5 -> 101.0 over the actual 5-day gap => velocity 0.1/day, not 0.5
    assert out.loc[2, "gamma_flip_velocity"] == pytest.approx(0.1)


def test_oi_change_by_dte_and_volume_to_prior_oi_scope_level():
    summary, ladder = _build_history()
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary)).sort_values("snapshot_date").reset_index(drop=True)

    # total_option_oi is the scope's (== dte-bucket's) total OI; oi_change_by_dte is its snapshot-over-snapshot delta.
    assert pd.isna(out.loc[0, "oi_change_by_dte"])
    assert out.loc[1, "oi_change_by_dte"] == pytest.approx(100.0)
    assert out.loc[2, "oi_change_by_dte"] == pytest.approx(100.0)

    # volume(t) / prior total_option_oi -- leak-free since volume(t) belongs to today's snapshot
    # and prior oi was captured strictly before today.
    assert out.loc[1, "volume_to_prior_oi"] == pytest.approx(210.0 / 1000.0)
    assert out.loc[2, "volume_to_prior_oi"] == pytest.approx(220.0 / 1100.0)


def test_level_stability_days_counts_consecutive_unchanged_walls():
    summary, ladder = _build_history()
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary)).sort_values("snapshot_date").reset_index(drop=True)

    assert out.loc[0, "level_stability_days"] == 1  # first known day
    assert out.loc[1, "level_stability_days"] == 2  # walls unchanged vs day1
    assert out.loc[2, "level_stability_days"] == 1  # call_wall moved -> resets


def test_level_stability_days_treats_missing_level_as_changed_not_equal():
    summary, ladder = _build_history()
    summary = summary.copy()
    summary.loc[summary["snapshot_date"] == pd.Timestamp("2026-07-03"), "call_wall"] = float("nan")
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary)).sort_values("snapshot_date").reset_index(drop=True)
    # day2 has a missing call_wall -> not "stable" vs day1, resets to 1.
    assert out.loc[1, "level_stability_days"] == 1
    # day3's call_wall (112.0) differs from day2's NaN -> also resets, not treated as equal to NaN.
    assert out.loc[2, "level_stability_days"] == 1


def test_distance_to_wall_atr_uses_injected_atr_and_is_nan_without_one():
    summary, ladder = _build_history()
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary)).sort_values("snapshot_date").reset_index(drop=True)

    # day2: spot=101, call_wall=110, put_wall=90, atr=2.0
    assert out.loc[1, "distance_to_call_wall_atr"] == pytest.approx(abs(101.0 - 110.0) / 2.0)
    assert out.loc[1, "distance_to_put_wall_atr"] == pytest.approx(abs(101.0 - 90.0) / 2.0)

    # No ATR available for this symbol/date -> NaN, never a fabricated distance.
    out_no_atr = compute_summary_dynamics(summary, ladder, atr_lookup={})
    assert out_no_atr["distance_to_call_wall_atr"].isna().all()
    assert out_no_atr["distance_to_put_wall_atr"].isna().all()


def test_iv_skew_change_uses_nearest_25_delta_call_and_put():
    summary, ladder = _build_history()
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary)).sort_values("snapshot_date").reset_index(drop=True)
    # All fixture rows use call_delta=0.25, put_delta=-0.25, call_iv=0.30, put_iv=0.35 -> skew=0.05 every day.
    assert out["iv_skew_25d"].round(4).tolist() == [0.05, 0.05, 0.05]
    assert pd.isna(out.loc[0, "iv_skew_change"])
    assert out.loc[1, "iv_skew_change"] == pytest.approx(0.0)


def test_near_level_option_volume_share():
    summary, ladder = _build_history()
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary)).sort_values("snapshot_date").reset_index(drop=True)
    # Day1: strikes 90 (== put_wall), 110 (== call_wall) are "near"; 112 is not (>0.5 band from any level).
    # near volume = 8+8=16 of total 24 -> 0.6667.
    assert out.loc[0, "near_level_option_volume_share"] == pytest.approx(16.0 / 24.0)


def test_oi_change_by_strike_matches_hand_computed_deltas_and_new_strike_is_nan():
    summary, ladder = _build_history()
    sd = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary))
    strike_out = compute_strike_dynamics(sd, ladder).sort_values(["snapshot_date", "strike"]).reset_index(drop=True)

    day2_strike90 = strike_out[(strike_out["snapshot_date"] == pd.Timestamp("2026-07-03")) & (strike_out["strike"] == 90.0)].iloc[0]
    assert day2_strike90["call_oi_change"] == pytest.approx(5.0)  # 105 - 100
    assert day2_strike90["put_oi_change"] == pytest.approx(2.0)  # 202 - 200
    assert day2_strike90["oi_change_by_strike"] == pytest.approx(7.0)
    assert day2_strike90["volume_to_prior_oi"] == pytest.approx(8.0 / 300.0)  # (5+3) / (100+200)

    day1_rows = strike_out[strike_out["snapshot_date"] == pd.Timestamp("2026-07-02")]
    assert day1_rows["call_oi_change"].isna().all()
    assert day1_rows["oi_change_by_strike"].isna().all()


def test_oi_change_by_strike_is_nan_for_a_strike_absent_from_the_prior_snapshot():
    summary = pd.DataFrame(
        [
            _summary_row(
                "2026-07-02", call_wall=110.0, put_wall=90.0, gamma_flip=100.0,
                gex_concentration_index=0.50, total_option_oi=1000.0, total_option_volume=200.0,
            ),
            _summary_row(
                "2026-07-03", call_wall=110.0, put_wall=90.0, gamma_flip=100.0,
                gex_concentration_index=0.50, total_option_oi=1005.0, total_option_volume=200.0,
            ),
        ]
    )
    ladder = pd.DataFrame(
        [
            _ladder_row("2026-07-02", 90.0, call_oi=100, put_oi=200),
            _ladder_row("2026-07-03", 90.0, call_oi=100, put_oi=200),
            _ladder_row("2026-07-03", 115.0, call_oi=5, put_oi=0),  # new strike, no prior row
        ]
    )
    summary, ladder = _normalize(summary), _normalize(ladder)
    sd = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary))
    strike_out = compute_strike_dynamics(sd, ladder)

    new_strike_row = strike_out[strike_out["strike"] == 115.0].iloc[0]
    assert pd.isna(new_strike_row["oi_change_by_strike"])
    assert pd.isna(new_strike_row["volume_to_prior_oi"])


def test_available_at_matches_captured_at_for_leakage_safe_joins():
    summary, ladder = _build_history()
    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary))
    assert (out["available_at"] == out["captured_at"]).all()

    sd = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary))
    strike_out = compute_strike_dynamics(sd, ladder)
    merged = strike_out.merge(
        out[["symbol", "scope", "snapshot_date", "captured_at"]], on=["symbol", "scope", "snapshot_date"]
    )
    assert (merged["available_at"] == merged["captured_at"]).all()


def test_scopes_are_never_mixed_when_computing_changes():
    rows = []
    for scope, call_wall in [("daily_week", 110.0), ("through_month", 130.0)]:
        rows.append(_summary_row("2026-07-02", call_wall=call_wall, put_wall=90.0, gamma_flip=100.0,
                                  gex_concentration_index=0.5, total_option_oi=1000.0, total_option_volume=200.0,
                                  scope=scope))
        rows.append(_summary_row("2026-07-03", call_wall=call_wall + 1.0, put_wall=90.0, gamma_flip=100.0,
                                  gex_concentration_index=0.5, total_option_oi=1000.0, total_option_volume=200.0,
                                  scope=scope))
    summary = _normalize(pd.DataFrame(rows))
    ladder = _normalize(pd.DataFrame([_ladder_row("2026-07-02", 90.0, call_oi=1, put_oi=1)]))

    out = compute_summary_dynamics(summary, ladder, atr_lookup=_atr_lookup(summary))
    day2 = out[out["snapshot_date"] == pd.Timestamp("2026-07-03")]
    # Each scope's wall_change_1d must be computed against its own scope's prior row (call_wall +1 each), never
    # cross-contaminated by the other scope's much larger absolute wall level.
    assert set(day2["wall_change_1d"].round(4)) == {0.5}  # mean(|+1|, |0|) = 0.5 for both scopes independently


def test_empty_history_returns_expected_columns_and_no_rows():
    empty_summary = _normalize(pd.DataFrame(columns=["captured_at", "snapshot_date", "symbol", "scope"]))
    empty_ladder = _normalize(pd.DataFrame(columns=["captured_at", "snapshot_date", "symbol", "scope", "strike"]))
    out = compute_summary_dynamics(empty_summary, empty_ladder)
    assert out.empty
    for col in ["wall_change_1d", "wall_change_3d", "gex_concentration_change", "gamma_flip_velocity",
                "distance_to_call_wall_atr", "distance_to_put_wall_atr", "level_stability_days",
                "oi_change_by_dte", "volume_to_prior_oi", "iv_skew_change", "near_level_option_volume_share"]:
        assert col in out.columns

    strike_out = compute_strike_dynamics(out, empty_ladder)
    assert strike_out.empty
