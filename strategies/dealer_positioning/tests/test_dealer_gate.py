from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from strategies.dealer_positioning.gate import (
    SCOPE_MONTHLY,
    SCOPE_NEAREST,
    DealerGateConfig,
    DealerLevelStore,
    evaluate_dealer_gate,
)


def _write_summary(tmp_path, rows, *, snapshot_date="2026-07-08"):
    d = tmp_path / snapshot_date.replace("-", "")
    d.mkdir(parents=True, exist_ok=True)
    path = d / "dealer_level_summary.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _row(symbol, scope, *, spot, call_wall, put_wall, magnet, snapshot_date="2026-07-08", direction="neutral"):
    return {
        "symbol": symbol, "scope": scope, "snapshot_date": snapshot_date, "spot": spot,
        "call_wall": call_wall, "put_wall": put_wall, "nearest_magnet": magnet,
        "gamma_flip": spot, "dealer_direction": direction,
    }


@pytest.fixture()
def store(tmp_path):
    rows = [
        # Call wall 0.5% overhead -> resistance for calls.
        _row("WALL", SCOPE_MONTHLY, spot=100.0, call_wall=100.5, put_wall=90.0, magnet=100.0),
        # Magnet 2% below -> pulls calls down.
        _row("MAG", SCOPE_MONTHLY, spot=100.0, call_wall=115.0, put_wall=90.0, magnet=98.0),
        # Clean runway upward.
        _row("CLEAN", SCOPE_MONTHLY, spot=100.0, call_wall=112.0, put_wall=90.0, magnet=101.0),
        _row("CLEAN", SCOPE_NEAREST, spot=100.0, call_wall=112.0, put_wall=90.0, magnet=101.0),
        # Put support just below -> resistance for puts.
        _row("PUTSUP", SCOPE_NEAREST, spot=100.0, call_wall=110.0, put_wall=99.5, magnet=100.0),
        # Wall 0.5% overhead but magnet ABOVE -> only the wall rule is in play.
        _row("WALLONLY", SCOPE_MONTHLY, spot=100.0, call_wall=100.5, put_wall=90.0, magnet=105.0),
    ]
    return DealerLevelStore(_write_summary(tmp_path, rows))


NOW = date(2026, 7, 8)
CFG = DealerGateConfig()


def test_call_wall_overhead_vetoes_call(store):
    v = evaluate_dealer_gate("WALL", "call", 100.0, SCOPE_MONTHLY, store=store, config=CFG, now=NOW)
    assert v.vetoed and v.reason == "call_wall_overhead"


def test_magnet_below_vetoes_call(store):
    v = evaluate_dealer_gate("MAG", "call", 100.0, SCOPE_MONTHLY, store=store, config=CFG, now=NOW)
    assert v.vetoed and v.reason == "magnet_below"


def test_clear_structure_allows_call(store):
    v = evaluate_dealer_gate("CLEAN", "call", 100.0, SCOPE_MONTHLY, store=store, config=CFG, now=NOW)
    assert v.action == "allow" and v.reason == "clear" and v.has_data


def test_put_wall_support_vetoes_put(store):
    v = evaluate_dealer_gate("PUTSUP", "put", 100.0, SCOPE_NEAREST, store=store, config=CFG, now=NOW)
    assert v.vetoed and v.reason == "put_wall_support"


def test_missing_symbol_fails_open(store):
    v = evaluate_dealer_gate("NOPE", "call", 100.0, SCOPE_MONTHLY, store=store, config=CFG, now=NOW)
    assert v.action == "allow" and v.reason == "no_dealer_data" and not v.has_data


def test_stale_snapshot_fails_open(tmp_path):
    rows = [_row("WALL", SCOPE_MONTHLY, spot=100.0, call_wall=100.5, put_wall=90.0, magnet=100.0,
                 snapshot_date="2026-06-01")]
    st = DealerLevelStore(_write_summary(tmp_path, rows, snapshot_date="2026-06-01"))
    v = evaluate_dealer_gate("WALL", "call", 100.0, SCOPE_MONTHLY, store=st, config=CFG, now=NOW)
    assert v.action == "allow" and v.reason == "stale_dealer_data" and v.stale


def test_live_price_moves_out_of_wall_zone(store):
    # WALLONLY: at spot the wall is 0.5% overhead (veto); once price runs above
    # the wall it is a breakout, and with the magnet above there is no veto.
    at_spot = evaluate_dealer_gate("WALLONLY", "call", 100.0, SCOPE_MONTHLY, store=store, config=CFG, now=NOW)
    assert at_spot.vetoed and at_spot.reason == "call_wall_overhead"
    broke_out = evaluate_dealer_gate("WALLONLY", "call", 101.0, SCOPE_MONTHLY, store=store, config=CFG, now=NOW)
    assert broke_out.action == "allow"


def test_bad_entry_price_fails_open(store):
    v = evaluate_dealer_gate("WALL", "call", 0.0, SCOPE_MONTHLY, store=store, config=CFG, now=NOW)
    assert v.action == "allow" and v.reason == "no_entry_price"


def test_numeric_direction_normalizes(store):
    long_v = evaluate_dealer_gate("WALL", 1, 100.0, SCOPE_MONTHLY, store=store, config=CFG, now=NOW)
    assert long_v.side == "call"
    short_v = evaluate_dealer_gate("PUTSUP", -1, 100.0, SCOPE_NEAREST, store=store, config=CFG, now=NOW)
    assert short_v.side == "put"
