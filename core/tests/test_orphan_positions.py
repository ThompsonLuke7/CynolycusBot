"""Positions no strategy claims must be findable, and swing must not be one.

dealer_ranker's TECK260724C00057000 expired ITM on 2026-07-24, auto-exercised
into 100 shares, and sat unowned for five weeks with no symptom — no module
sizes, stops or exits an orphan. An earlier batch (GRAB, U, SMCI, FIG) had to be
cleaned up by hand after the 2026-08-14 expiry. See
research/daily_live_reports/2026-08-26.md.
"""
from __future__ import annotations

import json
import logging

import pytest

from core.orphan_positions import (
    Orphan,
    find_orphans,
    log_orphans,
    managed_symbols,
    swing_book_symbols,
)


def _position(symbol, qty=100, mv=7000.0, pl=1200.0, asset_class="us_equity"):
    return {"symbol": symbol, "qty": str(qty), "market_value": str(mv),
            "unrealized_pl": str(pl), "asset_class": asset_class}


@pytest.fixture
def books(tmp_path):
    """One equity module, one option module, and swing's own position list."""

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"managed": {
        "CRWD": {"route": "equity", "symbol": "CRWD", "shares": 26},
    }}))
    dealer = tmp_path / "dealer.json"
    dealer.write_text(json.dumps({"managed": {
        "CDE": {"route": "option", "occ": "CDE260828C00022000", "contracts": 94},
    }}))
    swing = tmp_path / "swing.json"
    swing.write_text(json.dumps({"positions": [
        {"ticker": "DIA", "option_symbol": "DIA260911C00531000", "qty": 10},
    ]}))
    return {"state_paths": {"meta": meta, "dealer": dealer}, "swing_book_path": swing}


def test_an_unclaimed_equity_is_an_orphan(books) -> None:
    orphans = find_orphans([_position("TECK")], **books)

    assert [o.symbol for o in orphans] == ["TECK"]
    assert orphans[0].qty == 100
    assert orphans[0].unrealized_pl == 1200.0


def test_a_managed_equity_is_not_an_orphan(books) -> None:
    assert find_orphans([_position("CRWD")], **books) == []


def test_a_managed_option_is_matched_on_its_occ_symbol(books) -> None:
    """Option modules key managed state by ticker but hold an OCC symbol."""

    assert find_orphans(
        [_position("CDE260828C00022000", asset_class="us_option")], **books
    ) == []


def test_swings_own_book_is_never_reported(books) -> None:
    """swing holds most of the account's contracts and keeps them in its own
    file, not a `managed` map. A scan blind to it would cry wolf every pass."""

    assert find_orphans(
        [_position("DIA260911C00531000", asset_class="us_option")], **books
    ) == []


def test_orphans_are_ordered_by_exposure(books) -> None:
    orphans = find_orphans(
        [_position("AEVA", mv=1610.0), _position("TECK", mv=7112.0),
         _position("EVH", mv=6731.0)],
        **books,
    )

    assert [o.symbol for o in orphans] == ["TECK", "EVH", "AEVA"]


def test_a_zero_quantity_position_is_not_an_orphan(books) -> None:
    assert find_orphans([_position("TECK", qty=0)], **books) == []


def test_an_unreadable_state_file_over_reports_rather_than_under_reports(
    tmp_path, books
) -> None:
    """Failing closed would hide a real orphan; failing open only adds noise,
    and the caller is warned which book could not be read."""

    missing = tmp_path / "gone.json"
    orphans = find_orphans([_position("CRWD")],
                           state_paths={"meta": missing},
                           swing_book_path=books["swing_book_path"])

    assert [o.symbol for o in orphans] == ["CRWD"]


def test_an_unreadable_swing_book_does_not_raise(tmp_path) -> None:
    assert swing_book_symbols(tmp_path / "nope.json") == set()


def test_managed_symbols_names_every_claimant(books) -> None:
    claims = managed_symbols(books["state_paths"])

    assert claims["CRWD"] == ["meta"]
    assert claims["CDE260828C00022000"] == ["dealer"]


def test_a_clean_book_logs_at_info_not_warning(caplog) -> None:
    """A detector that warns when nothing is wrong gets muted by its readers."""

    with caplog.at_level(logging.INFO):
        log_orphans([])

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "every broker position is claimed" in caplog.text


def test_orphans_are_reported_at_warning_with_their_totals(caplog) -> None:
    orphans = [Orphan("TECK", 100, 7112.0, 1252.0, "us_equity"),
               Orphan("SU", 100, 6567.0, 227.0, "us_equity")]

    with caplog.at_level(logging.WARNING):
        log_orphans(orphans)

    assert "2 position(s)" in caplog.text
    assert "13,679" in caplog.text      # market value
    assert "+1,479" in caplog.text      # unrealized
    assert "TECK" in caplog.text and "SU" in caplog.text
    assert "startup_queue" in caplog.text
