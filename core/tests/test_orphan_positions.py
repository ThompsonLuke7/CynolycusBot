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
from pathlib import Path

import pytest

from core.orphan_positions import (
    Orphan,
    find_orphans,
    log_orphans,
    managed_symbols,
    spy_daytrader_symbols,
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
    # An empty live-runs root, so these tests never read the real SPY daytrader
    # book out of the repo and start depending on whatever it holds today.
    spy_runs = tmp_path / "live_runs"
    spy_runs.mkdir()
    return {"state_paths": {"meta": meta, "dealer": dealer},
            "swing_book_path": swing, "spy_runs_root": spy_runs}


def _spy_session(runs_root, name, rows):
    session = runs_root / name
    session.mkdir(parents=True)
    (session / "trade-events.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return session


def _spy_event(event, long_symbol=None, short_symbol=None):
    return {"stream": "trade_events", "recorded_at": "2026-08-31T13:52:10+00:00",
            "payload": {"symbol": "SPY", "result": {
                "event": event,
                "open_long_symbol": long_symbol,
                "open_short_symbol": short_symbol,
            }}}


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


# --- the SPY daytrader's book -------------------------------------------------
#
# It keeps no `managed` state file, so before 2026-09-01 the scan reported its
# intraday 0DTE as an orphan for as long as it was held — a different contract
# symbol every session, which read as orphans accumulating daily on top of a
# static set of 12. See research/daily_live_reports/2026-08-31.md.


def test_a_contract_the_spy_daytrader_holds_is_not_an_orphan(books, tmp_path) -> None:
    _spy_session(books["spy_runs_root"], "20260831_045208_live_spy", [
        _spy_event("intent_update"),
        _spy_event("open_long", long_symbol="SPY260831C00767000"),
    ])

    assert find_orphans([_position("SPY260831C00767000", qty=1,
                                   asset_class="us_option")], **books) == []


def test_a_contract_the_daytrader_already_closed_is_an_orphan_again(books) -> None:
    """Only the newest row is the book; a closed contract must stop being claimed."""

    _spy_session(books["spy_runs_root"], "20260831_045208_live_spy", [
        _spy_event("open_long", long_symbol="SPY260831C00767000"),
        _spy_event("close_long"),
    ])

    orphans = find_orphans([_position("SPY260831C00767000", qty=1,
                                      asset_class="us_option")], **books)
    assert [o.symbol for o in orphans] == ["SPY260831C00767000"]


def test_a_short_leg_is_claimed_too(books) -> None:
    _spy_session(books["spy_runs_root"], "20260828_045208_live_spy", [
        _spy_event("open_short", short_symbol="SPY260828P00771000"),
    ])

    assert find_orphans([_position("SPY260828P00771000", qty=1,
                                   asset_class="us_option")], **books) == []


def test_only_the_newest_session_is_read(books) -> None:
    """Session dirs are named by launch timestamp; a stale one must not claim."""

    _spy_session(books["spy_runs_root"], "20260820_040331_live_spy", [
        _spy_event("open_long", long_symbol="SPY260820C00700000"),
    ])
    _spy_session(books["spy_runs_root"], "20260831_045208_live_spy", [
        _spy_event("open_long", long_symbol="SPY260831C00767000"),
    ])

    orphans = find_orphans(
        [_position("SPY260820C00700000", qty=1, asset_class="us_option"),
         _position("SPY260831C00767000", qty=1, asset_class="us_option")],
        **books,
    )
    assert [o.symbol for o in orphans] == ["SPY260820C00700000"]


def test_no_session_directory_claims_nothing_and_does_not_raise(books) -> None:
    assert spy_daytrader_symbols(books["spy_runs_root"]) == set()


def test_a_session_that_never_traded_claims_nothing(books) -> None:
    _spy_session(books["spy_runs_root"], "20260831_045208_live_spy", [])

    assert spy_daytrader_symbols(books["spy_runs_root"]) == set()


def test_a_missing_trade_events_file_claims_nothing(books) -> None:
    (books["spy_runs_root"] / "20260831_045208_live_spy").mkdir(parents=True)

    assert spy_daytrader_symbols(books["spy_runs_root"]) == set()


def test_a_truncated_leading_fragment_does_not_break_the_read(books) -> None:
    """The tail read seeks mid-file, so the first line is usually a fragment."""

    session = books["spy_runs_root"] / "20260831_045208_live_spy"
    session.mkdir(parents=True)
    (session / "trade-events.jsonl").write_text(
        '{"payload": {"result": {"event": "trunc\n'
        + json.dumps(_spy_event("open_long", long_symbol="SPY260831C00767000")) + "\n"
    )

    assert spy_daytrader_symbols(books["spy_runs_root"]) == {"SPY260831C00767000"}


def test_a_flat_daytrader_claims_nothing(books) -> None:
    _spy_session(books["spy_runs_root"], "20260831_045208_live_spy", [
        _spy_event("intent_update"),
    ])

    assert spy_daytrader_symbols(books["spy_runs_root"]) == set()


def test_a_file_larger_than_the_tail_window_still_finds_the_current_book(books) -> None:
    """The real file is multi-MB; the read must seek and still parse the last row."""

    session = books["spy_runs_root"] / "20260831_045208_live_spy"
    session.mkdir(parents=True)
    filler = [_spy_event("intent_update") for _ in range(400)]
    rows = filler + [_spy_event("open_long", long_symbol="SPY260831C00767000")]
    body = "".join(json.dumps(r) + "\n" for r in rows)
    assert len(body.encode()) > 65_536, "filler must exceed the tail window"
    (session / "trade-events.jsonl").write_text(body)

    assert spy_daytrader_symbols(books["spy_runs_root"]) == {"SPY260831C00767000"}


# --- intraday_structure ownership (2026-09-01) ---------------------------------
#
# intraday_structure was in no claim reader. multi_ticker_swing's reconcile
# therefore saw its contracts as unowned, adopted six of them and force-exited
# four as `restored_unknown_expiring` within three minutes of entry —
# QQQ260901P00708000 bought 10:27:05 ET, sold out from under it 10:30:08 ET.

def _intraday_book(tmp_path, *entries):
    path = tmp_path / "open_option_positions.json"
    path.write_text(json.dumps({"open": {
        f"{occ}:setup": {"occ": occ, "ticker": ticker, "qty": 9}
        for occ, ticker in entries
    }}))
    return path


def test_intraday_structure_book_is_read(tmp_path):
    from core.orphan_positions import intraday_structure_symbols

    path = _intraday_book(tmp_path, ("QQQ260901P00708000", "QQQ"))
    assert intraday_structure_symbols(path) == {"QQQ260901P00708000"}


def test_intraday_structure_claims_the_contract_not_the_underlying(tmp_path):
    """Claiming bare "QQQ" would stop a sibling adopting an unrelated QQQ contract."""
    from core.orphan_positions import intraday_structure_symbols

    path = _intraday_book(tmp_path, ("QQQ260901P00708000", "QQQ"))
    assert "QQQ" not in intraday_structure_symbols(path)


def test_missing_intraday_book_is_not_an_error(tmp_path):
    from core.orphan_positions import intraday_structure_symbols

    assert intraday_structure_symbols(tmp_path / "nope.json") == set()


def test_intraday_contracts_are_not_orphans(tmp_path):
    """The regression: an intraday contract must read as claimed, not unowned."""
    path = _intraday_book(tmp_path, ("QQQ260901P00708000", "QQQ"))
    orphans = find_orphans(
        [_position("QQQ260901P00708000", qty=9, asset_class="us_option")],
        state_paths={},
        swing_book_path=tmp_path / "no_swing.json",
        spy_runs_root=tmp_path / "no_runs",
        intraday_book_path=path,
    )
    assert orphans == []


def test_claimed_symbols_can_exclude_the_asking_module(tmp_path):
    """Swing asks "what do my siblings claim" and must not get its own book back."""
    from core.orphan_positions import claimed_symbols

    swing = tmp_path / "swing.json"
    swing.write_text(json.dumps({"positions": [
        {"option_symbol": "TGT260911C00157500", "ticker": "TGT"},
    ]}))
    intraday = _intraday_book(tmp_path, ("QQQ260901P00708000", "QQQ"))
    kwargs = dict(state_paths={}, swing_book_path=swing,
                  spy_runs_root=tmp_path / "no_runs", intraday_book_path=intraday)

    everything = claimed_symbols(**kwargs)
    assert {"TGT260911C00157500", "QQQ260901P00708000"} <= everything

    siblings = claimed_symbols(**kwargs, exclude=("multi_ticker_swing",))
    assert "QQQ260901P00708000" in siblings
    assert "TGT260911C00157500" not in siblings


# A book that cannot be read is not a book that claims nothing
# (2026-09-02 11:50, multi_ticker_swing/open_positions.json read mid-write).


def test_a_book_read_mid_write_is_retried_before_giving_up(tmp_path, monkeypatch):
    """These files are rewritten in place while the scan reads them."""

    from core import orphan_positions

    book = tmp_path / "open_positions.json"
    book.write_text('{"positions": [{"ticker": "AAPL"}]}')

    calls = {"n": 0}
    real_read = Path.read_text

    def _truncated_once(self, *a, **k):
        if self == book:
            calls["n"] += 1
            if calls["n"] == 1:
                return ""  # caught mid-write
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _truncated_once)

    assert orphan_positions.swing_book_symbols(book) == {"AAPL"}
    assert calls["n"] > 1, "a single failed read must not be the final answer"


def test_a_persistently_unreadable_book_raises_rather_than_claiming_nothing(tmp_path):
    """The distinction the veto path depends on."""

    from core.orphan_positions import ClaimBookUnreadable, swing_book_symbols

    book = tmp_path / "open_positions.json"
    book.write_text("{not json")

    with pytest.raises(ClaimBookUnreadable):
        swing_book_symbols(book)


def test_a_missing_book_is_still_simply_empty(tmp_path):
    """Absent is a real answer: the module has never written a position."""

    from core.orphan_positions import swing_book_symbols

    assert swing_book_symbols(tmp_path / "never_written.json") == set()


def test_the_orphan_scan_stays_lenient_when_a_book_is_unreadable(tmp_path):
    """A detector that only warns should over-report, never fail to run."""

    from core.orphan_positions import claimed_symbols

    book = tmp_path / "open_positions.json"
    book.write_text("{not json")

    claimed = claimed_symbols(
        state_paths={},
        swing_book_path=book,
        spy_runs_root=tmp_path / "no_runs",
        intraday_book_path=tmp_path / "no_book.json",
        extra_claimed=("AAPL",),
    )

    assert claimed == {"AAPL"}


def test_a_veto_caller_asking_strictly_gets_the_failure(tmp_path):
    """multi_ticker_swing's reconcile must be able to fail closed."""

    from core.orphan_positions import ClaimBookUnreadable, claimed_symbols

    book = tmp_path / "open_positions.json"
    book.write_text("{not json")

    with pytest.raises(ClaimBookUnreadable):
        claimed_symbols(
            state_paths={},
            swing_book_path=book,
            spy_runs_root=tmp_path / "no_runs",
            intraday_book_path=tmp_path / "no_book.json",
            strict=True,
        )
