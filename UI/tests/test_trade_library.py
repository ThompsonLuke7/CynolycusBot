"""Position reconstruction from the closed-trades ledgers.

The ledger stores sell LEGS. A ``take_profit_+30%`` leg is a partial trim that
leaves the position open; a stop/horizon leg closes it. Folding them wrong is
what made the raw ledger read as a 46% win rate when the fully-closed round-trip
rate was 16% — every trim counted as a win, every full exit as a loss.
"""
from __future__ import annotations

import json

import pytest

from UI import trade_library


@pytest.fixture
def ledgers(tmp_path, monkeypatch):
    """Point the module at a throwaway ledger root and live-state map."""
    root = tmp_path / "inference"
    monkeypatch.setattr(trade_library, "LEDGER_ROOT", root)
    monkeypatch.setattr(trade_library, "LIVE_STATES", {})

    def write(module, rows, managed=None):
        d = root / module
        d.mkdir(parents=True, exist_ok=True)
        (d / "closed_trades.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        if managed is not None:
            state = tmp_path / f"{module}_state.json"
            state.write_text(json.dumps({"managed": managed}))
            trade_library.LIVE_STATES[module] = state
    return write


def _leg(ticker, ts, reason, pnl, *, entry_bar="2026-08-01T14:00:00+00:00",
         route="option", qty=5, entry=10.0, exit_px=6.0, module="m1"):
    return {"ts": ts, "module": module, "ticker": ticker, "order_symbol": ticker,
            "route": route, "qty": qty, "exit_reason": reason,
            "entry_avg_price": entry, "exit_fill_price": exit_px,
            "realized_pnl": pnl, "entry_bar": entry_bar, "order_id": "abc123"}


def test_a_single_stop_is_one_closed_position(ledgers):
    ledgers("m1", [_leg("AAA", "2026-08-05T18:00:00+00:00", "stop_-39%", -2000.0)])
    (p,) = trade_library.build_positions()
    assert p["status"] == "closed" and p["realized"] == -2000.0
    assert p["final_reason"] == "stop_-39%" and p["trims"] == 0


def test_trim_then_stop_is_ONE_position_not_a_win_and_a_loss(ledgers):
    """The core folding rule. Raw legs would score this 1 win + 1 loss."""
    ledgers("m1", [
        _leg("BBB", "2026-08-04T18:00:00+00:00", "take_profit_+30%", 300.0),
        _leg("BBB", "2026-08-06T18:00:00+00:00", "stop_-39%", -2000.0),
    ])
    (p,) = trade_library.build_positions()
    assert p["status"] == "closed"
    assert p["realized"] == -1700.0          # netted, not two rows
    assert p["trims"] == 1 and p["final_reason"] == "stop_-39%"
    assert len(p["legs"]) == 2


def test_a_trim_with_the_position_still_managed_stays_open(ledgers):
    ledgers("m1", [_leg("CCC", "2026-08-04T18:00:00+00:00", "take_profit_+30%", 300.0)],
            managed={"CCC": {"route": "option", "last_mark_price": 18.0,
                             "entry_bar": "2026-08-01T14:00:00+00:00"}})
    (p,) = trade_library.build_positions()
    assert p["status"] == "open"
    assert p["realized"] == 300.0            # booked trim only
    assert p["open_mark"] == 18.0
    assert p["open_ret_pct"] == pytest.approx(80.0)


def test_open_trims_are_never_counted_as_closed_pnl(ledgers):
    ledgers("m1", [
        _leg("CCC", "2026-08-04T18:00:00+00:00", "take_profit_+30%", 300.0),
        _leg("DDD", "2026-08-05T18:00:00+00:00", "stop_-39%", -900.0),
    ], managed={"CCC": {"route": "option", "last_mark_price": 18.0}})
    s = trade_library.summary(trade_library.build_positions())
    assert s["realized_closed"] == -900.0
    assert s["realized_on_open_trims"] == 300.0
    assert s["closed"] == 1 and s["open"] == 1
    assert s["win_rate"] == 0.0              # the trim must not inflate this


def test_a_trimmed_position_the_module_no_longer_manages_is_closed(ledgers):
    """No live-state entry means nothing is riding; do not claim a live mark."""
    ledgers("m1", [_leg("EEE", "2026-08-04T18:00:00+00:00", "take_profit_+30%", 300.0)],
            managed={})
    (p,) = trade_library.build_positions()
    assert p["status"] == "closed" and p["open_mark"] is None


def test_a_re_entry_is_a_separate_position(ledgers):
    """Same ticker, later entry_bar -> a new episode, not more of the old trade."""
    ledgers("m1", [
        _leg("FIG", "2026-07-23T18:00:00+00:00", "stop_-50%", -93.0,
             entry_bar="2026-07-21T14:00:00+00:00"),
        _leg("FIG", "2026-08-13T18:00:00+00:00", "take_profit_+20%", 6360.0,
             entry_bar="2026-08-06T14:00:00+00:00"),
    ])
    ps = sorted(trade_library.build_positions(), key=lambda p: p["entry_bar"])
    assert len(ps) == 2
    assert [p["realized"] for p in ps] == [-93.0, 6360.0]


def test_legacy_occ_rows_group_with_the_underlying(ledgers):
    ledgers("m1", [_leg("COHR260717C00387500", "2026-07-09T18:00:00+00:00",
                        "dropped_out", -19400.0)])
    (p,) = trade_library.build_positions()
    assert p["ticker"] == "COHR"


def test_audit_gap_markers_are_reported_not_traded(ledgers):
    ledgers("m1", [
        {"event": "audit_gap", "detail": "748 zeroed bytes", "repaired_at": "2026-08-04"},
        _leg("GGG", "2026-08-05T18:00:00+00:00", "stop_-39%", -100.0),
    ])
    assert len(trade_library.build_positions()) == 1        # marker is not a trade
    assert len(trade_library.ledger_health()["gaps"]) == 1  # but it IS surfaced


def test_rows_without_realized_pnl_are_counted_as_a_gap(ledgers):
    ledgers("m1", [_leg("HHH", "2026-08-05T18:00:00+00:00", "stop_-50%", None)])
    assert trade_library.ledger_health()["rows_missing_pnl"] == 1
    (p,) = trade_library.build_positions()
    assert p["realized"] == 0.0     # excluded, not guessed


def test_unparseable_lines_do_not_take_the_page_down(ledgers):
    ledgers("m1", [_leg("III", "2026-08-05T18:00:00+00:00", "stop_-39%", -50.0)])
    path = trade_library.LEDGER_ROOT / "m1" / "closed_trades.jsonl"
    path.write_text(path.read_text() + "{not json\n")
    assert len(trade_library.build_positions()) == 1


def test_price_series_reports_a_missing_cache_instead_of_raising():
    out = trade_library.price_series("ZZZZ_NOPE")
    assert out["bars"] == [] and "no 4h bar cache" in out["error"]


def test_price_series_reads_the_shared_cache(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(trade_library, "BARS_4H_DIR", tmp_path)
    idx = pd.date_range("2026-08-01", periods=6, freq="4h", tz="UTC")
    pd.DataFrame({"timestamp": idx, "open": 1.0, "high": 2.0, "low": 0.5,
                  "close": 1.5, "volume": 10}).to_parquet(tmp_path / "AAA.parquet")
    out = trade_library.price_series("AAA", days=0)
    assert len(out["bars"]) == 6 and out["bars"][0]["c"] == 1.5


# --- dashboard endpoints -------------------------------------------------------

def test_the_stop_line_is_drawn_only_when_the_runner_anchored_the_position(ledgers):
    """A position with no u_entry/u_atr predates the rule; inventing a level for
    it would draw a stop that never existed."""
    from UI.trade_library_dashboard import TradeLibraryApp
    ledgers("m1", [_leg("AAA", "2026-08-05T18:00:00+00:00", "take_profit_+30%", 10.0)],
            managed={"AAA": {"route": "option", "last_mark_price": 12.0,
                             "u_entry": 100.0, "u_atr": 4.0}})
    app = TradeLibraryApp()
    pid = app.positions({})["positions"][0]["id"]
    assert app.position({"id": [pid]})["underlying_stop"]["level"] == pytest.approx(94.0)

    trade_library.LIVE_STATES.clear()
    pid = app.positions({})["positions"][0]["id"]
    assert app.position({"id": [pid]})["underlying_stop"] is None


def test_filters_narrow_the_book(ledgers):
    from UI.trade_library_dashboard import TradeLibraryApp
    ledgers("m1", [
        _leg("AAA", "2026-08-05T18:00:00+00:00", "stop_-39%", -100.0),
        _leg("BBB", "2026-08-06T18:00:00+00:00", "horizon", 250.0, route="equity"),
    ])
    app = TradeLibraryApp()
    assert len(app.positions({})["positions"]) == 2
    assert len(app.positions({"outcome": ["win"]})["positions"]) == 1
    assert len(app.positions({"route": ["equity"]})["positions"]) == 1
    assert len(app.positions({"ticker": ["AAA"]})["positions"]) == 1
    assert len(app.positions({"reason": ["stop_"]})["positions"]) == 1
    # the unfiltered book travels alongside so the page can show both
    assert app.positions({"outcome": ["win"]})["book"]["closed"] == 2


def test_unknown_position_id_is_an_error_not_a_crash(ledgers):
    from UI.trade_library_dashboard import TradeLibraryApp
    ledgers("m1", [_leg("AAA", "2026-08-05T18:00:00+00:00", "stop_-39%", -1.0)])
    assert "error" in TradeLibraryApp().position({"id": ["nope"]})


def test_a_full_take_profit_closes_the_position_not_leaves_it_open(ledgers):
    """`take_profit_full_+30%` sells everything. Classifying it as a partial trim
    would strand it in the book as an open runner forever."""
    ledgers("m1", [_leg("SNDK", "2026-08-19T18:00:00+00:00",
                        "take_profit_full_+30%", 4200.0)])
    (p,) = trade_library.build_positions()
    assert p["status"] == "closed"
    assert p["trims"] == 0
    assert p["final_reason"] == "take_profit_full_+30%"
    s = trade_library.summary([p])
    assert s["realized_closed"] == 4200.0 and s["win_rate"] == 100.0


def test_a_partial_trim_then_a_full_take_profit_is_one_closed_position(ledgers):
    ledgers("m1", [
        _leg("AAA", "2026-08-18T18:00:00+00:00", "take_profit_+30%", 300.0),
        _leg("AAA", "2026-08-19T18:00:00+00:00", "take_profit_full_+30%", 900.0),
    ])
    (p,) = trade_library.build_positions()
    assert p["status"] == "closed" and p["trims"] == 1 and p["realized"] == 1200.0


# --- input-fingerprint memo -------------------------------------------------
# The fold is reused only while its input files are unchanged. Getting this
# wrong in the "too sticky" direction shows a stale book right after a runner
# closed a position, which is worse than the cost it saves.

def test_repeat_reads_do_not_reparse_unchanged_ledgers(ledgers, monkeypatch):
    ledgers("m1", [_leg("AAA", "2026-08-05T18:00:00+00:00", "stop_-39%", -2000.0)])
    trade_library.build_positions()

    calls: list[int] = []
    real = trade_library._open_book
    monkeypatch.setattr(trade_library, "_open_book",
                        lambda: (calls.append(1), real())[1])

    trade_library.build_positions()
    trade_library.build_positions()

    assert calls == [], "an unchanged ledger must not be re-folded"


def test_an_appended_leg_invalidates_immediately(ledgers):
    ledgers("m1", [_leg("AAA", "2026-08-05T18:00:00+00:00", "take_profit_+30%", 300.0)],
            managed={"AAA": {"qty": 3, "last_mark_price": 12.0}})
    first = trade_library.build_positions()
    assert first[0]["status"] == "open"

    # The runner closes it out: same file, one more line.
    ledgers("m1", [_leg("AAA", "2026-08-05T18:00:00+00:00", "take_profit_+30%", 300.0),
                   _leg("AAA", "2026-08-06T18:00:00+00:00", "stop_-39%", -500.0)],
            managed={})
    second = trade_library.build_positions()

    assert second[0]["status"] == "closed", "the memo must not outlive its inputs"
    assert second[0]["realized"] == -200.0


def test_ledger_health_is_memoized_separately(ledgers, monkeypatch):
    ledgers("m1", [_leg("AAA", "2026-08-05T18:00:00+00:00", "stop_-39%", -2000.0)])
    assert trade_library.ledger_health()["gaps"] == []

    def _boom():
        raise AssertionError("unchanged ledgers must not be re-read")

    monkeypatch.setattr(trade_library, "_ledger_health_uncached", _boom)
    assert trade_library.ledger_health()["gaps"] == []
