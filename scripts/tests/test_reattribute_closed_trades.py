"""A closed trade is booked under the module whose entry order opened it.

The account nets each symbol across modules, so one module can sell what
another bought. On 2026-10-07 Meta's ledger held +$3,960 on NTSK contracts HTF
had bought, the 30m swing's held eleven Intraday Structure trades it had
adopted, and two Intraday exits were in no ledger at all.
"""
from __future__ import annotations

import json

import pytest

from core.order_reconciliation import save_evidence
from scripts import reattribute_closed_trades as tool

ENTRY = "df7eab3d-5c2f-423b-a207-0c8a6a92ec71"
EXIT = "be946ab8-fcee-4856-903e-60b4aea187ff"
NTSK = "NTSK260918C00015000"


def _broker(**extra):
    orders = {
        ENTRY: {"id": ENTRY, "symbol": NTSK, "side": "buy", "filled_qty": "39",
                "filled_avg_price": "1.1", "filled_at": "2026-09-02T13:37:28Z"},
        EXIT: {"id": EXIT, "symbol": NTSK, "side": "sell", "filled_qty": "33",
               "filled_avg_price": "2.3", "filled_at": "2026-09-18T19:47:52Z"},
        **extra,
    }
    return orders.__getitem__


def _ledger(root, module, rows):
    path = root / module / "closed_trades.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _row(module="meta_ranker", **over):
    return {"module": module, "ts": "2026-09-18T19:52:56+00:00", "ticker": "NTSK",
            "order_symbol": NTSK, "route": "option", "qty": 33.0, "entry_avg_price": 1.1,
            "exit_fill_price": 2.3, "realized_pnl": 3960.0, "order_id": EXIT,
            "event_id": f"{EXIT}:close", **over}


@pytest.fixture
def books(tmp_path):
    root, logs = tmp_path / "inference", tmp_path / "logs"
    logs.mkdir()
    other = _row(ticker="CAVA", order_symbol="CAVA", order_id="other", event_id="other:close")
    _ledger(root, "meta_ranker", [other, _row()])
    _ledger(root, "multi_ticker_swing_htf", [_row("multi_ticker_swing_htf", qty=6.0,
                                                  order_id="ddb3eeab", event_id="ddb3eeab:close")])
    (logs / "server_20260902.log").write_text(
        "2026-09-02 09:35:45,304 INFO [core.live_4h_exec] meta_ranker pending-open: "
        f"submitted buy {NTSK} x39 id=?\n"
        "2026-09-02 09:37:28,077 INFO [core.live_4h_exec] multi_ticker_swing_htf pending-open: "
        f"submitted buy {NTSK} x39 id={ENTRY}\n"
        f"  OK buy 39 {NTSK}  id={ENTRY}\n")          # names no module: not evidence
    return root, logs


def _run(root, logs, *args, read_order=None, **kw):
    return tool.main(["--root", str(root), "--logs", str(logs), *args],
                     read_order=read_order or _broker(), **kw)


def _rows(root, module, name="closed_trades.jsonl"):
    path = root / module / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


MOVE = f"meta_ranker=multi_ticker_swing_htf:{EXIT}:{ENTRY}"


def test_a_dry_run_writes_nothing(books):
    root, logs = books
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    assert _run(root, logs, "--move", MOVE) == 0

    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_the_row_moves_to_the_module_that_sent_the_entry(books):
    root, logs = books
    fills = root / "meta_ranker" / "exit_fills.jsonl"
    fills.write_text(json.dumps({"module": "meta_ranker", "order_id": EXIT, "qty": 33.0,
                                 "proceeds": 7590.0}) + "\n")
    registry = root / "broker_reconciliation" / "paper" / "order_registry.jsonl"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"order_id": EXIT, "module": "meta_ranker", "side": "sell"}) + "\n")

    assert _run(root, logs, "--move", MOVE, "--apply") == 0

    assert [r["order_id"] for r in _rows(root, "meta_ranker")] == ["other"]
    moved = _rows(root, "multi_ticker_swing_htf")[-1]
    assert moved["module"] == "multi_ticker_swing_htf"
    assert (moved["order_id"], moved["qty"], moved["realized_pnl"]) == (EXIT, 33.0, 3960.0)
    assert moved["ledger_repair"]["from_module"] == "meta_ranker"
    assert moved["ledger_repair"]["entry_owner_evidence"] == "server_20260902.log:2"
    # The per-fill projection and the order's registered owner follow the row,
    # so the broker-fill certificate still finds the proceeds.
    assert _rows(root, "meta_ranker", "exit_fills.jsonl") == []
    assert _rows(root, "multi_ticker_swing_htf", "exit_fills.jsonl")[0]["module"] == "multi_ticker_swing_htf"
    assert json.loads(registry.read_text())["module"] == "multi_ticker_swing_htf"
    # Totals across the two ledgers are unchanged; the originals are kept.
    assert len(list((root / "meta_ranker").glob("closed_trades.jsonl.*.bak"))) == 1
    assert [r["to_module"] for r in _rows(root, "meta_ranker", "ledger_repairs.jsonl")] == [
        "multi_ticker_swing_htf"]


def test_a_second_run_finds_nothing_left_to_move(books):
    root, logs = books
    _run(root, logs, "--move", MOVE, "--apply")
    assert _run(root, logs, "--move", MOVE) == 1          # refused: the row is no longer in Meta
    assert len(_rows(root, "multi_ticker_swing_htf")) == 2


def test_refused_when_no_record_names_the_destination_as_the_buyer(books):
    root, logs = books
    (logs / "server_20260902.log").write_text(f"  OK buy 39 {NTSK}  id={ENTRY}\n")
    assert _run(root, logs, "--move", MOVE, "--apply") == 1
    assert len(_rows(root, "meta_ranker")) == 2


def test_refused_when_two_modules_are_on_record_for_the_entry(books):
    root, logs = books
    save_evidence(root, "meta_ranker", "some-exit", {
        "owner": "NTSK", "symbol": NTSK, "state": {"entry_order_id": ENTRY}})
    assert _run(root, logs, "--move", MOVE, "--apply") == 1
    assert len(_rows(root, "meta_ranker")) == 2


def test_refused_when_the_row_is_not_about_that_entry(books):
    root, logs = books
    _ledger(root, "meta_ranker", [_row(entry_avg_price=1.45)])     # a different lot's basis
    assert _run(root, logs, "--move", MOVE, "--apply") == 1


def test_refused_when_the_entry_did_not_come_first_or_was_too_small(books):
    root, logs = books
    late = {ENTRY: {"id": ENTRY, "symbol": NTSK, "side": "buy", "filled_qty": "39",
                    "filled_avg_price": "1.1", "filled_at": "2026-09-19T13:37:28Z"}}
    small = {ENTRY: {"id": ENTRY, "symbol": NTSK, "side": "buy", "filled_qty": "10",
                     "filled_avg_price": "1.1", "filled_at": "2026-09-02T13:37:28Z"}}
    for orders in (late, small):
        assert _run(root, logs, "--move", MOVE, "--apply", read_order=_broker(**orders)) == 1
    assert len(_rows(root, "meta_ranker")) == 2


def test_an_order_the_broker_cannot_show_is_a_refusal_not_a_crash(books):
    root, logs = books

    def unreadable(order_id):
        raise OSError("422 order_id is missing")

    assert _run(root, logs, "--move", MOVE, read_order=unreadable) == 1


# --- a ledger that records no order ids (the 30m swing), and Intraday's log line ---

QQQ, BUY, SELL = "QQQ260929P00735000", "efffe277-buy", "c2586acf-sell"


@pytest.fixture
def adopted(tmp_path):
    root, logs = tmp_path / "inference", tmp_path / "logs"
    logs.mkdir()
    _ledger(root, "multi_ticker_swing", [
        {"module": "multi_ticker_swing", "ts": "2026-09-29T15:10:02+00:00", "ticker": "QQQ",
         "order_symbol": QQQ, "route": "option", "qty": 19.0, "entry_avg_price": 0.51,
         "exit_fill_price": 0.59, "realized_pnl": 152.0, "order_id": None,
         "exit_reason": "restored_unknown_expiring"}])
    _ledger(root, "intraday_structure", [])
    (logs / "server_20260929.log").write_text(
        "2026-09-29 11:06:25,100 INFO [strategies.intraday_structure.execution] intraday "
        f"execution: BUY {QQQ} x19 filled @ 0.51 for QQQ:short:vwap_reclaim_continuation (limit 0.51)\n")
    orders = {
        BUY: {"id": BUY, "symbol": QQQ, "side": "buy", "filled_qty": "19",
              "filled_avg_price": "0.51", "filled_at": "2026-09-29T15:06:24.6Z"},
        SELL: {"id": SELL, "symbol": QQQ, "side": "sell", "filled_qty": "19",
               "filled_avg_price": "0.59", "filled_at": "2026-09-29T15:10:02.1Z"}}
    return root, logs, orders.__getitem__


def test_a_row_without_an_order_id_is_found_by_its_fill_and_gains_the_id(adopted):
    root, logs, read_order = adopted

    assert _run(root, logs, "--move", f"multi_ticker_swing=intraday_structure:{SELL}:{BUY}",
                "--apply", read_order=read_order) == 0

    assert _rows(root, "multi_ticker_swing") == []
    (row,) = _rows(root, "intraday_structure")
    assert (row["module"], row["order_id"], row["realized_pnl"]) == ("intraday_structure", SELL, 152.0)
    assert row["exit_reason"] == "restored_unknown_expiring"      # the row itself is unchanged


def test_intradays_buy_line_must_match_time_size_and_price(adopted):
    root, logs, read_order = adopted
    (logs / "server_20260929.log").write_text(
        "2026-09-29 11:06:25,100 INFO [x] intraday execution: BUY QQQ260929P00735000 x20 "
        "filled @ 0.51 for s\n"                                    # 20, not 19
        "2026-09-29 12:28:47,000 INFO [x] intraday execution: BUY QQQ260929P00735000 x19 "
        "filled @ 0.51 for s\n")                                   # 82 minutes later
    assert _run(root, logs, "--move", f"multi_ticker_swing=intraday_structure:{SELL}:{BUY}",
                "--apply", read_order=read_order) == 1
    assert len(_rows(root, "multi_ticker_swing")) == 1


# --- exits no ledger holds ------------------------------------------------------

def test_an_unbooked_broker_exit_is_written_for_the_entry_owner(adopted):
    root, logs, read_order = adopted
    (root / "multi_ticker_swing" / "closed_trades.jsonl").write_text("")   # nobody booked the sell

    assert _run(root, logs, "--book", f"intraday_structure:{SELL}:{BUY}", "--apply",
                read_order=read_order) == 0

    (row,) = _rows(root, "intraday_structure")
    assert (row["order_symbol"], row["ticker"], row["qty"]) == (QQQ, "QQQ", 19.0)
    assert row["realized_pnl"] == pytest.approx((0.59 - 0.51) * 19 * 100)
    assert (row["order_id"], row["entry_order_id"]) == (SELL, BUY)
    assert row["exit_reason"] == "booked_from_broker_fill"


def test_an_exit_already_in_a_ledger_is_not_booked_again(adopted):
    root, logs, read_order = adopted
    assert _run(root, logs, "--book", f"intraday_structure:{SELL}:{BUY}", "--apply",
                read_order=read_order) == 1
    assert _rows(root, "intraday_structure") == []


def _expiry(qty="-19", net="0"):
    return lambda symbol, expiry: [{"id": "opexp-1", "activity_type": "OPEXP", "symbol": symbol,
                                    "status": "executed", "qty": qty, "net_amount": net,
                                    "date": expiry.isoformat()}]


def test_a_worthless_expiry_is_booked_as_the_full_premium_lost(adopted):
    root, logs, read_order = adopted
    (root / "multi_ticker_swing" / "closed_trades.jsonl").write_text("")

    assert _run(root, logs, "--book-expiry", f"intraday_structure:{BUY}", "--apply",
                read_order=read_order, read_activities=_expiry()) == 0

    (row,) = _rows(root, "intraday_structure")
    assert (row["exit_fill_price"], row["realized_pnl"]) == (0.0, pytest.approx(-0.51 * 19 * 100))
    assert (row["settle_outcome"], row["activity_id"]) == ("expired_worthless", "opexp-1")
    # Idempotent: the position is now booked, so a second run refuses.
    assert _run(root, logs, "--book-expiry", f"intraday_structure:{BUY}", "--apply",
                read_order=read_order, read_activities=_expiry()) == 1
    assert len(_rows(root, "intraday_structure")) == 1


def test_an_expiry_that_does_not_cover_the_entry_is_refused(adopted):
    root, logs, read_order = adopted
    for activities in (_expiry(qty="-5"), _expiry(net="120"), lambda s, e: []):
        assert _run(root, logs, "--book-expiry", f"intraday_structure:{BUY}", "--apply",
                    read_order=read_order, read_activities=activities) == 1
    assert _rows(root, "intraday_structure") == []
