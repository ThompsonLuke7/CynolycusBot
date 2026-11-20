import json
from datetime import datetime, timedelta, timezone

import pytest

from core.personal_portfolio import trade_history as th
from core.personal_portfolio.data import ET

_ids = iter(range(1, 10_000))


def _fee(amount):
    return {"instrument": {"assetType": "CURRENCY", "symbol": "CURRENCY_USD"}, "amount": amount, "cost": -amount,
            "feeType": "COMMISSION"}


def _opt_inst(symbol, und, expiry, pc, strike, legacy=False):
    inst = {"assetType": "OPTION", "symbol": symbol, "underlyingSymbol": und, "putCall": pc, "strikePrice": strike,
            "optionPremiumMultiplier": 100, "description": f"{und} option"}
    if not legacy:
        inst["expirationDate"] = f"{expiry}T05:00:00+0000"
    return inst


def trade(time, inst, qty, price, effect=None, fee=0.66, mult=None, order=None):
    mult = mult or (100 if inst["assetType"] == "OPTION" else 1)
    item = {"instrument": inst, "amount": qty, "price": price, "cost": -qty * price * mult}
    if effect:
        item["positionEffect"] = effect
    return {"activityId": next(_ids), "time": time, "type": "TRADE", "status": "VALID", "accountNumber": "…560",
            "orderId": order or next(_ids), "netAmount": item["cost"] - fee, "transferItems": [_fee(fee), item]}


def rad(time, inst, qty, desc, effect=None):
    item = {"instrument": inst, "amount": qty, "price": 0.0, "cost": 0.0}
    if effect:
        item["positionEffect"] = effect
    return {"activityId": next(_ids), "time": time, "type": "RECEIVE_AND_DELIVER", "status": "VALID",
            "accountNumber": "…560", "description": desc, "netAmount": 0.0, "transferItems": [item]}


def stock(sym, kind="EQUITY"):
    return {"assetType": kind, "symbol": sym, "description": f"{sym} INC"}


def episodes(raw, smap=None):
    fills, issues = th.normalize_fills(raw)
    eps, checks = th.build_episodes(fills, smap)
    return eps, checks, issues


def test_parse_option_symbol_occ_and_td_legacy():
    assert th.parse_option_symbol("TQQQ  261218C00060000") == {"root": "TQQQ", "expiry": "2026-12-18",
                                                                "put_call": "CALL", "strike": 60.0}
    assert th.parse_option_symbol("NRDS_072123C12.5") == {"root": "NRDS", "expiry": "2023-07-21",
                                                           "put_call": "CALL", "strike": 12.5}
    assert th.parse_option_symbol("SPY_060721P421")["put_call"] == "PUT"
    assert th.parse_option_symbol("SPY") is None


def test_scale_in_partial_exit_and_expiry_make_one_episode_with_cash_pnl():
    inst = _opt_inst("TQQQ  261218C00060000", "TQQQ", "2026-12-18", "CALL", 60.0)
    raw = [trade("2026-11-02T15:00:00+0000", inst, 1, 4.00, "OPENING"),
           trade("2026-11-03T15:00:00+0000", inst, 1, 3.00, "OPENING"),
           trade("2026-11-05T15:00:00+0000", inst, -1, 6.00, "CLOSING"),
           rad("2026-12-21T05:00:00+0000", inst, -1, "Removed due to Expiration CALL TQQQ")]
    eps, checks, _ = episodes(raw)
    assert len(eps) == 1
    e = eps[0]
    assert e["status"] == "closed" and e["exit_reason"] == "expired" and e["n_entries"] == 2 and e["n_exits"] == 2
    assert e["avg_entry"] == pytest.approx(3.5) and e["avg_exit"] == pytest.approx(3.0)
    assert e["cost"] == pytest.approx(700) and e["proceeds"] == pytest.approx(600)
    assert e["pnl"] == pytest.approx(600 - 700 - 3 * 0.66) and e["return_pct"] == pytest.approx(e["pnl"] / 700)
    # the expiration closes at 16:00 ET on the expiry date, not when Schwab posts it
    assert datetime.fromisoformat(e["close_time"]).astimezone(ET).isoformat().startswith("2026-12-18T16:00")
    assert e["dte_at_entry"] == 46 and checks == {"effect_checked": 3, "effect_mismatch": 0, "corp_unhandled": 0}


def test_legacy_symbols_without_position_effect_are_inferred_and_keyed_like_occ():
    legacy = _opt_inst("MVST_081823C3", "MVST", "2023-08-18", "CALL", 3.0, legacy=True)
    occ = _opt_inst("MVST  230818C00003000", "MVST", "2023-08-18", "CALL", 3.0)
    raw = [trade("2023-07-10T14:00:00+0000", legacy, 3, 0.20),
           trade("2023-07-20T14:00:00+0000", occ, -3, 0.50)]
    eps, checks, _ = episodes(raw)
    assert len(eps) == 1 and eps[0]["key"] == "MVST 2023-08-18 C 3" and eps[0]["pnl"] == pytest.approx(3 * 30 - 2 * 0.66)
    assert checks["effect_checked"] == 0


def test_first_fill_that_closes_marks_the_episode_incomplete():
    inst = _opt_inst("PYPL_081823P63", "PYPL", "2023-08-18", "PUT", 63.0, legacy=True)
    eps, checks, _ = episodes([trade("2023-08-04T15:13:06+0000", inst, -1, 1.15, "CLOSING")])
    assert eps[0]["incomplete"] and eps[0]["status"] == "open" and checks["effect_mismatch"] == 1


def test_a_fill_that_crosses_zero_is_split_into_close_and_new_open():
    raw = [trade("2026-01-05T15:00:00+0000", stock("XYZ"), 10, 10.0, fee=1.0),
           trade("2026-01-06T15:00:00+0000", stock("XYZ"), -15, 12.0, fee=1.5)]
    eps, _, _ = episodes(raw)
    long, short = eps
    assert long["status"] == "closed" and long["pnl"] == pytest.approx(120 - 100 - 1.0 - 1.0)
    assert short["status"] == "open" and short["direction"] == "short" and short["open_qty"] == -5


def test_stock_split_adds_shares_inside_the_same_episode():
    raw = [trade("2021-01-19T15:00:00+0000", stock("GME"), 2, 40.0, "OPENING", fee=0),
           rad("2022-07-22T04:00:00+0000", stock("GME"), 6, "Stock Split"),
           trade("2023-05-26T15:00:00+0000", stock("GME"), -8, 23.5, "CLOSING", fee=0)]
    eps, _, _ = episodes(raw)
    assert len(eps) == 1 and eps[0]["status"] == "closed" and eps[0]["pnl"] == pytest.approx(8 * 23.5 - 80)
    assert not eps[0]["incomplete"] and any("Stock Split" in n for n in eps[0]["notes"])


def test_reverse_split_into_a_new_cusip_and_a_merger_carry_the_episode():
    raw = [trade("2023-08-11T15:00:00+0000", stock("96209A104"), 250, 0.21, "OPENING", fee=0),
           rad("2023-09-05T04:00:00+0000", stock("96209A104"), -250, "Mandatory Reverse Split"),
           rad("2023-09-05T04:00:00+0000", stock("96209A401"), 6, "Mandatory Reverse Split"),
           trade("2024-05-15T15:00:00+0000", stock("96209A401"), -6, 0.15, "CLOSING", fee=0),
           trade("2025-06-23T15:00:00+0000", stock("CLBR"), 8, 15.5, "OPENING", fee=0),
           rad("2025-07-01T04:00:00+0000", stock("CLBR"), -8, "CORP ACT EXP", "CLOSING"),
           rad("2025-07-01T04:00:00+0000", stock("CLBR"), 8, "CORP ACT EXP", "OPENING"),
           rad("2025-07-16T04:00:00+0000", stock("CLBR"), -8, "MANDATORY MERGER", "CLOSING"),
           rad("2025-07-16T04:00:00+0000", stock("PEW"), 8, "NEW CO", "OPENING"),
           trade("2025-07-16T15:00:00+0000", stock("PEW"), -8, 17.0, "CLOSING", fee=0)]
    eps, checks, _ = episodes(raw, {"96209A104": "WISA", "96209A401": "WISA"})
    wisa, clbr = eps
    assert wisa["status"] == "closed" and wisa["pnl"] == pytest.approx(0.9 - 52.5) and wisa["chart_symbol"] == "WISA"
    assert wisa["symbols"] == ["96209A104", "96209A401"] and not wisa["incomplete"]
    assert clbr["status"] == "closed" and clbr["key"] == "PEW" and clbr["pnl"] == pytest.approx(8 * 1.5)
    assert checks["corp_unhandled"] == 0


def test_normalize_ignores_cash_sweeps_and_reports_unattributed_records():
    sweep = {"activityId": 1, "time": "2021-03-01T05:00:00+0000", "type": "RECEIVE_AND_DELIVER", "status": "VALID",
             "accountNumber": "…560", "description": "CASH ALTERNATIVES PURCHASE",
             "transferItems": [{"instrument": {"assetType": "CASH_EQUIVALENT", "symbol": "MMDA1"}, "amount": 20.0}]}
    orphan = {"activityId": 2, "time": "2023-08-03T13:52:40+0000", "type": "TRADE", "status": "VALID",
              "accountNumber": "…560", "netAmount": 37.01}
    cancelled = {**trade("2026-01-05T15:00:00+0000", stock("XYZ"), 1, 10.0), "status": "CANCELED"}
    fills, issues = th.normalize_fills([sweep, orphan, cancelled])
    assert fills == [] and len(issues) == 2 and issues[0]["net_amount"] == 37.01


def test_adjusted_option_root_uses_the_plain_root_for_bars():
    inst = _opt_inst("HIMZ1 260320C00002000", "HIMZ1", "2026-03-20", "CALL", 2.0)
    eps, _, _ = episodes([trade("2026-03-02T15:00:00+0000", inst, 1, 0.5, "OPENING")])
    assert eps[0]["chart_symbol"] == "HIMZ" and not eps[0]["standard_deliverable"]


def _holdings(*positions):
    return {"accounts": [{"account": "…560", "positions": list(positions)}]}


def test_attach_marks_reconciles_and_flags_unresolved():
    opt = _opt_inst("SOXL  261218C00070000", "SOXL", "2026-12-18", "CALL", 70.0)
    gone = _opt_inst("SIRI_081823C6", "SIRI", "2023-08-18", "CALL", 6.0, legacy=True)
    raw = [trade("2026-11-02T15:00:00+0000", opt, 1, 5.0, "OPENING"),
           trade("2026-11-02T15:00:00+0000", stock("NEBX", "COLLECTIVE_INVESTMENT"), 10, 20.0, "OPENING"),
           trade("2023-07-31T18:56:35+0000", gone, 3, 0.33, "OPENING")]
    eps, _, _ = episodes(raw)
    held = _holdings({"kind": "option", "underlying": "SOXL", "expiry": "2026-12-18", "put_call": "CALL",
                      "strike": 70.0, "quantity": 1.0, "market_value": 650.0},
                     {"kind": "etf", "symbol": "NEBX", "quantity": 12.0, "market_value": 260.0},
                     {"kind": "equity", "symbol": "NEW", "quantity": 1.0, "market_value": 5.0})
    rec = th.attach_marks(eps, held)
    by = {e["key"]: e for e in eps}
    assert by["SOXL 2026-12-18 C 70"]["unrealized"] == pytest.approx(650 - 500 - 0.66)
    assert rec["qty_mismatch"] == [{"key": "NEBX", "history_qty": 10.0, "held_qty": 12.0}]
    assert rec["held_without_history"] == ["NEW"] and not rec["ok"]
    assert by["SIRI 2023-08-18 C 6"]["status"] == "unresolved" and by["SIRI 2023-08-18 C 6"]["incomplete"]


def test_raw_store_is_append_only_and_newest_copy_wins(tmp_path):
    a = {"activityId": 1, "time": "2026-01-02T15:00:00+0000", "netAmount": 1.0}
    assert th.append_raw([a], tmp_path) == 1 and th.append_raw([a], tmp_path) == 0
    assert th.append_raw([{**a, "netAmount": 2.0}], tmp_path) == 1
    lines = (tmp_path / "transactions.jsonl").read_text().splitlines()
    assert len(lines) == 2 and th.load_raw(tmp_path) == [{**a, "netAmount": 2.0}]
    with pytest.raises(ValueError):
        th.append_raw([{"time": "x"}], tmp_path)


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.text = body, status, json.dumps(body)

    def json(self):
        return self.body


class _GetOnlyClient:
    """Only the two GET methods exist; touching anything else (an order method) fails the test."""

    class Transactions:
        class TransactionType:
            TRADE, RECEIVE_AND_DELIVER = "TRADE", "RECEIVE_AND_DELIVER"

    def __init__(self, data_until_windows=2):
        self.calls, self.n = [], data_until_windows

    def get_account_numbers(self):
        return _Resp([{"accountNumber": "123456560", "hashValue": "HASH"}])

    def get_transactions(self, acct_hash, *, start_date, end_date, transaction_types):
        self.calls.append((acct_hash, start_date, end_date))
        k = len(self.calls)
        return _Resp([{"activityId": k, "time": end_date.isoformat(), "accountNumber": "123456560"}] if k <= self.n else [])


def test_backfill_walks_back_until_three_empty_windows_and_masks_accounts(tmp_path):
    client = _GetOnlyClient(data_until_windows=2)
    now = datetime(2026, 11, 16, 20, tzinfo=timezone.utc)
    out = th.sync_transactions(client, now, tmp_path)
    assert out["mode"] == "backfill" and len(client.calls) == 5 and out["written"] == 2
    stored = (tmp_path / "transactions.jsonl").read_text()
    assert "123456560" not in stored and "HASH" not in stored
    assert {r["accountNumber"] for r in th.load_raw(tmp_path)} == {"…560"}
    assert all(we - ws == timedelta(days=th.WINDOW_DAYS) for _, ws, we in client.calls)


def test_incremental_sync_starts_an_overlap_before_the_newest_record(tmp_path):
    th.append_raw([{"activityId": 1, "time": "2026-11-01T15:00:00+0000"}], tmp_path)
    client = _GetOnlyClient(data_until_windows=0)
    th.sync_transactions(client, datetime(2026, 11, 16, 20, tzinfo=timezone.utc), tmp_path)
    (_, start, end), = client.calls
    assert start == datetime(2026, 11, 1, 15, tzinfo=timezone.utc) - timedelta(days=th.INCREMENTAL_OVERLAP_DAYS)
