from __future__ import annotations

import json

import pytest

from core.broker_fill_reconciliation import (
    account_label_for_client, reconcile_account, register_order_ownership, require_clean_certificate,
)
from core.live_state import append_once


class Client:
    def __init__(self, rows):
        self.rows = rows

    def get_account_activities(self, **_kwargs):
        return self.rows


def _fill(*, activity_id="a1", order_id="o1", qty="2", price="10"):
    return {"id": activity_id, "activity_type": "FILL", "order_id": order_id,
            "symbol": "DELL260918C00470000", "side": "sell", "qty": qty, "price": price}


def test_reconciliation_passes_for_registered_projected_exit(tmp_path):
    register_order_ownership(order_id="o1", module="htf", symbol="DELL260918C00470000",
                             side="sell", qty=2, route="option", root=tmp_path)
    append_once(tmp_path / "htf" / "exit_fills.jsonl",
                {"order_id": "o1", "qty": 2, "proceeds": 2000}, identity="o1:fill:2")
    cert = reconcile_account(Client([_fill()]), root=tmp_path)
    assert cert["status"] == "PASS"
    assert require_clean_certificate(root=tmp_path)["status"] == "PASS"


def test_unattributed_fill_is_quarantined_and_blocks_reports(tmp_path):
    cert = reconcile_account(Client([_fill()]), root=tmp_path)
    assert cert["status"] == "FAIL"
    assert cert["unattributed_fills"][0]["id"] == "a1"
    quarantine = tmp_path / "broker_reconciliation" / "paper" / "quarantine.jsonl"
    assert json.loads(quarantine.read_text().strip())["reason"] == "unattributed_broker_fill"
    with pytest.raises(RuntimeError, match="blocked"):
        require_clean_certificate(root=tmp_path)


def test_broker_ledger_quantity_or_proceeds_mismatch_fails(tmp_path):
    register_order_ownership(order_id="o1", module="htf", symbol="DELL260918C00470000",
                             side="sell", qty=2, route="option", root=tmp_path)
    append_once(tmp_path / "htf" / "exit_fills.jsonl",
                {"order_id": "o1", "qty": 1, "proceeds": 1000}, identity="o1:fill:1")
    cert = reconcile_account(Client([_fill()]), root=tmp_path)
    assert cert["status"] == "FAIL"
    assert cert["ledger_mismatches"][0]["broker_qty"] == 2


def test_repeated_activity_is_idempotent(tmp_path):
    register_order_ownership(order_id="o1", module="htf", symbol="DELL260918C00470000",
                             side="sell", qty=2, route="option", root=tmp_path)
    append_once(tmp_path / "htf" / "exit_fills.jsonl",
                {"order_id": "o1", "qty": 2, "proceeds": 2000}, identity="o1:fill:2")
    client = Client([_fill()])
    reconcile_account(client, root=tmp_path)
    reconcile_account(client, root=tmp_path)
    journal = tmp_path / "broker_reconciliation" / "paper" / "fills.jsonl"
    assert len(journal.read_text().splitlines()) == 1


def test_client_account_partition_is_not_silently_paper_for_live():
    class LiveClient:
        _trading_base = "https://api.alpaca.markets"
    class PaperClient:
        _trading_base = "https://paper-api.alpaca.markets"
    assert account_label_for_client(LiveClient()) == "live"
    assert account_label_for_client(PaperClient()) == "paper"
