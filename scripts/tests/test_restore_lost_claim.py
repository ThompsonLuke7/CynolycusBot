"""A restored claim must survive the engine's own exit recovery.

`restore_lost_claim.py` put AMLX, SGMT and TENX back under Meta on 2026-10-06.
At 09:33 the next morning Meta's risk pass dropped all three again: each
position's earlier horizon exit is kept as evidence on disk, the restored entry
carried the same entry bar and no record of having settled that exit, so the
engine replayed it, saw a filled full exit, and closed the position.
"""
from __future__ import annotations

import json

from core.order_reconciliation import recover_pending_exits, save_evidence
from scripts.restore_lost_claim import settled_exit_ids

ENTRY_BAR = "2026-08-04 14:00:00+00:00"
EXIT_ID = "7ef9c7fe-0348-4435-b8f9-9f3cdc2cc741"


def _evidence(root, order_id=EXIT_ID, symbol="AMLX", entry_bar=ENTRY_BAR):
    """What the engine saved when Meta's 09-24 horizon exit of 191 AMLX filled."""
    save_evidence(root, "meta_ranker", order_id, {
        "owner": symbol, "symbol": symbol, "complete": True, "result": "closed",
        "state": {"route": "equity", "symbol": symbol, "shares": 0.0, "entry_bar": entry_bar,
                  "exit_pending": {"order_id": order_id, "qty": 191.0, "route": "equity",
                                   "status": "filled", "confirmed_qty": 191.0}},
    })


def _restored(**extra):
    return {"route": "equity", "symbol": "AMLX", "shares": 36.0, "runs_held": 68,
            "bars_out": 68, "trimmed": True, "entry_bar": ENTRY_BAR,
            "entry_avg_price": 22.118326, **extra}


def test_the_old_restore_was_reclosed_by_the_saved_exit(tmp_path):
    """The defect, pinned: without the list the engine re-attaches the old exit."""
    _evidence(tmp_path)
    managed = {"AMLX": _restored()}

    recover_pending_exits(tmp_path, "meta_ranker", managed)

    assert managed["AMLX"]["exit_pending"]["order_id"] == EXIT_ID


def test_a_restored_claim_lists_its_settled_exits_and_is_left_alone(tmp_path):
    _evidence(tmp_path)
    settled = settled_exit_ids(tmp_path, "meta_ranker", "AMLX", ENTRY_BAR)
    assert settled == [EXIT_ID]
    managed = {"AMLX": _restored(exit_reconciled_ids=settled)}

    recover_pending_exits(tmp_path, "meta_ranker", managed)

    assert "exit_pending" not in managed["AMLX"]
    assert managed["AMLX"]["shares"] == 36.0


def test_only_this_positions_exits_are_listed(tmp_path):
    _evidence(tmp_path)
    _evidence(tmp_path, order_id="other-symbol", symbol="SGMT")
    _evidence(tmp_path, order_id="earlier-trade", entry_bar="2026-05-01 14:00:00+00:00")
    (tmp_path / "meta_ranker" / "exit_orders" / "second.json").write_text(json.dumps({
        "owner": "AMLX", "symbol": "AMLX", "complete": True,
        "state": {"entry_bar": ENTRY_BAR, "exit_pending": {"order_id": "trim-1"}}}))

    assert sorted(settled_exit_ids(tmp_path, "meta_ranker", "AMLX", ENTRY_BAR)) == sorted(
        [EXIT_ID, "trim-1"])


def test_a_module_with_no_exit_evidence_lists_nothing(tmp_path):
    assert settled_exit_ids(tmp_path, "meta_ranker", "AMLX", ENTRY_BAR) == []
