"""The classifier that decides which ledger rows are test residue.

This script edits live P&L ledgers, so the risk that matters is a FALSE POSITIVE
— quarantining a real trade. Every case below that is ambiguous must be kept.
"""
from __future__ import annotations

import json

from scripts.quarantine_test_rows_from_ledgers import is_synthetic, main, scan

_UUID = "eec50d8b-9e2d-4e7a-9c1d-f104c4c1d948"


def _row(**kw):
    base = {"ts": "2026-08-13T19:52:55+00:00", "module": "dealer_ranker",
            "ticker": "FIG", "order_symbol": "FIG260814C00023500", "route": "option",
            "side": "sell", "qty": 30.0, "exit_reason": "take_profit_+20%",
            "entry_avg_price": 0.8, "exit_fill_price": 2.92, "realized_pnl": 6360.0,
            "order_id": _UUID}
    base.update(kw)
    return base


def test_real_broker_fill_is_kept():
    assert is_synthetic(_row()) is False


def test_fixture_row_is_flagged():
    assert is_synthetic(_row(ticker="AAA", order_symbol="AAA260724C00010000",
                             order_id="AAA260724C00010000-sell",
                             realized_pnl=None, exit_fill_price=None,
                             entry_avg_price=None)) is True
    assert is_synthetic(_row(ticker="OLD", order_id="OLD-sell",
                             realized_pnl=None, exit_fill_price=None,
                             entry_avg_price=None)) is True


def test_row_with_a_realized_pnl_is_never_removed():
    """A P&L came from a real fill price, whatever the order_id looks like."""
    assert is_synthetic(_row(order_id="weird-id", realized_pnl=-2016.0)) is False


def test_row_without_an_order_id_is_kept():
    """A real exit whose response carried no id — not a fixture."""
    for missing in (None, ""):
        assert is_synthetic(_row(order_id=missing, realized_pnl=None)) is False


def test_non_dict_is_not_synthetic():
    assert is_synthetic(None) is False
    assert is_synthetic(["not", "a", "record"]) is False


def test_scan_preserves_unparseable_lines(tmp_path):
    """The dealer ledger holds an `audit_gap` repair marker — never drop it."""
    p = tmp_path / "closed_trades.jsonl"
    marker = '{"event": "audit_gap", "reason": "nul_block_from_unclean_shutdown"}'
    p.write_text("\n".join([
        json.dumps(_row()),
        marker,
        json.dumps(_row(ticker="AAA", order_id="AAA-sell", realized_pnl=None)),
        "{not json",
    ]) + "\n")
    kept, removed, bad = scan(p)
    assert len(removed) == 1
    assert "{not json" in kept and bad == ["{not json"]
    assert marker in kept


def test_dry_run_does_not_modify_the_ledger(tmp_path, capsys):
    led = tmp_path / "Data" / "inference" / "dealer_ranker"
    led.mkdir(parents=True)
    p = led / "closed_trades.jsonl"
    original = "\n".join([json.dumps(_row()),
                          json.dumps(_row(ticker="AAA", order_id="AAA-sell",
                                          realized_pnl=None))]) + "\n"
    p.write_text(original)

    main(["--root", str(tmp_path)])
    assert p.read_text() == original
    assert "DRY RUN" in capsys.readouterr().out
    assert not list(led.glob("*.bak"))


def test_apply_quarantines_and_backs_up(tmp_path):
    led = tmp_path / "Data" / "inference" / "dealer_ranker"
    led.mkdir(parents=True)
    p = led / "closed_trades.jsonl"
    real, fixture = _row(), _row(ticker="AAA", order_id="AAA-sell", realized_pnl=None)
    p.write_text(json.dumps(real) + "\n" + json.dumps(fixture) + "\n")

    main(["--root", str(tmp_path), "--apply"])

    surviving = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert surviving == [real]
    quarantined = [json.loads(l) for l in
                   (led / "closed_trades.quarantine.jsonl").read_text().splitlines() if l.strip()]
    assert quarantined == [fixture]
    backups = list(led.glob("*.bak"))
    assert len(backups) == 1
    assert len(backups[0].read_text().strip().splitlines()) == 2   # the original, intact
