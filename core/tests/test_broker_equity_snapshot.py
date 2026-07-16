from datetime import datetime, timezone
import json

from core.broker_equity_snapshot import capture_snapshot


class _Client:
    def get_account(self):
        return {
            "equity": "100123.45", "last_equity": "100000", "portfolio_value": "100123.45",
            "cash": "50000", "buying_power": "200000", "account_number": "must_not_persist",
        }

    def get_positions(self):
        return [{
            "symbol": "ABC", "asset_class": "us_equity", "qty": "100", "side": "long",
            "avg_entry_price": "10", "current_price": "11", "market_value": "1100",
            "cost_basis": "1000", "unrealized_pl": "100", "unrealized_plpc": "0.1",
            "unrealized_intraday_pl": "25", "unrealized_intraday_plpc": "0.023",
        }]


def test_capture_snapshot_writes_account_and_position_marks_without_account_id(tmp_path):
    out = capture_snapshot(
        client=_Client(), account_label="paper", root=tmp_path,
        now=datetime(2026, 7, 14, 20, 5, tzinfo=timezone.utc),
    )
    path = tmp_path / "broker_equity_20260714_paper.jsonl"
    assert out["path"] == str(path)
    row = json.loads(path.read_text())
    assert row["session_date_et"] == "2026-07-14"
    assert row["account"]["equity"] == "100123.45"
    assert row["positions"][0]["symbol"] == "ABC"
    assert "account_number" not in json.dumps(row)
