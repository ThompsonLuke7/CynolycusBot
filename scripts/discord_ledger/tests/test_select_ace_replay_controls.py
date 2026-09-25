from __future__ import annotations

from scripts.discord_ledger.select_ace_replay_controls import select


def _row(message_id: str, decision: str) -> dict:
    return {"symbol": "SPY", "session_date_et": "2026-01-05", "direction_sign": 1,
            "source_message_ids": [message_id], "decision_at_utc": decision,
            "trade_ids": [f"trade:{message_id}"], "source_quote": "SPY call",
            "source_tags": [], "exclusive_bucket": "unlabelled", "source_session_file": "fixture.parquet"}


def test_control_selector_assigns_controls_without_replacement_within_group() -> None:
    alerts = [_row("a", "2026-01-05T16:00:00+00:00"), _row("b", "2026-01-05T16:05:00+00:00")]
    controls = []
    for message_id in ("a", "b"):
        controls.extend([_row(message_id, "2026-01-05T15:00:00+00:00"),
                         _row(message_id, "2026-01-05T15:05:00+00:00")])
    selected = select(alerts, controls)
    assert len(selected) == 2
    assert len({row["decision_at_utc"] for row in selected}) == 2
    assert {row["paired_source_alert_decision_at_utc"] for row in selected} == {
        "2026-01-05T16:00:00+00:00", "2026-01-05T16:05:00+00:00"}
