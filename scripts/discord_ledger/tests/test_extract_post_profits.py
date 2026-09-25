from __future__ import annotations

from scripts.discord_ledger.extract_post_profits import parse_claims


def test_extracts_each_contract_leg_from_recap_without_cross_assigning_strikes() -> None:
    message = {"text": "$AAPL 310C: $1.00 →2.28(+120%)\n$SPX 7685P: 1.30 → 2.00 (+54%)"}
    rows = parse_claims(message, {"AAPL", "SPX"})
    assert [(row["symbol"], row["strike"], row["option_type"], row["reported_entry_price"], row["reported_exit_price"], row["reported_return_pct"])
            for row in rows] == [("AAPL", 310.0, "call", 1.0, 2.28, 120.0), ("SPX", 7685.0, "put", 1.3, 2.0, 54.0)]


def test_marks_explicit_loss_as_negative_claim() -> None:
    message = {"text": "$SPX 7805C 1.20 → (-100%) Loss"}
    rows = parse_claims(message, {"SPX"})
    assert len(rows) == 1
    assert rows[0]["reported_return_pct"] == -100.0


def test_extracts_put_word_form_in_a_multileg_recap() -> None:
    message = {"text": "TSLA 415 put 2.42 → 2.98 (+23%)\nSPY 690 put 1.91 → 4.30 (+125%)"}
    rows = parse_claims(message, {"TSLA", "SPY"})
    assert [(row["symbol"], row["option_type"], row["reported_return_pct"]) for row in rows] == [
        ("TSLA", "put", 23.0), ("SPY", "put", 125.0)]
