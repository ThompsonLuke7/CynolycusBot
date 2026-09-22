from scripts.discord_ledger.curate import decision, source_contracts, source_price, source_stop, supplement


def source(text: str) -> dict:
    return {"message_id": "1458529779379863624", "author_id": "1079391263083733072",
            "channel_id": "1416372120388243516", "text": text,
            "timestamp_utc": "2026-01-07T18:36:37.577Z", "timestamp_raw": "1/7/2026 1:36 PM",
            "reply_to_message_id": None, "is_edited": False}


def draft(**overrides) -> dict:
    row = {"event_id": "1458529779379863624:0", "message_id": "1458529779379863624",
           "author_id": "1079391263083733072", "channel_id": "1416372120388243516",
           "event_type": "entry", "speech_act": "instruction", "symbol": "RKLB", "symbol_raw": "RKLB",
           "direction": None, "instrument_type": "unknown", "option_type": None, "strike": 90.0,
           "expiry_raw": "1/16", "expiry": None, "price": 1.9, "price_text": "at 1.9",
           "stop_price": None, "stop_text": None, "quantity": None, "quantity_text": None,
           "uncertainties": []}
    row.update(overrides)
    return row


def test_explicit_entry_price_not_confused_with_stop() -> None:
    text = "RKLB 90c 1/16\nEntry 2.7\nStops on this contract at 1.9\nOr a 1 hour close below 82\n@everyone"
    event, status = decision(draft(), source(text))
    assert status == "kept"
    assert event["instrument_type"] == "option"
    assert event["option_type"] == "call"
    assert event["expiry"] == "2026-01-16"
    assert event["price"] == 2.7
    assert event["stop_price"] == 1.9
    assert event["field_sources"]["price"] == [source(text)["message_id"]]


def test_bare_contract_premium_and_non_entry_language() -> None:
    contracts = source_contracts("QQQ 615 put .50 @everyone")
    assert [(c[1], c[2], c[3]) for c in contracts] == [("QQQ", 615.0, "put")]
    assert source_price("QQQ 615 put .50 @everyone", contracts[0][0])[0] == 0.5
    assert source_stop("QQQ 615 put .50 @everyone") == (None, None)
    assert source_contracts("I will close at 682.50 and enter 6950 calls") == []

    recap, _ = decision(draft(symbol="GLD", symbol_raw="GLD", strike=415.0),
                        source("GLD 415 P ALMOST 100% @everyone"))
    assert recap["event_type"] == "performance_claim"
    assert recap["price"] is None


def test_multicontract_ideas_are_watchlists_not_entries() -> None:
    idea, _ = decision(draft(symbol="MU", symbol_raw="MU", strike=950.0),
                       source("Lotto Friday Trade Ideas: $MU 950C above 910; $NVDA 222.5C above 220"))
    assert idea["event_type"] == "watchlist"
    assert idea["price"] is None


def test_supplement_preserves_bare_price_quantity_and_0dte_expiry() -> None:
    message = source("SPY 686 p 8 cons .24 0DTE\n@everyone")
    event = supplement(message)
    assert event is not None
    assert event["symbol"] == "SPY"
    assert event["price"] == 0.24
    assert event["quantity"] == 8.0
    assert event["expiry"] == "2026-01-07"


def test_supplement_does_not_promote_watch_or_large_integer_to_fill_price() -> None:
    assert supplement(source("Watching $ERO $45c")) is None
    event = supplement(source("SNDK 1800 c at 4400 WOW"))
    assert event is not None
    assert event["price"] is None
    assert event["price_text"] == "at 4400"
