from __future__ import annotations

from datetime import date

import pytest

from scripts.afterhour.extract_calls import dedupe, extract_post, parse_expiry


def post(title: str, body: str = "", tickers=("SPY",), when="2026-11-12T16:49:00Z", topic=None) -> dict:
    return {"username": "u", "post_id": "pst_1", "share_url": "", "posted_at_utc": when, "title": title,
            "body": body, "tickers": list(tickers), "topic": topic, "has_detail": True}


def one(p: dict) -> dict:
    calls = extract_post(p)
    assert len(calls) == 1, calls
    return calls[0]


def test_forward_contract_is_tier_b_put() -> None:
    c = one(post("$SPY $755 Puts For Next Monday Could Get Are Still In Play"))
    assert (c["tier"], c["direction"], c["instrument"], c["strike"]) == ("B", "short", "put", 755.0)
    assert c["expiry"] == "2026-11-16"  # posted Thu 2026-11-12 -> next Monday


def test_logged_option_entry_is_tier_a_with_expiry() -> None:
    c = one(post("I bought 1 $MSTR $415c 12/27 @ $28.25", tickers=("MSTR",), when="2024-12-13T15:00:00Z"))
    assert (c["tier"], c["direction"], c["strike"], c["expiry"]) == ("A", "long", 415.0, "2024-12-27")


def test_share_entry_and_short_entry() -> None:
    assert one(post("I bought 200 shares $KITT @ $2.11", tickers=("KITT",)))["direction"] == "long"
    assert one(post("$RGTI short", tickers=("RGTI",)))["direction"] == "short"


def test_cash_secured_put_is_bullish() -> None:
    c = one(post("$FN - Aug 21st $470 Cash Secured Put", tickers=("FN",), when="2026-08-18T14:00:00Z"))
    assert c["direction"] == "long" and c["expiry"] == "2026-08-21"


@pytest.mark.parametrize("title", [
    "Caught $MSFT 530c for a nice $100 move up",
    "I exited $NVDA $140c 11/8 for 25% profit",
    "Those $SPY Puts Have Popped $100 Since I Posted About Them Earlier",
    "When do we buy this $INTC?",
    "Sold $PEW for 14% gain",
])
def test_hindsight_and_questions_are_not_calls(title) -> None:
    assert extract_post(post(title, tickers=("MSFT", "NVDA", "SPY", "INTC", "PEW"))) == []


def test_mixed_clause_keeps_only_the_new_entry() -> None:
    calls = extract_post(post("I'm out of my $MSTR call & I just bought bought 3.5 shares of $IONQ", tickers=("MSTR", "IONQ")))
    assert [(c["symbol"], c["direction"]) for c in calls] == [("IONQ", "long")]


def test_gain_topic_posts_are_skipped() -> None:
    assert extract_post(post("I bought $SPY calls", topic="gain")) == []


def test_opinion_and_negated_opinion() -> None:
    assert one(post("$MSTR very undervalued", tickers=("MSTR",)))["tier"] == "C"
    assert extract_post(post("I'm not bullish on $MSTR here", tickers=("MSTR",))) == []


def test_untagged_cashtag_is_ignored_when_post_has_tags() -> None:
    assert extract_post(post("bought $ZZZZ", tickers=("SPY",))) == []


def test_month_year_expiry_is_third_friday() -> None:
    assert parse_expiry("$MARA Jan 2027 30C", date(2025, 9, 16)) == "2027-01-15"


def test_dedupe_keeps_first_per_day() -> None:
    a = {"username": "u", "symbol": "X", "direction": "long", "tier": "C", "posted_at_utc": "2026-01-05T15:00:00Z"}
    b = {**a, "tier": "A", "posted_at_utc": "2026-01-05T18:00:00Z"}
    assert dedupe([b, a]) == [a]


def test_short_squeeze_is_not_a_short_call() -> None:
    c = one(post("$GME short squeeze incoming 🚀", tickers=("GME",)))
    assert c["direction"] == "long"


@pytest.mark.parametrize("title,tickers,expected", [
    ("Bought more $SNOW, selling $PYPL and $SQ", ("SNOW", "PYPL", "SQ"), [("SNOW", "long")]),
    ("Selling this $HOOD I bought at the bottom", ("HOOD",), []),
    ("Sold $ASST leaps for +10%", ("ASST",), []),
    ("I sold $MSTR $390c 1/24 +30%", ("MSTR",), []),
    ("$BULL Analysis Update", ("BULL",), []),
    ("$GME OMG GUYS WHAT IS A MARGIN CALL", ("GME",), []),
    ("In honor of those who bought $BYND today", ("BYND",), []),
    ("Dips On $IWM Continues To Be Bought", ("IWM",), []),
    ("If You're Trading The $XOM breakout Then 2nd Target Has Been Hit", ("XOM",), []),
    ("Sold a bunch of CSP at open on $HIMS expiring 4/4", ("HIMS",), [("HIMS", "long")]),
    ("Sniped a $QQQ put", ("QQQ",), [("QQQ", "short")]),
])
def test_review_round_1_fixes(title, tickers, expected) -> None:
    assert [(c["symbol"], c["direction"]) for c in extract_post(post(title, tickers=tickers))] == expected


@pytest.mark.parametrize("title,tickers,when,expected", [
    ("I picked a $320 Jan 25 call $PANW", ("PANW",), "2024-02-27T11:06:00Z", ("PANW", 320.0, "2025-01-17")),
    ("$BABA $150 10/17 call 4,000+", ("BABA",), "2025-09-01T15:00:00Z", ("BABA", 150.0, "2025-10-17")),
    ("Picked some $99c for tomorrow on $NVDA", ("NVDA",), "2024-09-05T16:23:00Z", ("NVDA", 99.0, "2024-09-06")),
    ("$AAPL calls here, $SPY $600 puts for next week", ("AAPL", "SPY"), "2024-11-06T15:00:00Z", ("SPY", 600.0, "2024-11-15")),
])
def test_strike_and_expiry_parsing(title, tickers, when, expected) -> None:
    calls = extract_post(post(title, tickers=tickers, when=when))
    got = [(c["symbol"], c["strike"], c["expiry"]) for c in calls if c["strike"]]
    assert expected in got


@pytest.mark.parametrize("title,tickers", [
    ("Cat Saturday Livestream Today - $BTC $QQQ $IWM Analysis - 530p EST / 230p PST", ("BTC", "QQQ", "IWM")),
    ("Companies with a lower Price to Earnings ratio than $NKE at 27 P/E:", ("NKE",)),
])
def test_times_and_ratios_are_not_contracts(title, tickers) -> None:
    assert [c for c in extract_post(post(title, tickers=tickers)) if c["strike"]] == []


@pytest.mark.parametrize("body", [
    "$SNDK is down 20%, $AMD and $MU are down 15%.",
    "Buy rating reiterated with a PT of $2",
    "$QCOM fell hard after handset revenue dropped.",  # body opinion without any forward-looking word
])
def test_body_noise_is_not_a_call(body) -> None:
    assert extract_post(post("Market recap", body=body, tickers=("SNDK", "AMD", "MU", "HUMA", "QCOM"))) == []


def test_body_forward_opinion_is_kept() -> None:
    c = one(post("Recap", body="I think $QCOM is heading lower into earnings.", tickers=("QCOM",)))
    assert (c["direction"], c["source"]) == ("short", "body")


@pytest.mark.parametrize("title,body,tickers,expected", [
    ("Recap", "I am not buying $HOOD.", ("HOOD",), []),
    ("$DJT", "I wish I had a position before market opened", ("DJT",), []),
    ("$TSM - congratulations to those who got in", "", ("TSM",), []),
    ("Flow", "The one guy fading it is long $LITE puts.", ("LITE",), []),
    ("Flow", "• $RIOT — CALL BUY $21, exp 12/18/26, $1.40M (10:12 PT)", ("RIOT",), []),
    ("Cash", "$220k or so in cash puts me in a decent position to pick up any wild moves (looking at you $TSLA)", ("TSLA",), []),
    ("List", "Side note: if anyone tries to put $ONDS on the list, I'm throwing hands.", ("ONDS",), []),
    ("Recap", "🎯 $CVS - I said calls, market agreed", ("CVS",), []),
    ("RIP to $HOOD and $BULL", "", ("HOOD", "BULL"), []),
    ("$CRBG 🚀🚀🚀 Sheesh", "", ("CRBG",), []),
    ("Recap", "$AVGO used $314 as a springboard to shake out bears this week.", ("AVGO",), []),
    ("Weekly", "* $GOOGL CCs that I bought shares on Monday, expiring Friday", ("GOOGL",), []),
    ("Boxx", "While you guys were sweating over $NVDA earnings, I was quietly loading the boat on $BOXX.", ("NVDA", "BOXX"),
     [("BOXX", "long")]),
    ("$EMKR", "Here's a cheap one. A break above $1.35 and it can push towards $1.60.", ("EMKR",), [("EMKR", "long")]),
    ("Market", "My money is on $IGV with a little bit of $IBIT sprinkled in.", ("IGV", "IBIT"), [("IGV", "long"), ("IBIT", "long")]),
    ("Fill", "I paid $2.39 for 1 $SPY 600c 11/15", ("SPY",), [("SPY", "long")]),
])
def test_review_round_2_fixes(title, body, tickers, expected) -> None:
    assert [(c["symbol"], c["direction"]) for c in extract_post(post(title, body=body, tickers=tickers))] == expected


def test_conflicting_directions_in_one_post_keep_the_stronger_tier() -> None:
    calls = extract_post(post("$SPY $600 puts", body="Long term I think $SPY goes higher.", tickers=("SPY",)))
    assert [(c["tier"], c["direction"]) for c in calls] == [("B", "short")]


@pytest.mark.parametrize("title,body,tickers,expected", [
    ("Swing", "I realized some nice gains today swinging $BULL calls", ("BULL",), []),
    ("Day", "$GRAB about even today, $KMX calls up", ("GRAB", "KMX"), []),
    ("$BTC", "Calls 169% tomorrow", ("BTC",), []),
    ("$SPY", "Out of puts looking to enter NVDA AND OR AMZN", ("SPY",), []),
    ("$GME", "IF YOU GOT IN WHEN I BROUGHT ATTENTION TO IT", ("GME",), []),
    ("$AAPL", "I bought in 2005 and went all in.", ("AAPL",), []),
    ("Flow", "the $SPCX chase is getting silly ($8.19M of 225 calls for next June)", ("SPCX",), []),
    ("$SPY", "To all that bought puts thanks for the liquidity.", ("SPY",), []),
    ("Live On CNBC, Joe Terranova Announces Bought Apple!", "", ("AAPL",), []),
    ("$PLTR", "“When we gonna rug?”", ("PLTR",), []),
    ("$SPX", "bearish shooting star candle closes for $Silver next week", ("SPX",), []),
])
def test_review_round_3_fixes(title, body, tickers, expected) -> None:
    assert [(c["symbol"], c["direction"]) for c in extract_post(post(title, body=body, tickers=tickers))] == expected


def test_contract_count_is_not_a_strike_and_not_robinhood_is_not_hood() -> None:
    c = one(post("$AGL", body="I have 200 call options for 1/27 after the reverse split.", tickers=("AGL",)))
    assert c["strike"] is None
    assert [c["symbol"] for c in extract_post(post("$RH (this is NOT Robinhood) 420c", tickers=("HOOD",)))] == []
