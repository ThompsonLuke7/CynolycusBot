"""Rule-based extraction of forward-looking callouts from AfterHour posts.

No LLM: deterministic regex/lexicon rules over title + body, clause by clause.
Tiers (scored separately downstream):
  A entry    - explicit position opened: bought/entered/long/short/sold puts (CSP)...
  B contract - forward strike/side (e.g. "$755 puts for next Monday") without hindsight words
  C opinion  - ticker + unambiguous bullish/bearish wording (no question, no negation, no conflict)
Excluded: outcome/hindsight clauses (exited, closed, trimmed, caught, "for 25% profit",
"since I posted"...), questions, covered calls, and posts tagged gain/loss.
A post's tickers come from AfterHour's own `securities` tags, matched to cashtags,
tagged bare symbols, or a few company names inside the clause.

Subcommands:
  select   write the detail-fetch selection (posts with tickers or cashtags)
  extract  write calls.jsonl (+ posts.jsonl normalized) from users/ and details/
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.afterhour.collect import post_hash, read_jsonl, write_jsonl

ET = ZoneInfo("America/New_York")
SCHEMA = "afterhour_callouts_rules_v1"

CASHTAG = re.compile(r"\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b")
NAMES = {"tesla": "TSLA", "nvidia": "NVDA", "apple": "AAPL", "amazon": "AMZN", "microsoft": "MSFT",
         "google": "GOOGL", "alphabet": "GOOGL", "palantir": "PLTR", "netflix": "NFLX", "intel": "INTC",
         "boeing": "BA", "robinhood": "HOOD", "coinbase": "COIN", "microstrategy": "MSTR"}
# bare tagged symbols that are also ordinary words are only accepted as cashtags
BARE_STOP = {"A", "I", "AI", "ALL", "AM", "AN", "ARE", "AT", "BE", "BIG", "BULL", "BY", "CAN", "CAR", "DD", "EAT",
             "FOR", "FUN", "GO", "HAS", "HE", "HOOD", "IN", "IS", "IT", "LOVE", "LOW", "ME", "MY", "NEW", "NOW",
             "OF", "ON", "ONE", "OPEN", "OR", "OUT", "PLAY", "REAL", "RUN", "SO", "TO", "UP", "US", "WE", "YOU",
             "EOD", "EOW", "ATH", "CEO", "USA", "ET", "PM", "IPO", "ER", "GDP", "CPI", "FOMC", "PT", "TA", "IV"}

ACTION_AHEAD = r"(?=(?:and\s+)?(?:i\s+)?(?:just\s+)?(?:bought|buying|sold|selling|added|adding|trimmed|exited|closed|shorted|loaded|grabbed)\b)"
CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|;\s*|\s+&\s+|\s+but\s+|\s+\|\s+|,\s*" + ACTION_AHEAD + r"|\s+and\s+" + ACTION_AHEAD, re.I)

NUM = r"\$?(\d{1,5}(?:\.\d{1,2})?)"
# "530c" / "$5c" (not a date fragment "10/17", not a time "530p EST", not "P/E")
ATTACHED = re.compile(r"(?<![\d/.])\$?(\d{1,5}(?:\.\d{1,2})?)\s?([cp])\b(?!\s*/)(?!\s*(?:est|pst|cst|edt|pdt|et|pt|ct)\b)", re.I)
SIDE_WORD = re.compile(r"\b(calls?|puts?)\b", re.I)
DOLLAR_NUM = re.compile(r"\$\s?(\d{1,5}(?:\.\d{1,2})?)(?![\d/,]|\s?[kmb]\b)", re.I)
PUT_VERB = re.compile(r"\b(?:to|will|would|could|should|gonna|i|we|you|they|just|cash)\s+puts?\b|"
                      r"\bputs?\s+(?:me|you|him|her|us|them|it|this|that|my|your|our|their|together|pressure|out|up|TKR\s+on|"
                      r"in\s+(?:a|the|my)|a lot|money)\b", re.I)
BARE_BEFORE_SIDE = re.compile(r"(?<![\d/.$])(\d{1,5}(?:\.\d{1,2})?)\s+(calls?|puts?)\b", re.I)
MONTH_BEFORE = re.compile(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*$", re.I)
SIDE_ONLY = re.compile(r"\b(calls?|puts?|leaps?|csps?|cash[- ]secured puts?|covered calls?)\b", re.I)
SOLD_PUTS = re.compile(r"\b(sold|sell(?:ing)?|sto)\b[^.]{0,40}?\b(puts?|csps?|cash[- ]secured)|\bcsps?\b|\bcash[- ]secured puts?\b", re.I)
CSP_STRIKE = re.compile(NUM + r"\s*(?:strike\s*)?(?:cash[- ]secured puts?|csps?|puts?|p)\b", re.I)
SHORT_CALL = re.compile(r"\bshort(?:ed|ing)?\b(?![\s-]*(?:term|squeeze|interest|sellers?|float|dated|covering|report))", re.I)
COVERED = re.compile(r"\bcovered calls?\b|\bccs?\b|\b(sold|sell(?:ing)?)\b[^.]{0,30}?\bcalls?\b", re.I)

ENTRY = re.compile(r"\b(bought|buying|bot|bto|grabbed|sniped|entered|entering|added|adding|loaded|loading|"
                   r"open(?:ed|ing) (?:a|an|my|new|some|up|\d+|positions?|calls?|puts?|shares?)|my money is on|give it a shot|"
                   r"hedge (?:this |it )?with|"
                   r"picked up|scooped|started a position|starter position|took a position|got in|i'?m in|"
                   r"going long|went long|shorted|shorting|went short|going short|yolo'?d?|sold to open)\b", re.I)
HINDSIGHT = re.compile(
    r"\b(exit(?:ed|ing)?|closed|closing|trim(?:med|ming)?|took profits?|taking profits?|cashed|caught|printed|"
    r"hit (?:my|the|our)? ?(?:pt|target)|pt hit|target hit|since i posted|told (?:you|yall|y'all)|"
    r"called it|as i said|if you caught|nice move|popped|ran \d|up \d+%|\d+% (?:gain|profit|up)|for (?:a )?(?:\d+%|profit|gain)|"
    r"locked in|secured profits?|i(?:'m| am) out|stopped out|gain(?:s)? on|(?:has|have|had|got|was|been) hit|"
    r"than (?:i|we) (?:thought|expected)|went (?:crazy|up|down|off)|ripped|exploded|soared|skyrocketed|mooned|went from|"
    r"i said|market agreed|best play|(?:got|get|getting|that|it|which) paid|paid (?:off|out|me|us|well|nicely|big)|bagholding|bag holding|wish i|should have|would have|could have|if i had|"
    r"started (?:taking off|running|ripping)|been having|having fun|rip to|rest in peace|reali[sz]ed|"
    r"(?:calls?|puts?|shares?|position|it)\s+(?:are\s+|is\s+)?(?:up|green|printing)|\d+%\s+(?:today|tomorrow|yesterday)|"
    r"out of (?:my |the |all )?(?:calls?|puts?|positions?|shares?)|if you got in|got in when|who got in|"
    r"bought (?:in|it in|this in) (?:19|20)\d\d|gave (?:the|you|yall|y'all) (?:the )?(?:blueprint|play|alert)|signal(?:ed|led))\b|[+]\$?\d[\d,]*(?:\.\d+)?%?", re.I)
# selling/sold = closing a position (an exit), except selling puts / CSPs or a market "sell-off"
SELL_EXIT = re.compile(r"\b(sold|selling|sell)\b(?![\s-]*(?:off|pressure|signal|to open))", re.I)
# third-party or passive buying is not the author's entry
NOT_AUTHOR_ENTRY = re.compile(r"\b(who|those|people|they|he|she|whales?|someone|everyone|buyers|bulls|bears|funds?|insiders?|"
                              r"institutions?|analysts?|congress|pelosi|cathie|ark|citadel)\b\W+(?:\w+\W+){0,2}?"
                              r"(?:bought|buying|entered|loaded|added|grabbed)\b|\b(?:be|being|been|getting|is|was|were|are)\s+bought\b|"
                              r"\bbuying\s+(?:levels?|zones?|opportunit\w+|areas?|range|pressure|power)\b", re.I)
NOT_OPTION_CALL = re.compile(r"\b(?:margin|earnings|conference|wake[- ]up|close|judge?ment|cold|phone|zoom|good|bad|bold)\s+call\b|"
                             r"\bcall\s+(?:it|out|for|me|him|her|them|this|that|bs|on)\b", re.I)
QUESTION = re.compile(r"\?[\s\"'\u201d\u2019)]*$|[🤔🤷]")
THIRD_PARTY = re.compile(r"\b(congratulations|congrats|those who|the one guy|some ?one|some ?body|whales?|sweeps?|unusual|flow|"
                         r"open interest|oi|(?:call|put)s? (?:buy|sell)|guys? (?:is|are|was|fading|buying|selling)|chase|"
                         r"(?:to )?all (?:that|who|those)|announces?)\b|\$\d+(?:\.\d+)?\s?[mk]\b", re.I)
NEG_ENTRY = re.compile(r"\b(not|n't|never|no|didn'?t|won'?t)\b", re.I)
# someone else's call (analyst actions) and descriptions of moves that already happened
ANALYST = re.compile(r"\b(upgrade[sd]?|downgrade[sd]?|price targets?|pt (?:raised|cut|lowered|of)|(?:buy|sell|hold|outperform|underperform|neutral) rating|"
                     r"reiterat\w+|initiat\w+ coverage|analysts?|raises|lowers|cuts)\b", re.I)
DESCRIPTIVE_MOVE = re.compile(r"\b(?:is|are|was|were|been|closed|trading|traded|opened|finished|ended)\s+(?:up|down|higher|lower)\s+"
                              r"(?:about\s+|over\s+|nearly\s+|almost\s+|another\s+)?\$?\d", re.I)
# opinion clauses taken from a post BODY must say something about the future
FORWARD = re.compile(r"\b(will|gonna|going to|expect\w*|should|could|might|look(?:ing)? (?:for|to)|watch(?:ing)? for|target\w*|next|"
                     r"heading|head(?:ed)? (?:to|towards)|towards?|set ?up|imo|i think|i believe|likely|soon|incoming|about to|wants? to|possible|"
                     r"ready to|primed|poised|in play|by (?:eow|eod|eoy|friday|next)|next week|tomorrow|long term|accumulat\w+|bullish|bearish|"
                     r"buy (?:the|here|now|zone)|load\w*|entry|retest|continue\w*|undervalued|overvalued|cheap|expensive|discount\w*)\b", re.I)
QUOTED_TWEETS = re.compile(r"(?:x|twitter)\.com/|t\.co/", re.I)
NEGATION = re.compile(r"\b(not|n't|never|no|don'?t|isn'?t|aren'?t|wouldn'?t|won'?t)\b", re.I)

BULL = re.compile(r"\b(bullish|undervalued|cheap(?:est)?|discount(?:ed)?|accumulat\w*|breakout|break(?:ing)? out|"
                  r"moon(?:ing)?|ripping|squeeze|pop|bounce|rebound|upside|higher|load (?:up|the boat)|buy the dip|"
                  r"buy(?:ing)? opportunity|send it|long term hold|going up|uptrend|reversal up)\b|🚀", re.I)
BEAR = re.compile(r"\b(bearish|overvalued|expensive|dump(?:ing)?|crash(?:ing)?|downside|lower|drop(?:ping)?|"
                  r"fall(?:ing)?|sell[- ]off|top(?:ped)? out|red flag|breakdown|break(?:ing)? down|fade|downtrend|"
                  r"slippery slope|rug(?:ged)?|bubble)\b|📉", re.I)
TARGET = re.compile(r"\b(?:to|target(?:ing)?|pt|tp|price target)\s*:?\s*\$\s?(\d{1,5}(?:\.\d{1,2})?)", re.I)
YEAR_DEADLINE = re.compile(r"\b(?:in|by|before|end of|eoy)\s*(20\d\d)\b", re.I)

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_name) if m})
EXP_MD = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
EXP_MON_DAY = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
EXP_MON_YEAR = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(20\d\d)\b", re.I)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
EXP_REL = re.compile(r"\b(0\s?dte|today|tomorrow|this (?:week|friday)|eow|end of (?:the )?week|next (monday|tuesday|wednesday|thursday|friday|week))\b", re.I)


# ---------------------------------------------------------------- normalization
def normalize_post(username: str, item: dict, detail: dict | None) -> dict:
    post = item.get("post") or {}
    body = (detail or {}).get("body") or ""
    return {
        "username": username,
        "post_id": post.get("id") or item.get("id"),
        "hash": post_hash(post.get("shareUrl", "")),
        "share_url": post.get("shareUrl", ""),
        "posted_at_utc": (post.get("createdAt") or item.get("createdAt")),
        "title": post.get("title") or "",
        "body": body,
        "has_detail": detail is not None and "_error_status" not in detail,
        "tickers": [s.get("tickerSymbol") for s in (item.get("securities") or []) if s and s.get("tickerSymbol")],
        "topic": item.get("primaryTopicKey"),
        "media_count": len(item.get("mediaAttachments") or []),
        "comment_count": item.get("commentCount", 0),
        "reaction_count": sum((item.get("reactionCounts") or {}).values()),
        "view_count": item.get("viewCount", 0),
    }


def load_posts(raw: Path) -> list[dict]:
    posts = []
    for f in sorted((raw / "users").glob("*.jsonl")):
        user = f.stem
        for item in read_jsonl(f):
            h = post_hash((item.get("post") or {}).get("shareUrl", ""))
            d = raw / "details" / user / f"{h}.json"
            detail = json.loads(d.read_text(encoding="utf-8")) if h and d.exists() else None
            posts.append(normalize_post(user, item, detail))
    return posts


def wants_detail(p: dict) -> bool:
    return bool(p["tickers"]) or bool(CASHTAG.search(p["title"]))


# ---------------------------------------------------------------- expiry parsing
def _year_for(month: int, day: int, posted: date) -> date | None:
    for y in (posted.year, posted.year + 1):
        try:
            d = date(y, month, day)
        except ValueError:
            return None
        if d >= posted - timedelta(days=1):
            return d
    return None


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def parse_expiry(text: str, posted: date) -> str | None:
    m = EXP_MD.search(text)
    if m:
        mo, dy = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= dy <= 31:
            if m.group(3):
                y = int(m.group(3))
                y = y + 2000 if y < 100 else y
                try:
                    return date(y, mo, dy).isoformat()
                except ValueError:
                    return None
            d = _year_for(mo, dy, posted)
            return d.isoformat() if d else None
    m = EXP_MON_YEAR.search(text)
    if m:
        return third_friday(int(m.group(2)), MONTHS[m.group(1).lower()[:3]]).isoformat()
    m = EXP_MON_DAY.search(text)
    if m:
        mo, dd = MONTHS[m.group(1).lower()[:3]], int(m.group(2))
        d = _year_for(mo, dd, posted)
        # "Jan 25 call" posted in 2024 = the Jan 2025 monthly when "Jan 25th" would be a weekend
        if (d is None or d.weekday() >= 5) and posted.year % 100 <= dd <= posted.year % 100 + 2:
            return third_friday(2000 + dd, mo).isoformat()
        return d.isoformat() if d else None
    m = EXP_REL.search(text)
    if m:
        s = m.group(1).lower()
        if s.startswith("0") or s == "today":
            return posted.isoformat()
        if s == "tomorrow":
            return (posted + timedelta(days=3 if posted.weekday() == 4 else 1)).isoformat()
        this_friday = posted + timedelta(days=(4 - posted.weekday()) % 7)
        if s.startswith("next "):
            wd = m.group(2).lower()
            if wd == "week":
                return (this_friday + timedelta(days=7)).isoformat()
            return (posted + timedelta(days=(WEEKDAYS.index(wd) - posted.weekday()) % 7 or 7)).isoformat()
        return this_friday.isoformat()  # this week / this Friday / EOW
    return None


# ---------------------------------------------------------------- extraction
def clause_tickers(clause: str, tagged: set[str]) -> list[str]:
    found = []
    for m in CASHTAG.finditer(clause):
        t = m.group(1).upper()
        if t in tagged or not tagged:
            found.append(t)
    for t in tagged:
        if t not in found and t not in BARE_STOP and len(t) >= 2 and re.search(rf"(?<![\w$]){re.escape(t)}(?!\w)", clause):
            found.append(t)
    low = clause.lower()
    for name, t in NAMES.items():
        if t in tagged and t not in found and re.search(rf"(?<!not )\b{name}\b", low):
            found.append(t)
    return found


def find_contract(clause: str) -> tuple[str, float | None, int] | None:
    """(right, strike, position) of an option mention, or None."""
    m = ATTACHED.search(clause)
    if m:
        return ("put" if m.group(2).lower() == "p" else "call"), float(m.group(1)), m.start()
    side = next((m for m in SIDE_WORD.finditer(clause)
                 if not (m.group(1).lower().startswith("put") and PUT_VERB.search(clause[max(0, m.start() - 8):m.end() + 12]))), None)
    if not side:
        return None
    right = "put" if side.group(1).lower().startswith("p") else "call"
    best = None
    for d in DOLLAR_NUM.finditer(clause):
        if clause[:d.start()].rstrip().endswith("@"):  # "@ $2.39" is the premium, not the strike
            continue
        dist = abs(d.start() - side.start())
        if dist <= 40 and (best is None or dist < best[0]):
            best = (dist, float(d.group(1)), d.start())
    if best:
        return right, best[1], best[2]
    b = BARE_BEFORE_SIDE.search(clause)
    qty = b and re.search(r"\b(?:have|had|own|bought|buy|sold|got|holding|with|of|x)\s*$", clause[:b.start()], re.I)
    if b and not MONTH_BEFORE.search(clause[:b.start()]) and not qty and not re.match(r"\s*call options", clause[b.end(1):]):
        return right, float(b.group(1)), b.start()
    return right, None, side.start()


def ticker_position(clause: str, t: str) -> int:
    m = re.search(rf"\${re.escape(t)}\b|(?<![\w$]){re.escape(t)}(?!\w)", clause, re.I)
    if m:
        return m.start()
    for name, sym in NAMES.items():
        if sym == t and (m := re.search(rf"\b{name}\b", clause, re.I)):
            return m.start()
    return -1


def mask_tickers(clause: str, tickers: list[str]) -> str:
    """Hide symbols so a ticker like BULL or DOWN cannot trigger the lexicon."""
    out = CASHTAG.sub(" TKR ", clause)
    for t in tickers:
        out = re.sub(rf"(?<![\w$]){re.escape(t)}(?!\w)", " TKR ", out)
    return out


def classify_clause(clause: str) -> dict | None:
    """Direction/tier for one (ticker-masked) clause, or None when it is not a forward call."""
    sold_puts = SOLD_PUTS.search(clause)
    if QUESTION.search(clause) or HINDSIGHT.search(clause) or (SELL_EXIT.search(clause) and not sold_puts):
        return None
    if ANALYST.search(clause) or DESCRIPTIVE_MOVE.search(clause) or THIRD_PARTY.search(clause):
        return None
    if COVERED.search(clause) and not sold_puts:
        return None
    contract = find_contract(clause)
    if contract and contract[0] == "call" and contract[1] is None and NOT_OPTION_CALL.search(clause):
        contract = None
    side = SIDE_ONLY.search(clause)  # leaps / CSP wording without call/put
    if sold_puts:
        k = CSP_STRIKE.search(clause)
        return {"tier": "A", "direction": "long", "instrument": "short_put", "rule": "sold_puts",
                "strike": float(k.group(1)) if k else None}
    entry = ENTRY.search(clause) if not NOT_AUTHOR_ENTRY.search(clause) else None
    if entry and NEG_ENTRY.search(clause):
        return None  # "I am not buying $HOOD"
    if contract:
        right, strike, pos = contract
        kind = "contract" if strike is not None else "side"
        return {"tier": "A" if entry else "B", "direction": "long" if right == "call" else "short",
                "instrument": right, "strike": strike, "contract_pos": pos,
                "rule": f"{kind}_entry" if entry else kind}
    if side and side.group(1).lower().startswith("leap"):
        return {"tier": "A" if entry else "B", "direction": "long", "instrument": "call",
                "rule": "side_entry" if entry else "side"}
    short = SHORT_CALL.search(clause)
    if entry:
        return {"tier": "A", "direction": "short" if short else "long", "instrument": "stock", "rule": "stock_entry"}
    if short:
        terse = len(clause.split()) <= 4  # trade-log style "$RGTI short"
        return {"tier": "A" if terse else "C", "direction": "short", "instrument": "stock",
                "rule": "short_log" if terse else "opinion_short"}
    bull, bear = BULL.search(clause), BEAR.search(clause)
    if bull and bear:
        return None
    if (bull or bear) and NEGATION.search(clause):
        return None
    if (bull or bear) and not FORWARD.search(clause):
        return None  # bare celebration / description, e.g. "$CRBG rockets Sheesh"
    if bull or bear:
        return {"tier": "C", "direction": "long" if bull else "short", "instrument": "stock",
                "rule": "opinion_bull" if bull else "opinion_bear"}
    return None  # a bare price target without direction words is often a quoted analyst PT: skipped


def extract_post(p: dict) -> list[dict]:
    if (p.get("topic") or "").lower() in ("gain", "loss"):
        return []
    posted_utc = datetime.fromisoformat(p["posted_at_utc"].replace("Z", "+00:00"))
    posted = posted_utc.astimezone(ET).date()
    tagged = {t.upper() for t in p["tickers"] if t}
    quoted = bool(QUOTED_TWEETS.search(p["body"]))
    segments = [("title", c) for c in CLAUSE_SPLIT.split(p["title"])] + [("body", c) for c in CLAUSE_SPLIT.split(p["body"])]
    calls, seen = [], set()
    for source, clause in ((src, c.strip()) for src, c in segments if c):
        if len(clause) < 3:
            continue
        tickers = clause_tickers(clause, tagged)
        inherited = False
        if not tickers and source == "body" and len(tagged) == 1 and not re.search(r"\$[A-Za-z]", clause):
            tickers, inherited = sorted(tagged), True  # single-ticker post: the body is about that ticker
        if not tickers:
            continue
        masked = mask_tickers(clause, tickers)
        cls = classify_clause(masked)
        if not cls:
            continue
        if cls["tier"] == "A" and len(tickers) > 1 and (verb := ENTRY.search(masked)):
            # "...sweating over $NVDA, I was loading the boat on $BOXX": the entry is the ticker after the verb
            vpos = clause.lower().find(verb.group(0).lower())
            pos = {t: ticker_position(clause, t) for t in tickers}
            after = [t for t in tickers if pos[t] >= vpos]
            before = [t for t in tickers if 0 <= pos[t] < vpos]
            tickers = after or ([max(before, key=pos.get)] if before else tickers)
        if cls.get("contract_pos") is not None and len(tickers) > 1:
            # one option mention belongs to one ticker: the nearest one before it, else the nearest after
            pos = {t: ticker_position(clause, t) for t in tickers}
            before = [t for t in tickers if 0 <= pos[t] <= cls["contract_pos"]]
            after = [t for t in tickers if pos[t] > cls["contract_pos"]]
            tickers = [max(before, key=pos.get)] if before else ([min(after, key=pos.get)] if after else tickers[:1])
        target = TARGET.search(clause)
        year = YEAR_DEADLINE.search(clause)
        expiry = parse_expiry(clause, posted) if cls["instrument"] in ("call", "put", "short_put") else None
        for t in tickers[:3]:  # a clause naming >3 tickers is a list, not a call; keep the first three
            key = (t, cls["direction"], cls["tier"])
            if key in seen:
                continue
            seen.add(key)
            calls.append({
                "call_id": f"{p['post_id']}:{t}:{len(calls)}", "username": p["username"], "post_id": p["post_id"],
                "share_url": p["share_url"], "posted_at_utc": p["posted_at_utc"], "symbol": t,
                "direction": cls["direction"], "tier": cls["tier"], "rule": cls["rule"],
                "instrument": cls["instrument"], "strike": cls.get("strike"), "expiry": expiry,
                "target": float(target.group(1)) if target else None,
                "deadline": f"{year.group(1)}-12-31" if (target and year) else None,
                "clause": clause[:300], "source": source, "post_quotes_tweets": quoted, "ticker_inherited": inherited,
                "n_tickers_in_clause": len(tickers), "has_detail": p["has_detail"],
            })
    return resolve_conflicts(calls)


def resolve_conflicts(calls: list[dict]) -> list[dict]:
    """Per post and ticker keep only the strongest tier; if it still holds both directions, drop the ticker."""
    out = []
    for t in dict.fromkeys(c["symbol"] for c in calls):
        mine = [c for c in calls if c["symbol"] == t]
        best = min(c["tier"] for c in mine)
        top = [c for c in mine if c["tier"] == best]
        if len({c["direction"] for c in top}) == 1:
            out += top
    return out


def dedupe(calls: list[dict]) -> list[dict]:
    """One call per user/symbol/direction/ET day: the earliest, at its strongest tier."""
    calls = sorted(calls, key=lambda c: (c["posted_at_utc"], c["tier"]))
    out, seen = [], set()
    for c in calls:
        day = datetime.fromisoformat(c["posted_at_utc"].replace("Z", "+00:00")).astimezone(ET).date().isoformat()
        key = (c["username"], c["symbol"], c["direction"], day)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--since", default="2000-01-01")
    e = sub.add_parser("extract")
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    posts = load_posts(args.raw)
    if args.cmd == "select":
        rows = [{"username": p["username"], "hash": p["hash"]} for p in posts
                if wants_detail(p) and p["hash"] and p["posted_at_utc"][:10] >= args.since]
        write_jsonl(args.out, rows)
        print(json.dumps({"posts": len(posts), "selected": len(rows), "by_user": Counter(r["username"] for r in rows)}))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "posts.jsonl", posts)
    raw_calls = [c for p in posts for c in extract_post(p)]
    calls = dedupe(raw_calls)
    write_jsonl(args.out / "calls.jsonl", calls)
    (args.out / "extract_manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "raw": str(args.raw), "posts": len(posts),
        "posts_with_detail": sum(p["has_detail"] for p in posts), "raw_calls": len(raw_calls), "deduped_calls": len(calls),
        "by_tier": Counter(c["tier"] for c in calls), "by_rule": Counter(c["rule"] for c in calls),
        "by_user": Counter(c["username"] for c in calls)}, indent=2) + "\n")
    print(json.dumps({"posts": len(posts), "calls": len(calls), "by_tier": Counter(c["tier"] for c in calls)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
