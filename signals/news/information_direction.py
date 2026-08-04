"""Label whether a headline carries NEW information or just restates a past move.

Raised 2026-08-03: a lot of live catalyst records only recap a move that already
printed ("Shares Skyrocket", "Stock Price Up 10.5%", "Gains 2.45% Today"). By
definition that information is already in the price, so it cannot be a catalyst.

Measured on the live ledger (18,605 records, 2026-06-16 -> 2026-08-03, collapsed
to 9,646 one-per-ticker-day observations, entry at the first daily close STRICTLY
after the record, so nothing is priced off the bar the news landed in):

    bucket             +1d       +3d       +5d    win(1d)      n
    price_recap     -0.034%   -0.638%   -1.418%    50.0%     604
    forward_looking +0.108%   -0.039%   +0.167%    53.0%   1,955
    other           +0.004%   -0.305%   -0.211%    49.7%   7,012
    ALL (baseline)  +0.020%   -0.267%   -0.203%    50.3%   9,646

price_recap is a coin flip at one day and underperforms the baseline by 1.2pp by
day five; forward_looking is the only bucket positive at five days, a 1.59pp
spread. So price recaps are not merely uninformative — they lean negative, which
is what post-move mean reversion looks like.

Caveats before anyone acts on this: one 7-week window and one regime, raw returns
with no beta or sector adjustment, and names still cluster across days even after
the ticker-day collapse. Reproduce with
research/news_catalysts/measure_backward_looking_catalysts.py.

This module only LABELS. Nothing is dropped here — excluding price recaps from
the catalyst aggregate is a signal-design decision that should be validated
downstream first.
"""
from __future__ import annotations

import re

PRICE_RECAP = re.compile(
    r"("
    r"shares?\s+(?:gap\s+)?(?:up|down|rise|risen|rose|fall|fell|drop|dropped|surge|surged|"
    r"jump|jumped|slump|slumped|skyrocket|skyrockets?|plunge|plunged|gain|gains|climb|climbs)|"
    r"stock\s+price\s+(?:up|down)|"
    r"(?:stock|shares?)\s+(?:gains?|drops?|falls?|rises?)\s+\d|"
    r"(?:up|down|gains?|gained|lost|loses?)\s+\d+(?:\.\d+)?%|"
    r"\d+(?:\.\d+)?%\s+(?:higher|lower|gain|drop|jump|decline|surge)|"
    r"moved?\s+(?:up|down)\s+by|"
    r"trading\s+(?:up|down)|"
    r"hits?\s+(?:new\s+)?(?:52[-\s]week|all[-\s]time)\s+(?:high|low)|"
    r"(?:top|worst|best)\s+(?:gainers?|losers?)|"
    r"why\s+(?:is|are|did).*(?:soaring|surging|plunging|falling|rising|sinking|tumbling)"
    r")",
    re.IGNORECASE,
)

FORWARD_LOOKING = re.compile(
    r"("
    r"upgrade[sd]?|downgrade[sd]?|initiate[sd]?\s+coverage|price\s+target|"
    r"guidance|outlook|forecast|"
    # "to buy" must be an actual deal ("agrees to buy X"), not the editorial
    # "Time to Buy?" that ends half of MarketBeat's price-recap headlines.
    r"acquisition|acquires?|merger|agree[sd]?\s+to\s+(?:buy|acquire)|takeover|stake\s+in|"
    r"fda\s+(?:approval|approves|clearance)|phase\s+[123]|trial\s+results|"
    r"contract|partnership|agreement|deal\s+with|awarded|"
    r"offering|dilution|buyback|repurchase|dividend|split|"
    r"appoints?|names?\s+ceo|resigns?|steps?\s+down|"
    r"launch(?:es|ed)?|unveils?|announces?"
    r")",
    re.IGNORECASE,
)

VALID = ("price_recap", "forward_looking", "mixed", "other")


def classify_information_direction(headline: str) -> str:
    """One of price_recap / forward_looking / mixed / other.

    "mixed" means the headline both recaps a move and reports new information
    ("Shares Gap Up on Analyst Upgrade") — kept separate rather than forced into
    one bucket, because collapsing it would hide which half is doing the work.
    """
    h = str(headline or "")
    recap = bool(PRICE_RECAP.search(h))
    fwd = bool(FORWARD_LOOKING.search(h))
    if recap and fwd:
        return "mixed"
    if recap:
        return "price_recap"
    if fwd:
        return "forward_looking"
    return "other"


def add_information_direction(df, *, headline_col: str = "headline",
                              out_col: str = "information_direction"):
    """Stamp the label onto a records frame in place; returns the frame."""
    if df is None or len(df) == 0 or headline_col not in df:
        return df
    df[out_col] = [classify_information_direction(h) for h in df[headline_col]]
    return df
