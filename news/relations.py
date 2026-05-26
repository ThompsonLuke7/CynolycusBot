"""Ticker relation classification for news catalysts."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from meta_context.config import CONTEXT_BACKTEST_UNIVERSE_PATH


THEME_PATTERNS = (
    r"\bshares of\b.*\bstocks?\b",
    r"\bstocks? (?:are )?trading\b",
    r"\bsector\b",
    r"\bindustry\b",
    r"\bETF\b",
    r"\bpeers?\b",
    r"\brelated stocks?\b",
)

NEGATIVE_PATTERNS = (
    r"\blower\b",
    r"\bdrop\b",
    r"\bdrops\b",
    r"\bfalls?\b",
    r"\btank(?:ed|s|ing)?\b",
    r"\bshort report\b",
    r"\boffering\b",
    r"\bdilut",
    r"\blawsuit\b",
    r"\bprobe\b",
    r"\binvestigation\b",
    r"\bwarning\b",
    r"\bmiss(?:es|ed)?\b",
    r"\bchapter 11\b",
    r"\bbankrupt",
)

POSITIVE_PATTERNS = (
    r"\bhigher\b",
    r"\bsoar(?:s|ed|ing)?\b",
    r"\bsurge(?:s|d|ing)?\b",
    r"\bbeats?\b",
    r"\bwins?\b",
    r"\bcontract\b",
    r"\bapproval\b",
    r"\bpartnership\b",
    r"\braises? guidance\b",
    r"\bupgrade\b",
    r"\boutperform\b",
)

SUPPLY_DEMAND_PATTERNS = (
    r"\bshortage\b",
    r"\bsupply\b",
    r"\bdemand\b",
    r"\btariff\b",
    r"\bpolicy\b",
    r"\bcommodity\b",
    r"\boil\b",
    r"\blithium\b",
    r"\brare earth",
    r"\bbitcoin\b",
    r"\bcrypto\b",
)


def _tokens(value: object) -> set[str]:
    return {token for token in re.findall(r"[A-Z0-9]{2,}", str(value or "").upper()) if len(token) >= 2}


def _company_keywords(value: object) -> set[str]:
    text = re.sub(r"\b(INC|CORP|CORPORATION|LTD|PLC|SA|ADR|CLASS|CL|COMMON|REIT|THE|AND|GROUP|HOLDINGS|HLDG|CO)\b", " ", str(value or "").upper())
    return {token for token in re.findall(r"[A-Z0-9]{4,}", text) if len(token) >= 4}


def load_company_keyword_map(universe_path: Path | str = CONTEXT_BACKTEST_UNIVERSE_PATH) -> dict[str, set[str]]:
    path = Path(universe_path)
    if not path.exists():
        return {}
    universe = pd.read_csv(path)
    out: dict[str, set[str]] = {}
    for row in universe.itertuples(index=False):
        ticker = str(getattr(row, "ticker", "")).upper().replace("$", "").strip()
        if not ticker:
            continue
        terms = set()
        for field in ("notes", "sector", "type"):
            terms.update(_company_keywords(getattr(row, field, "")))
        out[ticker] = terms
    return out


def classify_news_relations(
    news: pd.DataFrame,
    *,
    company_keywords: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Classify whether a record is directly about its ticker or broader context."""
    if news.empty:
        out = news.copy()
        for col in ("relation_type", "relation_confidence", "is_direct_catalyst", "impact_role"):
            if col not in out.columns:
                out[col] = []
        return out
    company_keywords = company_keywords or load_company_keyword_map()
    out = news.copy()
    relation_types = []
    confidences = []
    direct_flags = []
    impact_roles = []
    for row in out.itertuples(index=False):
        ticker = str(getattr(row, "ticker", "")).upper().replace("$", "").strip()
        source = str(getattr(row, "source", "")).lower()
        text = " ".join(
            str(getattr(row, field, "") or "")
            for field in ("headline", "summary", "body", "url")
        )
        upper_text = text.upper()
        text_tokens = _tokens(upper_text)
        theme_like = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in THEME_PATTERNS)
        negative_like = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in NEGATIVE_PATTERNS)
        positive_like = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in POSITIVE_PATTERNS)
        supply_demand_like = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SUPPLY_DEMAND_PATTERNS)
        direct = bool(ticker and ticker in text_tokens)
        keywords = company_keywords.get(ticker, set())
        direct_by_name = bool(keywords and len(keywords.intersection(text_tokens)) >= 1)

        if source.startswith("sec"):
            relation_type = "sec_filing_direct"
            confidence = 1.0
            is_direct = 1.0
        elif direct or direct_by_name:
            relation_type = "direct_mention"
            confidence = 0.9 if direct else 0.75
            is_direct = 1.0
        elif theme_like:
            relation_type = "theme_macro"
            confidence = 0.45
            is_direct = 0.0
        elif source == "finnhub":
            relation_type = "peer_sector"
            confidence = 0.35
            is_direct = 0.0
        else:
            relation_type = "ambiguous"
            confidence = 0.2
            is_direct = 0.0

        relation_types.append(relation_type)
        confidences.append(float(confidence))
        direct_flags.append(float(is_direct))
        if is_direct and negative_like and not positive_like:
            impact_role = "direct_victim"
        elif is_direct and positive_like and not negative_like:
            impact_role = "direct_beneficiary"
        elif is_direct:
            impact_role = "company_specific"
        elif theme_like and negative_like:
            impact_role = "theme_pressure"
        elif theme_like and positive_like:
            impact_role = "theme_beneficiary"
        elif supply_demand_like:
            impact_role = "macro_supply_demand"
        else:
            impact_role = "peer_sympathy" if relation_type == "peer_sector" else "unknown"
        impact_roles.append(impact_role)
    out["relation_type"] = relation_types
    out["relation_confidence"] = confidences
    out["is_direct_catalyst"] = direct_flags
    out["impact_role"] = impact_roles
    return out
