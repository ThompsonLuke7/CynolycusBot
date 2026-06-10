"""Earnings-result and guidance enrichment for catalyst records."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from signals.events.forward_guidance.features.nlp import extract_forward_sections, extract_structured_guidance_features, normalize_text


EARNINGS_TERMS = (
    r"\bearnings\b",
    r"\bquarterly results?\b",
    r"\bfinancial results?\b",
    r"\brevenue\b",
    r"\bsales\b",
    r"\beps\b",
    r"\bearnings per share\b",
    r"\bguidance\b",
    r"\boutlook\b",
)

BEAT_TERMS = (
    r"\bbeats?\b",
    r"\bbeat estimates?\b",
    r"\btops? estimates?\b",
    r"\babove estimates?\b",
    r"\bexceeds? estimates?\b",
    r"\bbetter than expected\b",
    r"\brecord revenue\b",
)

MISS_TERMS = (
    r"\bmiss(?:es|ed)?\b",
    r"\bmiss estimates?\b",
    r"\bbelow estimates?\b",
    r"\bshort of estimates?\b",
    r"\bworse than expected\b",
    r"\bweak(?:er)? than expected\b",
)

GUIDANCE_RAISE_TERMS = (
    r"\braises? (?:full.year )?guidance\b",
    r"\blifts? (?:full.year )?guidance\b",
    r"\bincreases? (?:full.year )?outlook\b",
    r"\babove consensus\b",
)

GUIDANCE_CUT_TERMS = (
    r"\bcuts? (?:full.year )?guidance\b",
    r"\blowers? (?:full.year )?guidance\b",
    r"\breduces? (?:full.year )?outlook\b",
    r"\bwithdraws? guidance\b",
    r"\bsuspends? guidance\b",
)

SEC_EARNINGS_FORMS = {"10-Q", "10-K"}


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _count_any(text: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


def _sec_form(source: object) -> str:
    raw = str(source or "").lower().replace("sec_", "").replace("_", "-")
    return raw.upper()


def _score_beat_miss(text: str) -> float:
    positive = _count_any(text, BEAT_TERMS)
    negative = _count_any(text, MISS_TERMS)
    if positive == 0 and negative == 0:
        return 0.0
    return float(np.clip((positive - negative) / max(positive + negative, 1), -1.0, 1.0))


def _score_explicit_guidance(text: str) -> float:
    positive = _count_any(text, GUIDANCE_RAISE_TERMS)
    negative = _count_any(text, GUIDANCE_CUT_TERMS)
    if positive == 0 and negative == 0:
        return 0.0
    return float(np.clip((positive - negative) / max(positive + negative, 1), -1.0, 1.0))


def enrich_earnings_catalyst_fields(news: pd.DataFrame) -> pd.DataFrame:
    """Add earnings-result/guidance fields without requiring paid estimate data.

    Beat/miss versus analyst consensus is only available when the text says it
    explicitly. SEC company filings give reported facts, not Street consensus,
    so this layer separates language-derived beat/miss from filing-derived
    earnings relevance.
    """
    out = news.copy()
    columns = {
        "is_earnings_catalyst": 0.0,
        "earnings_catalyst_type": "",
        "earnings_forward_guidance_text": "",
        "earnings_embedding_text": "",
        "earnings_relevance_score": 0.0,
        "earnings_beat_miss_score": 0.0,
        "earnings_guidance_score": 0.0,
        "earnings_language_score": 0.0,
        "earnings_revenue_language_score": 0.0,
        "earnings_eps_language_score": 0.0,
    }
    if out.empty:
        for col, default in columns.items():
            out[col] = []
        return out

    values: dict[str, list[object]] = {col: [] for col in columns}
    impact_roles = []
    has_impact_role = "impact_role" in out.columns

    for row in out.itertuples(index=False):
        source = str(getattr(row, "source", "") or "")
        form = _sec_form(source)
        text = " ".join(
            str(getattr(row, field, "") or "")
            for field in ("headline", "summary", "body", "text")
        )
        earnings_text = _contains_any(text, EARNINGS_TERMS)
        sec_report = source.lower().startswith("sec") and form in SEC_EARNINGS_FORMS
        is_earnings = bool(earnings_text or sec_report)

        sections = extract_forward_sections(text if is_earnings else "")
        guidance_text = sections.get("forward_guidance", "")
        if not guidance_text and is_earnings:
            guidance_text = text
        guidance_features = extract_structured_guidance_features(guidance_text if is_earnings else "")
        structured_guidance = float(guidance_features.get("guidance_strength_score", 0.0) or 0.0)
        explicit_guidance = _score_explicit_guidance(guidance_text or text)
        guidance_score = float(np.clip(structured_guidance + explicit_guidance, -1.0, 1.0))
        beat_miss_score = _score_beat_miss(text)
        language_score = float(np.clip((0.55 * beat_miss_score) + (0.45 * guidance_score), -1.0, 1.0))

        if not is_earnings:
            catalyst_type = ""
            relevance = 0.0
        elif guidance_score >= 0.25:
            catalyst_type = "earnings_guidance_raise"
            relevance = 1.0
        elif guidance_score <= -0.25:
            catalyst_type = "earnings_guidance_cut"
            relevance = 1.0
        elif beat_miss_score >= 0.25:
            catalyst_type = "earnings_beat"
            relevance = 0.95
        elif beat_miss_score <= -0.25:
            catalyst_type = "earnings_miss"
            relevance = 0.95
        elif sec_report:
            catalyst_type = "earnings_report_filing"
            relevance = 0.8
        else:
            catalyst_type = "earnings_result"
            relevance = 0.7

        values["is_earnings_catalyst"].append(float(is_earnings))
        values["earnings_catalyst_type"].append(catalyst_type)
        values["earnings_forward_guidance_text"].append(normalize_text(guidance_text)[:12000] if is_earnings else "")
        if is_earnings and guidance_text.strip():
            values["earnings_embedding_text"].append(normalize_text(" ".join([str(getattr(row, "headline", "") or ""), str(getattr(row, "summary", "") or ""), guidance_text]))[:16000])
        else:
            values["earnings_embedding_text"].append("")
        values["earnings_relevance_score"].append(float(relevance))
        values["earnings_beat_miss_score"].append(float(beat_miss_score if is_earnings else 0.0))
        values["earnings_guidance_score"].append(float(guidance_score if is_earnings else 0.0))
        values["earnings_language_score"].append(float(language_score if is_earnings else 0.0))
        values["earnings_revenue_language_score"].append(float(guidance_features.get("guidance_revenue_raise_cut", 0.0) or 0.0))
        values["earnings_eps_language_score"].append(float(guidance_features.get("guidance_eps_raise_cut", 0.0) or 0.0))

        original_role = str(getattr(row, "impact_role", "") or "")
        impact_roles.append(catalyst_type or original_role)

    for col, col_values in values.items():
        out[col] = col_values
    if has_impact_role:
        out["impact_role"] = impact_roles
    return out
