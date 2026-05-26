"""Broad catalyst family and subtype classification."""

from __future__ import annotations

import re

import pandas as pd


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _source_form(source: object) -> str:
    return str(source or "").lower().replace("sec_", "").replace("_", " ").upper()


def classify_catalyst_types(news: pd.DataFrame) -> pd.DataFrame:
    """Assign a broad family before embedding-cluster analysis."""
    out = news.copy()
    if out.empty:
        out["catalyst_family"] = []
        out["catalyst_subtype"] = []
        return out

    families: list[str] = []
    subtypes: list[str] = []
    for row in out.itertuples(index=False):
        source = str(getattr(row, "source", "") or "")
        form = _source_form(source)
        earnings_type = str(getattr(row, "earnings_catalyst_type", "") or "")
        text = " ".join(
            str(getattr(row, field, "") or "")
            for field in ("headline", "summary", "body", "text", "earnings_forward_guidance_text")
        )

        if earnings_type.startswith("earnings_guidance") or str(getattr(row, "earnings_forward_guidance_text", "") or "").strip():
            family = "earnings_guidance"
            subtype = earnings_type or "earnings_guidance_language"
        elif earnings_type:
            family = "earnings_result"
            subtype = earnings_type
        elif source.startswith("sec") and form in {"424B3", "424B4", "424B5", "S-1", "S-3", "F-1"}:
            family = "financing_dilution"
            subtype = form.lower().replace(" ", "_")
        elif source.startswith("sec") and form in {"SC TO-I", "SC TO-T", "DEFM14A", "PREM14A", "S-4"}:
            family = "ma_proxy_tender"
            subtype = form.lower().replace(" ", "_")
        elif source.startswith("sec") and "SC 13D" in form:
            family = "activist_ownership"
            subtype = form.lower().replace(" ", "_").replace("/", "_")
        elif _has(text, r"\bfda\b", r"\bpdufa\b", r"\bclinical\b", r"\bphase [123]\b", r"\btrial\b", r"\bapproval\b"):
            family = "biotech_fda"
            subtype = "approval_trial_update"
        elif _has(text, r"\bupgrade\b", r"\bdowngrade\b", r"\bprice target\b", r"\banalyst\b", r"\binitiates?\b"):
            family = "analyst_action"
            subtype = "analyst_rating_target"
        elif _has(text, r"\bcontract\b", r"\baward\b", r"\bpartnership\b", r"\bcollaboration\b", r"\bcommercial momentum\b"):
            family = "contract_partnership"
            subtype = "commercial_deal"
        elif _has(text, r"\boffering\b", r"\bdilut", r"\bshelf\b", r"\bregistered direct\b"):
            family = "financing_dilution"
            subtype = "offering_language"
        elif _has(text, r"\btariff\b", r"\bpolicy\b", r"\boil\b", r"\bcrypto\b", r"\bbitcoin\b", r"\bsector\b", r"\bindustry\b"):
            family = "macro_theme"
            subtype = "theme_macro"
        elif source.startswith("sec"):
            family = "sec_other"
            subtype = form.lower().replace(" ", "_")
        else:
            family = "company_news"
            subtype = "general_company_news"

        families.append(family)
        subtypes.append(subtype)

    out["catalyst_family"] = families
    out["catalyst_subtype"] = subtypes
    return out
