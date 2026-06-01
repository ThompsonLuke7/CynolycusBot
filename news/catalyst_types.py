"""Broad catalyst family and subtype classification.

Two passes:

1. :func:`classify_catalyst_types` assigns a coarse family/subtype via regex
   + SEC form code. Used to seed clustering.
2. :func:`refine_catalyst_types_from_clusters` (called after KMeans clustering
   in the pipeline) re-derives family/subtype from each cluster's modal
   keywords so that records like an "Earnings 8/8" options-flow alert that
   the regex bucketed as ``earnings_guidance`` can be re-bucketed as
   ``options_flow`` if the cluster centroid points that way.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd


# Multi-label keyword tags used by the cluster-driven refinement pass. A
# cluster is reclassified into a tag if more than ``cluster_label_threshold``
# of its members hit that tag's regex set. Order matters — earlier tags win
# ties so that more specific labels (options_flow, biotech_fda) beat the
# generic "earnings_guidance" / "sec_other" buckets.
CLUSTER_KEYWORD_TAGS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "options_flow",
        "options_flow_alert",
        (
            r"\boption(?:s)? alert\b",
            r"\bsweep(?:s)?\b",
            r"\bunusual options\b",
            r"\b(?:call|put)s? (?:at|on) the (?:ask|bid)\b",
            r"\bopen interest\b",
            r"\bvs \d+ ?oi\b",
        ),
    ),
    (
        "biotech_fda",
        "approval_trial_update",
        (
            r"\bfda\b",
            r"\bpdufa\b",
            r"\bclinical\b",
            r"\bphase [123]\b",
            r"\btrial\b",
            r"\bapproval\b",
            r"\bemergency use authorization\b",
        ),
    ),
    (
        "contract_partnership",
        "commercial_deal",
        (
            r"\bcontract\b",
            r"\baward\b",
            r"\bpartnership\b",
            r"\bcollaboration\b",
            r"\bjoint venture\b",
            r"\bsupply agreement\b",
            r"\bcommercial momentum\b",
            r"\bdesign win\b",
        ),
    ),
    (
        "analyst_action",
        "analyst_rating_target",
        (
            r"\bupgrade\b",
            r"\bdowngrade\b",
            r"\bprice target\b",
            r"\bbuy rating\b",
            r"\bsell rating\b",
            r"\binitiates?\b",
            r"\breiterates?\b",
        ),
    ),
    (
        "ma_proxy_tender",
        "acquisition_announcement",
        (
            r"\bacquir(?:e|es|ed|ing|ition)\b",
            r"\bmerger\b",
            r"\btender offer\b",
            r"\bto be acquired\b",
            r"\bdefinitive agreement\b",
            r"\btake[- ]?private\b",
        ),
    ),
    (
        "earnings_guidance",
        "earnings_result",
        (
            r"\bearnings (?:results?|report|release)\b",
            r"\bquarterly results\b",
            r"\beps of\b",
            r"\brevenue of\b",
            r"\braises? (?:guidance|outlook|forecast)\b",
            r"\blowers? (?:guidance|outlook|forecast)\b",
            r"\bbeats? estimates\b",
            r"\bmisses? estimates\b",
        ),
    ),
    (
        "financing_dilution",
        "offering_language",
        (
            r"\boffering\b",
            r"\bdilut",
            r"\bshelf\b",
            r"\bregistered direct\b",
            r"\bat[- ]the[- ]market\b",
            r"\bsecondary offering\b",
        ),
    ),
    (
        "macro_theme",
        "theme_macro",
        (
            r"\btariff\b",
            r"\bfederal reserve\b",
            r"\binflation\b",
            r"\bsector rotation\b",
            r"\bcrypto\b",
            r"\bbitcoin\b",
        ),
    ),
)


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
        elif source.startswith("sec") and form in {"4", "4/A"}:
            # Default Form 4 to insider_buying; the catalyst-types refine pass
            # downgrades to insider_selling if the body text reveals a sale.
            family = "insider_activity"
            subtype = "insider_buying"
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


def _row_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(field, "") or "")
        for field in ("headline", "summary", "body", "text", "earnings_forward_guidance_text")
    )


# Subtypes that are too distinctive to lose to a cluster-majority vote.
# After cluster-driven refinement, any record whose text individually matches
# one of these patterns gets overridden, even if its cluster as a whole was
# dominated by something else. This catches sparse-but-distinctive catalysts
# (e.g. ~35 options-flow alerts in a 94K corpus) that can't form their own
# cluster under KMeans.
RECORD_LEVEL_OVERRIDES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "options_flow",
        "options_flow_alert",
        (
            r"\boption(?:s)? alert\b",
            r"\bunusual options\b",
            r"\b(?:call|put)s? (?:at|on) the (?:ask|bid)\b",
            r"\bvs \d+ ?oi\b",
            r"\bsweep(?:s)? (?:on|at|hit)\b",
            r"\bbullish (?:call|put) flow\b",
            r"\bbearish (?:call|put) flow\b",
        ),
    ),
)


def refine_catalyst_types_from_clusters(
    news: pd.DataFrame,
    *,
    cluster_label_threshold: float = 0.35,
    min_cluster_size: int = 4,
) -> pd.DataFrame:
    """Re-derive ``catalyst_family``/``catalyst_subtype`` from cluster modal tags,
    then apply record-level regex overrides for distinctive sparse catalysts.

    Two passes:
    1. For each cluster, count how many member records hit each
       ``CLUSTER_KEYWORD_TAGS`` tag. If any tag hits a fraction >=
       ``cluster_label_threshold``, the whole cluster takes that tag's
       family/subtype (earliest-wins on ties). Clusters smaller than
       ``min_cluster_size`` keep their regex-assigned label.
    2. For each ``RECORD_LEVEL_OVERRIDES`` entry, scan all records and reassign
       any whose text matches the override patterns. This ensures rare-but-
       distinctive catalysts (options-flow alerts, etc.) aren't drowned by
       the majority of their cluster.
    """
    if news.empty or "news_cluster_id" not in news.columns:
        return news
    out = news.copy()
    # Pass 1: cluster-majority voting
    for cluster_id, group in out.groupby("news_cluster_id", dropna=True):
        if len(group) < int(min_cluster_size):
            continue
        texts = group.apply(_row_text, axis=1).str.lower()
        n = len(texts)
        chosen_family: str | None = None
        chosen_subtype: str | None = None
        for family, subtype, patterns in CLUSTER_KEYWORD_TAGS:
            hits = sum(any(re.search(p, t) for p in patterns) for t in texts)
            if hits / max(n, 1) >= float(cluster_label_threshold):
                chosen_family = family
                chosen_subtype = subtype
                break
        if chosen_family is None:
            continue
        idx = group.index
        out.loc[idx, "catalyst_family"] = chosen_family
        out.loc[idx, "catalyst_subtype"] = chosen_subtype
    # Pass 2: record-level overrides for sparse-but-distinctive catalysts
    if RECORD_LEVEL_OVERRIDES:
        texts_all = out.apply(_row_text, axis=1).str.lower()
        for family, subtype, patterns in RECORD_LEVEL_OVERRIDES:
            hit_mask = texts_all.apply(lambda t: any(re.search(p, t) for p in patterns))
            if hit_mask.any():
                out.loc[hit_mask, "catalyst_family"] = family
                out.loc[hit_mask, "catalyst_subtype"] = subtype
    return out
