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


# Source quality tags. These DO NOT remove records — they tag each record with
# a category so downstream features can (a) weight high-alpha sources more in the
# catalyst classifier and (b) compute mention frequency from aggregators/listicles
# as a social-momentum / attention-peak signal in its own right.
SOURCE_QUALITY_AGGREGATORS = (
    "gurufocus",
    "simplywall.st",
    "stocktitan",
    "tipranks",
    "quiverquant",
    "kavout",
    "marketbeat",
    "zacks.com",
    "msn.com",
    "wallstreetzen",
    "stocknews",
    "chartmill",
    "trefis",
    "stockstotrade",
    "stockstory",
    "intellectia",
    "wallstcheatsheet",
)
SOURCE_QUALITY_OPINION_PATTERNS = (
    r"\bis\s+\(?[A-Z]{1,5}\)?\s+a\s+(?:buy|sell|hold)\b",
    r"\b[A-Z]{1,5}\s+stock(?:s)? to (?:buy|sell|watch)\b",
    r"\bhow\s+much\s+have\s+you\s+made\b",
    r"\bbest\s+(?:stocks?|dividend\s+stocks?)\s+for\b",
    r"\b[A-Z]{1,5}\s+to\s+\$\d+\??\b",  # "XOM To $120?"
    r"\bwhy\s+[a-z\s]+(?:could|might)\b",
    r"\b\(?[A-Z]{1,5}\)?\s+stock\s+price,?\s*quote\b",
    r"\bvaluation\s+check\b",
    r"\bdividend\s+aristocrats\s+in\s+focus\b",
    r"\bafter\s+(?:rapid|recent)\s+(?:multi[-\s]month\s+)?(?:share\s+price\s+)?surge\b",
)
SOURCE_QUALITY_BREAKING_PATTERNS = (
    r"\bbreaking\b",
    r"\b(?:announces|releases|reports|files)\b",
    r"\b(?:fda|pdufa)\s+(?:approval|grant|clearance)\b",
    r"\bcontract\s+(?:award|win|signing)\b",
    r"\bguidance\s+(?:raise|cut|update)\b",
    r"\bunusual\s+options\b",
    r"\bearnings\s+(?:beat|miss|result)\b",
    r"\bphase\s+[123]\s+(?:data|results|update)\b",
    r"\bmerger\s+(?:agreement|announcement)\b",
    r"\bdefinitive\s+agreement\b",
    r"\b8-?K\s+(?:filed|item)\b",
)


def classify_source_quality(news: pd.DataFrame) -> pd.DataFrame:
    """Tag each record with source_quality ∈ {breaking, high_alpha, opinion, aggregator}.

    NOT a filter — every record keeps its row, but the tag flows downstream as a
    feature (`source_quality`). Aggregator/opinion records contribute to the
    mention-frequency / social-buzz signal even when their text has no alpha.
    """
    if news.empty:
        return news
    out = news.copy()
    url_lower = out.get("url", pd.Series([""] * len(out))).fillna("").str.lower()
    headline_lower = out.get("headline", pd.Series([""] * len(out))).fillna("").str.lower()
    source_lower = out.get("source", pd.Series([""] * len(out))).fillna("").str.lower()

    is_agg = url_lower.apply(lambda u: any(a in u for a in SOURCE_QUALITY_AGGREGATORS)) | \
             headline_lower.str.contains(
                 "|".join([a.replace(".", r"\.") for a in SOURCE_QUALITY_AGGREGATORS]),
                 regex=True,
             )
    is_opinion = headline_lower.apply(
        lambda h: any(re.search(p, h, re.IGNORECASE) for p in SOURCE_QUALITY_OPINION_PATTERNS)
    )
    is_breaking = headline_lower.apply(
        lambda h: any(re.search(p, h, re.IGNORECASE) for p in SOURCE_QUALITY_BREAKING_PATTERNS)
    ) | source_lower.str.startswith("sec_") | source_lower.isin(
        {"fed_rss", "openfda", "clinicaltrials", "cboe_options_flow", "finra_short_spike"}
    )

    quality = pd.Series(["high_alpha"] * len(out), index=out.index)
    quality[is_agg] = "aggregator"
    quality[is_opinion & ~is_agg] = "opinion"
    quality[is_breaking] = "breaking"
    out["source_quality"] = quality
    return out


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

# Form 4 transactions default to insider_buying. Override to insider_selling
# when the body text shows a disposition. SEC Form 4 uses transaction codes:
#  S = open-market sale, D = disposition, F = payment of tax via shares,
#  G = gift, M = exercise of derivative, A = grant/award.
# We promote to insider_selling only on language patterns that unambiguously
# indicate the insider received cash for shares; routine F/M/G/A transactions
# stay tagged as the generic insider_activity / insider_buying default.
FORM4_SELLING_PATTERNS: tuple[str, ...] = (
    r"\binsider (?:sale|sells|sold|selling)\b",
    r"\b(?:director|ceo|cfo|coo|cto|president|officer|executive)\s+.{0,40}\bsell(?:s|ing)?\b",
    r"\bsell(?:s|ing|er)?\s+.{0,40}\bshares\b",
    r"\bsold\s+(?:about\s+)?[\d,]+ shares\b",
    r"\bsold\s+\$[\d,.]+[mk]?\s+(?:in|worth|of)\s+shares\b",
    r"\bdisposition of shares\b",
    r"\bopen[- ]market sale\b",
    r"\bdisposed of .{0,40} shares\b",
    r"\bform 144\b",
    r"\b10b5[- ]1\b.*\bsale\b",
    r"\bplanned sale\b",
    r"\btransaction code[\s:]+[\"\']?s[\"\']?\b",
)
FORM4_BUYING_PATTERNS: tuple[str, ...] = (
    r"\binsider (?:buy|buys|bought|buying|purchase|purchases|purchased)\b",
    r"\b(?:director|ceo|cfo|coo|cto|president|officer|executive)\s+.{0,40}\b(?:buy|bought|purchas)\b",
    r"\bopen[- ]market purchase\b",
    r"\bbought\s+(?:about\s+)?[\d,]+ shares\b",
    r"\bpurchased\s+(?:about\s+)?[\d,]+ shares\b",
    r"\bacquired\s+(?:about\s+)?[\d,]+ shares\b",
    r"\btransaction code[\s:]+[\"\']?p[\"\']?\b",
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

    # Pass 3: insider buying/selling detection across all sources.
    # Form 4 records typically have no body, but Google News and yfinance
    # cover insider transactions in headlines ("Director sells X shares",
    # "CFO sells $181k in shares", etc.). Reuse texts_all from pass 2.
    if RECORD_LEVEL_OVERRIDES:
        sell_hit = texts_all.apply(lambda t: any(re.search(p, t) for p in FORM4_SELLING_PATTERNS))
        buy_hit = texts_all.apply(lambda t: any(re.search(p, t) for p in FORM4_BUYING_PATTERNS))
        sell_only = sell_hit & ~buy_hit
        buy_only = buy_hit & ~sell_hit
        if sell_only.any():
            out.loc[sell_only, "catalyst_family"] = "insider_activity"
            out.loc[sell_only, "catalyst_subtype"] = "insider_selling"
        if buy_only.any():
            out.loc[buy_only, "catalyst_family"] = "insider_activity"
            out.loc[buy_only, "catalyst_subtype"] = "insider_buying"
    return out
