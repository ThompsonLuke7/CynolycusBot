"""Reliability weights attached to every gamma observation.

The point of this module is a single distinction:

    *Where* gamma sits can be high confidence while *who owns it* is low
    confidence, and the two must not share one number.

So a snapshot carries ``structure_confidence`` (did we see enough of the chain
to describe its shape?) separately from ``sign_confidence`` (do we believe the
dealer-side assumption for this symbol?), plus ``data_freshness`` (is the
snapshot recent enough for the decision reading it?).

These are **documented reliability weights in [0, 1], not probabilities.** The
nervous system forbids mapping uncalibrated heuristics into probability fields,
and none of these are calibrated against outcomes. They exist so a consumer can
down-weight a weak observation instead of treating every snapshot as equally
authoritative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Sign-confidence priors
#
# STATED PRIORS, NOT FITTED VALUES. They encode one belief: dealer-side
# inference from open interest degrades as a name gets less liquid and its flow
# less identifiable. Index and large ETF flow is dominated by recognizable
# customer hedging; a normal single name's open interest could belong to anyone.
#
# Do not tune these against a backtest. If they ever need to move, they should
# move because a measurement said so, and then they stop being priors.
# ---------------------------------------------------------------------------
SIGN_CONFIDENCE_TIERS: dict[str, float] = {
    "index_etf": 0.60,      # SPY, QQQ
    "liquid_etf": 0.45,     # IWM, GLD, SLV, sector ETFs
    "mega_cap": 0.30,       # very heavily optioned single names
    "normal_equity": 0.20,
    "illiquid": 0.10,
}

INDEX_ETFS = frozenset({"SPY", "QQQ", "SPX", "SPXW", "NDX"})
LIQUID_ETFS = frozenset({"IWM", "GLD", "SLV", "DIA", "XLF", "XLE", "XLK", "SMH", "EEM", "TLT", "HYG", "USO"})

MEGA_CAP_MIN_MARKET_CAP = 5.0e11
MEGA_CAP_MIN_DOLLAR_VOLUME = 2.0e9
NORMAL_MIN_DOLLAR_VOLUME = 2.0e7

# A snapshot older than this contributes no freshness credit at all.
DEFAULT_MAX_AGE_DAYS = 4.0


@dataclass(frozen=True)
class ChainQuality:
    """Countable facts about the chain rows a snapshot was built from."""

    rows_total: int
    rows_zero_gamma: int
    rows_missing_iv: int
    rows_zero_oi: int
    strikes: int
    zero_gamma_oi_share: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceBlock:
    """The reliability weights that travel with a gamma snapshot."""

    structure_confidence: float
    sign_confidence: float
    data_freshness: float
    liquidity_tier: str
    chain_quality: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_chain_quality(frame: pd.DataFrame | None) -> ChainQuality:
    """Count the defects in a contract-level chain frame.

    ``zero_gamma_oi_share`` is the important one: zero-gamma rows are harmless
    when they are deep-ITM contracts nobody holds, and material when they carry
    real open interest, because that open interest silently contributes no
    exposure.
    """
    if frame is None or frame.empty:
        return ChainQuality(0, 0, 0, 0, 0, None)
    work = frame.copy()
    for col in ("gamma", "open_interest", "iv", "strike"):
        if col not in work.columns:
            work[col] = None
        work[col] = pd.to_numeric(work[col], errors="coerce")
    zero_gamma = work["gamma"].fillna(0.0) == 0.0
    total_oi = float(work["open_interest"].fillna(0.0).abs().sum())
    zero_gamma_oi = float(work.loc[zero_gamma, "open_interest"].fillna(0.0).abs().sum())
    return ChainQuality(
        rows_total=int(len(work)),
        rows_zero_gamma=int(zero_gamma.sum()),
        rows_missing_iv=int(work["iv"].isna().sum()),
        rows_zero_oi=int((work["open_interest"].fillna(0.0) == 0.0).sum()),
        strikes=int(work["strike"].dropna().nunique()),
        zero_gamma_oi_share=(zero_gamma_oi / total_oi) if total_oi > 0 else None,
    )


def liquidity_tier(
    symbol: str,
    *,
    avg_dollar_volume_20d: float | None = None,
    market_cap: float | None = None,
) -> str:
    """Classify a symbol into a sign-confidence tier.

    Membership lists win over the size thresholds: an index ETF is an index ETF
    regardless of what its dollar volume happened to be that day.
    """
    upper = (symbol or "").upper()
    if upper in INDEX_ETFS:
        return "index_etf"
    if upper in LIQUID_ETFS:
        return "liquid_etf"
    adv = float(avg_dollar_volume_20d) if avg_dollar_volume_20d else 0.0
    cap = float(market_cap) if market_cap else 0.0
    if cap >= MEGA_CAP_MIN_MARKET_CAP and adv >= MEGA_CAP_MIN_DOLLAR_VOLUME:
        return "mega_cap"
    if adv >= NORMAL_MIN_DOLLAR_VOLUME:
        return "normal_equity"
    return "illiquid"


def sign_confidence(tier: str, *, convention_dispersion: float | None = None) -> float:
    """Confidence in the dealer-side assumption for this symbol.

    ``convention_dispersion`` is the disagreement between alternative sign
    conventions in [0, 1] when several have been computed. Disagreement can only
    lower confidence -- agreement between two guesses is not evidence.
    """
    base = SIGN_CONFIDENCE_TIERS.get(tier, SIGN_CONFIDENCE_TIERS["illiquid"])
    if convention_dispersion is None:
        return float(base)
    penalty = max(0.0, min(1.0, float(convention_dispersion)))
    return float(base * (1.0 - penalty))


def structure_confidence(
    quality: ChainQuality,
    *,
    strike_coverage: float | None = None,
    min_strikes: int = 10,
) -> float:
    """How well the snapshot describes the shape of the chain.

    Three independent ways a snapshot can fail to describe structure, combined
    multiplicatively because any one of them alone is disqualifying:

    * too few strikes to see a shape at all,
    * open interest sitting on rows that report no gamma,
    * a strike window that did not cover the region around spot.
    """
    if quality.rows_total <= 0 or quality.strikes <= 0:
        return 0.0
    breadth = min(1.0, quality.strikes / float(min_strikes))
    lost = quality.zero_gamma_oi_share or 0.0
    integrity = max(0.0, 1.0 - float(lost))
    coverage = 1.0 if strike_coverage is None else max(0.0, min(1.0, float(strike_coverage)))
    return float(max(0.0, min(1.0, breadth * integrity * coverage)))


def data_freshness(age_days: float | None, *, max_age_days: float = DEFAULT_MAX_AGE_DAYS) -> float:
    """Linear decay from 1.0 at zero age to 0.0 at ``max_age_days``.

    A missing age is treated as fully stale rather than fully fresh: an unknown
    timestamp is not evidence of recency.
    """
    if age_days is None or max_age_days <= 0:
        return 0.0
    age = float(age_days)
    if age <= 0.0:
        return 1.0
    if age >= float(max_age_days):
        return 0.0
    return float(1.0 - age / float(max_age_days))


def build_confidence(
    *,
    symbol: str,
    frame: pd.DataFrame | None,
    age_days: float | None,
    avg_dollar_volume_20d: float | None = None,
    market_cap: float | None = None,
    strike_coverage: float | None = None,
    convention_dispersion: float | None = None,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
) -> ConfidenceBlock:
    """Assemble the confidence block for one snapshot."""
    quality = assess_chain_quality(frame)
    tier = liquidity_tier(symbol, avg_dollar_volume_20d=avg_dollar_volume_20d, market_cap=market_cap)
    return ConfidenceBlock(
        structure_confidence=structure_confidence(quality, strike_coverage=strike_coverage),
        sign_confidence=sign_confidence(tier, convention_dispersion=convention_dispersion),
        data_freshness=data_freshness(age_days, max_age_days=max_age_days),
        liquidity_tier=tier,
        chain_quality=quality.to_dict(),
    )
