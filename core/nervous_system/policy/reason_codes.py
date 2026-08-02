"""Stable machine reason codes and their human detail.

Persisted policy decisions carry only the machine code.  The human sentence
lives here so that an audit trail written today still reads correctly when the
wording is improved, and so that no free text is ever hashed into a decision.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping


class ReasonCode(str, Enum):
    """Every reason the policy engine may emit."""

    # 1 environment permission
    ENV_PRODUCTION_LIVE_DISABLED_MVP = "ENV_PRODUCTION_LIVE_DISABLED_MVP"
    ENV_STRATEGY_NOT_PERMITTED = "ENV_STRATEGY_NOT_PERMITTED"
    POLICY_OFF_AUDIT_ONLY = "POLICY_OFF_AUDIT_ONLY"

    # 2 snapshot validity and freshness
    SNAPSHOT_INVALID = "SNAPSHOT_INVALID"
    SNAPSHOT_PROFILE_MISMATCH = "SNAPSHOT_PROFILE_MISMATCH"
    SNAPSHOT_LINEAGE_MISMATCH = "SNAPSHOT_LINEAGE_MISMATCH"
    SNAPSHOT_DECISION_TIME_AFTER_INTENT = "SNAPSHOT_DECISION_TIME_AFTER_INTENT"
    SNAPSHOT_REQUIRED_STATE_STALE = "SNAPSHOT_REQUIRED_STATE_STALE"
    SNAPSHOT_REQUIRED_STATE_MISSING = "SNAPSHOT_REQUIRED_STATE_MISSING"

    # 3 operational readiness
    READINESS_STATE_MISSING = "READINESS_STATE_MISSING"
    READINESS_NOT_READY = "READINESS_NOT_READY"
    READINESS_JOB_NOT_REQUIRED = "READINESS_JOB_NOT_REQUIRED"

    # 4 instrument and structure permission
    INSTRUMENT_PREFERENCE_MISSING = "INSTRUMENT_PREFERENCE_MISSING"
    INSTRUMENT_FAMILY_NOT_PERMITTED = "INSTRUMENT_FAMILY_NOT_PERMITTED"
    STRUCTURE_UNKNOWN_MAXIMUM_LOSS = "STRUCTURE_UNKNOWN_MAXIMUM_LOSS"
    STRUCTURE_NAKED_SHORT_OPTION = "STRUCTURE_NAKED_SHORT_OPTION"
    STRUCTURE_UNCOVERED_RATIO = "STRUCTURE_UNCOVERED_RATIO"

    # 5 broker and account constraints
    BROKER_PAPER_ACCOUNT_REQUIRED = "BROKER_PAPER_ACCOUNT_REQUIRED"
    BROKER_PORTFOLIO_STATE_MISSING = "BROKER_PORTFOLIO_STATE_MISSING"
    BROKER_ACCOUNT_IDENTITY_MISMATCH = "BROKER_ACCOUNT_IDENTITY_MISMATCH"
    BROKER_DUPLICATE_IDEMPOTENCY_KEY = "BROKER_DUPLICATE_IDEMPOTENCY_KEY"
    BROKER_OPEN_ORDERS_NOT_OBSERVED = "BROKER_OPEN_ORDERS_NOT_OBSERVED"
    BROKER_INSUFFICIENT_BUYING_POWER = "BROKER_INSUFFICIENT_BUYING_POWER"

    # 6 hard portfolio limits
    PORTFOLIO_MAX_DAILY_LOSS_BREACH = "PORTFOLIO_MAX_DAILY_LOSS_BREACH"
    PORTFOLIO_MAX_GROSS_NOTIONAL_BREACH = "PORTFOLIO_MAX_GROSS_NOTIONAL_BREACH"
    PORTFOLIO_EXPOSURE_UNKNOWN = "PORTFOLIO_EXPOSURE_UNKNOWN"

    # 7 liquidity and data-quality limits
    DATA_QUALITY_BLOCKING = "DATA_QUALITY_BLOCKING"
    LIQUIDITY_TICKER_STATE_MISSING = "LIQUIDITY_TICKER_STATE_MISSING"
    LIQUIDITY_METRIC_UNKNOWN = "LIQUIDITY_METRIC_UNKNOWN"
    LIQUIDITY_BELOW_MINIMUM = "LIQUIDITY_BELOW_MINIMUM"

    # 8 contextual size modifiers
    CONTEXT_MARKET_REGIME_MODIFIER = "CONTEXT_MARKET_REGIME_MODIFIER"
    CONTEXT_THEME_REGIME_MODIFIER = "CONTEXT_THEME_REGIME_MODIFIER"
    CONTEXT_DEALER_REGIME_MODIFIER = "CONTEXT_DEALER_REGIME_MODIFIER"
    CONTEXT_DATA_QUALITY_MODIFIER = "CONTEXT_DATA_QUALITY_MODIFIER"
    CONTEXT_MARKET_UNAVAILABLE = "CONTEXT_MARKET_UNAVAILABLE"
    CONTEXT_THEME_UNAVAILABLE = "CONTEXT_THEME_UNAVAILABLE"
    CONTEXT_DEALER_UNAVAILABLE = "CONTEXT_DEALER_UNAVAILABLE"

    # 9 final caps and minimum executable size
    SIZE_CAPPED_TO_MAX_POSITION_NOTIONAL = "SIZE_CAPPED_TO_MAX_POSITION_NOTIONAL"
    SIZE_QUANTIZED_TO_MONEY_QUANTUM = "SIZE_QUANTIZED_TO_MONEY_QUANTUM"
    SIZE_BELOW_MINIMUM_EXECUTABLE = "SIZE_BELOW_MINIMUM_EXECUTABLE"

    # Outcomes
    POLICY_APPROVED_AS_REQUESTED = "POLICY_APPROVED_AS_REQUESTED"
    POLICY_APPROVED_REDUCED = "POLICY_APPROVED_REDUCED"
    EXIT_RISK_REDUCING_PERMITTED = "EXIT_RISK_REDUCING_PERMITTED"


_DETAIL: Mapping[ReasonCode, str] = MappingProxyType(
    {
        ReasonCode.ENV_PRODUCTION_LIVE_DISABLED_MVP: (
            "Production-live submission is denied in every policy mode for this MVP."
        ),
        ReasonCode.ENV_STRATEGY_NOT_PERMITTED: (
            "The originating strategy is not enabled for this environment."
        ),
        ReasonCode.POLICY_OFF_AUDIT_ONLY: (
            "Policy mode is OFF: the unmodified baseline is recorded for parity and "
            "is never executable."
        ),
        ReasonCode.SNAPSHOT_INVALID: (
            "The context snapshot is marked invalid by the snapshot builder."
        ),
        ReasonCode.SNAPSHOT_PROFILE_MISMATCH: (
            "The snapshot was built with a freshness profile this policy version "
            "does not accept."
        ),
        ReasonCode.SNAPSHOT_LINEAGE_MISMATCH: (
            "The intent references a different snapshot than the one supplied."
        ),
        ReasonCode.SNAPSHOT_DECISION_TIME_AFTER_INTENT: (
            "The snapshot decision time is after the intent decision time, which "
            "would be look-ahead evidence."
        ),
        ReasonCode.SNAPSHOT_REQUIRED_STATE_STALE: (
            "At least one required context state is stale at the decision time."
        ),
        ReasonCode.SNAPSHOT_REQUIRED_STATE_MISSING: (
            "At least one required context state is missing at the decision time."
        ),
        ReasonCode.READINESS_STATE_MISSING: (
            "No operational readiness state is present in the snapshot."
        ),
        ReasonCode.READINESS_NOT_READY: (
            "The nightly data readiness gate did not report ready."
        ),
        ReasonCode.READINESS_JOB_NOT_REQUIRED: (
            "The readiness state reports a job this policy version does not require."
        ),
        ReasonCode.INSTRUMENT_PREFERENCE_MISSING: (
            "The intent requested no instrument family, so nothing can be permitted."
        ),
        ReasonCode.INSTRUMENT_FAMILY_NOT_PERMITTED: (
            "No requested instrument family is permitted by this policy version."
        ),
        ReasonCode.STRUCTURE_UNKNOWN_MAXIMUM_LOSS: (
            "A requested structure has no defined maximum loss at intent time."
        ),
        ReasonCode.STRUCTURE_NAKED_SHORT_OPTION: (
            "A requested structure is classified as a naked short option."
        ),
        ReasonCode.STRUCTURE_UNCOVERED_RATIO: (
            "A requested structure is classified as an uncovered ratio."
        ),
        ReasonCode.BROKER_PAPER_ACCOUNT_REQUIRED: (
            "QA-paper requires a configured paper account alias and credentials."
        ),
        ReasonCode.BROKER_PORTFOLIO_STATE_MISSING: (
            "No broker-authoritative portfolio state is present in the snapshot."
        ),
        ReasonCode.BROKER_ACCOUNT_IDENTITY_MISMATCH: (
            "The observed broker account alias differs from the configured account."
        ),
        ReasonCode.BROKER_DUPLICATE_IDEMPOTENCY_KEY: (
            "An open broker order already carries this intent idempotency key."
        ),
        ReasonCode.BROKER_OPEN_ORDERS_NOT_OBSERVED: (
            "Open orders could not be observed, so a duplicate or conflicting "
            "order cannot be ruled out before a new entry."
        ),
        ReasonCode.BROKER_INSUFFICIENT_BUYING_POWER: (
            "Observed buying power is below the requested risk budget."
        ),
        ReasonCode.PORTFOLIO_MAX_DAILY_LOSS_BREACH: (
            "The account has reached the configured maximum daily loss."
        ),
        ReasonCode.PORTFOLIO_MAX_GROSS_NOTIONAL_BREACH: (
            "The proposed entry would exceed the configured gross notional limit."
        ),
        ReasonCode.PORTFOLIO_EXPOSURE_UNKNOWN: (
            "An open position carries no market value, so gross exposure cannot "
            "be established and the limit cannot be enforced."
        ),
        ReasonCode.DATA_QUALITY_BLOCKING: (
            "A context state carries a data-quality issue at a blocking severity."
        ),
        ReasonCode.LIQUIDITY_TICKER_STATE_MISSING: (
            "No ticker state is present, so liquidity cannot be established."
        ),
        ReasonCode.LIQUIDITY_METRIC_UNKNOWN: (
            "The configured liquidity metric is absent from the ticker state."
        ),
        ReasonCode.LIQUIDITY_BELOW_MINIMUM: (
            "Observed liquidity is below the configured minimum for an entry."
        ),
        ReasonCode.CONTEXT_MARKET_REGIME_MODIFIER: (
            "The market regime applied a contextual size multiplier."
        ),
        ReasonCode.CONTEXT_THEME_REGIME_MODIFIER: (
            "The most conservative theme regime applied a contextual size multiplier."
        ),
        ReasonCode.CONTEXT_DEALER_REGIME_MODIFIER: (
            "The dealer regime applied a contextual size multiplier."
        ),
        ReasonCode.CONTEXT_DATA_QUALITY_MODIFIER: (
            "A non-blocking data-quality issue applied a contextual size multiplier."
        ),
        ReasonCode.CONTEXT_MARKET_UNAVAILABLE: (
            "No market state was available, so no market modifier was applied."
        ),
        ReasonCode.CONTEXT_THEME_UNAVAILABLE: (
            "No theme state was available, so no theme modifier was applied."
        ),
        ReasonCode.CONTEXT_DEALER_UNAVAILABLE: (
            "No dealer state was available, so no dealer modifier was applied."
        ),
        ReasonCode.SIZE_CAPPED_TO_MAX_POSITION_NOTIONAL: (
            "The risk budget was capped to the configured maximum position notional."
        ),
        ReasonCode.SIZE_QUANTIZED_TO_MONEY_QUANTUM: (
            "The risk budget was rounded down to the configured money quantum."
        ),
        ReasonCode.SIZE_BELOW_MINIMUM_EXECUTABLE: (
            "The final risk budget is below the configured minimum executable size."
        ),
        ReasonCode.POLICY_APPROVED_AS_REQUESTED: (
            "The strategy request was approved without reduction."
        ),
        ReasonCode.POLICY_APPROVED_REDUCED: (
            "The strategy request was approved at a reduced risk budget."
        ),
        ReasonCode.EXIT_RISK_REDUCING_PERMITTED: (
            "A risk-reducing exit stays operable under degraded context and is "
            "exempt from entry-only vetoes and contextual modifiers."
        ),
    }
)

_MISSING = tuple(code for code in ReasonCode if code not in _DETAIL)
if _MISSING:  # pragma: no cover - import-time contract
    raise RuntimeError(f"reason codes without human detail: {_MISSING}")


def describe(code: ReasonCode | str) -> str:
    """Return the human detail for a stable machine reason code."""

    try:
        resolved = ReasonCode(code)
    except ValueError as exc:
        raise KeyError(f"unknown reason code: {code!r}") from exc
    return _DETAIL[resolved]


__all__ = ["ReasonCode", "describe"]
