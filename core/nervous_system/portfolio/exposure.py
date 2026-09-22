"""Canonical portfolio exposure.

Pure function of the broker portfolio observation, the context snapshot, and
versioned configuration.  Unknown inputs are never imputed: a position with no
market value or an option with no Greeks produces an explicit unknown-exposure
issue and a breached ``exposure.known`` limit, because silently treating it as
zero would understate risk.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from uuid import UUID

from core.nervous_system.config.portfolio import PortfolioConfig
from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import AssetClass, DataQualitySeverity
from core.nervous_system.contracts.portfolio import ExposureLimitResult, ExposureReport
from core.nervous_system.contracts.quality import DataQualityIssue, DataQualitySummary
from core.nervous_system.contracts.states import PortfolioPosition, PortfolioState


UNALLOCATED = "UNALLOCATED"
_COMPONENT = "portfolio.exposure"
_GREEK_NAMES = ("delta", "gamma", "vega", "theta")
_ZERO = Decimal("0")


def _decimal(value: float | int | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _quantize(value: Decimal, quantum: Decimal) -> Decimal:
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def _add(bucket: dict[str, Decimal], key: str, value: Decimal) -> None:
    bucket[key] = bucket.get(key, _ZERO) + value


def calculate_exposure(
    portfolio: PortfolioState,
    context: ContextSnapshot,
    *,
    config: PortfolioConfig,
    report_id: UUID,
    proposed_position: PortfolioPosition | None = None,
) -> ExposureReport:
    """Compute exposure for one broker observation against one context."""

    issues: list[DataQualityIssue] = []
    theme_weights_by_ticker = _theme_weights_by_ticker(context)

    existing = _aggregate(
        portfolio.positions, config, theme_weights_by_ticker, issues
    )

    proposed = None
    if proposed_position is not None:
        proposed = _aggregate(
            (proposed_position,), config, theme_weights_by_ticker, issues
        )

    combined = _combine(existing, proposed)
    quantum = config.money_quantum

    # Only ERROR/CRITICAL evidence means exposure is genuinely unmeasurable.  A
    # WARNING (an unmapped sector, say) is reported but must not breach the
    # known-exposure limit, or every ticker outside the curated map would be
    # blocked for a benign gap.
    quality = DataQualitySummary(issues=tuple(issues))
    limit_results = _evaluate_limits(
        combined, config, unknown=not quality.is_usable
    )

    incremental: dict[str, Decimal] = {}
    if proposed is not None:
        incremental["GROSS"] = _quantize(proposed.gross, quantum)
        incremental["NET"] = _quantize(proposed.net, quantum)
        for key, value in sorted(proposed.symbol.items()):
            incremental[f"SYMBOL:{key}"] = _quantize(value, quantum)
        for key, value in sorted(proposed.sector.items()):
            incremental[f"SECTOR:{key}"] = _quantize(value, quantum)
        for key, value in sorted(proposed.theme.items()):
            incremental[f"THEME:{key}"] = _quantize(value, quantum)
        for key, value in sorted(proposed.factor.items()):
            incremental[f"FACTOR:{key}"] = _quantize(value, quantum)

    return ExposureReport.create(
        report_id=report_id,
        portfolio_state_id=portfolio.state_id,
        snapshot_id=context.snapshot_id,
        calculated_at=portfolio.as_of,
        gross_notional=_quantize(existing.gross, quantum),
        net_notional=_quantize(existing.net, quantum),
        long_notional=_quantize(existing.long, quantum),
        short_notional=_quantize(existing.short, quantum),
        symbol_notional=_quantize_map(existing.symbol, quantum),
        underlying_equivalent=_quantize_map(existing.underlying, quantum),
        sector_notional=_quantize_map(existing.sector, quantum),
        theme_notional=_quantize_map(existing.theme, quantum),
        factor_notional=_quantize_map(existing.factor, quantum),
        option_greeks=_quantize_map(existing.greeks, quantum),
        proposed_incremental_exposure=incremental,
        limit_results=limit_results,
        quality=quality,
        config_version=config.config_version,
    )


def evaluate_proposed_concentration(
    portfolio: PortfolioState,
    context: ContextSnapshot,
    *,
    config: PortfolioConfig,
    proposed_underlying: str,
    proposed_notional: Decimal,
) -> tuple[ExposureLimitResult, ...]:
    """Evaluate concentration limits for the portfolio PLUS one proposed entry.

    Pure, and deliberately separate from ``calculate_exposure``: the policy
    engine must not mint a report id or persist anything to decide a veto, and
    a post-trade report must not be the thing a pre-trade gate depends on.

    The proposal is added as notional to the buckets its underlying maps to,
    rather than as a fabricated ``PortfolioPosition``, because a synthetic
    position would need an invented quantity and price to exist at all -- and
    the limits are stated in notional, so the invention would buy nothing.
    """

    if not isinstance(proposed_underlying, str) or not proposed_underlying.strip():
        raise ValueError("proposed_underlying must be a non-empty string")
    notional = _decimal(proposed_notional)
    if notional < _ZERO:
        raise ValueError("proposed_notional must not be negative")

    issues: list[DataQualityIssue] = []
    theme_weights_by_ticker = _theme_weights_by_ticker(context)
    aggregate = _aggregate(portfolio.positions, config, theme_weights_by_ticker, issues)

    if notional > _ZERO:
        aggregate.gross += notional
        aggregate.long += notional
        aggregate.net += notional
        _add(aggregate.symbol, proposed_underlying, notional)
        sector_id = config.sector_for(proposed_underlying)
        _add(aggregate.sector, sector_id if sector_id is not None else UNALLOCATED, notional)
        _allocate_theme(
            aggregate,
            PortfolioPosition(
                broker_position_id=f"PROPOSED:{proposed_underlying}",
                symbol=proposed_underlying,
                underlying=proposed_underlying,
                asset_class=AssetClass.EQUITY,
                quantity=Decimal("1"),
            ),
            notional,
            theme_weights_by_ticker,
        )
        for factor_id in config.factors_for(proposed_underlying):
            _add(aggregate.factor, factor_id, notional)

    quality = DataQualitySummary(issues=tuple(issues))
    return _evaluate_limits(aggregate, config, unknown=not quality.is_usable)


def _quantize_map(bucket: Mapping[str, Decimal], quantum: Decimal) -> dict[str, Decimal]:
    return {key: _quantize(value, quantum) for key, value in sorted(bucket.items())}


class _Aggregate:
    __slots__ = (
        "gross",
        "net",
        "long",
        "short",
        "symbol",
        "underlying",
        "sector",
        "theme",
        "factor",
        "greeks",
    )

    def __init__(self) -> None:
        self.gross = _ZERO
        self.net = _ZERO
        self.long = _ZERO
        self.short = _ZERO
        self.symbol: dict[str, Decimal] = {}
        self.underlying: dict[str, Decimal] = {}
        self.sector: dict[str, Decimal] = {}
        self.theme: dict[str, Decimal] = {}
        self.factor: dict[str, Decimal] = {}
        self.greeks: dict[str, Decimal] = {}


def _aggregate(
    positions: tuple[PortfolioPosition, ...],
    config: PortfolioConfig,
    theme_weights_by_ticker: Mapping[str, Mapping[str, Decimal]],
    issues: list[DataQualityIssue],
) -> _Aggregate:
    result = _Aggregate()
    # Sort so the report never depends on broker payload ordering.
    for position in sorted(positions, key=lambda item: (item.symbol, item.broker_position_id)):
        if position.market_value is None:
            issues.append(
                DataQualityIssue(
                    code="POSITION_NOTIONAL_UNKNOWN",
                    severity=DataQualitySeverity.ERROR,
                    component=_COMPONENT,
                    message=f"position {position.symbol} has no market value",
                )
            )
            continue

        notional = _decimal(position.market_value)
        result.net += notional
        if notional >= _ZERO:
            result.long += notional
        else:
            result.short += -notional
        result.gross += abs(notional)
        _add(result.symbol, position.symbol, notional)

        sector_id = config.sector_for(position.underlying)
        if sector_id is None:
            issues.append(
                DataQualityIssue(
                    code="SECTOR_UNMAPPED",
                    severity=DataQualitySeverity.WARNING,
                    component=_COMPONENT,
                    message=f"no canonical sector for {position.underlying}",
                    fallback_used=UNALLOCATED,
                )
            )
            _add(result.sector, UNALLOCATED, notional)
        else:
            _add(result.sector, sector_id, notional)

        _allocate_theme(result, position, notional, theme_weights_by_ticker)

        for factor_id in config.factors_for(position.underlying):
            _add(result.factor, factor_id, notional)

        if position.asset_class is AssetClass.OPTION:
            _accumulate_option(result, position, config, issues)
        else:
            _add(result.underlying, position.underlying, _decimal(position.quantity))

    return result


def _allocate_theme(
    result: _Aggregate,
    position: PortfolioPosition,
    notional: Decimal,
    theme_weights_by_ticker: Mapping[str, Mapping[str, Decimal]],
) -> None:
    """Split one position's notional across its own normalised theme weights."""

    theme_weights = theme_weights_by_ticker.get(position.underlying)
    if not theme_weights:
        # No membership was carried for this underlying, so its theme is
        # genuinely unknown. UNALLOCATED says so rather than guessing.
        _add(result.theme, UNALLOCATED, notional)
        return
    allocated = _ZERO
    keys = sorted(theme_weights)
    for index, theme_id in enumerate(keys):
        if index == len(keys) - 1:
            # The final theme absorbs the rounding residue so the split always
            # sums back to the full position notional.
            share = notional - allocated
        else:
            share = notional * theme_weights[theme_id]
            allocated += share
        _add(result.theme, theme_id, share)


def _accumulate_option(
    result: _Aggregate,
    position: PortfolioPosition,
    config: PortfolioConfig,
    issues: list[DataQualityIssue],
) -> None:
    multiplier = (
        _decimal(position.contract_multiplier)
        if position.contract_multiplier is not None
        else config.default_contract_multiplier
    )
    contracts = _decimal(position.quantity)
    missing = [name for name in _GREEK_NAMES if name not in position.greeks]
    if missing:
        issues.append(
            DataQualityIssue(
                code="OPTION_GREEKS_UNKNOWN",
                severity=DataQualitySeverity.ERROR,
                component=_COMPONENT,
                message=(
                    f"option {position.symbol} is missing Greeks: "
                    + ", ".join(sorted(missing))
                ),
            )
        )
    for name in _GREEK_NAMES:
        raw = position.greeks.get(name)
        if raw is None:
            continue
        _add(result.greeks, name, _decimal(raw) * multiplier * contracts)
    delta = position.greeks.get("delta")
    if delta is None:
        # No delta means no defensible underlying-equivalent contribution.  It
        # is left out and flagged rather than counted as zero.
        return
    _add(result.underlying, position.underlying, _decimal(delta) * multiplier * contracts)


def _combine(existing: _Aggregate, proposed: _Aggregate | None) -> _Aggregate:
    if proposed is None:
        return existing
    combined = _Aggregate()
    combined.gross = existing.gross + proposed.gross
    combined.net = existing.net + proposed.net
    combined.long = existing.long + proposed.long
    combined.short = existing.short + proposed.short
    for name in ("symbol", "underlying", "sector", "theme", "factor", "greeks"):
        merged: dict[str, Decimal] = dict(getattr(existing, name))
        for key, value in getattr(proposed, name).items():
            merged[key] = merged.get(key, _ZERO) + value
        setattr(combined, name, merged)
    return combined


def _evaluate_limits(
    aggregate: _Aggregate,
    config: PortfolioConfig,
    *,
    unknown: bool,
) -> tuple[ExposureLimitResult, ...]:
    quantum = config.money_quantum
    results: list[ExposureLimitResult] = [
        ExposureLimitResult(
            limit_id="exposure.gross",
            scope="GROSS",
            scope_id="PORTFOLIO",
            observed=_quantize(aggregate.gross, quantum),
            limit_value=config.max_gross_notional,
            breached=aggregate.gross > config.max_gross_notional,
            reason_code="PORTFOLIO_MAX_GROSS_NOTIONAL_BREACH",
        )
    ]
    scopes = (
        ("SYMBOL", aggregate.symbol, config.max_symbol_notional, "exposure.symbol",
         "PORTFOLIO_MAX_SYMBOL_NOTIONAL_BREACH"),
        ("SECTOR", aggregate.sector, config.max_sector_notional, "exposure.sector",
         "PORTFOLIO_MAX_SECTOR_NOTIONAL_BREACH"),
        ("THEME", aggregate.theme, config.max_theme_notional, "exposure.theme",
         "PORTFOLIO_MAX_THEME_NOTIONAL_BREACH"),
        ("FACTOR", aggregate.factor, config.max_factor_notional, "exposure.factor",
         "PORTFOLIO_MAX_FACTOR_NOTIONAL_BREACH"),
    )
    for scope, bucket, limit_value, limit_id, reason_code in scopes:
        for scope_id in sorted(bucket):
            observed = abs(bucket[scope_id])
            results.append(
                ExposureLimitResult(
                    limit_id=limit_id,
                    scope=scope,
                    scope_id=scope_id,
                    observed=_quantize(observed, quantum),
                    limit_value=limit_value,
                    breached=observed > limit_value,
                    reason_code=reason_code,
                )
            )
    results.append(
        ExposureLimitResult(
            limit_id="exposure.known",
            scope="QUALITY",
            scope_id="PORTFOLIO",
            observed=Decimal("1") if unknown else _ZERO,
            limit_value=_ZERO,
            breached=unknown,
            reason_code="PORTFOLIO_EXPOSURE_UNKNOWN",
        )
    )
    return tuple(results)


def _theme_weights_by_ticker(context: ContextSnapshot) -> dict[str, dict[str, Decimal]]:
    """Normalised theme weights per underlying, for the context ticker and peers.

    Before peers existed this returned weights for the context ticker alone, so
    every OTHER holding fell to ``UNALLOCATED`` and a theme limit could only
    ever see one position.  Sector and factor buckets never had that problem --
    they come from portfolio-wide config maps -- which is why theme was the one
    scope that could not be enforced.
    """

    grouped: dict[str, dict[str, Decimal]] = {}
    for membership in (*context.theme_memberships, *context.peer_theme_memberships):
        if membership.weight <= 0:
            continue
        grouped.setdefault(membership.ticker, {})[membership.theme_id] = _decimal(
            membership.weight
        )
    normalised: dict[str, dict[str, Decimal]] = {}
    for ticker, positive in grouped.items():
        total = sum(positive.values(), _ZERO)
        if total <= _ZERO:
            continue
        normalised[ticker] = {
            theme_id: weight / total for theme_id, weight in positive.items()
        }
    return normalised


__all__ = ["UNALLOCATED", "calculate_exposure", "evaluate_proposed_concentration"]
