"""Pure causal eligibility and freshness evaluation for snapshot inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Sequence
import json
from zoneinfo import ZoneInfo

from core.calendar.us_market_calendar import is_trading_day, prev_trading_day
from core.nervous_system.config.freshness import FreshnessRule, SnapshotProfile
from core.nervous_system.contracts.context import FreshnessResult, RejectedCandidate
from core.nervous_system.contracts.enums import MissingStateAction, StateType
from core.nervous_system.contracts.states import (
    CatalystEvent,
    CatalystPressure,
    StateEnvelope,
    ThemeMembership,
)


_ET = ZoneInfo("America/New_York")
_UTC = timezone.utc
_SINGLETON_TYPES = frozenset(
    {
        StateType.MARKET,
        StateType.TICKER,
        StateType.DEALER,
        StateType.PORTFOLIO,
        StateType.READINESS,
    }
)
_BAR_BOUND_TYPES = frozenset(
    {
        StateType.TICKER,
        StateType.MARKET,
        StateType.SECTOR,
        StateType.THEME,
        StateType.THEME_MEMBERSHIP,
    }
)


@dataclass(frozen=True)
class SnapshotEntityScope:
    """Stable entity routing for state types without a build argument."""

    market_entity_id: str = "US"
    portfolio_entity_id: str = "paper"
    readiness_entity_id: str = "nightly_data_readiness"
    sector_entity_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "market_entity_id",
            "portfolio_entity_id",
            "readiness_entity_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        normalized = tuple(self.sector_entity_ids)
        if any(not isinstance(value, str) or not value.strip() for value in normalized):
            raise ValueError("sector_entity_ids must contain non-empty strings")
        object.__setattr__(self, "sector_entity_ids", normalized)


@dataclass(frozen=True)
class RequirementEvaluation:
    selected_states: tuple[StateEnvelope, ...]
    requirement_results: tuple[FreshnessResult, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    stale_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    valid: bool


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(_UTC)


def decision_session(decision_time: datetime) -> str:
    """Return the ET trading-session label without 24-hour arithmetic."""

    et = _aware(decision_time, "decision_time").astimezone(_ET)
    session_date = et.date()
    if not is_trading_day(session_date):
        session_date = prev_trading_day(session_date)
    return session_date.isoformat()


def _profile_boundary(profile: SnapshotProfile, decision_time: datetime) -> datetime | None:
    if profile.dealer_capture_after_et is None:
        return None
    et = _aware(decision_time, "decision_time").astimezone(_ET)
    if not is_trading_day(et.date()):
        return None
    return datetime.combine(
        et.date(),
        profile.dealer_capture_after_et,
        tzinfo=_ET,
    ).astimezone(_UTC)


def _expected_market_session(
    profile: SnapshotProfile,
    decision_time: datetime,
) -> str | None:
    if profile.market_session_lag == 0:
        return None
    et_date = _aware(decision_time, "decision_time").astimezone(_ET).date()
    session_date = et_date if is_trading_day(et_date) else prev_trading_day(et_date)
    for _ in range(profile.market_session_lag):
        session_date = prev_trading_day(session_date)
    return session_date.isoformat()


def _expected_entities(
    state_type: StateType,
    *,
    entity_id: str,
    scope: SnapshotEntityScope,
) -> tuple[str, ...] | None:
    if state_type is StateType.MARKET:
        return (scope.market_entity_id,)
    if state_type in {StateType.TICKER, StateType.DEALER}:
        return (entity_id,)
    if state_type is StateType.PORTFOLIO:
        return (scope.portfolio_entity_id,)
    if state_type is StateType.READINESS:
        return (scope.readiness_entity_id,)
    if state_type is StateType.SECTOR:
        return scope.sector_entity_ids
    return None


def _entity_matches(
    candidate: StateEnvelope,
    rule: FreshnessRule,
    *,
    entity_id: str,
    scope: SnapshotEntityScope,
) -> bool:
    expected = _expected_entities(rule.state_type, entity_id=entity_id, scope=scope)
    if expected is not None and candidate.entity_id not in expected:
        return False
    if rule.state_type is StateType.THEME_MEMBERSHIP and isinstance(candidate, ThemeMembership):
        return candidate.ticker == entity_id
    if rule.state_type is StateType.CATALYST_EVENT and isinstance(candidate, CatalystEvent):
        return candidate.ticker is None or candidate.ticker == entity_id
    if rule.state_type is StateType.CATALYST_PRESSURE and isinstance(candidate, CatalystPressure):
        scope_type = candidate.scope_type.upper()
        if scope_type == "TICKER":
            return candidate.scope_id == entity_id
        if scope_type == "MARKET":
            return candidate.scope_id == scope.market_entity_id
        return False
    return True


def producer_version(candidate: StateEnvelope) -> str:
    """Return one persisted, explicit version discriminator for tie-breaking."""

    string_parts = (
        candidate.producer,
        candidate.model_version,
        candidate.feature_version,
        candidate.config_version,
    )
    if any(not isinstance(value, str) or not value.strip() for value in string_parts):
        raise ValueError("producer and state versions must be non-empty strings")
    return json.dumps(
        (
            candidate.producer,
            f"{candidate.schema_version:020d}",
            candidate.model_version,
            candidate.feature_version,
            candidate.config_version,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def candidate_tie_key(candidate: StateEnvelope) -> tuple[datetime, datetime, str, str]:
    """The exact deterministic selection key required by Task 13."""

    return (
        _aware(candidate.available_at, "state.available_at"),
        _aware(candidate.generated_at, "state.generated_at"),
        producer_version(candidate),
        str(candidate.state_id),
    )


def _selection_key(candidate: StateEnvelope) -> tuple[str, str, str, str]:
    if hasattr(candidate, "ticker"):
        business = str(getattr(candidate, "ticker"))
    elif hasattr(candidate, "sector_id"):
        business = str(getattr(candidate, "sector_id"))
    elif hasattr(candidate, "theme_id"):
        business = str(getattr(candidate, "theme_id"))
    elif hasattr(candidate, "scope_id"):
        business = str(getattr(candidate, "scope_id"))
    elif hasattr(candidate, "account_alias"):
        business = str(getattr(candidate, "account_alias"))
    elif hasattr(candidate, "job"):
        business = str(getattr(candidate, "job"))
    else:
        business = candidate.entity_id
    return (candidate.state_type.value, business, str(candidate.state_id), producer_version(candidate))


def _effective_key(candidate: StateEnvelope) -> tuple[str, ...]:
    if candidate.state_type is StateType.THEME_MEMBERSHIP:
        return (
            candidate.state_type.value,
            str(getattr(candidate, "ticker", candidate.entity_id)),
            str(getattr(candidate, "theme_id", candidate.entity_id)),
        )
    if candidate.state_type is StateType.SECTOR:
        return (candidate.state_type.value, candidate.entity_id)
    if candidate.state_type is StateType.THEME:
        return (candidate.state_type.value, str(getattr(candidate, "theme_id", candidate.entity_id)))
    if candidate.state_type is StateType.CATALYST_PRESSURE:
        return (
            candidate.state_type.value,
            str(getattr(candidate, "scope_type", "")),
            str(getattr(candidate, "scope_id", candidate.entity_id)),
        )
    if candidate.state_type is StateType.CATALYST_EVENT:
        return (candidate.state_type.value, str(getattr(candidate, "event_id", candidate.state_id)))
    return (candidate.state_type.value, candidate.entity_id)


def _dealer_boundary_reason(
    candidate: StateEnvelope,
    profile: SnapshotProfile,
    decision_time: datetime,
) -> str | None:
    if candidate.state_type is not StateType.DEALER:
        return None
    if not profile.dealer_allowed:
        return "DEALER_NOT_ALLOWED"
    boundary = _profile_boundary(profile, decision_time)
    if boundary is None:
        return None
    if _aware(candidate.as_of, "state.as_of") <= boundary:
        return "DEALER_CAPTURE_NOT_POST_BOUNDARY"
    return None


def _requirement_key(rule: FreshnessRule) -> str:
    return rule.state_type.value


def _status_for_rejections(
    rule: FreshnessRule,
    rejection_codes: Sequence[str],
) -> tuple[str, str]:
    if "EXPIRED" in rejection_codes:
        return "STALE", "EXPIRED"
    if "STALE" in rejection_codes:
        return "STALE", "STALE"
    if "FUTURE_BAR" in rejection_codes:
        return "MISSING", "FUTURE_BAR"
    if "DEALER_NOT_ALLOWED" in rejection_codes:
        return "MISSING", "DEALER_NOT_ALLOWED"
    if "DEALER_CAPTURE_NOT_POST_BOUNDARY" in rejection_codes:
        return "MISSING", "DEALER_CAPTURE_NOT_POST_BOUNDARY"
    reason = "MISSING_REQUIRED_STATE" if rule.required else "OPTIONAL_STATE_MISSING"
    return "MISSING", reason


def _freshness_age_seconds(decision_time: datetime, candidate: StateEnvelope) -> float:
    return (_aware(decision_time, "decision_time") - _aware(candidate.available_at, "state.available_at")).total_seconds()


def _candidate_rejection_reason(
    candidate: StateEnvelope,
    rule: FreshnessRule,
    *,
    entity_id: str,
    decision_time_utc: datetime,
    decision_bar_utc: datetime,
    profile: SnapshotProfile,
    scope: SnapshotEntityScope,
    selected_theme_ids: frozenset[str] = frozenset(),
) -> str | None:
    if not _entity_matches(candidate, rule, entity_id=entity_id, scope=scope):
        return "WRONG_ENTITY"
    expected_market_session = _expected_market_session(profile, decision_time_utc)
    if (
        rule.state_type is StateType.MARKET
        and expected_market_session is not None
        and _aware(candidate.as_of, "state.as_of").astimezone(_ET).date().isoformat()
        != expected_market_session
    ):
        return "MARKET_SESSION_MISMATCH"
    if rule.state_type is StateType.THEME and str(
        getattr(candidate, "theme_id", candidate.entity_id)
    ) not in selected_theme_ids:
        return "THEME_NOT_IN_TICKER_MEMBERSHIPS"
    if isinstance(candidate, ThemeMembership):
        if decision_time_utc < _aware(candidate.effective_from, "state.effective_from"):
            return "MEMBERSHIP_NOT_YET_EFFECTIVE"
        if candidate.effective_until is not None and decision_time_utc >= _aware(
            candidate.effective_until,
            "state.effective_until",
        ):
            return "MEMBERSHIP_EFFECTIVE_EXPIRED"
    if rule.state_type in _BAR_BOUND_TYPES and _aware(candidate.as_of, "state.as_of") > decision_bar_utc:
        return "FUTURE_BAR"
    if _aware(candidate.available_at, "state.available_at") > decision_time_utc:
        return "NOT_AVAILABLE"
    if decision_time_utc >= _aware(candidate.valid_until, "state.valid_until"):
        return "EXPIRED"
    age_seconds = _freshness_age_seconds(decision_time_utc, candidate)
    if age_seconds < 0:
        return "NOT_AVAILABLE"
    if age_seconds > rule.max_age.total_seconds():
        return "STALE"
    dealer_reason = _dealer_boundary_reason(candidate, profile, decision_time_utc)
    if dealer_reason is not None:
        return dealer_reason
    return None


def _selected_ticker_theme_ids(
    candidates: Sequence[StateEnvelope],
    *,
    rule: FreshnessRule | None,
    entity_id: str,
    decision_time_utc: datetime,
    decision_bar_utc: datetime,
    profile: SnapshotProfile,
    scope: SnapshotEntityScope,
) -> frozenset[str]:
    if rule is None:
        return frozenset()
    eligible = [
        candidate
        for candidate in candidates
        if _candidate_rejection_reason(
            candidate,
            rule,
            entity_id=entity_id,
            decision_time_utc=decision_time_utc,
            decision_bar_utc=decision_bar_utc,
            profile=profile,
            scope=scope,
        )
        is None
    ]
    grouped: dict[tuple[str, ...], list[StateEnvelope]] = {}
    for candidate in eligible:
        grouped.setdefault(_effective_key(candidate), []).append(candidate)
    selected = [max(group, key=candidate_tie_key) for group in grouped.values()]
    return frozenset(str(getattr(candidate, "theme_id")) for candidate in selected)


def evaluate_requirements(
    candidates: Sequence[StateEnvelope],
    *,
    entity_id: str,
    decision_time: datetime,
    decision_bar: datetime,
    profile: SnapshotProfile,
    scope: SnapshotEntityScope | None = None,
) -> RequirementEvaluation:
    """Select causal states from an already-loaded candidate pool.

    This function performs no I/O and has no mutation.  It is intentionally
    the only place that decides whether a candidate is eligible.
    """

    decision_time_utc = _aware(decision_time, "decision_time")
    decision_bar_utc = _aware(decision_bar, "decision_bar")
    if decision_bar_utc > decision_time_utc:
        raise ValueError("decision_bar must not be after decision_time")
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValueError("entity_id must be a non-empty string")
    scope = scope or SnapshotEntityScope()

    by_type: dict[StateType, list[StateEnvelope]] = {rule.state_type: [] for rule in profile.rules}
    for candidate in candidates:
        if not isinstance(candidate, StateEnvelope):
            raise TypeError("candidate pool must contain StateEnvelope values")
        if candidate.state_type in by_type:
            by_type[candidate.state_type].append(candidate)

    membership_rule = next(
        (rule for rule in profile.rules if rule.state_type is StateType.THEME_MEMBERSHIP),
        None,
    )
    selected_theme_ids = _selected_ticker_theme_ids(
        by_type.get(StateType.THEME_MEMBERSHIP, ()),
        rule=membership_rule,
        entity_id=entity_id,
        decision_time_utc=decision_time_utc,
        decision_bar_utc=decision_bar_utc,
        profile=profile,
        scope=scope,
    )

    selected: list[StateEnvelope] = []
    results: list[FreshnessResult] = []
    rejected: list[RejectedCandidate] = []
    stale_inputs: list[str] = []
    missing_inputs: list[str] = []
    valid = True

    for rule in profile.rules:
        eligible: list[StateEnvelope] = []
        rule_rejections: list[tuple[StateEnvelope, str]] = []
        for candidate in by_type[rule.state_type]:
            reason_code = _candidate_rejection_reason(
                candidate,
                rule,
                entity_id=entity_id,
                decision_time_utc=decision_time_utc,
                decision_bar_utc=decision_bar_utc,
                profile=profile,
                scope=scope,
                selected_theme_ids=selected_theme_ids,
            )
            if reason_code is not None:
                rule_rejections.append((candidate, reason_code))
                continue
            eligible.append(candidate)

        selected_for_rule: list[StateEnvelope] = []
        if rule.state_type in _SINGLETON_TYPES:
            if eligible:
                selected_for_rule = [max(eligible, key=candidate_tie_key)]
        else:
            grouped: dict[tuple[str, ...], list[StateEnvelope]] = {}
            for candidate in eligible:
                grouped.setdefault(_effective_key(candidate), []).append(candidate)
            for group in grouped.values():
                selected_for_rule.append(max(group, key=candidate_tie_key))
            selected_for_rule.sort(key=_selection_key)

        selected.extend(selected_for_rule)
        selected_ids = {candidate.state_id for candidate in selected_for_rule}
        for candidate in eligible:
            if candidate.state_id not in selected_ids:
                rule_rejections.append((candidate, "NOT_SELECTED"))

        for candidate, reason_code in rule_rejections:
            rejected.append(
                RejectedCandidate(
                    state_id=candidate.state_id,
                    state_type=candidate.state_type,
                    entity_id=candidate.entity_id,
                    reason_code=reason_code,
                )
            )

        if selected_for_rule:
            selected_for_rule.sort(key=candidate_tie_key, reverse=True)
            selected_candidate = selected_for_rule[0]
            results.append(
                FreshnessResult(
                    state_type=rule.state_type,
                    entity_id=selected_candidate.entity_id,
                    required=rule.required,
                    status="FRESH",
                    selected_state_id=selected_candidate.state_id,
                    age_seconds=_freshness_age_seconds(decision_time_utc, selected_candidate),
                    max_age_seconds=float(rule.max_age.total_seconds()),
                    reason_code="FRESH",
                )
            )
            continue

        codes = [reason_code for _candidate, reason_code in rule_rejections]
        status, reason_code = _status_for_rejections(rule, codes)
        age_candidates = [
            candidate
            for candidate in by_type[rule.state_type]
            if _entity_matches(candidate, rule, entity_id=entity_id, scope=scope)
            and _aware(candidate.available_at, "state.available_at") <= decision_time_utc
        ]
        age_seconds = (
            min((_freshness_age_seconds(decision_time_utc, candidate) for candidate in age_candidates), default=None)
        )
        results.append(
            FreshnessResult(
                state_type=rule.state_type,
                entity_id=(
                    _expected_entities(rule.state_type, entity_id=entity_id, scope=scope) or (entity_id,)
                )[0],
                required=rule.required,
                status=status,
                selected_state_id=None,
                age_seconds=age_seconds,
                max_age_seconds=float(rule.max_age.total_seconds()),
                reason_code=reason_code,
            )
        )
        key = _requirement_key(rule)
        if status == "STALE":
            stale_inputs.append(key)
        elif status == "MISSING":
            missing_inputs.append(key)
        if rule.required or rule.fallback is MissingStateAction.REJECT:
            valid = False

    rejected.sort(key=lambda item: (item.state_type.value, item.entity_id, str(item.state_id), item.reason_code))
    results.sort(key=lambda item: item.state_type.value)
    return RequirementEvaluation(
        selected_states=tuple(selected),
        requirement_results=tuple(results),
        rejected_candidates=tuple(rejected),
        stale_inputs=tuple(dict.fromkeys(stale_inputs)),
        missing_inputs=tuple(dict.fromkeys(missing_inputs)),
        valid=valid,
    )


# Short aliases make the pure selector easy to discover for tests and callers.
select_causal_states = evaluate_requirements
select_candidates = evaluate_requirements


__all__ = [
    "RequirementEvaluation",
    "SnapshotEntityScope",
    "candidate_tie_key",
    "decision_session",
    "evaluate_requirements",
    "producer_version",
    "select_candidates",
    "select_causal_states",
]
