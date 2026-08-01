"""Versioned freshness requirements for causal context snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from core.nervous_system.contracts.enums import MissingStateAction, StateType


@dataclass(frozen=True)
class FreshnessRule:
    state_type: StateType
    required: bool
    max_age: timedelta
    fallback: MissingStateAction

    def __post_init__(self) -> None:
        if not isinstance(self.state_type, StateType):
            raise TypeError("state_type must be a StateType")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")
        if not isinstance(self.max_age, timedelta) or self.max_age <= timedelta(0):
            raise ValueError("max_age must be a positive timedelta")
        if not isinstance(self.fallback, MissingStateAction):
            raise TypeError("fallback must be a MissingStateAction")


@dataclass(frozen=True)
class SnapshotProfile:
    profile_id: str
    rules: tuple[FreshnessRule, ...]
    dealer_allowed: bool = True
    dealer_capture_after_et: time | None = None
    market_session_lag: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        if "@" not in self.profile_id:
            raise ValueError("profile_id must be versioned with '@'")
        normalized_rules = tuple(self.rules)
        if not normalized_rules:
            raise ValueError("snapshot profile must contain at least one rule")
        if any(not isinstance(rule, FreshnessRule) for rule in normalized_rules):
            raise TypeError("rules must contain FreshnessRule values")
        state_types = [rule.state_type for rule in normalized_rules]
        if len(set(state_types)) != len(state_types):
            raise ValueError("snapshot profile cannot repeat a state_type")
        if type(self.dealer_allowed) is not bool:
            raise TypeError("dealer_allowed must be a bool")
        if self.dealer_capture_after_et is not None:
            if not isinstance(self.dealer_capture_after_et, time):
                raise TypeError("dealer_capture_after_et must be a time or None")
            if self.dealer_capture_after_et.tzinfo is not None:
                raise ValueError("dealer_capture_after_et must be a naive ET wall-clock time")
            if not self.dealer_allowed:
                raise ValueError("dealer_capture_after_et requires dealer_allowed=True")
        if type(self.market_session_lag) is not int:
            raise TypeError("market_session_lag must be an int")
        if self.market_session_lag < 0:
            raise ValueError("market_session_lag must be non-negative")
        object.__setattr__(self, "rules", normalized_rules)

    @property
    def content_hash(self) -> str:
        return profile_content_hash(self)

    @property
    def profile_hash(self) -> str:
        return self.content_hash


def canonical_profile_payload(profile: SnapshotProfile) -> dict[str, object]:
    """Return the stable, JSON-compatible profile representation."""

    return {
        "profile_id": profile.profile_id,
        "policy": {
            "dealer_allowed": profile.dealer_allowed,
            "dealer_capture_after_et": (
                profile.dealer_capture_after_et.isoformat(timespec="microseconds")
                if profile.dealer_capture_after_et is not None
                else None
            ),
            "market_session_lag": profile.market_session_lag,
        },
        "rules": [
            {
                "state_type": rule.state_type.value,
                "required": rule.required,
                "max_age_microseconds": rule.max_age // timedelta(microseconds=1),
                "fallback": rule.fallback.value,
            }
            for rule in profile.rules
        ],
    }


def canonical_profile_json(profile: SnapshotProfile) -> str:
    return json.dumps(
        canonical_profile_payload(profile),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def profile_content_hash(profile: SnapshotProfile) -> str:
    return hashlib.sha256(canonical_profile_json(profile).encode("utf-8")).hexdigest()


profile_hash = profile_content_hash


# Versioned MVP defaults.  Session-derived evidence uses a 96-hour elapsed
# allowance so a prior Friday/holiday session remains eligible on Monday;
# profile IDs carry the version of these policy values into snapshot hashes.
MVP_POLICY_DEFAULTS: Mapping[StateType, timedelta] = MappingProxyType(
    {
        StateType.TICKER: timedelta(hours=6),
        StateType.PORTFOLIO: timedelta(hours=24),
        StateType.DEALER: timedelta(hours=4),
        StateType.MARKET: timedelta(hours=96),
        StateType.SECTOR: timedelta(hours=96),
        StateType.READINESS: timedelta(hours=96),
        StateType.THEME: timedelta(hours=96),
        StateType.THEME_MEMBERSHIP: timedelta(hours=96),
        StateType.CATALYST_EVENT: timedelta(hours=96),
        StateType.CATALYST_PRESSURE: timedelta(hours=96),
    }
)

_REQUIRED_RULES = (
    FreshnessRule(
        StateType.TICKER,
        True,
        MVP_POLICY_DEFAULTS[StateType.TICKER],
        MissingStateAction.REJECT,
    ),
    FreshnessRule(
        StateType.MARKET,
        True,
        MVP_POLICY_DEFAULTS[StateType.MARKET],
        MissingStateAction.REJECT,
    ),
    FreshnessRule(
        StateType.SECTOR,
        True,
        MVP_POLICY_DEFAULTS[StateType.SECTOR],
        MissingStateAction.REJECT,
    ),
    FreshnessRule(
        StateType.PORTFOLIO,
        True,
        MVP_POLICY_DEFAULTS[StateType.PORTFOLIO],
        MissingStateAction.REJECT,
    ),
    FreshnessRule(
        StateType.READINESS,
        True,
        MVP_POLICY_DEFAULTS[StateType.READINESS],
        MissingStateAction.REJECT,
    ),
)

_OPTIONAL_RULES = (
    FreshnessRule(
        StateType.THEME_MEMBERSHIP,
        False,
        MVP_POLICY_DEFAULTS[StateType.THEME_MEMBERSHIP],
        MissingStateAction.WARN,
    ),
    FreshnessRule(
        StateType.THEME,
        False,
        MVP_POLICY_DEFAULTS[StateType.THEME],
        MissingStateAction.WARN,
    ),
    FreshnessRule(
        StateType.CATALYST_EVENT,
        False,
        MVP_POLICY_DEFAULTS[StateType.CATALYST_EVENT],
        MissingStateAction.WARN,
    ),
    FreshnessRule(
        StateType.CATALYST_PRESSURE,
        False,
        MVP_POLICY_DEFAULTS[StateType.CATALYST_PRESSURE],
        MissingStateAction.WARN,
    ),
    FreshnessRule(
        StateType.DEALER,
        False,
        MVP_POLICY_DEFAULTS[StateType.DEALER],
        MissingStateAction.WARN,
    ),
)


META_4H_1420_PROFILE = SnapshotProfile(
    profile_id="meta_4h_1420@1",
    rules=_REQUIRED_RULES + _OPTIONAL_RULES,
    dealer_allowed=False,
    dealer_capture_after_et=None,
    market_session_lag=1,
)
META_4H_1620_PROFILE = SnapshotProfile(
    profile_id="meta_4h_1620@1",
    rules=_REQUIRED_RULES + _OPTIONAL_RULES,
    dealer_allowed=True,
    dealer_capture_after_et=time(14, 20),
    market_session_lag=1,
)

# Public aliases keep the registry discoverable without making callers depend
# on one spelling of the profile constants.
META_4H_1420 = META_4H_1420_PROFILE
META_4H_1620 = META_4H_1620_PROFILE
SNAPSHOT_PROFILES: Mapping[str, SnapshotProfile] = MappingProxyType(
    {
        META_4H_1420_PROFILE.profile_id: META_4H_1420_PROFILE,
        META_4H_1620_PROFILE.profile_id: META_4H_1620_PROFILE,
    }
)
FRESHNESS_PROFILES = SNAPSHOT_PROFILES
PROFILE_REGISTRY = SNAPSHOT_PROFILES


def get_snapshot_profile(profile_id: str) -> SnapshotProfile:
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be a non-empty string")
    try:
        return SNAPSHOT_PROFILES[profile_id]
    except KeyError as exc:
        raise KeyError(f"unknown snapshot profile: {profile_id}") from exc


__all__ = [
    "FRESHNESS_PROFILES",
    "FreshnessRule",
    "META_4H_1420",
    "META_4H_1420_PROFILE",
    "META_4H_1620",
    "META_4H_1620_PROFILE",
    "MVP_POLICY_DEFAULTS",
    "PROFILE_REGISTRY",
    "SNAPSHOT_PROFILES",
    "SnapshotProfile",
    "canonical_profile_json",
    "canonical_profile_payload",
    "get_snapshot_profile",
    "profile_content_hash",
    "profile_hash",
]
