from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any

from multi_ticker_swing.live.scanner import Signal


@dataclass(frozen=True)
class RiskProfilePolicyConfig:
    enabled: bool = True
    soft_qqq_ret_16: float = 0.0
    hard_qqq_ret_16: float = -0.01
    late_short_floor_qqq_ret_16: float = -0.02
    require_short_qqq_relative_weakness: bool = True
    block_high_beta_longs_in_balanced: bool = True
    block_high_beta_longs_in_defensive: bool = True
    require_long_qqq_relative_strength_in_balanced: bool = True
    require_long_qqq_relative_strength_in_defensive: bool = True
    high_beta_bucket_min: float = 2.0
    beta_like_spy_min: float = 1.2


@dataclass(frozen=True)
class RiskProfileDecision:
    allowed: bool
    profile: str
    reason: str
    details: dict[str, Any]


def config_from_env() -> RiskProfilePolicyConfig:
    def _bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    return RiskProfilePolicyConfig(
        enabled=_bool("MULTITICKER_RISK_PROFILE_POLICY", True),
        soft_qqq_ret_16=_float("MULTITICKER_RISK_SOFT_QQQ_RET_16", 0.0),
        hard_qqq_ret_16=_float("MULTITICKER_RISK_HARD_QQQ_RET_16", -0.01),
        late_short_floor_qqq_ret_16=_float("MULTITICKER_RISK_LATE_SHORT_FLOOR_QQQ_RET_16", -0.02),
        require_short_qqq_relative_weakness=_bool("MULTITICKER_RISK_REQUIRE_SHORT_QQQ_RS_WEAK", True),
        block_high_beta_longs_in_balanced=_bool("MULTITICKER_RISK_BLOCK_HIGH_BETA_LONGS_BALANCED", True),
        block_high_beta_longs_in_defensive=_bool("MULTITICKER_RISK_BLOCK_HIGH_BETA_LONGS_DEFENSIVE", True),
        require_long_qqq_relative_strength_in_balanced=_bool("MULTITICKER_RISK_REQUIRE_LONG_QQQ_RS_BALANCED", True),
        require_long_qqq_relative_strength_in_defensive=_bool("MULTITICKER_RISK_REQUIRE_LONG_QQQ_RS_DEFENSIVE", True),
        high_beta_bucket_min=_float("MULTITICKER_RISK_HIGH_BETA_BUCKET_MIN", 2.0),
        beta_like_spy_min=_float("MULTITICKER_RISK_BETA_LIKE_SPY_MIN", 1.2),
    )


class RiskProfilePolicy:
    def __init__(self, config: RiskProfilePolicyConfig | None = None) -> None:
        self.config = config or config_from_env()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self.config)

    def evaluate(self, signal: Signal) -> RiskProfileDecision:
        cfg = self.config
        features = signal.features or {}
        qqq_ret_16 = _num(features.get("qqq_ret_16"), 0.0)
        rel_str_qqq_4 = _num(features.get("rel_str_qqq_4"), 0.0)
        beta_bucket = _num(features.get("stock_beta_bucket"), 0.0)
        beta_like_spy = _num(features.get("beta_like_spy_64"), 1.0)
        high_beta = beta_bucket >= cfg.high_beta_bucket_min or beta_like_spy >= cfg.beta_like_spy_min
        profile = _profile(qqq_ret_16, cfg)
        details = {
            "policy": "risk_profile_v1",
            "enabled": bool(cfg.enabled),
            "profile": profile,
            "qqq_ret_16": qqq_ret_16,
            "rel_str_qqq_4": rel_str_qqq_4,
            "stock_beta_bucket": beta_bucket,
            "beta_like_spy_64": beta_like_spy,
            "is_high_beta": bool(high_beta),
        }
        if not cfg.enabled:
            return RiskProfileDecision(True, profile, "risk_profile_policy_disabled", details)

        if signal.direction < 0:
            if cfg.require_short_qqq_relative_weakness and rel_str_qqq_4 >= 0.0:
                return RiskProfileDecision(False, profile, "risk_short_requires_qqq_relative_weakness", details)
            if qqq_ret_16 <= cfg.late_short_floor_qqq_ret_16:
                return RiskProfileDecision(False, profile, "risk_late_short_after_qqq_drop", details)
            return RiskProfileDecision(True, profile, "risk_short_allowed", details)

        if signal.direction > 0:
            if profile == "defensive":
                if cfg.block_high_beta_longs_in_defensive and high_beta:
                    return RiskProfileDecision(False, profile, "risk_defensive_blocks_high_beta_long", details)
                if cfg.require_long_qqq_relative_strength_in_defensive and rel_str_qqq_4 <= 0.0:
                    return RiskProfileDecision(False, profile, "risk_defensive_long_requires_qqq_relative_strength", details)
            if profile == "balanced":
                if cfg.block_high_beta_longs_in_balanced and high_beta:
                    return RiskProfileDecision(False, profile, "risk_balanced_blocks_high_beta_long", details)
                if cfg.require_long_qqq_relative_strength_in_balanced and rel_str_qqq_4 <= 0.0:
                    return RiskProfileDecision(False, profile, "risk_balanced_long_requires_qqq_relative_strength", details)
            return RiskProfileDecision(True, profile, "risk_long_allowed", details)

        return RiskProfileDecision(False, profile, "risk_zero_direction", details)


def _profile(qqq_ret_16: float, cfg: RiskProfilePolicyConfig) -> str:
    if qqq_ret_16 <= cfg.hard_qqq_ret_16:
        return "defensive"
    if qqq_ret_16 <= cfg.soft_qqq_ret_16:
        return "balanced"
    return "aggressive"


def _num(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default
