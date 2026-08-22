from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from strategies.multi_ticker_swing.live.scanner import Signal


_REPO_ROOT = Path(__file__).resolve().parents[3]
# Anchored to REPO ROOT (not CWD): if the server isn't started from the repo
# root, a CWD-relative path silently resolves to a nonexistent file, and
# EVCalibrationTable.from_path() then just returns an empty table with no
# error — every entry gets tagged ev_bucket_unseen and sizing/blocks run on
# defaults. Sibling modules (momentum_expansion, meta_ranker, HTF) anchor the
# same way.
DEFAULT_CALIBRATION_PATH = _REPO_ROOT / "Data/inference/multi_ticker_swing/signal_ev_calibration.json"


@dataclass(frozen=True)
class SignalPolicyConfig:
    enabled: bool = True
    enforce: bool = False
    apply_sizing: bool = False
    calibration_path: Path = DEFAULT_CALIBRATION_PATH
    min_bucket_trades: int = 20
    min_bucket_win_rate: float = 0.45
    min_bucket_avg_return: float = 0.0
    max_contracts_per_trade: int = 1
    max_option_breakeven_to_expected_move: float = 2.0
    # Measured over 492 filled option entries in UI/swing_audit (2026-05..08):
    # the median quoted spread is 12.1% of mid and 63% of entries exceed 10%, so
    # an 18% gate kept 87% of entries and avoided $1,809 of $22,025 in
    # cumulative fill-versus-mid slippage — it was very nearly inert. Slippage is
    # superlinear in the spread, concentrated in a wide-spread tail: mean $45 per
    # entry against a median of $7, worst single entry $437. Modelled over the
    # same sample, this gate keeps 49% of entries and avoids $15,000 (68%) of the
    # slippage. 10% would avoid 81% but keep only 37%.
    #
    # Unlike a ladder change, the effect does not depend on fill-engine fidelity:
    # a contract that is never bought cannot cost its spread, whether the fills
    # are simulated or real.
    max_entry_spread_pct_mid: float = 0.12


@dataclass(frozen=True)
class CalibrationBucket:
    key: str
    count: int
    win_rate: float | None = None
    avg_return: float | None = None
    max_drawdown: float | None = None
    avg_holding_minutes: float | None = None
    option_return_estimate: float | None = None
    recent_ev_multiplier: float | None = None


@dataclass(frozen=True)
class SignalPolicyDecision:
    enabled: bool
    action: str
    reason: str
    score_bucket: str
    side: str
    regime: str
    size_multiplier: float
    recommended_qty: int
    confidence_multiplier: float
    regime_multiplier: float
    liquidity_multiplier: float
    reasons: tuple[str, ...]
    inputs: dict[str, Any]
    calibration: dict[str, Any] | None = None
    option_translation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_env() -> SignalPolicyConfig:
    def _bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    return SignalPolicyConfig(
        enabled=_bool("MULTITICKER_SIGNAL_POLICY", True),
        enforce=_bool("MULTITICKER_SIGNAL_POLICY_ENFORCE", False),
        apply_sizing=_bool("MULTITICKER_SIGNAL_POLICY_APPLY_SIZING", False),
        calibration_path=Path(os.getenv("MULTITICKER_SIGNAL_POLICY_CALIBRATION", str(DEFAULT_CALIBRATION_PATH))),
        min_bucket_trades=_int("MULTITICKER_SIGNAL_POLICY_MIN_BUCKET_TRADES", 20),
        min_bucket_win_rate=_float("MULTITICKER_SIGNAL_POLICY_MIN_BUCKET_WIN_RATE", 0.45),
        min_bucket_avg_return=_float("MULTITICKER_SIGNAL_POLICY_MIN_BUCKET_AVG_RETURN", 0.0),
        max_contracts_per_trade=_int("MULTITICKER_SIGNAL_POLICY_MAX_CONTRACTS", 1),
        max_option_breakeven_to_expected_move=_float("MULTITICKER_SIGNAL_POLICY_MAX_BREAKEVEN_TO_EXPECTED_MOVE", 2.0),
        max_entry_spread_pct_mid=_float("MULTITICKER_SIGNAL_POLICY_MAX_SPREAD_PCT_MID", 0.18),
    )


class EVCalibrationTable:
    def __init__(self, buckets: dict[str, CalibrationBucket] | None = None) -> None:
        self._buckets = buckets or {}

    @classmethod
    def from_path(cls, path: Path) -> "EVCalibrationTable":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        rows = raw.get("buckets", raw) if isinstance(raw, dict) else raw
        buckets: dict[str, CalibrationBucket] = {}
        if isinstance(rows, dict):
            iterable = [dict(v, key=k) if isinstance(v, dict) else {"key": k} for k, v in rows.items()]
        else:
            iterable = rows if isinstance(rows, list) else []
        for row in iterable:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key") or _bucket_key_from_row(row))
            count = _as_int(row.get("count") or row.get("n") or row.get("trades"), 0)
            buckets[key] = CalibrationBucket(
                key=key,
                count=count,
                win_rate=_nullable_float(row.get("win_rate")),
                avg_return=_nullable_float(row.get("avg_return")),
                max_drawdown=_nullable_float(row.get("max_drawdown")),
                avg_holding_minutes=_nullable_float(row.get("avg_holding_minutes") or row.get("holding_minutes")),
                option_return_estimate=_nullable_float(row.get("option_return_estimate")),
                recent_ev_multiplier=_nullable_float(row.get("recent_ev_multiplier")),
            )
        return cls(buckets)

    def lookup(self, *, module: str, side: str, regime: str, score_bucket: str) -> CalibrationBucket | None:
        candidates = (
            f"{module}|{side}|{regime}|{score_bucket}",
            f"{module}|{side}|*|{score_bucket}",
            f"{module}|*|{regime}|{score_bucket}",
            f"{module}|*|*|{score_bucket}",
            f"*|{side}|{regime}|{score_bucket}",
            f"*|{side}|*|{score_bucket}",
            f"*|*|{regime}|{score_bucket}",
            f"*|*|*|{score_bucket}",
            score_bucket,
        )
        for key in candidates:
            if key in self._buckets:
                return self._buckets[key]
        return None


class SignalPolicyLayer:
    def __init__(
        self,
        config: SignalPolicyConfig | None = None,
        calibration: EVCalibrationTable | None = None,
    ) -> None:
        self.config = config or config_from_env()
        self._calibration = calibration if calibration is not None else EVCalibrationTable.from_path(self.config.calibration_path)

    def snapshot(self) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "calibration_path": str(self.config.calibration_path),
        }

    def evaluate_signal(self, signal: Signal, *, module: str = "multi_ticker_swing") -> SignalPolicyDecision:
        side = "long" if int(signal.direction) > 0 else "short"
        features = signal.features or {}
        qqq_ret_16 = _num(features.get("qqq_ret_16"), 0.0)
        rel_str_qqq_4 = _num(features.get("rel_str_qqq_4"), 0.0)
        rel_str_spy_16 = _num(features.get("rel_str_spy_16"), 0.0)
        beta_bucket = _num(features.get("stock_beta_bucket"), 0.0)
        beta_like_spy = _num(features.get("beta_like_spy_64"), 1.0)
        range_pos_20 = _num(features.get("range_pos_20"), 0.5)
        daily_range_pos_20 = _num(features.get("daily_range_pos_20"), 0.5)
        zscore_close_64 = _num(features.get("zscore_close_64"), 0.0)
        regime = _regime(qqq_ret_16)
        score_bucket = score_bucket_for(float(signal.p_dir))

        reasons: list[str] = []
        action = "ALLOW"
        confidence_mult = _confidence_multiplier(float(signal.p_dir))
        regime_mult = _regime_multiplier(side=side, regime=regime, qqq_ret_16=qqq_ret_16)
        liquidity_mult = 1.0

        if side == "long" and regime == "defensive":
            reasons.append("bad_tape_for_longs")
            action = _worse_action(action, "REDUCE")
        if side == "short" and regime == "aggressive":
            reasons.append("risk_on_tape_for_shorts")
            action = _worse_action(action, "REDUCE")
        if side == "long" and (daily_range_pos_20 >= 0.90 or range_pos_20 >= 0.95) and zscore_close_64 >= 2.0:
            reasons.append("overextended_entry")
            action = _worse_action(action, "REDUCE")
            regime_mult = min(regime_mult, 0.50)
        if side == "long" and rel_str_spy_16 < -5.0:
            reasons.append("weak_vs_spy")
            action = _worse_action(action, "REDUCE")
        if side == "short" and rel_str_qqq_4 > 0.0:
            reasons.append("short_without_qqq_relative_weakness")
            action = _worse_action(action, "REDUCE")
        if side == "long" and (beta_bucket >= 2.0 or beta_like_spy >= 1.2) and regime == "defensive":
            reasons.append("high_beta_long_in_defensive_regime")
            action = _worse_action(action, "BLOCK")

        bucket = self._calibration.lookup(module=module, side=side, regime=regime, score_bucket=score_bucket)
        calibration_payload = asdict(bucket) if bucket is not None else None
        calibration_mult = 1.0
        if bucket is not None and bucket.count >= self.config.min_bucket_trades:
            if bucket.win_rate is not None and bucket.win_rate < self.config.min_bucket_win_rate:
                reasons.append("ev_bucket_low_win_rate")
                action = _worse_action(action, "BLOCK")
            if bucket.avg_return is not None and bucket.avg_return < self.config.min_bucket_avg_return:
                reasons.append("ev_bucket_negative_return")
                action = _worse_action(action, "BLOCK")
            if bucket.recent_ev_multiplier is not None and math.isfinite(bucket.recent_ev_multiplier):
                calibration_mult = max(0.0, min(1.5, float(bucket.recent_ev_multiplier)))
        elif bucket is None:
            reasons.append("ev_bucket_unseen")

        if not self.config.enabled:
            action = "ALLOW"
            reasons.append("signal_policy_disabled")

        if action == "BLOCK":
            size_mult = 0.0
        else:
            size_mult = confidence_mult * regime_mult * liquidity_mult * calibration_mult
            if action == "REDUCE":
                size_mult = min(size_mult, 0.50)
            size_mult = max(0.0, min(1.50, size_mult))

        recommended_qty = _recommended_qty(size_mult, max_contracts=self.config.max_contracts_per_trade)
        reason = reasons[0] if reasons else "signal_policy_allowed"
        return SignalPolicyDecision(
            enabled=bool(self.config.enabled),
            action=action,
            reason=reason,
            score_bucket=score_bucket,
            side=side,
            regime=regime,
            size_multiplier=float(size_mult),
            recommended_qty=recommended_qty,
            confidence_multiplier=float(confidence_mult),
            regime_multiplier=float(regime_mult),
            liquidity_multiplier=float(liquidity_mult),
            reasons=tuple(reasons),
            inputs={
                "p_dir": float(signal.p_dir),
                "ev_score": float(signal.ev_score),
                "qqq_ret_16": qqq_ret_16,
                "rel_str_qqq_4": rel_str_qqq_4,
                "rel_str_spy_16": rel_str_spy_16,
                "stock_beta_bucket": beta_bucket,
                "beta_like_spy_64": beta_like_spy,
                "range_pos_20": range_pos_20,
                "daily_range_pos_20": daily_range_pos_20,
                "zscore_close_64": zscore_close_64,
            },
            calibration=calibration_payload,
        )

    def with_entry_context(
        self,
        decision: SignalPolicyDecision,
        *,
        signal: Signal,
        option_meta: dict[str, Any],
        quote_meta: dict[str, Any],
    ) -> SignalPolicyDecision:
        spread_pct = _nullable_float((quote_meta or {}).get("spread_pct_mid"))
        liquidity_mult = _liquidity_multiplier(spread_pct, self.config.max_entry_spread_pct_mid)
        action = decision.action
        reasons = list(decision.reasons)
        if spread_pct is None:
            reasons.append("option_spread_missing")
        elif spread_pct > self.config.max_entry_spread_pct_mid:
            reasons.append("option_spread_too_wide")
            action = _worse_action(action, "BLOCK")

        option_translation = _option_translation(
            signal=signal,
            option_meta=option_meta or {},
            quote_meta=quote_meta or {},
            max_breakeven_ratio=self.config.max_option_breakeven_to_expected_move,
            spread_ok=(spread_pct is not None and spread_pct <= self.config.max_entry_spread_pct_mid),
        )
        if option_translation["route"] == "stock_only":
            reasons.append("options_breakeven_exceeds_expected_move")
            action = _worse_action(action, "REDUCE")
        elif option_translation["route"] == "skip_options":
            reasons.append("options_translation_skip")
            action = _worse_action(action, "BLOCK")

        if action == "BLOCK":
            size_mult = 0.0
        else:
            size_mult = (
                decision.confidence_multiplier
                * decision.regime_multiplier
                * liquidity_mult
                * _calibration_multiplier(decision.calibration, min_bucket_trades=self.config.min_bucket_trades)
            )
            if action == "REDUCE":
                size_mult = min(size_mult, 0.50)
            size_mult = max(0.0, min(1.50, size_mult))

        return SignalPolicyDecision(
            enabled=decision.enabled,
            action=action,
            reason=reasons[0] if reasons else decision.reason,
            score_bucket=decision.score_bucket,
            side=decision.side,
            regime=decision.regime,
            size_multiplier=float(size_mult),
            recommended_qty=_recommended_qty(size_mult, max_contracts=self.config.max_contracts_per_trade),
            confidence_multiplier=decision.confidence_multiplier,
            regime_multiplier=decision.regime_multiplier,
            liquidity_multiplier=float(liquidity_mult),
            reasons=tuple(reasons),
            inputs=decision.inputs,
            calibration=decision.calibration,
            option_translation=option_translation,
        )


def score_bucket_for(score: float) -> str:
    if not math.isfinite(score):
        return "p_dir_nan"
    clipped = max(0.0, min(1.0, float(score)))
    lo = math.floor(clipped / 0.05) * 0.05
    hi = min(1.0, lo + 0.05)
    if clipped >= 1.0:
        lo, hi = 0.95, 1.0
    return f"p_dir_{lo:.2f}_{hi:.2f}"


def _option_translation(
    *,
    signal: Signal,
    option_meta: dict[str, Any],
    quote_meta: dict[str, Any],
    max_breakeven_ratio: float,
    spread_ok: bool,
) -> dict[str, Any]:
    ask = _nullable_float(quote_meta.get("ask"))
    mid = _nullable_float(quote_meta.get("mid"))
    premium = ask if ask is not None and ask > 0 else mid
    underlying = _nullable_float(option_meta.get("underlying_price_at_selection"))
    strike = _nullable_float(option_meta.get("strike"))
    dte = _as_int(option_meta.get("dte"), -1)
    expected_move = max(abs(float(signal.ev_score or 0.0)), 0.005)
    route = "call_option" if signal.direction > 0 else "put_option"
    breakeven_move = None
    breakeven_ratio = None

    if not spread_ok:
        route = "skip_options"
    elif premium is None or underlying is None or strike is None or underlying <= 0:
        route = "stock_only"
    else:
        if signal.direction > 0:
            breakeven = strike + premium
            breakeven_move = max(0.0, (breakeven - underlying) / underlying)
        else:
            breakeven = strike - premium
            breakeven_move = max(0.0, (underlying - breakeven) / underlying)
        breakeven_ratio = breakeven_move / expected_move if expected_move > 0 else None
        if breakeven_ratio is not None and breakeven_ratio > max_breakeven_ratio:
            route = "stock_only"
        elif dte == 0 and expected_move < 0.01:
            route = "stock_only"

    if route == "call_option" and breakeven_ratio is not None and breakeven_ratio > 1.25:
        route = "call_debit_spread"

    return {
        "route": route,
        "expected_stock_move_pct": float(expected_move),
        "option_breakeven_move_pct": float(breakeven_move) if breakeven_move is not None else None,
        "breakeven_to_expected_move": float(breakeven_ratio) if breakeven_ratio is not None else None,
        "dte": dte if dte >= 0 else None,
        "spread_ok": bool(spread_ok),
    }


def _regime(qqq_ret_16: float) -> str:
    if qqq_ret_16 <= -0.01:
        return "defensive"
    if qqq_ret_16 <= 0.0:
        return "balanced"
    return "aggressive"


def _confidence_multiplier(p_dir: float) -> float:
    if p_dir >= 0.80:
        return 1.25
    if p_dir >= 0.70:
        return 1.00
    if p_dir >= 0.60:
        return 0.75
    return 0.50


def _regime_multiplier(*, side: str, regime: str, qqq_ret_16: float) -> float:
    if side == "long":
        return {"aggressive": 1.0, "balanced": 0.50, "defensive": 0.25}.get(regime, 0.50)
    if regime == "aggressive":
        return 0.50
    if qqq_ret_16 <= -0.02:
        return 0.50
    return 1.0


def _liquidity_multiplier(spread_pct: float | None, max_spread: float) -> float:
    if spread_pct is None:
        return 0.50
    if spread_pct > max_spread:
        return 0.0
    if spread_pct <= max_spread * 0.50:
        return 1.0
    return 0.75


def _calibration_multiplier(calibration: dict[str, Any] | None, *, min_bucket_trades: int = 0) -> float:
    if not calibration:
        return 1.0
    # Same sample-size gate evaluate_signal applies before trusting
    # recent_ev_multiplier — without it, a bucket too small to trust (e.g. 3
    # trades) can still size a trade via with_entry_context.
    if _as_int(calibration.get("count"), 0) < min_bucket_trades:
        return 1.0
    value = _nullable_float(calibration.get("recent_ev_multiplier"))
    if value is None:
        return 1.0
    return max(0.0, min(1.5, value))


def _recommended_qty(size_multiplier: float, *, max_contracts: int) -> int:
    if size_multiplier <= 0.0:
        return 0
    return max(1, min(int(max_contracts), int(math.floor(size_multiplier + 1e-9)) or 1))


def _worse_action(current: str, new: str) -> str:
    rank = {"ALLOW": 0, "REDUCE": 1, "BLOCK": 2}
    return new if rank.get(new, 0) > rank.get(current, 0) else current


def _bucket_key_from_row(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(k, "*"))
        for k in ("module", "side", "regime", "score_bucket")
    )


def _num(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _nullable_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
