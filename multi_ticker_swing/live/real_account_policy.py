from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from multi_ticker_swing.live.scanner import Signal


ET = ZoneInfo("America/New_York")
DEFAULT_STATE_PATH = Path("Data/inference/multi_ticker_swing/real_account_book.json")


@dataclass(frozen=True)
class RealAccountPolicyConfig:
    enabled: bool = False
    calls_only: bool = True
    max_open_positions: int = 2
    max_new_trades_per_day: int = 4
    max_contracts_per_trade: int = 1
    max_premium_per_trade: float = 350.0
    max_open_premium: float = 700.0
    min_account_equity: float = 500.0
    min_cash_after_entry: float = 250.0
    max_spread_pct_mid: float = 0.18
    max_abs_theta: float = 20.0
    min_abs_delta: float = 0.35
    max_abs_delta: float = 0.65
    min_atr_pct: float = 0.010
    min_dte: int = 0
    max_dte: int = 7
    block_after_time: str = "15:15"
    block_0dte_after_time: str = "12:30"
    require_fresh_entry: bool = True
    min_confirmation_body_frac: float = 0.30
    state_path: Path = DEFAULT_STATE_PATH


@dataclass
class RealAccountDecision:
    allowed: bool
    reason: str
    qty: int = 0
    premium_at_risk: float = 0.0
    details: dict[str, Any] | None = None


def config_from_env(*, enabled: bool | None = None, state_path: str | None = None) -> RealAccountPolicyConfig:
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

    return RealAccountPolicyConfig(
        enabled=_bool("MULTITICKER_REAL_ACCOUNT_POLICY", False) if enabled is None else bool(enabled),
        calls_only=_bool("MULTITICKER_REAL_CALLS_ONLY", True),
        max_open_positions=_int("MULTITICKER_REAL_MAX_OPEN_POSITIONS", 2),
        max_new_trades_per_day=_int("MULTITICKER_REAL_MAX_NEW_TRADES_PER_DAY", 4),
        max_contracts_per_trade=_int("MULTITICKER_REAL_MAX_CONTRACTS_PER_TRADE", 1),
        max_premium_per_trade=_float("MULTITICKER_REAL_MAX_PREMIUM_PER_TRADE", 350.0),
        max_open_premium=_float("MULTITICKER_REAL_MAX_OPEN_PREMIUM", 700.0),
        min_account_equity=_float("MULTITICKER_REAL_MIN_ACCOUNT_EQUITY", 500.0),
        min_cash_after_entry=_float("MULTITICKER_REAL_MIN_CASH_AFTER_ENTRY", 250.0),
        max_spread_pct_mid=_float("MULTITICKER_REAL_MAX_SPREAD_PCT_MID", 0.18),
        max_abs_theta=_float("MULTITICKER_REAL_MAX_ABS_THETA", 20.0),
        min_abs_delta=_float("MULTITICKER_REAL_MIN_ABS_DELTA", 0.35),
        max_abs_delta=_float("MULTITICKER_REAL_MAX_ABS_DELTA", 0.65),
        min_atr_pct=_float("MULTITICKER_REAL_MIN_ATR_PCT", 0.010),
        min_dte=_int("MULTITICKER_REAL_MIN_DTE", 0),
        max_dte=_int("MULTITICKER_REAL_MAX_DTE", 7),
        block_after_time=os.getenv("MULTITICKER_REAL_BLOCK_AFTER_TIME", "15:15"),
        block_0dte_after_time=os.getenv("MULTITICKER_REAL_BLOCK_0DTE_AFTER_TIME", "12:30"),
        require_fresh_entry=_bool("MULTITICKER_REAL_REQUIRE_FRESH_ENTRY", True),
        min_confirmation_body_frac=_float("MULTITICKER_REAL_MIN_CONFIRMATION_BODY_FRAC", 0.30),
        state_path=Path(state_path) if state_path else DEFAULT_STATE_PATH,
    )


class RealAccountBookkeeper:
    """Small persistent guardrail for real-money option entries.

    It is intentionally conservative: it only controls new entries. Exits are
    never blocked, because risk controls should not trap an open position.
    """

    def __init__(self, config: RealAccountPolicyConfig) -> None:
        if not isinstance(config.state_path, Path):
            config = RealAccountPolicyConfig(
                **{
                    **asdict(config),
                    "state_path": Path(config.state_path),
                }
            )
        self.config = config
        self._state: dict[str, Any] = self._load_state()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def snapshot(self) -> dict[str, Any]:
        state = self._day_state()
        return {
            "enabled": self.enabled,
            "config": {
                **asdict(self.config),
                "state_path": str(self.config.state_path),
            },
            "state": state,
        }

    def evaluate_entry(
        self,
        *,
        signal: Signal,
        option_symbol: str,
        option_meta: dict[str, Any],
        quote_meta: dict[str, Any],
        limit_prices: list[float],
        open_positions_count: int,
        account: dict[str, Any] | None,
        entry_quality: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> RealAccountDecision:
        if not self.enabled:
            return RealAccountDecision(True, "policy_disabled", qty=1, premium_at_risk=0.0)

        now_et = (now or datetime.now(timezone.utc)).astimezone(ET)
        day_state = self._day_state(now_et)
        max_limit = max([float(x) for x in limit_prices if _finite(x)], default=float("nan"))
        ask = _as_float((quote_meta or {}).get("ask"))
        mid = _as_float((quote_meta or {}).get("mid"))
        spread_pct = _as_float((quote_meta or {}).get("spread_pct_mid"))
        fill_budget_price = max(x for x in [max_limit, ask, mid] if _finite(x)) if any(_finite(x) for x in [max_limit, ask, mid]) else float("nan")

        if signal.direction < 0 and self.config.calls_only:
            return self._deny("real_policy_calls_only", signal=signal, option_symbol=option_symbol)
        body_frac = _as_float((entry_quality or {}).get("body_frac"))
        if (
            _finite(body_frac)
            and self.config.min_confirmation_body_frac > 0
            and body_frac < self.config.min_confirmation_body_frac
        ):
            return self._deny(
                "real_policy_confirmation_body_too_small",
                signal=signal,
                option_symbol=option_symbol,
                confirmation_body_frac=body_frac,
                min_confirmation_body_frac=self.config.min_confirmation_body_frac,
            )
        underlying_price = _as_float(option_meta.get("underlying_price_at_selection"))
        atr = _as_float(signal.atr)
        atr_pct = atr / underlying_price if _finite(atr) and _finite(underlying_price) and underlying_price > 0 else None
        if atr_pct is not None and atr_pct < self.config.min_atr_pct:
            return self._deny("real_policy_atr_pct_too_low", signal=signal, option_symbol=option_symbol, atr_pct=atr_pct)
        if open_positions_count >= self.config.max_open_positions:
            return self._deny("real_policy_max_open_positions", signal=signal, option_symbol=option_symbol, open_positions_count=open_positions_count)
        if int(day_state.get("new_entries", 0)) >= self.config.max_new_trades_per_day:
            return self._deny("real_policy_max_new_trades_per_day", signal=signal, option_symbol=option_symbol)
        if _parse_hhmm(now_et) >= _parse_hhmm_text(self.config.block_after_time):
            return self._deny("real_policy_after_entry_cutoff", signal=signal, option_symbol=option_symbol, now_et=now_et.isoformat())

        dte = _as_int(option_meta.get("dte"))
        if dte is not None:
            if dte < self.config.min_dte:
                return self._deny("real_policy_dte_too_low", signal=signal, option_symbol=option_symbol, dte=dte)
            if dte > self.config.max_dte:
                return self._deny("real_policy_dte_too_high", signal=signal, option_symbol=option_symbol, dte=dte)
            if dte == 0 and _parse_hhmm(now_et) >= _parse_hhmm_text(self.config.block_0dte_after_time):
                return self._deny("real_policy_0dte_after_cutoff", signal=signal, option_symbol=option_symbol, dte=dte, now_et=now_et.isoformat())

        delta = _selected_abs_delta(option_meta)
        if delta is not None and (delta < self.config.min_abs_delta or delta > self.config.max_abs_delta):
            return self._deny("real_policy_delta_out_of_range", signal=signal, option_symbol=option_symbol, selected_abs_delta=delta)
        theta = _abs_theta(option_meta)
        if theta is not None and theta > self.config.max_abs_theta:
            return self._deny("real_policy_theta_too_large", signal=signal, option_symbol=option_symbol, theta=theta)
        if _finite(spread_pct) and spread_pct > self.config.max_spread_pct_mid:
            return self._deny("real_policy_spread_too_wide", signal=signal, option_symbol=option_symbol, spread_pct_mid=spread_pct)
        if not _finite(fill_budget_price) or fill_budget_price <= 0:
            return self._deny("real_policy_missing_entry_price", signal=signal, option_symbol=option_symbol)

        premium_per_contract = fill_budget_price * 100.0
        max_by_trade = int(math.floor(self.config.max_premium_per_trade / premium_per_contract))
        open_premium = float(day_state.get("open_premium", 0.0) or 0.0)
        max_by_open = int(math.floor(max(0.0, self.config.max_open_premium - open_premium) / premium_per_contract))
        max_by_account = self._qty_allowed_by_account(account, premium_per_contract)
        qty = min(self.config.max_contracts_per_trade, max_by_trade, max_by_open, max_by_account)
        if qty < 1:
            return self._deny(
                "real_policy_premium_budget_exceeded",
                signal=signal,
                option_symbol=option_symbol,
                premium_per_contract=premium_per_contract,
                open_premium=open_premium,
                max_by_trade=max_by_trade,
                max_by_open=max_by_open,
                max_by_account=max_by_account,
            )

        premium_at_risk = premium_per_contract * qty
        return RealAccountDecision(
            True,
            "real_policy_allowed",
            qty=int(qty),
            premium_at_risk=float(premium_at_risk),
            details={
                "option_symbol": option_symbol,
                "premium_per_contract": float(premium_per_contract),
                "open_premium_before": float(open_premium),
                "new_entries_today": int(day_state.get("new_entries", 0) or 0),
                "dte": dte,
                "selected_abs_delta": delta,
                "theta": theta,
                "spread_pct_mid": spread_pct if _finite(spread_pct) else None,
                "account": _account_snapshot(account),
                "atr_pct": atr_pct,
            },
        )

    def record_entry(
        self,
        *,
        ticker: str,
        option_symbol: str,
        qty: int,
        premium_at_risk: float,
        reason: str,
        now: datetime | None = None,
    ) -> None:
        if not self.enabled:
            return
        state = self._day_state(now)
        state["new_entries"] = int(state.get("new_entries", 0) or 0) + 1
        state["open_premium"] = float(state.get("open_premium", 0.0) or 0.0) + float(premium_at_risk)
        state.setdefault("entries", []).append(
            {
                "ts": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
                "ticker": str(ticker).upper(),
                "option_symbol": str(option_symbol).upper(),
                "qty": int(qty),
                "premium_at_risk": float(premium_at_risk),
                "reason": reason,
            }
        )
        self._persist()

    def mark_position_closed(
        self,
        *,
        ticker: str,
        option_symbol: str,
        entry_premium: float | None,
        qty: int | None,
        now: datetime | None = None,
    ) -> None:
        if not self.enabled:
            return
        premium = self._recorded_open_premium(option_symbol)
        if premium is None:
            premium = _as_float(entry_premium) * 100.0 * int(qty or 1)
        if not _finite(premium) or premium <= 0:
            premium = 0.0
        state = self._day_state(now)
        state["open_premium"] = max(0.0, float(state.get("open_premium", 0.0) or 0.0) - premium)
        state.setdefault("closes", []).append(
            {
                "ts": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
                "ticker": str(ticker).upper(),
                "option_symbol": str(option_symbol).upper(),
                "released_premium": float(premium),
            }
        )
        self._persist()

    def _recorded_open_premium(self, option_symbol: str) -> float | None:
        target = str(option_symbol or "").strip().upper()
        if not target:
            return None
        state = self._day_state()
        closed_count = sum(
            1
            for row in state.get("closes", [])
            if str(row.get("option_symbol", "")).strip().upper() == target
        )
        matches = [
            row
            for row in state.get("entries", [])
            if str(row.get("option_symbol", "")).strip().upper() == target
        ]
        if len(matches) <= closed_count:
            return None
        premium = _as_float(matches[closed_count].get("premium_at_risk"))
        return float(premium) if _finite(premium) else None

    def _qty_allowed_by_account(self, account: dict[str, Any] | None, premium_per_contract: float) -> int:
        if not account:
            return self.config.max_contracts_per_trade
        equity = _as_float(account.get("equity"))
        cash = _as_float(account.get("cash", account.get("buying_power")))
        buying_power = _as_float(account.get("buying_power"))
        usable_cash = cash if _finite(cash) else buying_power
        if _finite(equity) and equity < self.config.min_account_equity:
            return 0
        if _finite(usable_cash):
            usable_after_reserve = usable_cash - self.config.min_cash_after_entry
            return int(math.floor(max(0.0, usable_after_reserve) / premium_per_contract))
        return self.config.max_contracts_per_trade

    def _deny(self, reason: str, **details: Any) -> RealAccountDecision:
        return RealAccountDecision(False, reason, qty=0, premium_at_risk=0.0, details=details)

    def _day_state(self, now: datetime | None = None) -> dict[str, Any]:
        now_et = (now or datetime.now(timezone.utc)).astimezone(ET)
        day = now_et.date().isoformat()
        current = self._state.get("day")
        if current != day:
            self._state = {"day": day, "new_entries": 0, "open_premium": 0.0, "entries": [], "closes": []}
            self._persist()
        return self._state

    def _load_state(self) -> dict[str, Any]:
        path = self.config.state_path
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {"day": "", "new_entries": 0, "open_premium": 0.0, "entries": [], "closes": []}

    def _persist(self) -> None:
        path = self.config.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")


def _selected_abs_delta(meta: dict[str, Any]) -> float | None:
    value = _as_float(meta.get("selected_abs_delta"))
    if _finite(value):
        return abs(value)
    greeks = meta.get("greeks") if isinstance(meta.get("greeks"), dict) else {}
    value = _as_float(greeks.get("delta"))
    return abs(value) if _finite(value) else None


def _abs_theta(meta: dict[str, Any]) -> float | None:
    greeks = meta.get("greeks") if isinstance(meta.get("greeks"), dict) else {}
    value = _as_float(greeks.get("theta"))
    return abs(value) if _finite(value) else None


def _account_snapshot(account: dict[str, Any] | None) -> dict[str, float | None]:
    if not account:
        return {}
    return {
        "equity": _finite_or_none(account.get("equity")),
        "cash": _finite_or_none(account.get("cash")),
        "buying_power": _finite_or_none(account.get("buying_power")),
    }


def _parse_hhmm(now_et: datetime) -> int:
    return now_et.hour * 60 + now_et.minute


def _parse_hhmm_text(value: str) -> int:
    try:
        hh, mm = str(value).strip().split(":", 1)
        return int(hh) * 60 + int(mm)
    except Exception:
        return 24 * 60


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _as_int(value: Any) -> int | None:
    try:
        out = int(float(value))
    except Exception:
        return None
    return out


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _finite_or_none(value: Any) -> float | None:
    out = _as_float(value)
    return float(out) if _finite(out) else None
