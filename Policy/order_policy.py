from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Callable

import math
import re
import time as time_mod
from zoneinfo import ZoneInfo

from API.Alpaca_API.options.options_api import AlpacaOptionsClient


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _parse_hhmm(hhmm: str) -> time:
    parts = (hhmm or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time: {hhmm}")
    return time(hour=int(parts[0]), minute=int(parts[1]))


def _next_business_day(d: date, *, n: int = 1) -> date:
    out = d
    for _ in range(max(0, int(n))):
        out = out + timedelta(days=1)
        while out.weekday() >= 5:
            out = out + timedelta(days=1)
    return out


@dataclass(frozen=True)
class OptionOrderPolicyConfig:
    underlying: str = "SPY"
    env_file: str = ".env"
    tz_name: str = "America/New_York"
    atr_length: int = 14
    atr_multiplier: float = 1.0
    dte_cutoff_hhmm: str = "14:00"
    qty: int = 1
    order_type: str = "market"
    time_in_force: str = "day"
    close_on_flat: bool = True
    close_on_flip: bool = True
    same_side_reentry_grace_bars: int = 1
    opposite_confirm_bars: int = 2
    opposite_min_abs_action: float = 0.10
    opposite_min_prob_edge: float = 0.05
    submit_orders: bool = True
    long_options_only: bool = True
    verify_submitted_orders: bool = True
    verify_timeout_sec: float = 15.0
    verify_poll_sec: float = 0.75
    resubmit_on_terminal_fail: bool = True
    max_resubmit_attempts: int = 1
    action_deadband: float = 0.05
    ema_alpha: float = 0.85
    rebalance_deadband: float = 0.10
    max_step_contracts: int = 2
    price_mode: str = "ask"  # ask|mid|bid|last|mark
    max_contracts_fallback: int = 1
    max_contracts_cap: int = 0  # <=0 disables hard cap


class OptionOrderPolicy:
    """
    Maps agent direction actions in [-1, 0, 1] to option orders:
      - long  -> buy call
      - short -> buy put
      - flat  -> optional sell-to-close
    Direction-only execution:
      - convert action to signed direction via deadband
      - ignore action magnitude for order sizing
      - size by fixed qty (bounded by max-contract checks)
      - trade only signed delta with per-step cap
    """

    def __init__(self, config: OptionOrderPolicyConfig) -> None:
        self.cfg = config
        self._tz = ZoneInfo(config.tz_name)
        self._cutoff = _parse_hhmm(config.dte_cutoff_hhmm)
        self._client = AlpacaOptionsClient(env_file=config.env_file)

        self._long_contracts = 0
        self._short_contracts = 0
        self._long_symbol: str | None = None
        self._short_symbol: str | None = None
        self._pos = 0
        self._signed_contracts = 0
        self._open_symbol: str | None = None
        self._bars_15m: list[tuple[float, float, float]] = []
        self._action_ema: float | None = None
        self._action_effective: float = 0.0
        self._pending_flat_side: int = 0
        self._pending_flat_bars: int = 0
        self._pending_opposite_side: int = 0
        self._pending_opposite_bars: int = 0

    def _refresh_legacy_state(self) -> None:
        self._signed_contracts = int(self._long_contracts) - int(self._short_contracts)
        if self._long_contracts > 0 and self._short_contracts <= 0:
            self._pos = 1
            self._open_symbol = self._long_symbol
        elif self._short_contracts > 0 and self._long_contracts <= 0:
            self._pos = -1
            self._open_symbol = self._short_symbol
        elif self._long_contracts <= 0 and self._short_contracts <= 0:
            self._pos = 0
            self._open_symbol = None
        else:
            self._pos = 0
            self._open_symbol = None

    def _to_local_ts(self, ts: Any) -> datetime:
        dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(self._tz)

    def _update_bar_state(self, closed_bar: dict[str, Any]) -> float:
        h = _as_float(closed_bar.get("high"))
        l = _as_float(closed_bar.get("low"))
        c = _as_float(closed_bar.get("close"))
        if not (math.isfinite(h) and math.isfinite(l) and math.isfinite(c)):
            return float("nan")
        self._bars_15m.append((h, l, c))
        max_keep = max(250, self.cfg.atr_length * 6)
        if len(self._bars_15m) > max_keep:
            self._bars_15m = self._bars_15m[-max_keep:]
        return self._compute_atr()

    def on_15m_bar(self, *, closed_bar: dict[str, Any]) -> float:
        """
        Update internal 15m bar state (ATR warmup) regardless of whether
        an action is produced on this bar.
        """
        return self._update_bar_state(closed_bar)

    def _compute_atr(self) -> float:
        n = int(self.cfg.atr_length)
        if n < 1 or len(self._bars_15m) < n + 1:
            return float("nan")

        trs: list[float] = []
        prev_close = self._bars_15m[0][2]
        for high, low, close in self._bars_15m:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(float(tr))
            prev_close = close

        seed = trs[:n]
        if not seed:
            return float("nan")
        atr = float(sum(seed) / n)
        for tr in trs[n:]:
            atr = ((n - 1) * atr + tr) / n
        return atr

    @staticmethod
    def _directional_prob_edge(*, closed_bar: dict[str, Any], desired_pos: int) -> float:
        """
        Compute directional confidence edge from available 15m probability columns.
        Positive edge favors the desired side.
        """
        meta_long = _as_float(closed_bar.get("p_enter_long"))
        meta_short = _as_float(closed_bar.get("p_enter_short"))
        if math.isfinite(meta_long) and math.isfinite(meta_short):
            edge = float(meta_long - meta_short)
            return edge if int(desired_pos) > 0 else -edge

        p_long_vals = [
            _as_float(closed_bar.get("p_pivot_long")),
            _as_float(closed_bar.get("p_tb_long")),
        ]
        p_short_vals = [
            _as_float(closed_bar.get("p_pivot_short")),
            _as_float(closed_bar.get("p_tb_short")),
        ]
        p_long = max((v for v in p_long_vals if math.isfinite(v)), default=float("nan"))
        p_short = max((v for v in p_short_vals if math.isfinite(v)), default=float("nan"))
        if not (math.isfinite(p_long) and math.isfinite(p_short)):
            return float("nan")
        edge = float(p_long - p_short)
        return edge if int(desired_pos) > 0 else -edge

    def _resolve_expiration(self, local_ts: datetime) -> date:
        dte = 0 if local_ts.time() < self._cutoff else 1
        session_day = local_ts.date()
        return session_day if dte == 0 else _next_business_day(session_day, n=1)

    def _sim_contract_symbol(
        self,
        *,
        option_type: str,
        expiration: date,
        strike: float,
    ) -> str:
        cp = "C" if str(option_type).strip().lower() == "call" else "P"
        strike_int = int(max(0, round(float(strike))))
        return f".SIM_{self.cfg.underlying}_{expiration.strftime('%y%m%d')}_{cp}_{strike_int}"

    @staticmethod
    def _extract_positions(resp: Any) -> list[dict[str, Any]]:
        if isinstance(resp, list):
            return [x for x in resp if isinstance(x, dict)]
        if isinstance(resp, dict):
            for key in ("positions", "data"):
                value = resp.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        return []

    @staticmethod
    def _option_cp(symbol: str) -> str | None:
        """
        Extract option type (C/P) from OCC symbol, e.g. SPY251031P00680000.
        """
        m = re.match(r"^[A-Z]+(\d{6})([CP])(\d{8})$", symbol.strip().upper())
        if not m:
            return None
        return m.group(2)

    @staticmethod
    def _extract_buying_power(resp: Any) -> float:
        if not isinstance(resp, dict):
            return float("nan")
        for key in ("buying_power", "non_marginable_buying_power", "regt_buying_power"):
            val = _as_float(resp.get(key))
            if math.isfinite(val) and val > 0.0:
                return val
        return float("nan")

    @staticmethod
    def _extract_quotes(resp: Any, *, symbol: str) -> list[dict[str, Any]]:
        sym = str(symbol).strip().upper()
        if isinstance(resp, dict):
            # Common format: {"quotes": {"SYMBOL": [{...}]}}
            quotes_obj = resp.get("quotes")
            if isinstance(quotes_obj, dict):
                for key, value in quotes_obj.items():
                    if str(key).strip().upper() != sym:
                        continue
                    if isinstance(value, list):
                        return [q for q in value if isinstance(q, dict)]
                    if isinstance(value, dict):
                        return [value]
            if isinstance(quotes_obj, list):
                return [
                    q for q in quotes_obj
                    if isinstance(q, dict) and str(q.get("symbol", "")).strip().upper() == sym
                ]
            # Alternate wrappers.
            for key in ("data",):
                value = resp.get(key)
                if isinstance(value, list):
                    return [
                        q for q in value
                        if isinstance(q, dict) and str(q.get("symbol", "")).strip().upper() == sym
                    ]
                if isinstance(value, dict):
                    maybe = value.get(sym) or value.get(sym.lower()) or value.get(sym.upper())
                    if isinstance(maybe, list):
                        return [q for q in maybe if isinstance(q, dict)]
                    if isinstance(maybe, dict):
                        return [maybe]
        if isinstance(resp, list):
            return [
                q for q in resp
                if isinstance(q, dict) and str(q.get("symbol", "")).strip().upper() == sym
            ]
        return []

    @staticmethod
    def _quote_price(quote: dict[str, Any], *, mode: str) -> float:
        mode_key = str(mode or "ask").strip().lower()
        ask = _as_float(
            quote.get("ask_price", quote.get("ap", quote.get("ask")))
        )
        bid = _as_float(
            quote.get("bid_price", quote.get("bp", quote.get("bid")))
        )
        last = _as_float(quote.get("last_price", quote.get("lp")))
        mark = _as_float(quote.get("mark_price", quote.get("mark")))
        if mode_key == "bid":
            return bid
        if mode_key == "last":
            return last
        if mode_key == "mark":
            return mark if math.isfinite(mark) else (0.5 * (bid + ask) if math.isfinite(bid) and math.isfinite(ask) else float("nan"))
        if mode_key == "mid":
            if math.isfinite(bid) and math.isfinite(ask):
                return 0.5 * (bid + ask)
            if math.isfinite(mark):
                return mark
            if math.isfinite(last):
                return last
            return ask if math.isfinite(ask) else bid
        # conservative default: ask
        if math.isfinite(ask):
            return ask
        if math.isfinite(mark):
            return mark
        if math.isfinite(last):
            return last
        if math.isfinite(bid):
            return bid
        return float("nan")

    def _get_buying_power(self) -> float:
        try:
            resp = self._client.get_account()
        except Exception:
            return float("nan")
        return self._extract_buying_power(resp)

    def _get_contract_price(
        self,
        *,
        symbol: str,
        logger: Callable[[str], None],
        mode: str | None = None,
    ) -> float:
        try:
            resp = self._client.get_option_quotes(symbols=symbol, limit=1)
            quotes = self._extract_quotes(resp, symbol=symbol)
            if not quotes:
                return float("nan")
            # Use the most recent quote if timestamps are present.
            quote = quotes[-1]
            price_mode = str(mode or self.cfg.price_mode)
            return self._quote_price(quote, mode=price_mode)
        except Exception as exc:
            logger(f"[order_policy] quote fetch failed symbol={symbol}: {exc}")
            return float("nan")

    def _contracts_max_for_symbol(self, *, symbol: str, logger: Callable[[str], None]) -> tuple[int, float, float]:
        if not self.cfg.submit_orders:
            max_ct = max(0, int(self.cfg.max_contracts_fallback))
            if self.cfg.max_contracts_cap and int(self.cfg.max_contracts_cap) > 0:
                max_ct = min(max_ct, int(self.cfg.max_contracts_cap))
            return max_ct, float("nan"), float("nan")
        price = self._get_contract_price(symbol=symbol, logger=logger)
        bp = self._get_buying_power()
        max_ct = 0
        if math.isfinite(bp) and bp > 0.0 and math.isfinite(price) and price > 0.0:
            max_ct = int(math.floor(bp / (price * 100.0)))
        if max_ct <= 0:
            max_ct = max(0, int(self.cfg.max_contracts_fallback))
        if self.cfg.max_contracts_cap and int(self.cfg.max_contracts_cap) > 0:
            max_ct = min(max_ct, int(self.cfg.max_contracts_cap))
        return max_ct, bp, price

    @staticmethod
    def _meta_threshold_value(closed_bar: dict[str, Any], *keys: str) -> float:
        for key in keys:
            value = _as_float(closed_bar.get(key))
            if math.isfinite(value):
                return value
        return float("nan")

    @classmethod
    def _has_meta_side_thresholds(cls, closed_bar: dict[str, Any]) -> bool:
        return (
            math.isfinite(cls._meta_threshold_value(closed_bar, "thr_enter_long", "enter_long_threshold"))
            and math.isfinite(cls._meta_threshold_value(closed_bar, "thr_enter_short", "enter_short_threshold"))
            and math.isfinite(cls._meta_threshold_value(closed_bar, "thr_exit_long", "exit_long_threshold"))
            and math.isfinite(cls._meta_threshold_value(closed_bar, "thr_exit_short", "exit_short_threshold"))
        )

    def _smooth_action(self, action: float) -> float:
        alpha = min(max(float(self.cfg.ema_alpha), 0.0), 0.9999)
        if self._action_ema is None or not math.isfinite(self._action_ema):
            self._action_ema = float(action)
        else:
            self._action_ema = alpha * float(self._action_ema) + (1.0 - alpha) * float(action)
        return float(self._action_ema)

    def sync_from_broker(self, *, logger: Callable[[str], None] = print) -> dict[str, Any]:
        """
        Seed in-memory position state from Alpaca open positions for this underlying.
        """
        try:
            resp = self._client.get_positions()
            positions = self._extract_positions(resp)
            under = self.cfg.underlying.strip().upper()

            long_candidates: list[tuple[float, str, float]] = []
            short_candidates: list[tuple[float, str, float]] = []
            ignored_short_count = 0
            for p in positions:
                symbol = str(p.get("symbol", "")).strip().upper()
                if not symbol.startswith(under):
                    continue
                cp = self._option_cp(symbol)
                if cp is None:
                    continue

                qty_val = _as_float(p.get("qty"))
                side_raw = str(p.get("side", "")).strip().lower()
                if self.cfg.long_options_only and side_raw == "short":
                    ignored_short_count += 1
                    continue
                side_mult = 1
                if side_raw == "short":
                    side_mult = -1
                elif side_raw == "long":
                    side_mult = 1
                elif math.isfinite(qty_val) and qty_val < 0:
                    side_mult = -1

                qty_abs = abs(qty_val) if math.isfinite(qty_val) else 0.0
                avg_entry = _as_float(p.get("avg_entry_price"))
                if qty_abs <= 0.0:
                    continue

                # Long call => bullish bucket, long put => bearish bucket.
                # Short option inventory is ignored when long_options_only=True.
                if cp == "C" and side_mult > 0:
                    long_candidates.append((qty_abs, symbol, avg_entry))
                elif cp == "P" and side_mult > 0:
                    short_candidates.append((qty_abs, symbol, avg_entry))

            if not long_candidates and not short_candidates:
                self._long_contracts = 0
                self._short_contracts = 0
                self._long_symbol = None
                self._short_symbol = None
                self._refresh_legacy_state()
                self._pending_flat_side = 0
                self._pending_flat_bars = 0
                self._pending_opposite_side = 0
                self._pending_opposite_bars = 0
                logger(f"[order_policy] Startup sync: no open {under} long option positions found.")
                return {
                    "synced": True,
                    "position": 0,
                    "signed_contracts": 0,
                    "long_contracts": 0,
                    "short_contracts": 0,
                    "long_symbol": None,
                    "short_symbol": None,
                    "avg_entry_price_long": None,
                    "avg_entry_price_short": None,
                    "ignored_short_positions": ignored_short_count,
                }

            long_candidates.sort(key=lambda x: x[0], reverse=True)
            short_candidates.sort(key=lambda x: x[0], reverse=True)
            long_qty, long_symbol, long_avg_entry = long_candidates[0] if long_candidates else (0.0, None, float("nan"))
            short_qty, short_symbol, short_avg_entry = short_candidates[0] if short_candidates else (0.0, None, float("nan"))
            self._long_contracts = int(round(long_qty)) if long_symbol else 0
            self._short_contracts = int(round(short_qty)) if short_symbol else 0
            self._long_symbol = long_symbol
            self._short_symbol = short_symbol
            self._refresh_legacy_state()
            self._pending_flat_side = 0
            self._pending_flat_bars = 0
            self._pending_opposite_side = 0
            self._pending_opposite_bars = 0

            if len(long_candidates) > 1 or len(short_candidates) > 1:
                logger(
                    f"[order_policy] Startup sync warning: multiple open {under} option positions "
                    f"found (long={len(long_candidates)}, short={len(short_candidates)}); "
                    "using largest qty symbol per side."
                )
            logger(
                f"[order_policy] Startup sync: restored long={self._long_contracts} symbol={self._long_symbol} "
                f"short={self._short_contracts} symbol={self._short_symbol} "
                f"signed_contracts={self._signed_contracts}"
            )
            return {
                "synced": True,
                "position": self._pos,
                "signed_contracts": self._signed_contracts,
                "long_contracts": self._long_contracts,
                "short_contracts": self._short_contracts,
                "long_symbol": self._long_symbol,
                "short_symbol": self._short_symbol,
                "qty_long": long_qty if long_symbol else 0.0,
                "qty_short": short_qty if short_symbol else 0.0,
                "avg_entry_price_long": float(long_avg_entry) if math.isfinite(long_avg_entry) else None,
                "avg_entry_price_short": float(short_avg_entry) if math.isfinite(short_avg_entry) else None,
                "multiple_positions": (len(long_candidates) > 1 or len(short_candidates) > 1),
                "ignored_short_positions": ignored_short_count,
            }
        except Exception as exc:
            logger(f"[order_policy] Startup sync failed: {exc}")
            return {"synced": False, "error": str(exc)}

    @staticmethod
    def _extract_contracts(resp: Any) -> list[dict[str, Any]]:
        if isinstance(resp, dict):
            for key in ("option_contracts", "contracts", "data"):
                value = resp.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        if isinstance(resp, list):
            return [x for x in resp if isinstance(x, dict)]
        return []

    @staticmethod
    def _status_key(status: Any) -> str:
        return str(status or "").strip().lower()

    @staticmethod
    def _order_is_success(status: Any) -> bool:
        return OptionOrderPolicy._status_key(status) in {"filled", "partially_filled"}

    @staticmethod
    def _order_is_terminal_fail(status: Any) -> bool:
        return OptionOrderPolicy._status_key(status) in {
            "canceled",
            "cancelled",
            "expired",
            "rejected",
            "failed",
            "suspended",
        }

    def _has_open_long_position(self, *, symbol: str) -> bool:
        try:
            resp = self._client.get_positions()
            positions = self._extract_positions(resp)
        except Exception:
            return False

        target = str(symbol).strip().upper()
        for p in positions:
            sym = str(p.get("symbol", "")).strip().upper()
            if sym != target:
                continue
            side_raw = str(p.get("side", "")).strip().lower()
            qty_val = _as_float(p.get("qty"))
            if self.cfg.long_options_only:
                if side_raw == "short":
                    continue
                if side_raw == "long":
                    return True
                if math.isfinite(qty_val) and qty_val > 0:
                    return True
                continue
            if side_raw in {"long", "short"}:
                return True
            if math.isfinite(qty_val) and abs(qty_val) > 0.0:
                return True
        return False

    def _verify_order_submission(
        self,
        *,
        submitted_resp: dict[str, Any],
        symbol: str,
        intent: str,
        logger: Callable[[str], None],
    ) -> dict[str, Any]:
        order_id = str(submitted_resp.get("id", "")).strip()
        last = submitted_resp
        status = self._status_key(last.get("status"))
        timeout_sec = max(1.0, float(self.cfg.verify_timeout_sec))
        poll_sec = max(0.2, float(self.cfg.verify_poll_sec))
        deadline = time_mod.monotonic() + timeout_sec

        if self._order_is_success(status):
            return {
                "verified": True,
                "status": status,
                "order_id": order_id,
                "via": "submit_response",
                "retryable": False,
                "order": last,
            }
        if self._order_is_terminal_fail(status):
            return {
                "verified": False,
                "status": status,
                "order_id": order_id,
                "via": "submit_response",
                "retryable": True,
                "order": last,
            }

        while time_mod.monotonic() < deadline and order_id:
            time_mod.sleep(poll_sec)
            try:
                current = self._client.get_order(order_id)
                if isinstance(current, dict):
                    last = current
                    status = self._status_key(current.get("status"))
            except Exception as exc:
                logger(f"[order_policy] verify poll warning order_id={order_id}: {exc}")
                continue

            if self._order_is_success(status):
                return {
                    "verified": True,
                    "status": status,
                    "order_id": order_id,
                    "via": "order_poll",
                    "retryable": False,
                    "order": last,
                }
            if self._order_is_terminal_fail(status):
                return {
                    "verified": False,
                    "status": status,
                    "order_id": order_id,
                    "via": "order_poll",
                    "retryable": True,
                    "order": last,
                }

        # Timeout fallback: reconcile with actual positions to reduce false negatives.
        try:
            has_pos = self._has_open_long_position(symbol=symbol)
            if intent == "open" and has_pos:
                return {
                    "verified": True,
                    "status": status or "unknown",
                    "order_id": order_id,
                    "via": "positions_reconcile",
                    "retryable": False,
                    "order": last,
                }
            if intent == "close" and not has_pos:
                return {
                    "verified": True,
                    "status": status or "unknown",
                    "order_id": order_id,
                    "via": "positions_reconcile",
                    "retryable": False,
                    "order": last,
                }
        except Exception:
            pass

        return {
            "verified": False,
            "status": status or "unknown",
            "order_id": order_id,
            "via": "timeout",
            "retryable": False,
            "order": last,
        }

    def _select_contract(
        self,
        *,
        option_type: str,
        expiration: date,
        target_strike: float,
        atr: float,
    ) -> tuple[str, float]:
        window = max(0.50, abs(float(atr)) * 2.0)
        lower = max(0.01, target_strike - window)
        upper = target_strike + window
        exp_str = expiration.isoformat()

        resp = self._client.get_option_contracts(
            underlying_symbol=self.cfg.underlying,
            expiration_date=exp_str,
            type=option_type,
            strike_price_gte=lower,
            strike_price_lte=upper,
            limit=200,
        )
        contracts = self._extract_contracts(resp)
        if not contracts:
            resp = self._client.get_option_contracts(
                underlying_symbol=self.cfg.underlying,
                expiration_date=exp_str,
                type=option_type,
                limit=1000,
            )
            contracts = self._extract_contracts(resp)
        if not contracts:
            raise RuntimeError(
                f"No option contracts found for {self.cfg.underlying} {option_type} exp={exp_str}"
            )

        candidates: list[tuple[float, str, float]] = []
        for contract in contracts:
            symbol = str(contract.get("symbol", "")).strip()
            strike = _as_float(contract.get("strike_price"))
            if not symbol or not math.isfinite(strike):
                continue
            # Prefer nearest strike to target.
            score = abs(strike - target_strike)
            candidates.append((score, symbol, strike))
        if not candidates:
            raise RuntimeError(
                f"Contracts returned but none had usable symbol/strike for exp={exp_str}"
            )

        candidates.sort(key=lambda x: x[0])
        _score, symbol, strike = candidates[0]
        return symbol, strike

    def _submit_order(
        self,
        *,
        symbol: str,
        side: str,
        intent: str = "open",
        qty: int | None = None,
        logger: Callable[[str], None] = print,
    ) -> dict[str, Any]:
        intent_key = str(intent).strip().lower()
        if intent_key not in {"open", "close"}:
            raise ValueError(f"Unknown order intent: {intent}")
        side_key = str(side).strip().lower()
        if self.cfg.long_options_only:
            if intent_key == "open" and side_key != "buy":
                raise ValueError("long_options_only=True requires buy-to-open orders.")
            if intent_key == "close" and side_key != "sell":
                raise ValueError("long_options_only=True requires sell-to-close orders.")

        order_qty = int(qty if qty is not None else self.cfg.qty)
        # Price ladder policy:
        # - start at midpoint
        # - +$0.01 per retry for opens (buy-to-open)
        # - -$0.01 per retry for closes (sell-to-close)
        base_limit = self._get_contract_price(symbol=symbol, logger=logger, mode="mid")
        if not math.isfinite(base_limit) or base_limit <= 0.0:
            fallback_mode = "ask" if intent_key == "open" else "bid"
            base_limit = self._get_contract_price(symbol=symbol, logger=logger, mode=fallback_mode)
        if not self.cfg.submit_orders and (not math.isfinite(base_limit) or base_limit <= 0.0):
            base_limit = 1.0
        if not math.isfinite(base_limit) or base_limit <= 0.0:
            raise RuntimeError(
                f"no_quote_for_limit_pricing intent={intent_key} symbol={symbol}"
            )
        tick = 0.01

        if not self.cfg.submit_orders:
            limit_price = round(float(base_limit), 2)
            payload = {
                "symbol": symbol,
                "qty": order_qty,
                "side": side_key,
                "intent": intent_key,
                "type": "limit",
                "time_in_force": self.cfg.time_in_force,
                "limit_price": limit_price,
            }
            logger(f"[order_policy] SIMULATED ORDER {payload}")
            return {"simulated": True, "payload": payload}
        max_attempts = max(0, int(self.cfg.max_resubmit_attempts))
        attempts = max_attempts + 1
        last_verify: dict[str, Any] | None = None
        for attempt in range(1, attempts + 1):
            offset = (attempt - 1) * tick
            if intent_key == "open":
                limit_price = base_limit + offset
            else:
                limit_price = max(tick, base_limit - offset)
            limit_price = round(float(limit_price), 2)

            resp = self._client.submit_option_order(
                symbol=symbol,
                qty=order_qty,
                side=side_key,
                order_type="limit",
                time_in_force=self.cfg.time_in_force,
                limit_price=limit_price,
            )
            status = self._status_key(resp.get("status") if isinstance(resp, dict) else None)
            oid = str(resp.get("id", "")).strip() if isinstance(resp, dict) else ""
            logger(
                "[order_policy] ORDER SUBMITTED "
                f"intent={intent_key} side={side_key} qty={order_qty} symbol={symbol} "
                f"order_id={oid or 'n/a'} status={status or 'n/a'} "
                f"limit_price={limit_price:.2f} attempt={attempt}/{attempts}"
            )

            verify_result: dict[str, Any] | None = None
            if self.cfg.verify_submitted_orders:
                verify_result = self._verify_order_submission(
                    submitted_resp=resp if isinstance(resp, dict) else {},
                    symbol=symbol,
                    intent=intent_key,
                    logger=logger,
                )
                last_verify = verify_result
                if verify_result.get("verified"):
                    logger(
                        "[order_policy] ORDER VERIFIED "
                        f"intent={intent_key} symbol={symbol} "
                        f"status={verify_result.get('status')} via={verify_result.get('via')}"
                    )
                    return {
                        "simulated": False,
                        "intent": intent_key,
                        "response": resp,
                        "verification": verify_result,
                        "side": side_key,
                        "qty": order_qty,
                        "symbol": symbol,
                    }

                retryable = bool(verify_result.get("retryable"))
                can_retry = (
                    self.cfg.resubmit_on_terminal_fail
                    and retryable
                    and attempt < attempts
                )
                logger(
                    "[order_policy] ORDER NOT VERIFIED "
                    f"intent={intent_key} symbol={symbol} "
                    f"status={verify_result.get('status')} via={verify_result.get('via')} "
                    f"retrying={can_retry}"
                )
                if can_retry:
                    continue

                raise RuntimeError(
                    "order_not_verified:"
                    f" intent={intent_key} symbol={symbol} status={verify_result.get('status')} "
                    f"via={verify_result.get('via')} order_id={verify_result.get('order_id')}"
                )

            return {
                "simulated": False,
                "intent": intent_key,
                "response": resp,
                "side": side_key,
                "qty": order_qty,
                "symbol": symbol,
            }

        raise RuntimeError(f"order_submit_failed intent={intent_key} symbol={symbol} verify={last_verify}")

    def snapshot_state(self) -> dict[str, Any]:
        """
        Return a lightweight policy state snapshot for monitoring UIs.
        """
        atr = self._compute_atr()
        return {
            "underlying": self.cfg.underlying,
            "position": int(self._pos),
            "signed_contracts": int(self._signed_contracts),
            "open_symbol": self._open_symbol,
            "long_contracts": int(self._long_contracts),
            "short_contracts": int(self._short_contracts),
            "open_long_symbol": self._long_symbol,
            "open_short_symbol": self._short_symbol,
            "atr": float(atr) if math.isfinite(atr) else None,
            "bars_15m": int(len(self._bars_15m)),
            "submit_orders": bool(self.cfg.submit_orders),
            "qty": int(self.cfg.qty),
            "price_mode": str(self.cfg.price_mode),
            "action_ema": float(self._action_ema) if self._action_ema is not None and math.isfinite(self._action_ema) else None,
            "action_effective": float(self._action_effective),
            "pending_flat_side": int(self._pending_flat_side),
            "pending_flat_bars": int(self._pending_flat_bars),
            "same_side_reentry_grace_bars": int(self.cfg.same_side_reentry_grace_bars),
            "pending_opposite_side": int(self._pending_opposite_side),
            "pending_opposite_bars": int(self._pending_opposite_bars),
            "opposite_confirm_bars": int(self.cfg.opposite_confirm_bars),
            "opposite_min_abs_action": float(self.cfg.opposite_min_abs_action),
            "opposite_min_prob_edge": float(self.cfg.opposite_min_prob_edge),
        }

    def snapshot_broker_state(self, *, orders_limit: int = 20) -> dict[str, Any]:
        """
        Pull current broker-side open positions and recent orders for this underlying.
        """
        under = self.cfg.underlying.strip().upper()
        out: dict[str, Any] = {
            "underlying": under,
            "positions": [],
            "recent_orders": [],
            "ok": True,
        }
        try:
            pos_resp = self._client.get_positions()
            positions = self._extract_positions(pos_resp)
            filtered_positions: list[dict[str, Any]] = []
            for p in positions:
                symbol = str(p.get("symbol", "")).strip().upper()
                if not symbol.startswith(under):
                    continue
                qty_val = _as_float(p.get("qty"))
                avg_entry = _as_float(p.get("avg_entry_price"))
                market_value = _as_float(p.get("market_value"))
                unrealized = _as_float(p.get("unrealized_pl"))
                filtered_positions.append(
                    {
                        "symbol": symbol,
                        "side": str(p.get("side", "")).strip().lower() or None,
                        "qty": qty_val if math.isfinite(qty_val) else None,
                        "avg_entry_price": avg_entry if math.isfinite(avg_entry) else None,
                        "market_value": market_value if math.isfinite(market_value) else None,
                        "unrealized_pl": unrealized if math.isfinite(unrealized) else None,
                    }
                )
            out["positions"] = filtered_positions
        except Exception as exc:
            out["ok"] = False
            out["positions_error"] = str(exc)

        try:
            limit = max(1, int(orders_limit))
            ord_resp = self._client.get_orders(status="all", limit=limit, direction="desc")
            orders: list[dict[str, Any]] = []
            if isinstance(ord_resp, list):
                raw_orders = [x for x in ord_resp if isinstance(x, dict)]
            elif isinstance(ord_resp, dict):
                data = ord_resp.get("orders")
                raw_orders = [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
            else:
                raw_orders = []
            for o in raw_orders:
                symbol = str(o.get("symbol", "")).strip().upper()
                if not symbol.startswith(under):
                    continue
                orders.append(
                    {
                        "id": str(o.get("id", "")).strip() or None,
                        "symbol": symbol,
                        "side": str(o.get("side", "")).strip().lower() or None,
                        "status": str(o.get("status", "")).strip().lower() or None,
                        "qty": str(o.get("qty", "")).strip() or None,
                        "filled_qty": str(o.get("filled_qty", "")).strip() or None,
                        "filled_avg_price": str(o.get("filled_avg_price", "")).strip() or None,
                        "submitted_at": str(o.get("submitted_at", "")).strip() or None,
                        "filled_at": str(o.get("filled_at", "")).strip() or None,
                    }
                )
            out["recent_orders"] = orders
        except Exception as exc:
            out["ok"] = False
            out["orders_error"] = str(exc)
        return out

    def _target_contracts_for_side(
        self,
        *,
        side: str,
        closed_bar: dict[str, Any],
    ) -> int:
        side_key = str(side).strip().lower()
        if side_key not in {"long", "short"}:
            return 0
        current_qty = self._long_contracts if side_key == "long" else self._short_contracts
        enter_prob = _as_float(closed_bar.get(f"p_enter_{side_key}"))
        exit_prob = _as_float(closed_bar.get(f"p_exit_{side_key}"))
        enter_thr = self._meta_threshold_value(closed_bar, f"thr_enter_{side_key}", f"enter_{side_key}_threshold")
        exit_thr = self._meta_threshold_value(closed_bar, f"thr_exit_{side_key}", f"exit_{side_key}_threshold")
        desired_qty = max(0, int(self.cfg.qty))
        if current_qty > 0:
            if math.isfinite(exit_prob) and math.isfinite(exit_thr) and exit_prob >= exit_thr:
                return 0
            return desired_qty
        if math.isfinite(enter_prob) and math.isfinite(enter_thr) and enter_prob >= enter_thr:
            return desired_qty
        return 0

    def _select_side_contract(
        self,
        *,
        side: str,
        close: float,
        atr: float,
        local_ts: datetime,
    ) -> tuple[str, str, date, float, float]:
        option_type = "call" if side == "long" else "put"
        strike_target = (
            close + self.cfg.atr_multiplier * atr
            if side == "long"
            else close - self.cfg.atr_multiplier * atr
        )
        expiration = self._resolve_expiration(local_ts)
        try:
            contract_symbol, picked_strike = self._select_contract(
                option_type=option_type,
                expiration=expiration,
                target_strike=strike_target,
                atr=atr,
            )
        except Exception as exc:
            if self.cfg.submit_orders:
                raise
            contract_symbol = self._sim_contract_symbol(
                option_type=option_type,
                expiration=expiration,
                strike=strike_target,
            )
            picked_strike = strike_target
            raise RuntimeError(
                f"sim_fallback:{option_type}:{expiration.isoformat()}:{strike_target:.2f}:{exc}"
            ) from exc
        return contract_symbol, option_type, expiration, strike_target, picked_strike

    def _on_independent_meta_decision(
        self,
        *,
        closed_bar: dict[str, Any],
        logger: Callable[[str], None],
        close: float,
        atr: float,
        local_ts: datetime,
    ) -> dict[str, Any]:
        if not math.isfinite(close):
            return {"event": "error", "reason": "invalid_close"}

        target_long = self._target_contracts_for_side(side="long", closed_bar=closed_bar)
        target_short = self._target_contracts_for_side(side="short", closed_bar=closed_bar)
        current_long = int(self._long_contracts)
        current_short = int(self._short_contracts)
        side_state: dict[str, dict[str, Any]] = {
            "long": {
                "current": current_long,
                "target": target_long,
                "symbol": self._long_symbol,
                "option_type": "call",
            },
            "short": {
                "current": current_short,
                "target": target_short,
                "symbol": self._short_symbol,
                "option_type": "put",
            },
        }

        orders: list[dict[str, Any]] = []
        side_events: list[str] = []
        buying_power = self._get_buying_power()

        for side in ("long", "short"):
            current_qty = int(side_state[side]["current"])
            target_qty = int(side_state[side]["target"])
            symbol = side_state[side]["symbol"]
            if current_qty > target_qty:
                if not symbol:
                    return {"event": "error", "reason": f"missing_{side}_symbol_for_close"}
                close_qty = current_qty - target_qty
                close_resp = self._submit_order(
                    symbol=symbol,
                    side="sell",
                    intent="close",
                    qty=close_qty,
                    logger=logger,
                )
                orders.append(
                    {
                        "type": f"close_{side}",
                        "side_key": side,
                        "symbol": symbol,
                        "qty": close_qty,
                        "response": close_resp,
                    }
                )
                side_events.append(f"close_{side}")
                if side == "long":
                    self._long_contracts = target_qty
                    if target_qty == 0:
                        self._long_symbol = None
                else:
                    self._short_contracts = target_qty
                    if target_qty == 0:
                        self._short_symbol = None

        for side in ("long", "short"):
            current_qty = int(self._long_contracts if side == "long" else self._short_contracts)
            target_qty = int(side_state[side]["target"])
            if target_qty <= current_qty:
                continue
            symbol = self._long_symbol if side == "long" else self._short_symbol
            option_type = "call" if side == "long" else "put"
            picked_strike = None
            strike_target = None
            expiration = None
            if not symbol:
                if not math.isfinite(atr) or atr <= 0.0:
                    return {"event": "error", "reason": "atr_unavailable", "close": close, "atr": atr}
                try:
                    symbol, option_type, expiration, strike_target, picked_strike = self._select_side_contract(
                        side=side,
                        close=close,
                        atr=atr,
                        local_ts=local_ts,
                    )
                except RuntimeError as exc:
                    msg = str(exc)
                    if msg.startswith("sim_fallback:"):
                        _, option_type, exp_str, strike_str, reason = msg.split(":", 4)
                        expiration = date.fromisoformat(exp_str)
                        strike_target = float(strike_str)
                        picked_strike = strike_target
                        symbol = self._sim_contract_symbol(
                            option_type=option_type,
                            expiration=expiration,
                            strike=strike_target,
                        )
                        logger(
                            "[order_policy] SIM fallback contract "
                            f"type={option_type} exp={expiration.isoformat()} "
                            f"strike={strike_target:.2f} reason={reason}"
                        )
                    else:
                        raise
            contracts_max, _bp_side, contract_price = self._contracts_max_for_symbol(
                symbol=symbol,
                logger=logger,
            )
            desired_final = min(target_qty, contracts_max) if contracts_max > 0 else target_qty
            if desired_final <= current_qty:
                continue
            open_qty = desired_final - current_qty
            open_resp = self._submit_order(
                symbol=symbol,
                side="buy",
                intent="open",
                qty=open_qty,
                logger=logger,
            )
            orders.append(
                {
                    "type": f"open_{side}",
                    "side_key": side,
                    "symbol": symbol,
                    "qty": open_qty,
                    "response": open_resp,
                    "contract_price": contract_price if math.isfinite(contract_price) else None,
                    "selected_option_type": option_type,
                    "expiration": expiration.isoformat() if isinstance(expiration, date) else None,
                    "target_strike": strike_target if strike_target is not None and math.isfinite(strike_target) else None,
                    "picked_strike": picked_strike if picked_strike is not None and math.isfinite(picked_strike) else None,
                }
            )
            side_events.append(f"open_{side}")
            if side == "long":
                self._long_contracts = desired_final
                self._long_symbol = symbol
            else:
                self._short_contracts = desired_final
                self._short_symbol = symbol

        self._refresh_legacy_state()
        if not orders:
            return {
                "event": "hold",
                "mode": "independent_meta",
                "position": int(self._pos),
                "signed_contracts": int(self._signed_contracts),
                "long_contracts": int(self._long_contracts),
                "short_contracts": int(self._short_contracts),
                "open_long_symbol": self._long_symbol,
                "open_short_symbol": self._short_symbol,
                "target_long_contracts": int(target_long),
                "target_short_contracts": int(target_short),
                "buying_power": buying_power if math.isfinite(buying_power) else None,
                "close": close,
                "atr": atr,
            }

        event = "multi_update" if len(side_events) > 1 else side_events[0]
        return {
            "event": event,
            "mode": "independent_meta",
            "position": int(self._pos),
            "signed_contracts": int(self._signed_contracts),
            "long_contracts": int(self._long_contracts),
            "short_contracts": int(self._short_contracts),
            "open_long_symbol": self._long_symbol,
            "open_short_symbol": self._short_symbol,
            "target_long_contracts": int(target_long),
            "target_short_contracts": int(target_short),
            "buying_power": buying_power if math.isfinite(buying_power) else None,
            "orders": orders,
            "close": close,
            "atr": atr,
        }

    def on_decision(
        self,
        *,
        action: float,
        closed_bar: dict[str, Any],
        logger: Callable[[str], None] = print,
        update_bar_state: bool = True,
    ) -> dict[str, Any]:
        """
        Process one 15m close + PPO action.

        Returns a dict with an `event` key:
          hold | enter | rebalance | flip | flat | error
        """
        try:
            raw_action = _as_float(action)
            if not math.isfinite(raw_action):
                return {"event": "error", "reason": f"invalid_action:{action}"}
            act = max(-1.0, min(1.0, float(raw_action)))
            deadband = max(0.0, float(self.cfg.action_deadband))
            desired_pos = 0 if abs(act) <= deadband else (1 if act > 0.0 else -1)
            # Direction-only mode: ignore action magnitude for execution.
            # Keep these fields for monitoring/debug compatibility.
            smooth_action = float(desired_pos)
            effective_action = float(desired_pos)
            prev_effective = float(self._action_effective)
            smoothed_change_applied = abs(effective_action - prev_effective) > 0.0
            self._action_ema = smooth_action
            self._action_effective = effective_action

            local_ts = self._to_local_ts(closed_bar.get("timestamp"))
            close = _as_float(closed_bar.get("close"))
            atr = self._update_bar_state(closed_bar) if update_bar_state else self._compute_atr()

            if not math.isfinite(close):
                return {"event": "error", "reason": "invalid_close"}
            if self._has_meta_side_thresholds(closed_bar):
                return self._on_independent_meta_decision(
                    closed_bar=closed_bar,
                    logger=logger,
                    close=close,
                    atr=atr,
                    local_ts=local_ts,
                )
            current_signed = int(self._signed_contracts)
            current_pos = 1 if current_signed > 0 else (-1 if current_signed < 0 else 0)

            flat_blocked = False
            flip_blocked = False
            if desired_pos == 0 and current_signed != 0 and not self.cfg.close_on_flat:
                flat_blocked = True
                desired_pos = current_pos
            if desired_pos != 0 and current_signed != 0 and desired_pos != current_pos and not self.cfg.close_on_flip:
                flip_blocked = True
                desired_pos = current_pos
            redundant_roundtrip_hold = False
            same_side_reentry_grace_bars = max(0, int(self.cfg.same_side_reentry_grace_bars))
            if current_signed == 0:
                self._pending_flat_side = 0
                self._pending_flat_bars = 0
            elif desired_pos == current_pos:
                self._pending_flat_side = 0
                self._pending_flat_bars = 0
            elif (
                desired_pos == 0
                and current_pos != 0
                and same_side_reentry_grace_bars > 0
                and self.cfg.close_on_flat
            ):
                if self._pending_flat_side != current_pos:
                    self._pending_flat_side = int(current_pos)
                    self._pending_flat_bars = 1
                else:
                    self._pending_flat_bars += 1
                if self._pending_flat_bars <= same_side_reentry_grace_bars:
                    redundant_roundtrip_hold = True
                    desired_pos = current_pos
                else:
                    self._pending_flat_side = 0
                    self._pending_flat_bars = 0
            else:
                self._pending_flat_side = 0
                self._pending_flat_bars = 0

            opposite_quality_blocked = False
            opposite_confirmation_pending = False
            opposite_prob_edge = float("nan")
            opposite_confirm_bars = max(1, int(self.cfg.opposite_confirm_bars))
            opposite_min_abs_action = max(0.0, float(self.cfg.opposite_min_abs_action))
            opposite_min_prob_edge = max(0.0, float(self.cfg.opposite_min_prob_edge))
            if current_signed == 0 or desired_pos == 0 or desired_pos == current_pos:
                self._pending_opposite_side = 0
                self._pending_opposite_bars = 0
            else:
                opposite_prob_edge = self._directional_prob_edge(
                    closed_bar=closed_bar,
                    desired_pos=desired_pos,
                )
                abs_quality_ok = abs(act) >= opposite_min_abs_action
                prob_quality_ok = (
                    opposite_min_prob_edge <= 0.0
                    or (math.isfinite(opposite_prob_edge) and opposite_prob_edge >= opposite_min_prob_edge)
                )
                if not (abs_quality_ok and prob_quality_ok):
                    opposite_quality_blocked = True
                    desired_pos = current_pos
                    self._pending_opposite_side = 0
                    self._pending_opposite_bars = 0
                else:
                    if self._pending_opposite_side != desired_pos:
                        self._pending_opposite_side = int(desired_pos)
                        self._pending_opposite_bars = 1
                    else:
                        self._pending_opposite_bars += 1
                    if self._pending_opposite_bars < opposite_confirm_bars:
                        opposite_confirmation_pending = True
                        desired_pos = current_pos
                    else:
                        self._pending_opposite_side = 0
                        self._pending_opposite_bars = 0

            option_type: str | None = None
            strike_target: float | None = None
            expiration: date | None = None
            contract_symbol: str | None = None
            picked_strike: float | None = None
            contracts_max = 0
            buying_power = float("nan")
            contract_price = float("nan")
            target_signed = 0

            if desired_pos != 0:
                if desired_pos == current_pos and self._open_symbol:
                    contract_symbol = self._open_symbol
                else:
                    if not math.isfinite(atr) or atr <= 0.0:
                        return {"event": "error", "reason": "atr_unavailable", "close": close, "atr": atr}
                    option_type = "call" if desired_pos > 0 else "put"
                    strike_target = (
                        close + self.cfg.atr_multiplier * atr
                        if desired_pos > 0
                        else close - self.cfg.atr_multiplier * atr
                    )
                    expiration = self._resolve_expiration(local_ts)
                    try:
                        contract_symbol, picked_strike = self._select_contract(
                            option_type=option_type,
                            expiration=expiration,
                            target_strike=strike_target,
                            atr=atr,
                        )
                    except Exception as exc:
                        if self.cfg.submit_orders:
                            raise
                        contract_symbol = self._sim_contract_symbol(
                            option_type=option_type,
                            expiration=expiration,
                            strike=strike_target,
                        )
                        picked_strike = strike_target
                        logger(
                            "[order_policy] SIM fallback contract "
                            f"type={option_type} exp={expiration.isoformat()} "
                            f"strike={strike_target:.2f} reason={exc}"
                        )

                if not contract_symbol:
                    return {"event": "error", "reason": "missing_contract_symbol"}

                contracts_max, buying_power, contract_price = self._contracts_max_for_symbol(
                    symbol=contract_symbol,
                    logger=logger,
                )
                # Size by fixed contract qty, not action magnitude.
                target_abs = max(0, int(self.cfg.qty))
                if contracts_max > 0:
                    target_abs = min(target_abs, int(contracts_max))
                target_signed = int(desired_pos * target_abs)
            else:
                buying_power = self._get_buying_power()
                target_signed = 0

            delta = int(target_signed - current_signed)
            step_cap = int(self.cfg.max_step_contracts)
            if step_cap > 0:
                step_delta = max(-step_cap, min(step_cap, delta))
            else:
                step_delta = delta
            step_target = int(current_signed + step_delta)
            step_pos = 1 if step_target > 0 else (-1 if step_target < 0 else 0)

            if step_target == current_signed:
                return {
                    "event": "hold",
                    "pos": current_pos,
                    "signed_contracts": current_signed,
                    "target_signed_contracts": target_signed,
                    "step_target_signed_contracts": step_target,
                    "exposure_raw": act,
                    "exposure_smooth": smooth_action,
                    "exposure_effective": effective_action,
                    "smoothed_change_applied": smoothed_change_applied,
                    "contracts_max": contracts_max,
                    "buying_power": buying_power if math.isfinite(buying_power) else None,
                    "contract_price": contract_price if math.isfinite(contract_price) else None,
                    "contract": contract_symbol or self._open_symbol,
                    "close": close,
                    "atr": atr,
                    "flat_blocked": flat_blocked,
                    "flip_blocked": flip_blocked,
                    "redundant_roundtrip_hold": redundant_roundtrip_hold,
                    "pending_flat_side": int(self._pending_flat_side),
                    "pending_flat_bars": int(self._pending_flat_bars),
                    "same_side_reentry_grace_bars": int(same_side_reentry_grace_bars),
                    "opposite_quality_blocked": opposite_quality_blocked,
                    "opposite_confirmation_pending": opposite_confirmation_pending,
                    "opposite_prob_edge": (
                        float(opposite_prob_edge) if math.isfinite(opposite_prob_edge) else None
                    ),
                    "pending_opposite_side": int(self._pending_opposite_side),
                    "pending_opposite_bars": int(self._pending_opposite_bars),
                    "opposite_confirm_bars": int(opposite_confirm_bars),
                    "opposite_min_abs_action": float(opposite_min_abs_action),
                    "opposite_min_prob_edge": float(opposite_min_prob_edge),
                }

            orders: list[dict[str, Any]] = []
            old_symbol = self._open_symbol

            # Leg 1: close/reduce existing side.
            if current_signed != 0 and (step_pos != current_pos or abs(step_target) < abs(current_signed)):
                if not self._open_symbol:
                    return {"event": "error", "reason": "missing_open_symbol_for_close"}
                close_qty = (
                    abs(current_signed)
                    if step_pos != current_pos
                    else abs(current_signed) - abs(step_target)
                )
                if close_qty > 0:
                    close_resp = self._submit_order(
                        symbol=self._open_symbol,
                        side="sell",
                        intent="close",
                        qty=close_qty,
                        logger=logger,
                    )
                    orders.append(
                        {
                            "type": "close",
                            "symbol": self._open_symbol,
                            "qty": close_qty,
                            "response": close_resp,
                        }
                    )
                if step_pos != current_pos:
                    self._open_symbol = None

            # Leg 2: open/increase desired side.
            if step_target != 0 and (step_pos != current_pos or abs(step_target) > abs(current_signed)):
                if step_pos == current_pos and self._open_symbol:
                    open_symbol = self._open_symbol
                    open_qty = abs(step_target) - abs(current_signed)
                else:
                    if not contract_symbol:
                        if not math.isfinite(atr) or atr <= 0.0:
                            return {"event": "error", "reason": "atr_unavailable", "close": close, "atr": atr}
                        option_type = "call" if step_pos > 0 else "put"
                        strike_target = (
                            close + self.cfg.atr_multiplier * atr
                            if step_pos > 0
                            else close - self.cfg.atr_multiplier * atr
                        )
                        expiration = self._resolve_expiration(local_ts)
                        try:
                            contract_symbol, picked_strike = self._select_contract(
                                option_type=option_type,
                                expiration=expiration,
                                target_strike=strike_target,
                                atr=atr,
                            )
                        except Exception as exc:
                            if self.cfg.submit_orders:
                                raise
                            contract_symbol = self._sim_contract_symbol(
                                option_type=option_type,
                                expiration=expiration,
                                strike=strike_target,
                            )
                            picked_strike = strike_target
                            logger(
                                "[order_policy] SIM fallback contract "
                                f"type={option_type} exp={expiration.isoformat()} "
                                f"strike={strike_target:.2f} reason={exc}"
                            )
                    open_symbol = contract_symbol
                    open_qty = abs(step_target) if step_pos != current_pos else max(0, abs(step_target) - abs(current_signed))

                if not open_symbol:
                    return {"event": "error", "reason": "missing_open_symbol_for_open"}

                if open_qty > 0:
                    open_resp = self._submit_order(
                        symbol=open_symbol,
                        side="buy",
                        intent="open",
                        qty=open_qty,
                        logger=logger,
                    )
                    orders.append(
                        {
                            "type": "open",
                            "symbol": open_symbol,
                            "qty": open_qty,
                            "response": open_resp,
                        }
                    )
                self._open_symbol = open_symbol

            self._signed_contracts = int(step_target)
            self._pos = 1 if self._signed_contracts > 0 else (-1 if self._signed_contracts < 0 else 0)
            if self._signed_contracts == 0:
                self._open_symbol = None
                self._long_contracts = 0
                self._short_contracts = 0
                self._long_symbol = None
                self._short_symbol = None
            elif self._signed_contracts > 0:
                self._long_contracts = int(abs(self._signed_contracts))
                self._short_contracts = 0
                self._long_symbol = self._open_symbol
                self._short_symbol = None
            else:
                self._long_contracts = 0
                self._short_contracts = int(abs(self._signed_contracts))
                self._long_symbol = None
                self._short_symbol = self._open_symbol

            if current_signed == 0 and self._signed_contracts != 0:
                event = "enter"
            elif self._signed_contracts == 0:
                event = "flat"
            elif current_pos != 0 and self._pos != current_pos:
                event = "flip"
            else:
                event = "rebalance"

            return {
                "event": event,
                "pos": self._pos,
                "signed_contracts": self._signed_contracts,
                "prev_signed_contracts": current_signed,
                "target_signed_contracts": target_signed,
                "step_target_signed_contracts": step_target,
                "delta_signed_contracts": delta,
                "step_delta_signed_contracts": step_delta,
                "exposure_raw": act,
                "exposure_smooth": smooth_action,
                "exposure_effective": effective_action,
                "smoothed_change_applied": smoothed_change_applied,
                "contracts_max": int(contracts_max),
                "buying_power": buying_power if math.isfinite(buying_power) else None,
                "contract_price": contract_price if math.isfinite(contract_price) else None,
                "contract": self._open_symbol,
                "selected_contract": contract_symbol,
                "selected_option_type": option_type,
                "expiration": expiration.isoformat() if isinstance(expiration, date) else None,
                "target_strike": strike_target if strike_target is not None and math.isfinite(strike_target) else None,
                "picked_strike": picked_strike if picked_strike is not None and math.isfinite(picked_strike) else None,
                "orders": orders,
                "closed_symbol": old_symbol if self._open_symbol != old_symbol else None,
                "close": close,
                "atr": atr,
                "flat_blocked": flat_blocked,
                "flip_blocked": flip_blocked,
                "redundant_roundtrip_hold": redundant_roundtrip_hold,
                "pending_flat_side": int(self._pending_flat_side),
                "pending_flat_bars": int(self._pending_flat_bars),
                "same_side_reentry_grace_bars": int(same_side_reentry_grace_bars),
                "opposite_quality_blocked": opposite_quality_blocked,
                "opposite_confirmation_pending": opposite_confirmation_pending,
                "opposite_prob_edge": (
                    float(opposite_prob_edge) if math.isfinite(opposite_prob_edge) else None
                ),
                "pending_opposite_side": int(self._pending_opposite_side),
                "pending_opposite_bars": int(self._pending_opposite_bars),
                "opposite_confirm_bars": int(opposite_confirm_bars),
                "opposite_min_abs_action": float(opposite_min_abs_action),
                "opposite_min_prob_edge": float(opposite_min_prob_edge),
            }
        except Exception as exc:
            return {"event": "error", "reason": str(exc)}
