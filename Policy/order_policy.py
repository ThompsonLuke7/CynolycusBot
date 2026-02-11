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
    Maps agent target exposure actions in [-1, 1] to option orders:
      - long  -> buy call
      - short -> buy put
      - flat  -> optional sell-to-close
    Continuous execution:
      - smooth action with EMA
      - apply deadband on action changes
      - map exposure magnitude to target contracts
      - trade only signed delta with per-step cap
    """

    def __init__(self, config: OptionOrderPolicyConfig) -> None:
        self.cfg = config
        self._tz = ZoneInfo(config.tz_name)
        self._cutoff = _parse_hhmm(config.dte_cutoff_hhmm)
        self._client = AlpacaOptionsClient(env_file=config.env_file)

        self._pos = 0
        self._signed_contracts = 0
        self._open_symbol: str | None = None
        self._bars_15m: list[tuple[float, float, float]] = []
        self._action_ema: float | None = None
        self._action_effective: float = 0.0

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

    def _resolve_expiration(self, local_ts: datetime) -> date:
        dte = 0 if local_ts.time() < self._cutoff else 1
        session_day = local_ts.date()
        return session_day if dte == 0 else _next_business_day(session_day, n=1)

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

    def _get_contract_price(self, *, symbol: str, logger: Callable[[str], None]) -> float:
        try:
            resp = self._client.get_option_quotes(symbols=symbol, limit=1)
            quotes = self._extract_quotes(resp, symbol=symbol)
            if not quotes:
                return float("nan")
            # Use the most recent quote if timestamps are present.
            quote = quotes[-1]
            return self._quote_price(quote, mode=self.cfg.price_mode)
        except Exception as exc:
            logger(f"[order_policy] quote fetch failed symbol={symbol}: {exc}")
            return float("nan")

    def _contracts_max_for_symbol(self, *, symbol: str, logger: Callable[[str], None]) -> tuple[int, float, float]:
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

            candidates: list[tuple[float, int, str, float]] = []
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

                # Map option position to directional policy state.
                # Long call => +1, long put => -1, short call => -1, short put => +1.
                pos_sign = side_mult if cp == "C" else -side_mult
                qty_abs = abs(qty_val) if math.isfinite(qty_val) else 0.0
                avg_entry = _as_float(p.get("avg_entry_price"))
                candidates.append((qty_abs, pos_sign, symbol, avg_entry))

            if not candidates:
                self._pos = 0
                self._signed_contracts = 0
                self._open_symbol = None
                logger(f"[order_policy] Startup sync: no open {under} long option positions found.")
                return {
                    "synced": True,
                    "position": 0,
                    "signed_contracts": 0,
                    "symbol": None,
                    "ignored_short_positions": ignored_short_count,
                }

            candidates.sort(key=lambda x: x[0], reverse=True)
            qty_abs, pos_sign, symbol, _avg_entry = candidates[0]
            self._pos = int(1 if pos_sign > 0 else -1)
            self._signed_contracts = int(round(qty_abs)) * self._pos
            self._open_symbol = symbol

            if len(candidates) > 1:
                logger(
                    f"[order_policy] Startup sync warning: multiple open {under} option positions "
                    f"found ({len(candidates)}); using largest qty symbol={symbol}."
                )
            logger(
                f"[order_policy] Startup sync: restored pos={self._pos} symbol={symbol} "
                f"qty={qty_abs:g} signed_contracts={self._signed_contracts}"
            )
            return {
                "synced": True,
                "position": self._pos,
                "signed_contracts": self._signed_contracts,
                "symbol": symbol,
                "qty": qty_abs,
                "multiple_positions": len(candidates) > 1,
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
        if not self.cfg.submit_orders:
            payload = {
                "symbol": symbol,
                "qty": order_qty,
                "side": side_key,
                "intent": intent_key,
                "type": self.cfg.order_type,
                "time_in_force": self.cfg.time_in_force,
            }
            logger(f"[order_policy] SIMULATED ORDER {payload}")
            return {"simulated": True, "payload": payload}
        max_attempts = max(0, int(self.cfg.max_resubmit_attempts))
        attempts = max_attempts + 1
        last_verify: dict[str, Any] | None = None
        for attempt in range(1, attempts + 1):
            resp = self._client.submit_option_order(
                symbol=symbol,
                qty=order_qty,
                side=side_key,
                order_type=self.cfg.order_type,
                time_in_force=self.cfg.time_in_force,
            )
            status = self._status_key(resp.get("status") if isinstance(resp, dict) else None)
            oid = str(resp.get("id", "")).strip() if isinstance(resp, dict) else ""
            logger(
                "[order_policy] ORDER SUBMITTED "
                f"intent={intent_key} side={side_key} qty={order_qty} symbol={symbol} "
                f"order_id={oid or 'n/a'} status={status or 'n/a'} attempt={attempt}/{attempts}"
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
            "atr": float(atr) if math.isfinite(atr) else None,
            "bars_15m": int(len(self._bars_15m)),
            "submit_orders": bool(self.cfg.submit_orders),
            "qty": int(self.cfg.qty),
            "price_mode": str(self.cfg.price_mode),
            "action_ema": float(self._action_ema) if self._action_ema is not None and math.isfinite(self._action_ema) else None,
            "action_effective": float(self._action_effective),
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
            smooth_action = self._smooth_action(act)
            prev_effective = float(self._action_effective)
            rebalance_deadband = max(0.0, float(self.cfg.rebalance_deadband))
            smoothed_change_applied = abs(smooth_action - prev_effective) >= rebalance_deadband
            effective_action = smooth_action if smoothed_change_applied else prev_effective
            self._action_effective = float(effective_action)

            deadband = max(0.0, float(self.cfg.action_deadband))
            desired_pos = 0 if abs(effective_action) <= deadband else (1 if effective_action > 0.0 else -1)
            local_ts = self._to_local_ts(closed_bar.get("timestamp"))
            close = _as_float(closed_bar.get("close"))
            atr = self._update_bar_state(closed_bar) if update_bar_state else self._compute_atr()

            if not math.isfinite(close):
                return {"event": "error", "reason": "invalid_close"}
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
                    contract_symbol, picked_strike = self._select_contract(
                        option_type=option_type,
                        expiration=expiration,
                        target_strike=strike_target,
                        atr=atr,
                    )

                if not contract_symbol:
                    return {"event": "error", "reason": "missing_contract_symbol"}

                contracts_max, buying_power, contract_price = self._contracts_max_for_symbol(
                    symbol=contract_symbol,
                    logger=logger,
                )
                target_abs = int(round(abs(effective_action) * max(0, contracts_max)))
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
                        contract_symbol, picked_strike = self._select_contract(
                            option_type=option_type,
                            expiration=expiration,
                            target_strike=strike_target,
                            atr=atr,
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
            }
        except Exception as exc:
            return {"event": "error", "reason": str(exc)}
