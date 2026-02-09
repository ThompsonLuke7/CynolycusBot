from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Callable

import math
import re
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


class OptionOrderPolicy:
    """
    Maps agent target-position actions (0 flat, 1 long, 2 short) to option orders:
      - long  -> buy call
      - short -> buy put
      - flat  -> optional sell-to-close
    """

    def __init__(self, config: OptionOrderPolicyConfig) -> None:
        self.cfg = config
        self._tz = ZoneInfo(config.tz_name)
        self._cutoff = _parse_hhmm(config.dte_cutoff_hhmm)
        self._client = AlpacaOptionsClient(env_file=config.env_file)

        self._pos = 0
        self._open_symbol: str | None = None
        self._bars_15m: list[tuple[float, float, float]] = []

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
                self._open_symbol = None
                logger(f"[order_policy] Startup sync: no open {under} long option positions found.")
                return {
                    "synced": True,
                    "position": 0,
                    "symbol": None,
                    "ignored_short_positions": ignored_short_count,
                }

            candidates.sort(key=lambda x: x[0], reverse=True)
            qty_abs, pos_sign, symbol, _avg_entry = candidates[0]
            self._pos = int(1 if pos_sign > 0 else -1)
            self._open_symbol = symbol

            if len(candidates) > 1:
                logger(
                    f"[order_policy] Startup sync warning: multiple open {under} option positions "
                    f"found ({len(candidates)}); using largest qty symbol={symbol}."
                )
            logger(
                f"[order_policy] Startup sync: restored pos={self._pos} symbol={symbol} "
                f"qty={qty_abs:g}"
            )
            return {
                "synced": True,
                "position": self._pos,
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

        resp = self._client.submit_option_order(
            symbol=symbol,
            qty=order_qty,
            side=side_key,
            order_type=self.cfg.order_type,
            time_in_force=self.cfg.time_in_force,
        )
        logger(
            "[order_policy] ORDER SUBMITTED "
            f"intent={intent_key} side={side_key} qty={order_qty} symbol={symbol}"
        )
        return {
            "simulated": False,
            "intent": intent_key,
            "response": resp,
            "side": side_key,
            "qty": order_qty,
            "symbol": symbol,
        }

    def on_decision(
        self,
        *,
        action: int,
        closed_bar: dict[str, Any],
        logger: Callable[[str], None] = print,
        update_bar_state: bool = True,
    ) -> dict[str, Any]:
        """
        Process one 15m close + PPO action.

        Returns a dict with an `event` key:
          hold | no_change | enter | flip | flat | error
        """
        try:
            act = int(action)
            if act not in (0, 1, 2):
                return {"event": "error", "reason": f"invalid_action:{action}"}

            desired_pos = 0 if act == 0 else (1 if act == 1 else -1)
            local_ts = self._to_local_ts(closed_bar.get("timestamp"))
            close = _as_float(closed_bar.get("close"))
            atr = self._update_bar_state(closed_bar) if update_bar_state else self._compute_atr()

            if not math.isfinite(close):
                return {"event": "error", "reason": "invalid_close"}

            if desired_pos == self._pos:
                return {"event": "hold", "pos": self._pos, "close": close, "atr": atr}

            # Optional close on flat.
            if desired_pos == 0:
                if self._pos != 0 and self._open_symbol and self.cfg.close_on_flat:
                    close_resp = self._submit_order(
                        symbol=self._open_symbol,
                        side="sell",
                        intent="close",
                        logger=logger,
                    )
                    old_symbol = self._open_symbol
                    self._open_symbol = None
                    self._pos = 0
                    return {
                        "event": "flat",
                        "closed_symbol": old_symbol,
                        "close_order": close_resp,
                        "close": close,
                        "atr": atr,
                    }
                self._pos = 0
                self._open_symbol = None
                return {"event": "no_change", "pos": 0, "close": close, "atr": atr}

            # desired_pos is +/-1 here.
            if not math.isfinite(atr) or atr <= 0.0:
                return {"event": "error", "reason": "atr_unavailable", "close": close, "atr": atr}

            event = "enter"
            flip_close_resp: dict[str, Any] | None = None
            if self._pos != 0 and self._pos != desired_pos:
                event = "flip"
                if self._open_symbol and self.cfg.close_on_flip:
                    flip_close_resp = self._submit_order(
                        symbol=self._open_symbol,
                        side="sell",
                        intent="close",
                        logger=logger,
                    )
                    self._open_symbol = None
                self._pos = 0

            option_type = "call" if desired_pos > 0 else "put"
            strike_target = close + self.cfg.atr_multiplier * atr if desired_pos > 0 else close - self.cfg.atr_multiplier * atr
            expiration = self._resolve_expiration(local_ts)
            contract_symbol, picked_strike = self._select_contract(
                option_type=option_type,
                expiration=expiration,
                target_strike=strike_target,
                atr=atr,
            )
            open_resp = self._submit_order(
                symbol=contract_symbol,
                side="buy",
                intent="open",
                logger=logger,
            )
            self._open_symbol = contract_symbol
            self._pos = desired_pos

            return {
                "event": event,
                "pos": self._pos,
                "contract": contract_symbol,
                "option_type": option_type,
                "expiration": expiration.isoformat(),
                "target_strike": strike_target,
                "picked_strike": picked_strike,
                "atr": atr,
                "close": close,
                "flip_close_order": flip_close_resp,
                "open_order": open_resp,
            }
        except Exception as exc:
            return {"event": "error", "reason": str(exc)}
