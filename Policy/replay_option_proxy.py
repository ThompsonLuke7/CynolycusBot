from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, time
import math
import re
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
_SIM_RE = re.compile(r"^\.SIM_([A-Z]+)_(\d{6})_([CP])_([0-9]+(?:\.[0-9]+)?)$")


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _parse_hhmm(hhmm: str) -> time:
    parts = str(hhmm or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time: {hhmm}")
    return time(hour=int(parts[0]), minute=int(parts[1]))


@dataclass(frozen=True)
class ParsedOptionSymbol:
    root: str
    expiry: pd.Timestamp
    right: str
    strike: float


def parse_occ_option_symbol(symbol: str) -> ParsedOptionSymbol | None:
    text = str(symbol or "").strip().upper()
    sim_match = _SIM_RE.match(text)
    if sim_match is not None:
        root, yymmdd, right, strike_raw = sim_match.groups()
        expiry = pd.to_datetime(yymmdd, format="%y%m%d", errors="coerce")
        if pd.isna(expiry):
            return None
        return ParsedOptionSymbol(
            root=root,
            expiry=pd.Timestamp(expiry).normalize(),
            right=right,
            strike=float(strike_raw),
        )
    match = _OCC_RE.match(text)
    if match is None:
        return None
    root, yymmdd, right, strike_raw = match.groups()
    expiry = pd.to_datetime(yymmdd, format="%y%m%d", errors="coerce")
    if pd.isna(expiry):
        return None
    return ParsedOptionSymbol(
        root=root,
        expiry=pd.Timestamp(expiry).normalize(),
        right=right,
        strike=float(int(strike_raw)) / 1000.0,
    )


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    tau_years: float,
    iv: float,
    right: str,
    risk_free_rate: float = 0.0,
) -> float:
    if not (math.isfinite(spot) and spot > 0.0 and math.isfinite(strike) and strike > 0.0):
        return float("nan")
    right_key = str(right or "").strip().upper()
    intrinsic = max(0.0, spot - strike) if right_key == "C" else max(0.0, strike - spot)
    tau = max(0.0, float(tau_years))
    sigma = max(0.0, float(iv))
    if tau <= 0.0 or sigma <= 0.0:
        return intrinsic
    vol_sqrt_t = sigma * math.sqrt(tau)
    if vol_sqrt_t <= 0.0:
        return intrinsic
    rate = float(risk_free_rate)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    discount = math.exp(-rate * tau)
    if right_key == "C":
        return spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    return strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


class ReplayOptionPriceProxy:
    """
    Deterministic option premium proxy for replay/simulated option orders.

    The proxy prices OCC-style option symbols from the current replay underlying
    bar using Black-Scholes. IV is estimated from recently streamed 1m spot
    returns and clipped so quiet replay segments do not collapse 0DTE premium to
    near-zero immediately.
    """

    def __init__(
        self,
        *,
        tz_name: str = "America/New_York",
        expiry_hhmm: str = "15:40",
        iv_floor: float = 0.12,
        iv_ceiling: float = 0.90,
        iv_multiplier: float = 1.50,
        fallback_iv: float = 0.30,
        min_dte_minutes: float = 1.0,
        quote_spread_bps: float = 0.0,
        history_len: int = 390,
    ) -> None:
        self._tz = ZoneInfo(str(tz_name or "America/New_York"))
        self._expiry_time = _parse_hhmm(expiry_hhmm)
        self._iv_floor = max(0.0, float(iv_floor))
        self._iv_ceiling = max(self._iv_floor, float(iv_ceiling))
        self._iv_multiplier = max(0.0, float(iv_multiplier))
        self._fallback_iv = min(self._iv_ceiling, max(self._iv_floor, float(fallback_iv)))
        self._min_dte_minutes = max(0.0, float(min_dte_minutes))
        self._quote_spread_bps = max(0.0, float(quote_spread_bps))
        self._bars: dict[str, dict[str, Any]] = {}
        self._closes: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max(20, int(history_len))))

    def update_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        root = str(symbol or bar.get("symbol", "")).strip().upper()
        close = _as_float(bar.get("close"))
        ts = pd.to_datetime(bar.get("timestamp"), utc=True, errors="coerce")
        if not root or not math.isfinite(close) or pd.isna(ts):
            return
        local_ts = pd.Timestamp(ts).tz_convert(self._tz)
        self._bars[root] = {
            "timestamp": local_ts,
            "close": float(close),
            "high": _as_float(bar.get("high")),
            "low": _as_float(bar.get("low")),
        }
        self._closes[root].append(float(close))

    def price(self, symbol: str, mode: str | None = None) -> float:
        parsed = parse_occ_option_symbol(symbol)
        if parsed is None:
            return float("nan")
        bar = self._bars.get(parsed.root)
        if not bar:
            return float("nan")
        spot = _as_float(bar.get("close"))
        local_ts = bar.get("timestamp")
        if not (math.isfinite(spot) and isinstance(local_ts, pd.Timestamp)):
            return float("nan")

        expiry_local = pd.Timestamp.combine(parsed.expiry.date(), self._expiry_time).tz_localize(self._tz)
        minutes_left = (expiry_local - local_ts).total_seconds() / 60.0
        minutes_left = max(float(self._min_dte_minutes), float(minutes_left))
        tau_years = minutes_left / (365.0 * 24.0 * 60.0)
        iv = self._realized_iv(parsed.root)
        mid = black_scholes_price(
            spot=spot,
            strike=parsed.strike,
            tau_years=tau_years,
            iv=iv,
            right=parsed.right,
        )
        if not math.isfinite(mid):
            return float("nan")
        price = max(0.01, float(mid))
        spread = price * (self._quote_spread_bps / 10000.0)
        mode_key = str(mode or "mid").strip().lower()
        if mode_key in {"bid"}:
            return max(0.01, price - 0.5 * spread)
        if mode_key in {"ask"}:
            return max(0.01, price + 0.5 * spread)
        return price

    def _realized_iv(self, root: str) -> float:
        closes = list(self._closes.get(str(root).upper(), ()))
        if len(closes) < 12:
            return self._fallback_iv
        returns: list[float] = []
        prev = closes[0]
        for cur in closes[1:]:
            if prev > 0.0 and cur > 0.0:
                returns.append(math.log(cur / prev))
            prev = cur
        if len(returns) < 10:
            return self._fallback_iv
        mean = sum(returns) / len(returns)
        var = sum((x - mean) * (x - mean) for x in returns) / max(1, len(returns) - 1)
        realized = math.sqrt(max(0.0, var)) * math.sqrt(252.0 * 390.0)
        iv = realized * self._iv_multiplier
        if not math.isfinite(iv) or iv <= 0.0:
            iv = self._fallback_iv
        return min(self._iv_ceiling, max(self._iv_floor, iv))
