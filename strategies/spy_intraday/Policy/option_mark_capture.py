"""Append-only forward capture of live SPY option bid/ask marks.

Historical option trade bars are not marks.  This module records the live
quotes used around an active SPY option position, from both available brokers,
so future exit research can evaluate premium paths without substituting stale
trades or a synthetic option-price proxy.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


_OCC = re.compile(r"^(?P<root>[A-Z]+)(?P<tail>\d{6}[CP]\d{8})$")


def schwab_option_symbol(occ_symbol: str) -> str:
    """Convert an OCC symbol to Schwab's six-character-root quote syntax."""
    compact = str(occ_symbol).replace(" ", "").upper()
    match = _OCC.fullmatch(compact)
    if match is None:
        raise ValueError(f"not an OCC option symbol: {occ_symbol!r}")
    return f"{match.group('root'):<6}{match.group('tail')}"


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _quote_fields(quote: dict[str, Any]) -> dict[str, float | None]:
    bid = _num(quote.get("bid_price", quote.get("bidPrice", quote.get("bp", quote.get("bid")))))
    ask = _num(quote.get("ask_price", quote.get("askPrice", quote.get("ap", quote.get("ask")))))
    mark = _num(quote.get("mark_price", quote.get("mark")))
    last = _num(quote.get("last_price", quote.get("lastPrice", quote.get("lp"))))
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else mark
    spread = ask - bid if bid is not None and ask is not None else None
    return {"bid": bid, "ask": ask, "mid": mid, "mark": mark, "last": last, "spread": spread}


def _alpaca_quote(payload: Any, symbol: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    target = symbol.upper()
    quotes = payload.get("quotes", payload.get("data", payload))
    if isinstance(quotes, dict):
        value = quotes.get(target) or quotes.get(symbol) or quotes.get(target.lower())
        if isinstance(value, list):
            return next((x for x in reversed(value) if isinstance(x, dict)), None)
        if isinstance(value, dict):
            return value
    if isinstance(quotes, list):
        candidates = [q for q in quotes if isinstance(q, dict) and str(q.get("symbol", "")).upper() == target]
        return candidates[-1] if candidates else None
    return None


def _schwab_quote(payload: Any, requested_symbol: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    item = payload.get(requested_symbol)
    if not isinstance(item, dict):
        item = next((v for v in payload.values() if isinstance(v, dict)), None)
    if not isinstance(item, dict):
        return None
    quote = item.get("quote")
    return quote if isinstance(quote, dict) else item


class OptionMarkCapture:
    """Fetch and persist both Alpaca and Schwab marks for active contracts only."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        alpaca_factory: Callable[[], Any] | None = None,
        schwab_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._alpaca_factory = alpaca_factory or self._default_alpaca
        self._schwab_factory = schwab_factory or self._default_schwab
        self._alpaca: Any | None = None
        self._schwab: Any | None = None

    @staticmethod
    def _default_alpaca() -> Any:
        from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
        return AlpacaOptionsClient()

    @staticmethod
    def _default_schwab() -> Any:
        from core.API.Schwab_API.schwab_client import SchwabClient
        return SchwabClient()

    def _path(self, captured_at: datetime) -> Path:
        return self._output_dir / f"spy_option_marks_{captured_at.astimezone(timezone.utc):%Y%m%d}.jsonl"

    def _write(self, row: dict[str, Any], captured_at: datetime) -> None:
        path = self._path(captured_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")

    def _capture_alpaca(self, symbol: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            if self._alpaca is None:
                self._alpaca = self._alpaca_factory()
            quote = _alpaca_quote(self._alpaca.get_option_quotes(symbols=symbol, limit=1), symbol)
            if quote is None:
                return None, "no_quote"
            return {**_quote_fields(quote), "source_quote_timestamp": quote.get("timestamp") or quote.get("t")}, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"[:300]

    def _capture_schwab(self, symbol: str) -> tuple[dict[str, Any] | None, str, str | None]:
        requested = schwab_option_symbol(symbol)
        try:
            if self._schwab is None:
                self._schwab = self._schwab_factory()
            quote = _schwab_quote(self._schwab.get_quotes([requested]), requested)
            if quote is None:
                return None, requested, "no_quote"
            return {**_quote_fields(quote), "source_quote_timestamp": quote.get("quoteTimeInLong")}, requested, None
        except Exception as exc:
            return None, requested, f"{type(exc).__name__}: {exc}"[:300]

    def capture_active(self, *, underlying: str, bar: dict[str, Any], policy_state: dict[str, Any], phase: str) -> int:
        """Persist a row per source/contract. Returns the number of quote rows."""
        symbols = sorted({str(policy_state.get(k)).strip().upper() for k in ("open_symbol", "open_long_symbol", "open_short_symbol") if policy_state.get(k)})
        if not symbols:
            return 0
        captured_at = datetime.now(timezone.utc)
        bar_ts = pd.to_datetime(bar.get("timestamp"), utc=True, errors="coerce")
        context = {
            "captured_at": captured_at.isoformat(),
            "bar_timestamp": bar_ts.isoformat() if not pd.isna(bar_ts) else None,
            "underlying": str(underlying).upper(), "underlying_close": _num(bar.get("close")),
            "phase": phase,
            "policy_position": policy_state.get("position"),
            "long_decision_reason": policy_state.get("long_decision_reason"),
            "short_decision_reason": policy_state.get("short_decision_reason"),
        }
        count = 0
        for symbol in symbols:
            for source in ("alpaca", "schwab"):
                if source == "alpaca":
                    quote, error = self._capture_alpaca(symbol)
                    requested = symbol
                else:
                    quote, requested, error = self._capture_schwab(symbol)
                self._write({**context, "contract": symbol, "source": source, "source_symbol": requested, "quote": quote, "error": error}, captured_at)
                count += int(quote is not None)
        return count
