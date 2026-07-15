from __future__ import annotations

from datetime import date
from typing import Any

from strategies.dealer_positioning.chain import target_expiration_dates
from strategies.dealer_positioning.config import DealerPositioningConfig


class SchwabDealerDataClient:
    """Small adapter around the repo's Schwab client.

    The schwab package is optional in this environment, so import happens only
    when the live runner starts.
    """

    def __init__(self, config: DealerPositioningConfig) -> None:
        from core.API.Schwab_API.schwab_client import SchwabClient

        self._client = SchwabClient()
        self._config = config

    def get_quote_price(self, symbol: str) -> float | None:
        quotes = self._client.get_quotes([symbol])
        item = quotes.get(symbol) if isinstance(quotes, dict) else None
        if not isinstance(item, dict):
            return None
        quote = item.get("quote") if isinstance(item.get("quote"), dict) else item
        for key in ("lastPrice", "mark", "closePrice", "regularMarketLastPrice"):
            raw = quote.get(key)
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    continue
        return None

    def get_option_chain(
        self,
        symbol: str,
        ref_date: date,
        *,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> dict[str, Any]:
        if from_date is None or to_date is None:
            start, end = target_expiration_dates(ref_date, self._config.dte_offsets)
        else:
            start, end = from_date, to_date
        raw_client = getattr(self._client, "client", None)
        if raw_client is None or not hasattr(raw_client, "get_option_chain"):
            raise RuntimeError("Schwab client does not expose get_option_chain; install/update schwab-py.")

        candidates = _chain_symbol_candidates(symbol)
        if not candidates:
            raise RuntimeError(
                f"{symbol.upper()} is not queryable on the Schwab chains endpoint "
                f"(cash-settled index root); skipping without an HTTP request."
            )

        options = getattr(raw_client, "Options", None)
        contract_type = getattr(getattr(options, "ContractType", None), "ALL", "ALL")
        strategy = getattr(getattr(options, "Strategy", None), "SINGLE", "SINGLE")
        tried: list[str] = []
        last_error: str | None = None
        for chain_symbol in candidates:
            tried.append(chain_symbol)
            resp = raw_client.get_option_chain(
                symbol=chain_symbol,
                contract_type=contract_type,
                strike_count=self._config.chain_strike_count,
                include_underlying_quote=True,
                strategy=strategy,
                from_date=start,
                to_date=end,
            )
            if 200 <= int(resp.status_code) < 300:
                payload = resp.json()
                if isinstance(payload, dict):
                    payload["_requested_symbol"] = symbol.upper()
                    payload["_chain_symbol"] = chain_symbol
                return payload
            body = ""
            try:
                body = (resp.text or "")[:300]
            except Exception:
                body = ""
            last_error = f"{resp.status_code} {body}".strip()

        raise RuntimeError(
            f"Schwab chain lookup failed for {symbol.upper()} after trying {tried}: {last_error or 'unknown error'}"
        )


# Cash-settled index roots the Schwab chains endpoint does not serve on this
# account (every alias — SPX, SPXW, $SPX, ^SPX — returns HTTP 400). Requesting
# them only spams Bad Request responses, so short-circuit with no HTTP call.
_UNSUPPORTED_CHAIN_ROOTS = frozenset({"SPX", "SPXW", "$SPX", "^SPX", "NDX", "RUT", "VIX"})


def _chain_symbol_candidates(symbol: str) -> list[str]:
    upper = symbol.strip().upper()
    if upper in _UNSUPPORTED_CHAIN_ROOTS:
        return []
    return [upper]
