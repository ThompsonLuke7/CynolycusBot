"""Daily option bars from Schwab, for contracts Alpaca will not serve.

Why this exists: Alpaca's /v1beta1/options/bars returns full history for EXPIRED
contracts but rejects currently-listed ones with
``HTTP 403 {"message":"OPRA agreement is not signed"}``. Schwab is the exact
complement -- it serves LIVE contracts and returns nothing for expired ones.
Measured 2026-09-08 on LCID:

    contract                       Alpaca            Schwab
    LCID260821C00006000 (expired)  37 bars           0 candles, empty=True
    LCID260918C00006000 (live)     403 OPRA          45 candles

So the pair covers the whole surface, and neither alone does.

SYMBOL FORMAT -- the trap. Schwab wants the underlying root LEFT-JUSTIFIED IN SIX
CHARACTERS, space-padded, before the OCC date/right/strike::

    'LCID  260918C00006000'   -> 45 candles
    'LCID260918C00006000'     -> HTTP 200, candles=[], empty=True

The unpadded form does not error. It returns a successful, empty response, which
reads identically to "this contract never traded". Per AGENTS.md a 200 is not
evidence of fitness for purpose; `to_schwab_symbol` exists so no caller has to
rediscover that.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

#: OCC: root, YYMMDD, C|P, strike * 1000 in 8 digits.
_OCC = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<rest>\d{6}[CP]\d{8})$")


def to_schwab_symbol(occ: str) -> str:
    """'LCID260918C00006000' -> 'LCID  260918C00006000'.

    Already-padded input is returned unchanged so this is safe to apply twice.
    """
    s = str(occ).strip().upper()
    m = _OCC.match(s.replace(" ", ""))
    if not m:
        raise ValueError(f"not an OCC option symbol: {occ!r}")
    return f"{m.group('root'):<6}{m.group('rest')}"


def fetch_daily_bars(
    client: Any,
    occ: str,
    start: dt.date,
    end: dt.date,
) -> tuple[list[dict], str]:
    """Daily candles for one contract. Returns (bars, status).

    `client` is the raw schwab-py Client (``SchwabClient().client``). Bars are
    normalised to the same shape the Alpaca path emits -- t/o/h/l/c/v -- so the
    two sources can be concatenated without a translation layer at every call
    site.

    An empty result is reported as ``no_candles`` rather than raised: for a
    genuinely untraded contract that is the correct answer, and the caller
    decides whether that disqualifies the observation.
    """
    try:
        sym = to_schwab_symbol(occ)
    except ValueError as exc:
        return [], f"bad_symbol({exc})"
    try:
        resp = client.get_price_history_every_day(
            sym,
            start_datetime=dt.datetime(start.year, start.month, start.day),
            end_datetime=dt.datetime(end.year, end.month, end.day),
        )
    except Exception as exc:  # noqa: BLE001
        return [], f"error({type(exc).__name__})"
    if getattr(resp, "status_code", None) != 200:
        return [], f"http_{getattr(resp, 'status_code', '?')}"
    payload = resp.json()
    candles = payload.get("candles") or []
    if not candles:
        return [], "no_candles"
    out = []
    for c in candles:
        # Schwab stamps candles in epoch MILLIseconds, US/Eastern session dates.
        ts = dt.datetime.fromtimestamp(int(c["datetime"]) / 1000, tz=dt.timezone.utc)
        out.append({"t": ts.date().isoformat(), "o": c.get("open"), "h": c.get("high"),
                    "l": c.get("low"), "c": c.get("close"), "v": c.get("volume"),
                    "n": None})
    return out, "ok"
