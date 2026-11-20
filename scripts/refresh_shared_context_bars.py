"""Refresh and validate live 4H context caches, and the daily market-regime inputs.

The base 4H modules read SPY/QQQ/IWM/VIXY/sector context from
``Data/shared/bars/4h``.  Those files are distinct from the ticker universe,
so this explicit step is required alongside the full-universe bar catch-up.

The daily regime (signals/market_regime) reads ``Data/shared/bars/1d``, which
only the universe catch-up refreshed. IEI, the credit-leg denominator, is not a
universe name: its bars stopped on 2026-08-06, and risk_appetite_z,
liquidity_stress_z and credit_risk_z ran on a forward-filled HYG/IEI for 61
days (their `*_stale_days` columns said so; nothing read them). Those inputs
are refreshed here too. A stale one is reported loudly but does not fail the
step: the regime stage is non-fatal by design.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from signals.market_regime.config import (
    BACKFILL_SHORT_HISTORY_SYMBOLS,
    CREDIT_DENOMINATOR_FALLBACK,
    REQUIRED_REGIME_SYMBOLS,
)
from strategies.momentum_expansion.config.momentum_config import CONTEXT_TICKERS, SECTOR_ETFS
from strategies.momentum_expansion.data.bars import fetch_context_bars, fetch_one


REPO = Path(__file__).resolve().parents[1]
BARS_4H = REPO / "Data/shared/bars/4h"
BARS_1D = REPO / "Data/shared/bars/1d"
REGIME_1D_SYMBOLS = tuple(sorted(
    set(REQUIRED_REGIME_SYMBOLS) | set(BACKFILL_SHORT_HISTORY_SYMBOLS) | {CREDIT_DENOMINATOR_FALLBACK}
))
# The SIP feed refuses the most recent ~15 minutes (see catchup_shared_bars.py).
SIP_DELAY = dt.timedelta(minutes=20)


def _latest_bar(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["timestamp"])
    if df.empty:
        return None
    return pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()


def refresh_and_validate(*, max_stale_days: float = 5.0) -> dict[str, int]:
    counts = fetch_context_bars()
    now = pd.Timestamp.now(tz="UTC")
    stale: list[str] = []
    for symbol in list(CONTEXT_TICKERS) + list(SECTOR_ETFS):
        latest = _latest_bar(BARS_4H / f"{symbol}.parquet")
        age_days = float("inf") if latest is None else (now - latest).total_seconds() / 86_400.0
        if age_days > max_stale_days:
            stale.append(f"{symbol}={latest} ({age_days:.1f}d)")
    if stale:
        raise RuntimeError("stale live context 4H cache(s): " + ", ".join(stale))
    return counts


def refresh_regime_daily(*, max_stale_days: float = 5.0) -> list[str]:
    """Bring every daily regime input current; return the ones still stale."""

    end = (dt.datetime.now(dt.timezone.utc) - SIP_DELAY).isoformat()
    stale: list[str] = []
    now = pd.Timestamp.now(tz="UTC")
    for symbol in REGIME_1D_SYMBOLS:
        try:
            fetch_one(ticker=symbol, kind="1d", end=end)
        except Exception as exc:  # noqa: BLE001 - reported below as stale
            print(f"regime input {symbol}: fetch failed ({type(exc).__name__}: {exc})")
        latest = _latest_bar(BARS_1D / f"{symbol}.parquet")
        age_days = float("inf") if latest is None else (now - latest).total_seconds() / 86_400.0
        if age_days > max_stale_days:
            stale.append(f"{symbol}={latest} ({age_days:.1f}d)")
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh and validate live shared context bars.")
    parser.add_argument("--max-stale-days", type=float, default=5.0)
    parser.add_argument(
        "--context-only", action="store_true",
        help="Refresh the 1H/4H context caches only and skip the daily regime inputs. "
             "This is the intraday mode: a daily fetch during the session would write "
             "today's PARTIAL daily bar into Data/shared/bars/1d.",
    )
    args = parser.parse_args()
    print(refresh_and_validate(max_stale_days=args.max_stale_days))
    if args.context_only:
        return
    stale = refresh_regime_daily(max_stale_days=args.max_stale_days)
    if stale:
        print("WARNING: STALE DAILY REGIME INPUT(S) — regime composites will forward-fill: " + ", ".join(stale))
    else:
        print(f"daily regime inputs current: {', '.join(REGIME_1D_SYMBOLS)}")


if __name__ == "__main__":
    main()
