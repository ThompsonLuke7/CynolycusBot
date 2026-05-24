"""Automated earnings-event discovery.

Two free discovery paths are supported:
  - SEC submissions: good for historical/backfill candidates from 8-K/10-Q/10-K
    filings. This is robust and official, but dates are filing dates and report
    time is UNKNOWN.
  - yfinance calendar/history: useful for upcoming/recent earnings dates when
    Yahoo data is available. This is optional and best-effort.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from events.forward_guidance.config import DISCOVERED_EVENTS_CSV, EVENTS_PATH, ensure_data_dirs
from events.forward_guidance.data.ingest_events import write_events
from events.forward_guidance.data.schema import EarningsEvent
from events.forward_guidance.data.sec_client import SecClient
from events.forward_guidance.utils.io import write_dataframe
from events.forward_guidance.utils.universe import load_universe, ticker_sector_etf

logger = logging.getLogger(__name__)


SEC_EARNINGS_FORMS = ("8-K", "10-Q", "10-K")


@dataclass(frozen=True)
class DiscoveryConfig:
    start: str
    end: str
    source: str = "both"
    tickers: tuple[str, ...] = ()
    limit: int | None = None


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ticker in tickers:
        clean = str(ticker).upper().replace("$", "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def tickers_from_universe(limit: int | None = None, *, include_funds: bool = False) -> list[str]:
    df = load_universe()
    if df.empty:
        return []
    if not include_funds and "type" in df.columns:
        asset_type = df["type"].astype(str).str.upper()
        df = df.loc[~asset_type.isin({"ETF", "ETN", "FUND", "INDEX"})].copy()
    tickers = _normalize_tickers(df["ticker"].tolist())
    return tickers[: int(limit)] if limit else tickers


def _sector_for_ticker(ticker: str) -> tuple[str | None, str | None]:
    df = load_universe()
    if df.empty:
        return None, None
    rows = df.loc[df["ticker"].astype(str).str.upper() == ticker.upper()]
    if rows.empty:
        return None, ticker_sector_etf(ticker, df)
    sector = rows.iloc[0].get("sector") if "sector" in rows.columns else None
    sector_etf = rows.iloc[0].get("sector_etf") if "sector_etf" in rows.columns else ticker_sector_etf(ticker, df)
    if pd.isna(sector):
        sector = None
    if pd.isna(sector_etf):
        sector_etf = None
    return (str(sector) if sector else None, str(sector_etf).upper() if sector_etf else None)


def _event_key(event: EarningsEvent) -> tuple[str, str]:
    return event.clean_ticker, str(pd.Timestamp(event.earnings_date).date())


def dedupe_events(events: Iterable[EarningsEvent]) -> list[EarningsEvent]:
    by_key: dict[tuple[str, str], EarningsEvent] = {}
    for event in events:
        key = _event_key(event)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = event
            continue
        if existing.report_time == "UNKNOWN" and event.report_time != "UNKNOWN":
            by_key[key] = event
    return sorted(by_key.values(), key=lambda e: (e.earnings_date, e.clean_ticker))


def discover_sec_filing_events(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    forms: tuple[str, ...] = SEC_EARNINGS_FORMS,
    sec_client: SecClient | None = None,
) -> list[EarningsEvent]:
    """Discover historical candidates from SEC submissions recent filings."""
    sec = sec_client or SecClient()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    events: list[EarningsEvent] = []
    for ticker in _normalize_tickers(tickers):
        try:
            cik = sec.ticker_to_cik(ticker)
            if not cik:
                logger.info("[%s] no SEC CIK found", ticker)
                continue
            sub = sec.submissions(cik)
            recent = sub.get("filings", {}).get("recent", {})
            df = pd.DataFrame(recent)
            if df.empty or "filingDate" not in df.columns:
                continue
            df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
            mask = df["form"].isin(forms) & df["filingDate"].between(start_ts, end_ts)
            sector, sector_etf = _sector_for_ticker(ticker)
            for _, row in df.loc[mask].iterrows():
                filing_date = row["filingDate"].date().isoformat()
                events.append(
                    EarningsEvent(
                        ticker=ticker,
                        earnings_date=filing_date,
                        report_time="UNKNOWN",
                        sector=sector,
                        sector_etf=sector_etf,
                        cik=cik,
                        source_url=None,
                        source_type=f"sec_{row.get('form')}",
                        metadata={
                            "discovery_source": "sec_submissions",
                            "form": row.get("form"),
                            "accession_number": row.get("accessionNumber"),
                            "primary_document": row.get("primaryDocument"),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("[%s] SEC event discovery failed: %s", ticker, exc)
    return dedupe_events(events)


def _parse_yfinance_earnings_dates(ticker: str, raw: object, *, start: str, end: str) -> list[EarningsEvent]:
    if raw is None:
        return []
    df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    if df.empty:
        return []
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    date_col = None
    for candidate in ("Earnings Date", "Earnings Date UTC", "index", "Date", "date"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df.columns[0]
    df["earnings_date_norm"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df.loc[df["earnings_date_norm"].between(start_ts, end_ts)]
    sector, sector_etf = _sector_for_ticker(ticker)
    events: list[EarningsEvent] = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["earnings_date_norm"])
        report_time = "UNKNOWN"
        if ts.hour < 12:
            report_time = "BMO"
        elif ts.hour >= 16:
            report_time = "AMC"
        events.append(
            EarningsEvent(
                ticker=ticker,
                earnings_date=ts.tz_convert("America/New_York").date().isoformat(),
                report_time=report_time,
                sector=sector,
                sector_etf=sector_etf,
                source_type="yfinance_earnings_calendar",
                metadata={"discovery_source": "yfinance", "raw_columns": sorted(map(str, df.columns))},
            )
        )
    return events


def discover_yfinance_events(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    per_ticker_limit: int = 16,
) -> list[EarningsEvent]:
    """Discover scheduled/recent earnings dates with yfinance when available."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for yfinance event discovery.") from exc
    events: list[EarningsEvent] = []
    for ticker in _normalize_tickers(tickers):
        try:
            t = yf.Ticker(ticker)
            raw = None
            if hasattr(t, "get_earnings_dates"):
                raw = t.get_earnings_dates(limit=per_ticker_limit)
            if raw is None or getattr(raw, "empty", False):
                raw = getattr(t, "earnings_dates", None)
            events.extend(_parse_yfinance_earnings_dates(ticker, raw, start=start, end=end))
        except Exception as exc:
            logger.warning("[%s] yfinance event discovery failed: %s", ticker, exc)
    return dedupe_events(events)


def discover_events(
    *,
    start: str,
    end: str,
    source: str = "both",
    tickers: Iterable[str] | None = None,
    limit: int | None = None,
    include_funds: bool = False,
) -> list[EarningsEvent]:
    ensure_data_dirs()
    symbols = _normalize_tickers(tickers or tickers_from_universe(limit=limit, include_funds=include_funds))
    if limit and tickers is not None:
        symbols = symbols[: int(limit)]
    if not symbols:
        raise ValueError("No tickers supplied and no reusable universe CSV was found.")
    all_events: list[EarningsEvent] = []
    if source in {"sec", "both"}:
        all_events.extend(discover_sec_filing_events(symbols, start=start, end=end))
    if source in {"yfinance", "both"}:
        all_events.extend(discover_yfinance_events(symbols, start=start, end=end))
    if source not in {"sec", "yfinance", "both"}:
        raise ValueError(f"Unknown discovery source: {source}")
    return dedupe_events(all_events)


def write_discovered_events(
    events: Iterable[EarningsEvent],
    *,
    csv_path: Path | str = DISCOVERED_EVENTS_CSV,
    parquet_path: Path | str = EVENTS_PATH,
) -> pd.DataFrame:
    events_list = list(events)
    df = write_events(events_list, parquet_path)
    write_dataframe(df, csv_path)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover earnings events from free sources.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--source", choices=["sec", "yfinance", "both"], default="both")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-funds", action="store_true", help="Include ETFs/funds from the reusable universe.")
    parser.add_argument("--output", default=str(DISCOVERED_EVENTS_CSV))
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()), format="%(asctime)s %(levelname)s %(message)s")
    events = discover_events(
        start=args.start,
        end=args.end,
        source=args.source,
        tickers=args.tickers,
        limit=args.limit,
        include_funds=args.include_funds,
    )
    df = write_discovered_events(events, csv_path=args.output)
    print(f"discovered {len(df)} events -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
