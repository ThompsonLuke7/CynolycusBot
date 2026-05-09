"""SEC EDGAR client used for free-source earnings ingestion.

The client intentionally stays small and dependency-free. It fetches SEC JSON
APIs, filing HTML/text, and performs conservative text cleanup. Runtime calls
respect the SEC User-Agent requirement through the SEC_USER_AGENT environment
variable when available.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
import gzip
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from forward_guidance.utils.io import ensure_parent, read_json, write_json


DEFAULT_USER_AGENT = "CynolycusBot forward_guidance/0.1 research@example.com"


def decompress_response_bytes(data: bytes, encoding: str = "") -> bytes:
    """Decode gzip/deflate HTTP response payloads returned by SEC endpoints."""
    enc = str(encoding or "").lower()
    if enc == "gzip" or data.startswith(b"\x1f\x8b"):
        return gzip.decompress(data)
    if enc == "deflate":
        try:
            return zlib.decompress(data)
        except zlib.error:
            return zlib.decompress(data, -zlib.MAX_WBITS)
    return data


def _clean_cik(value: str | int) -> str:
    raw = str(value).strip()
    if raw.endswith(".0") and raw[:-2].isdigit():
        raw = raw[:-2]
    return raw.lstrip("0").zfill(10)


def html_to_text(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass(frozen=True)
class SecFiling:
    cik: str
    accession_number: str
    filing_date: str
    form: str
    primary_document: str
    description: str | None = None

    @property
    def accession_nodash(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def document_url(self) -> str:
        return (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{self.accession_nodash}/{self.primary_document}"
        )


class SecClient:
    def __init__(
        self,
        *,
        user_agent: str | None = None,
        cache_dir: Path | str | None = None,
        min_interval_s: float = 0.12,
    ) -> None:
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT") or DEFAULT_USER_AGENT
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.min_interval_s = float(min_interval_s)
        self._last_request_ts = 0.0

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

    def _request_json(self, url: str, *, cache_name: str | None = None, force: bool = False) -> Any:
        cache_path = self.cache_dir / cache_name if self.cache_dir and cache_name else None
        if cache_path and cache_path.exists() and not force:
            return read_json(cache_path)
        raw = self._request_bytes(url, host_header=urllib.parse.urlparse(url).netloc)
        payload = json.loads(raw.decode("utf-8"))
        if cache_path:
            write_json(payload, cache_path)
        return payload

    def _request_text(self, url: str, *, cache_path: Path | None = None, force: bool = False) -> str:
        if cache_path and cache_path.exists() and not force:
            return cache_path.read_text(encoding="utf-8", errors="ignore")
        raw = self._request_bytes(url, host_header=urllib.parse.urlparse(url).netloc)
        text = raw.decode("utf-8", errors="ignore")
        if cache_path:
            ensure_parent(cache_path).write_text(text, encoding="utf-8")
        return text

    def _request_bytes(self, url: str, *, host_header: str) -> bytes:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        headers = self._headers()
        headers["Host"] = host_header
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as resp:
            data = resp.read()
            encoding = str(resp.headers.get("Content-Encoding", "")).lower()
        self._last_request_ts = time.monotonic()
        return decompress_response_bytes(data, encoding)

    def company_tickers(self, *, force: bool = False) -> pd.DataFrame:
        data = self._request_json(
            "https://www.sec.gov/files/company_tickers.json",
            cache_name="company_tickers.json",
            force=force,
        )
        rows = list(data.values()) if isinstance(data, dict) else data
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["ticker", "cik_str", "title"])
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["cik_str"] = df["cik_str"].astype(str).map(_clean_cik)
        return df

    def ticker_to_cik(self, ticker: str) -> str | None:
        df = self.company_tickers()
        rows = df.loc[df["ticker"] == str(ticker).upper().replace("$", "")]
        if rows.empty:
            return None
        return str(rows.iloc[0]["cik_str"])

    def submissions(self, cik: str | int, *, force: bool = False) -> dict[str, Any]:
        clean = _clean_cik(cik)
        return self._request_json(
            f"https://data.sec.gov/submissions/CIK{clean}.json",
            cache_name=f"submissions_{clean}.json",
            force=force,
        )

    def companyfacts(self, cik: str | int, *, force: bool = False) -> dict[str, Any]:
        clean = _clean_cik(cik)
        return self._request_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{clean}.json",
            cache_name=f"companyfacts_{clean}.json",
            force=force,
        )

    def find_nearby_filings(
        self,
        *,
        cik: str | int,
        earnings_date: str,
        forms: tuple[str, ...] = ("8-K", "10-Q", "10-K"),
        days_before: int = 3,
        days_after: int = 7,
    ) -> list[SecFiling]:
        clean = _clean_cik(cik)
        sub = self.submissions(clean)
        recent = sub.get("filings", {}).get("recent", {})
        if not recent:
            return []
        df = pd.DataFrame(recent)
        if df.empty or "filingDate" not in df.columns:
            return []
        target = pd.Timestamp(earnings_date).normalize()
        start = target - pd.Timedelta(days=days_before)
        end = target + pd.Timedelta(days=days_after)
        df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
        mask = df["form"].isin(forms) & df["filingDate"].between(start, end)
        out: list[SecFiling] = []
        for _, row in df.loc[mask].iterrows():
            out.append(
                SecFiling(
                    cik=clean,
                    accession_number=str(row.get("accessionNumber")),
                    filing_date=str(row.get("filingDate").date()),
                    form=str(row.get("form")),
                    primary_document=str(row.get("primaryDocument")),
                    description=row.get("primaryDocDescription"),
                )
            )
        return out

    def download_filing_text(
        self,
        filing: SecFiling,
        *,
        cache_path: Path | None = None,
        force: bool = False,
    ) -> str:
        raw = self._request_text(filing.document_url, cache_path=cache_path, force=force)
        return html_to_text(raw)


def extract_basic_xbrl_metrics(companyfacts: dict[str, Any], fiscal_period: str | None = None) -> dict[str, Any]:
    """Extract a small, comparable set of recent EPS/revenue facts from companyfacts."""
    facts = companyfacts.get("facts", {}).get("us-gaap", {}) if companyfacts else {}
    tags = {
        "revenue_actual": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        "eps_actual": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
        "net_income": ("NetIncomeLoss",),
        "gross_profit": ("GrossProfit",),
        "operating_income": ("OperatingIncomeLoss",),
    }
    out: dict[str, Any] = {}
    for out_name, candidates in tags.items():
        out[out_name] = None
        out[f"{out_name}_asof"] = None
        for tag in candidates:
            entry = facts.get(tag, {})
            units = entry.get("units", {})
            unit_values = []
            for values in units.values():
                unit_values.extend(values if isinstance(values, list) else [])
            if not unit_values:
                continue
            df = pd.DataFrame(unit_values)
            if df.empty or "val" not in df.columns:
                continue
            if fiscal_period and "fp" in df.columns:
                filtered = df.loc[df["fp"].astype(str).str.upper() == str(fiscal_period).upper()]
                if not filtered.empty:
                    df = filtered
            if "filed" in df.columns:
                df = df.sort_values("filed")
            row = df.iloc[-1]
            out[out_name] = row.get("val")
            out[f"{out_name}_asof"] = row.get("filed") or row.get("end")
            break
    return out
