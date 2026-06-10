"""One-shot backfill: for each ``sec_8-k`` record without a URL, look up the
matching accession on EDGAR by (ticker, filing_date), set the URL, then run
the EX-99 enricher to fill the body with the attached press-release prose.

Run once after the catalyst-module collection rerun. Safe to interrupt and
re-run — it skips records that already have URL + body populated.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

import pandas as pd
import urllib.request
import json

from signals.news.config import NEWS_RECORDS_PATH
from signals.news.sources import enrich_sec_8k_ex99_text


HEADERS = {"User-Agent": "CynolycusBot research@example.com"}
MIN_INTERVAL = 0.2  # SEC fair-use limit


def load_ticker_map() -> dict[str, str]:
    req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    return {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def fetch_submissions(cik: str) -> pd.DataFrame:
    req = urllib.request.Request(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception:
        return pd.DataFrame()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "form": recent.get("form", []),
            "filingDate": recent.get("filingDate", []),
            "accessionNumber": recent.get("accessionNumber", []),
            "primaryDocument": recent.get("primaryDocument", []),
        }
    )


def build_url(cik: str, accession: str, primary_doc: str) -> str:
    acc_no = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{primary_doc}"


def main() -> int:
    nr = pd.read_parquet(NEWS_RECORDS_PATH)
    needs_url = (nr["source"].astype(str) == "sec_8-k") & (nr["url"].fillna("").str.len() == 0)
    targets = nr[needs_url].copy()
    print(f"sec_8-k records needing URL: {len(targets)} of {len(nr[nr['source']=='sec_8-k']):,}")
    if targets.empty:
        return 0

    ticker_map = load_ticker_map()
    print(f"ticker map loaded: {len(ticker_map)} tickers")

    submission_cache: dict[str, pd.DataFrame] = {}
    last_req = 0.0
    matched = 0

    for idx, row in targets.iterrows():
        ticker = str(row["ticker"]).upper()
        cik = ticker_map.get(ticker)
        if not cik:
            continue
        if cik not in submission_cache:
            elapsed = time.monotonic() - last_req
            if elapsed < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - elapsed)
            submission_cache[cik] = fetch_submissions(cik)
            last_req = time.monotonic()
        sub = submission_cache[cik]
        if sub.empty:
            continue
        filing_date = str(pd.to_datetime(row["timestamp"]).date())
        match = sub[(sub["form"] == "8-K") & (sub["filingDate"] == filing_date)]
        if match.empty:
            continue
        m = match.iloc[0]
        url = build_url(cik, str(m["accessionNumber"]), str(m["primaryDocument"]))
        nr.at[idx, "url"] = url
        matched += 1
        if matched % 100 == 0:
            print(f"  matched {matched} URLs so far...", flush=True)

    print(f"URL backfill done: {matched} of {len(targets)} records got URLs")

    # Now run EX-99 enrichment on the newly URL-populated rows
    print("running EX-99 enrichment on backfilled rows...")
    enrich_mask = (nr["source"].astype(str) == "sec_8-k") & (nr["url"].fillna("").str.len() > 0) & (nr["body"].fillna("").str.len() < 200)
    to_enrich = nr[enrich_mask].copy()
    print(f"  candidate rows: {len(to_enrich)}")
    if not to_enrich.empty:
        enriched = enrich_sec_8k_ex99_text(to_enrich)
        added = enriched.attrs.get("ex99_enriched_count", 0)
        print(f"  EX-99 bodies added: {added}")
        nr.loc[enriched.index, "body"] = enriched["body"].values
        if "text" in nr.columns:
            nr.loc[enriched.index, "text"] = enriched["text"].values

    out_path = Path(NEWS_RECORDS_PATH)
    nr.to_parquet(out_path, index=False)
    print(f"saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
