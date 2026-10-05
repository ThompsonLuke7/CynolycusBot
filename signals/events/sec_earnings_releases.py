"""Fetch earnings press releases: 8-K Item 2.02 -> Exhibit 99.x, full universe.

This replaces what forward_guidance ingest stored as "press_release". That pipeline took
the filing's PRIMARY document, which for an 8-K is the cover page (addresses and checkboxes)
and for a 10-Q is the accounting notes. Neither contains results or guidance.

    .venv/bin/python -m signals.events.sec_earnings_releases                  # universe, 2016+
    .venv/bin/python -m signals.events.sec_earnings_releases --tickers NVDA META

Raw text: Data/raw/sec_earnings_releases/{TICKER}/{accession}.txt.gz. This is immutable and
already-fetched accessions are skipped.
Manifest: Data/raw/sec_earnings_releases/_manifest.parquet, one row per 8-K, including
failures. `acceptance_ts` (EDGAR acceptanceDateTime, UTC) is the AVAILABILITY time.
It is exact to the second, so an after-close release is not usable until the next session.

SEC fair access allows <=10 requests/s in total. Do not run this concurrently with
sec_fundamentals --fetch.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import re
from pathlib import Path

import pandas as pd

from signals.events.forward_guidance.data.sec_client import SecClient, html_to_text

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "Data" / "raw" / "sec_earnings_releases"
META_CACHE = REPO / "Data" / "raw" / "sec_companyfacts" / "_meta"
MANIFEST = RAW_DIR / "_manifest.parquet"
UNIVERSE_CSV = REPO / "Data" / "shared" / "universe" / "shared_universe.csv"
START = "2016-01-01"
EXHIBIT_RE = re.compile(r"(ex|exhibit|dex)[-_]?99", re.I)

logger = logging.getLogger("sec_earnings_releases")


def earnings_8ks(filings: pd.DataFrame, start: str = START) -> pd.DataFrame:
    """8-K / 8-K/A filings whose items include 2.02 (Results of Operations)."""
    if filings.empty or not {"form", "items", "filingDate", "accessionNumber"} <= set(filings.columns):
        return pd.DataFrame()
    f = filings[filings["form"].isin(["8-K", "8-K/A"])].copy()
    f = f[f["items"].astype(str).str.contains(r"\b2\.02\b", regex=True)]
    f["filingDate"] = pd.to_datetime(f["filingDate"], errors="coerce")
    return f[f["filingDate"] >= pd.Timestamp(start)].drop_duplicates("accessionNumber")


def pick_exhibit(index_items: list[dict]) -> str | None:
    """First Exhibit-99 document in a filing index (99.1 sorts before 99.2)."""
    names = sorted(str(i.get("name", "")) for i in index_items)
    for n in names:
        if EXHIBIT_RE.search(n) and n.lower().endswith((".htm", ".html", ".txt")):
            return n
    return None


def fetch_ticker(sec: SecClient, ticker: str, cik: str, done: set[str]) -> list[dict]:
    rows = []
    try:
        filings = sec.all_submission_filings(cik)
    except Exception as exc:  # noqa: BLE001
        return [{"ticker": ticker, "cik": cik, "status": f"submissions_error: {exc}"[:200]}]
    for _, f in earnings_8ks(filings).iterrows():
        acc = str(f["accessionNumber"])
        if acc in done:
            continue
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}"
        row = {"ticker": ticker, "cik": cik, "accession": acc, "form": f["form"],
               "filing_date": f["filingDate"], "items": str(f["items"]),
               "acceptance_ts": pd.to_datetime(f.get("acceptanceDateTime"), utc=True, errors="coerce")}
        try:
            idx = sec._request_json(f"{base}/index.json")
            ex = pick_exhibit(idx.get("directory", {}).get("item", []))
            if ex is None:
                rows.append({**row, "status": "no_ex99"})
                continue
            text = html_to_text(sec._request_text(f"{base}/{ex}"))
            out = RAW_DIR / ticker / f"{acc}.txt.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(out, "wt", encoding="utf-8") as fh:
                fh.write(text)
            rows.append({**row, "exhibit": ex, "chars": len(text), "status": "ok"})
        except Exception as exc:  # noqa: BLE001
            rows.append({**row, "status": f"error: {exc}"[:200]})
    return rows


def universe_map(sec: SecClient, tickers: list[str] | None) -> list[tuple[str, str]]:
    if tickers is None:
        u = pd.read_csv(UNIVERSE_CSV)
        tickers = sorted(u.loc[u["type"].fillna("Stock") != "ETF", "ticker"].dropna().astype(str).str.upper().unique())
    ct = sec.company_tickers()
    m = ct[ct["ticker"].isin([t.upper() for t in tickers])]
    return list(zip(m["ticker"], m["cik_str"]))


def _write_manifest(rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    old = pd.read_parquet(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    allm = pd.concat([old, new], ignore_index=True)
    if "accession" in allm:
        # A later ok row supersedes an earlier failure for the same accession.
        allm = (allm.assign(_ok=allm["status"].eq("ok"))
                    .sort_values("_ok").drop_duplicates(["ticker", "accession"], keep="last")
                    .drop(columns="_ok"))
    allm.to_parquet(MANIFEST, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="SEC 8-K Item 2.02 earnings press releases")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--min-interval", type=float, default=0.12,
                    help="seconds between SEC requests; raise it when sharing the 10 req/s budget "
                         "with the live server's forward-guidance feed (503s otherwise)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta = SecClient(cache_dir=META_CACHE, min_interval_s=args.min_interval)
    sec = SecClient(min_interval_s=args.min_interval)  # uncached: submissions/index JSON are not worth keeping
    done = set()
    if MANIFEST.exists():
        m = pd.read_parquet(MANIFEST)
        done = set(m.loc[m["status"].isin(["ok", "no_ex99"]), "accession"].dropna())
    pairs = universe_map(meta, args.tickers)
    buf: list[dict] = []
    for i, (t, cik) in enumerate(pairs, 1):
        buf.extend(fetch_ticker(sec, t, cik, done))
        if i % 50 == 0 or i == len(pairs):
            _write_manifest(buf)
            buf = []
            logger.info("%d/%d tickers", i, len(pairs))
    m = pd.read_parquet(MANIFEST)
    logger.info("manifest: %d rows, status %s", len(m), m["status"].str.split(":").str[0].value_counts().to_dict())


if __name__ == "__main__":
    main()
