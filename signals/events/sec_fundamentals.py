"""Point-in-time quarterly fundamentals from SEC XBRL companyfacts.

    .venv/bin/python -m signals.events.sec_fundamentals --fetch        # raw JSON for the universe
    .venv/bin/python -m signals.events.sec_fundamentals --extract      # rebuild the PIT parquet
    .venv/bin/python -m signals.events.sec_fundamentals --fetch --extract --tickers META NVDA

Raw: Data/raw/sec_companyfacts/CIK##########.json.gz. It is immutable and is never refetched
unless --force is given.
Out: Data/research/fundamentals/sec_quarterly_facts.parquet, one row per
     (ticker, metric, period_end), with `available_at` = SEC `filed` date.

Time-correctness rules:
  * AS-FIRST-REPORTED: a period's value is taken from the EARLIEST filing that
    carries it. Later restatements (the same period re-reported in a later 10-Q/10-K)
    are ignored, so a model never sees a revised number before it existed.
  * available_at = filed date. For most names this is days to weeks AFTER the
    earnings press release, so it is conservative (late), never early.
  * Q4 is rarely filed as a 3-month fact. It is DERIVED as FY - (Q1+Q2+Q3), and
    available_at is the 10-K filed date. Such rows are flagged derived=True. Derived
    EPS is approximate because the share count changes within the year.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from signals.events.forward_guidance.data.sec_client import SecClient

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "Data" / "raw" / "sec_companyfacts"
OUT_PATH = REPO / "Data" / "research" / "fundamentals" / "sec_quarterly_facts.parquet"
UNIVERSE_CSV = REPO / "Data" / "shared" / "universe" / "shared_universe.csv"

# metric -> us-gaap tags in priority order (issuers switch tags across years).
METRIC_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
        "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense",
    ),
    "gross_profit": ("GrossProfit",),
    # Many issuers (META, ORCL) never tag GrossProfit; revenue - cost_of_revenue
    # recovers it at feature time.
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                        "CostOfServices"),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_diluted": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"),
}
QUARTER_DAYS = (80, 100)
YEAR_DAYS = (350, 380)
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F", "6-K"}

logger = logging.getLogger("sec_fundamentals")


def _raw_path(cik: str) -> Path:
    return RAW_DIR / f"CIK{cik}.json.gz"


def load_raw(cik: str) -> dict | None:
    p = _raw_path(cik)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _tag_facts(companyfacts: dict, tag: str) -> pd.DataFrame:
    entry = companyfacts.get("facts", {}).get("us-gaap", {}).get(tag, {})
    rows = []
    for unit, values in entry.get("units", {}).items():
        if unit.startswith("USD"):
            rows.extend(values)
    df = pd.DataFrame(rows)
    need = {"start", "end", "val", "filed", "form"}
    if df.empty or not need <= set(df.columns):
        return pd.DataFrame()
    df = df[df["form"].isin(FORMS)].copy()
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["start", "end", "filed", "val"])
    df["days"] = (df["end"] - df["start"]).dt.days
    df["tag"] = tag
    return df


def _first_reported(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (start, end): the earliest filing wins (as-first-reported)."""
    return (df.sort_values(["filed", "accn"] if "accn" in df else ["filed"])
              .drop_duplicates(["start", "end"], keep="first"))


def _metric_periods(companyfacts: dict, tags: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarterly and annual facts for a metric, merging tags by priority per period."""
    frames = []
    for rank, tag in enumerate(tags):
        f = _tag_facts(companyfacts, tag)
        if not f.empty:
            frames.append(_first_reported(f).assign(rank=rank))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    allf = pd.concat(frames, ignore_index=True)
    # A period is reported under the highest-priority tag that has it. Its
    # availability is the earliest filing of THAT tag (checked above per tag).
    allf = allf.sort_values(["rank", "filed"]).drop_duplicates(["start", "end"], keep="first")
    q = allf[allf["days"].between(*QUARTER_DAYS)]
    y = allf[allf["days"].between(*YEAR_DAYS)]
    return q, y


def _derive_q4(q: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """FY minus the three reported quarters inside it, for fiscal years lacking a Q4 fact."""
    out = []
    for _, fy in y.iterrows():
        inside = q[(q["start"] >= fy["start"] - pd.Timedelta(days=7))
                   & (q["end"] <= fy["end"] + pd.Timedelta(days=7))]
        if len(inside) != 3 or (inside["end"] >= fy["end"] - pd.Timedelta(days=7)).any():
            continue  # Q4 already reported (or quarters incomplete): nothing to derive
        out.append({
            "start": inside["end"].max() + pd.Timedelta(days=1), "end": fy["end"],
            "val": float(fy["val"]) - float(inside["val"].sum()),
            "filed": max(fy["filed"], inside["filed"].max()),
            "form": fy["form"], "tag": fy["tag"], "derived": True,
        })
    return pd.DataFrame(out)


def extract_quarterly_facts(companyfacts: dict, ticker: str, cik: str) -> pd.DataFrame:
    rows = []
    for metric, tags in METRIC_TAGS.items():
        q, y = _metric_periods(companyfacts, tags)
        if q.empty and y.empty:
            continue
        q = q.assign(derived=False)[["start", "end", "val", "filed", "form", "tag", "derived"]]
        q4 = _derive_q4(q, y) if not y.empty else pd.DataFrame()
        m = pd.concat([q, q4], ignore_index=True) if not q4.empty else q
        rows.append(m.assign(metric=metric))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={"start": "period_start", "end": "period_end", "val": "value",
                            "filed": "available_at"})
    df["ticker"], df["cik"] = ticker, cik
    df["value"] = df["value"].astype(float)
    return df[["ticker", "cik", "metric", "period_start", "period_end", "value",
               "available_at", "form", "tag", "derived"]].sort_values(["metric", "period_end"])


def universe_ciks(client: SecClient, tickers: list[str] | None) -> pd.DataFrame:
    ct = client.company_tickers()
    if tickers is None:
        u = pd.read_csv(UNIVERSE_CSV)
        u = u[u["type"].fillna("Stock") != "ETF"]
        tickers = sorted(u["ticker"].dropna().astype(str).str.upper().unique())
    m = ct[ct["ticker"].isin([t.upper() for t in tickers])][["ticker", "cik_str"]]
    missing = sorted(set(t.upper() for t in tickers) - set(m["ticker"]))
    logger.info("CIK map: %d of %d tickers (%d unmapped: ETFs/funds/foreign mostly)",
                len(m), len(tickers), len(missing))
    return m.rename(columns={"cik_str": "cik"})


def fetch(client: SecClient, cmap: pd.DataFrame, force: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fails = 0
    for i, cik in enumerate(sorted(cmap["cik"].unique()), 1):
        p = _raw_path(cik)
        if p.exists() and not force:
            continue
        try:
            data = client.companyfacts(cik)
        except Exception as exc:  # noqa: BLE001 - 404 = no XBRL (funds, some foreign issuers)
            fails += 1
            logger.debug("CIK %s: %s", cik, exc)
            continue
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(data, fh)
        tmp.replace(p)
        if i % 200 == 0:
            logger.info("fetched %d/%d (fails %d)", i, cmap["cik"].nunique(), fails)
    logger.info("fetch done; %d CIKs without companyfacts", fails)


def extract(cmap: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, cik in cmap[["ticker", "cik"]].itertuples(index=False):
        data = load_raw(cik)
        if data:
            f = extract_quarterly_facts(data, ticker, cik)
            if not f.empty:
                frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("wrote %s: %d rows, %d tickers, periods %s..%s, derived Q4 share %.1f%%",
                OUT_PATH, len(out), out["ticker"].nunique(), out["period_end"].min().date(),
                out["period_end"].max().date(), 100 * out["derived"].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SEC XBRL point-in-time fundamentals")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Cached client for company_tickers.json only; companyfacts go through an
    # uncached client so each payload is stored once, gzipped, in RAW_DIR.
    cmap = universe_ciks(SecClient(cache_dir=RAW_DIR / "_meta"), args.tickers)
    if args.fetch:
        fetch(SecClient(), cmap, args.force)
    if args.extract:
        extract(cmap)


if __name__ == "__main__":
    main()
