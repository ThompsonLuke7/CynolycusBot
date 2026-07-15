"""
Fetch US Treasury constant-maturity yields from FRED (free, no API key) and write
the parquet the Meta Ranker matrix expects.

Series: DGS3MO, DGS2, DGS10, DGS30 (daily, percent). FRED exposes a keyless CSV
download endpoint, so this needs no credentials.

Output columns: date, month3, year2, year10, year30, spread_2s10s, spread_3m10y, inverted
  -> signals/meta_context/data/processed/fmp_treasury_rates.parquet

  PYTHONPATH=. python scripts/fetch_treasury_rates.py
"""
from __future__ import annotations

import io
import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "signals/meta_context/data/processed/fmp_treasury_rates.parquet"
SERIES = {"DGS3MO": "month3", "DGS2": "year2", "DGS10": "year10", "DGS30": "year30"}
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
TREASURY_XML = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    "?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)
FRED_HEADERS = {
    # fred.stlouisfed.org has repeatedly timed out from urllib with a generic
    # browser UA, while the same endpoint responds immediately to curl.
    "User-Agent": "curl/8.5.0",
    "Accept": "text/csv,*/*",
    "Connection": "close",
}
RETRIES = 3
TIMEOUT = 30
STALE_WARN_DAYS = 7
STALE_FAIL_DAYS = 30
TREASURY_START_YEAR = int(os.getenv("TREASURY_START_YEAR", "1990"))


def _cached_staleness_days(path: Path) -> float:
    df = pd.read_parquet(path, columns=["date"])
    if df.empty:
        return float("inf")
    last = pd.to_datetime(df["date"]).max()
    return (pd.Timestamp.now().normalize() - last.normalize()) / pd.Timedelta(days=1)


def _fetch_series(sid: str) -> pd.Series:
    # FRED's keyless endpoint intermittently read-times-out; retry with backoff
    # before giving up so one slow response doesn't fail the whole refresh.
    last_exc: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(FRED.format(sid=sid), headers=FRED_HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                df = pd.read_csv(io.BytesIO(resp.read()))
            break
        except Exception as exc:  # noqa: BLE001 — network read timeouts, etc.
            last_exc = exc
            print(f"  ! {sid} attempt {attempt}/{RETRIES} failed: {type(exc).__name__}: {exc}")
            if attempt < RETRIES:
                time.sleep(2 * attempt)
    else:
        raise last_exc  # type: ignore[misc]
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = sid.lower() if sid.lower() in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")  # FRED uses "." for holidays
    return df.set_index(date_col)[val_col].rename(SERIES[sid])


def _fetch_fred_rates() -> pd.DataFrame:
    cols = [_fetch_series(sid) for sid in SERIES]
    return pd.concat(cols, axis=1).sort_index()


def _fetch_treasury_year(year: int) -> pd.DataFrame:
    req = urllib.request.Request(
        TREASURY_XML.format(year=year),
        headers={"User-Agent": "curl/8.5.0", "Accept": "application/xml,*/*", "Connection": "close"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        raw = resp.read()
    root = ET.fromstring(raw)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    }
    rows = []
    for props in root.findall(".//m:properties", ns):
        def text(tag: str) -> str | None:
            node = props.find(f"d:{tag}", ns)
            return None if node is None else node.text

        rows.append({
            "date": text("NEW_DATE"),
            "month3": text("BC_3MONTH"),
            "year2": text("BC_2YEAR"),
            "year10": text("BC_10YEAR"),
            "year30": text("BC_30YEAR"),
        })
    if not rows:
        return pd.DataFrame(columns=["date", "month3", "year2", "year10", "year30"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["month3", "year2", "year10", "year30"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).set_index("date").sort_index()


def _fetch_treasury_rates(start_year: int = TREASURY_START_YEAR) -> pd.DataFrame:
    current_year = pd.Timestamp.now().year
    frames = []
    last_exc: Exception | None = None
    for year in range(start_year, current_year + 1):
        try:
            yearly = _fetch_treasury_year(year)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"  ! Treasury XML {year} failed: {type(exc).__name__}: {exc}")
            continue
        if not yearly.empty:
            frames.append(yearly)
    if not frames:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Treasury XML fallback returned no rows")
    return pd.concat(frames).sort_index()


def main():
    tr: pd.DataFrame | None = None
    source = ""
    fetch_exc: Exception | None = None
    try:
        tr = _fetch_fred_rates()
        source = "FRED"
        try:
            recent = _fetch_treasury_rates(start_year=max(TREASURY_START_YEAR, pd.Timestamp.now().year - 1))
        except Exception as overlay_exc:  # noqa: BLE001
            print(f"WARNING: Treasury XML overlay failed ({type(overlay_exc).__name__}: {overlay_exc}); using FRED only")
        else:
            tr = pd.concat([tr, recent]).groupby(level=0).last().sort_index()
            source = "FRED + Treasury XML overlay"
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: FRED fetch failed ({type(exc).__name__}: {exc}); trying Treasury XML fallback")
        try:
            tr = _fetch_treasury_rates()
            source = "Treasury XML"
        except Exception as fallback_exc:  # noqa: BLE001
            fetch_exc = fallback_exc
    if tr is None:
        # Keep the last-good parquet rather than failing the nightly: these yields
        # move slowly and the matrix builder already falls back to the cached file.
        # But that fallback must not be unbounded — fail loudly once the cache is
        # old enough that "keep serving it" would itself be a silent data-quality bug.
        if OUT.exists():
            stale_days = _cached_staleness_days(OUT)
            if stale_days > STALE_FAIL_DAYS:
                raise RuntimeError(
                    f"treasury fetch failed ({type(fetch_exc).__name__}: {fetch_exc}) and cached {OUT.name} "
                    f"is {stale_days:.0f}d stale (> {STALE_FAIL_DAYS}d) — refusing to keep serving it."
                ) from fetch_exc
            level = "WARNING" if stale_days > STALE_WARN_DAYS else "info"
            print(f"{level}: treasury fetch failed ({type(fetch_exc).__name__}: {fetch_exc}); "
                  f"keeping last-good {OUT.name} ({stale_days:.0f}d stale)")
            return
        raise RuntimeError(f"treasury fetch failed and no cached data exists: {fetch_exc}") from fetch_exc
    tr = tr.ffill().dropna(how="all")
    tr.index.name = "date"
    tr = tr.reset_index()
    tr["spread_2s10s"] = tr["year10"] - tr["year2"]
    tr["spread_3m10y"] = tr["year10"] - tr["month3"]
    tr["inverted"] = (tr["spread_3m10y"] < 0).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(OUT, index=False)
    print(f"wrote {len(tr):,} rows -> {OUT}")
    print(f"  source {source}")
    print(f"  range {tr['date'].min().date()} .. {tr['date'].max().date()}")
    print(tr.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
