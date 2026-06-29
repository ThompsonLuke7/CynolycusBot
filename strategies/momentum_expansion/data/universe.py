"""
Weekly universe selector for momentum_expansion.

Approach:
  - Start from a curated candidate pool of liquid US tickers (reuses
    multi_ticker_swing's universe CSV — 213 names — as the seed pool).
  - On each rebuild date, score every name with daily bars available
    on (avg dollar volume, 5/20/60d relative strength vs SPY, ATR
    expansion).
  - Take the top N as that week's universe and write a snapshot CSV.
  - Backtest reads the snapshot dated <= the bar in question (no future
    constituents).

We intentionally accept survivorship bias from Alpaca's currently-listed
universe; flagged in the plan / report.
"""
from __future__ import annotations

import logging
import re
from zipfile import ZipFile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from strategies.momentum_expansion.config.momentum_config import (
    CONTEXT_ONLY_TICKERS,
    REPO_ROOT,
    RAW_1D_DIR,
    UNIVERSE_CONFIG,
    UNIVERSE_DIR,
)

logger = logging.getLogger(__name__)
_CONTEXT_ONLY_SET = {t.upper() for t in CONTEXT_ONLY_TICKERS}


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

_HAND_CURATED_TICKERS = [
        "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA",
        "AVGO", "AMD", "NFLX", "ADBE", "CRM", "ORCL", "INTC", "MU",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP",
        "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT",
        "XOM", "CVX", "COP", "OXY", "SLB", "EOG", "MPC", "PSX",
        "CAT", "DE", "BA", "GE", "HON", "LMT", "RTX",
        "WMT", "COST", "TGT", "HD", "LOW", "NKE", "MCD", "SBUX",
        "PG", "KO", "PEP", "PM", "MO",
        "DIS", "CMCSA", "T", "VZ", "TMUS",
        "F", "GM", "RIVN", "LCID",
        "COIN", "PYPL", "SQ", "SHOP",
        "PLTR", "NET", "DDOG", "SNOW", "CRWD", "PANW", "ZS",
        "UBER", "LYFT", "DASH", "ABNB",
        "BABA", "JD", "PDD", "BIDU",
        "TSM", "ASML", "ARM",
        "SMCI", "MSTR", "MARA", "RIOT",
]


def _clean_ticker(value: object) -> str:
    return str(value).strip().upper().replace("$", "")


def _swing_universe_path() -> Path:
    return REPO_ROOT / "strategies" / "multi_ticker_swing" / "config" / "swing_trader_universe_v3.csv"


def _parse_market_cap_bucket(market_cap: object) -> str:
    """
    Convert Thinkorswim-style market-cap strings (for example "1,597 M")
    into broad buckets compatible with the swing model's behavioral identity.
    """
    s = str(market_cap).strip().replace(",", "")
    match = re.match(r"([-+0-9.]+)\s*([KMBT]?)", s, flags=re.IGNORECASE)
    if not match:
        return "Unknown"
    value = float(match.group(1))
    unit = match.group(2).upper()
    cap_m = value * {"": 1.0, "K": 1e-3, "M": 1.0, "B": 1e3, "T": 1e6}[unit]
    if cap_m >= 200_000:
        return "Mega"
    if cap_m >= 10_000:
        return "Large"
    if cap_m >= 2_000:
        return "Mid"
    if cap_m >= 300:
        return "Small"
    return "Micro"


def _read_tos_xlsx(path: Path) -> pd.DataFrame:
    """
    Lightweight XLSX reader for the Thinkorswim scan export.

    The project environment does not require openpyxl. TOS exports the first
    worksheet as ordinary shared-string XLSX XML, so a small reader is enough
    for ticker metadata.
    """
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with ZipFile(path) as zf:
        names = zf.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                text = "".join(
                    t.text or ""
                    for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                )
                shared.append(text)

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet = workbook.find("a:sheets", ns)[0]
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}[rel_id]
        sheet_path = "xl/" + target if not target.startswith("xl/") else target
        sheet_root = ET.fromstring(zf.read(sheet_path))

        def col_num(cell_ref: str) -> int:
            letters = "".join(ch for ch in cell_ref if ch.isalpha())
            n = 0
            for ch in letters:
                n = n * 26 + ord(ch.upper()) - 64
            return n - 1

        rows: list[list[str]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", ns):
            values: list[str] = []
            for cell in row.findall("a:c", ns):
                idx = col_num(cell.attrib["r"])
                while len(values) <= idx:
                    values.append("")
                value_node = cell.find("a:v", ns)
                value = "" if value_node is None else value_node.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                values[idx] = value
            rows.append(values)

    records: list[dict] = []
    for row in rows:
        if not row:
            continue
        ticker = _clean_ticker(row[0])
        if not ticker or ticker in {"SYMBOL", "TICKER"}:
            continue
        records.append({
            "ticker": ticker,
            "sector": "Unknown",
            "market_cap_bucket": _parse_market_cap_bucket(row[11] if len(row) > 11 else ""),
            "type": "Stock",
            "notes": str(row[1]).strip() if len(row) > 1 else "Thinkorswim scan export",
        })
    return pd.DataFrame.from_records(records)


def _read_external_candidate_pool(path: Path | str | None) -> pd.DataFrame:
    if path is None or str(path).strip() in {"", "None"}:
        return pd.DataFrame()
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists() or p.name.startswith("~$"):
        return pd.DataFrame()
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in {".xlsx", ".xlsm"}:
        df = _read_tos_xlsx(p)
    else:
        logger.warning("Unsupported candidate pool format: %s", p)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()
    if "ticker" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ticker"})
    df["ticker"] = df["ticker"].map(_clean_ticker)
    df = df[df["ticker"] != ""].copy()
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "market_cap_bucket" not in df.columns:
        cap_col = "cap_bucket" if "cap_bucket" in df.columns else None
        df["market_cap_bucket"] = df[cap_col] if cap_col else "Unknown"
    if "type" not in df.columns:
        type_col = "asset_type" if "asset_type" in df.columns else None
        df["type"] = df[type_col] if type_col else "Stock"
    if "notes" not in df.columns:
        df["notes"] = f"External candidate pool: {p.name}"
    return df[["ticker", "sector", "market_cap_bucket", "type", "notes"]]


def load_candidate_metadata(cfg: dict | None = None) -> pd.DataFrame:
    """
    Return ticker metadata for the momentum-expansion training universe.

    Sources are unioned, in priority order:
      1. Thinkorswim/exported candidate pool from config.
      2. Full multi_ticker_swing universe, including ETFs.
      3. Hand-curated liquid/high-beta additions.
    """
    cfg = {**UNIVERSE_CONFIG, **(cfg or {})}
    frames: list[pd.DataFrame] = []

    external = _read_external_candidate_pool(cfg.get("candidate_pool_csv"))
    if not external.empty:
        external["_source_priority"] = 0
        frames.append(external)

    swing_csv = _swing_universe_path()
    if swing_csv.exists():
        swing = pd.read_csv(swing_csv)
        if "cap_bucket" in swing.columns and "market_cap_bucket" not in swing.columns:
            swing = swing.rename(columns={"cap_bucket": "market_cap_bucket"})
        if "asset_type" in swing.columns and "type" not in swing.columns:
            swing = swing.rename(columns={"asset_type": "type"})
        for col, default in (
            ("sector", "Unknown"),
            ("market_cap_bucket", "Unknown"),
            ("type", "Stock"),
            ("notes", "multi_ticker_swing universe"),
        ):
            if col not in swing.columns:
                swing[col] = default
        swing["ticker"] = swing["ticker"].map(_clean_ticker)
        swing["_source_priority"] = 1
        frames.append(swing[["ticker", "sector", "market_cap_bucket", "type", "notes", "_source_priority"]])

    hand = pd.DataFrame({
        "ticker": _HAND_CURATED_TICKERS,
        "sector": "Unknown",
        "market_cap_bucket": "Unknown",
        "type": "Stock",
        "notes": "hand-curated momentum expansion addition",
    })
    hand["_source_priority"] = 2
    frames.append(hand)

    meta = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    if meta.empty:
        return meta
    meta["ticker"] = meta["ticker"].map(_clean_ticker)
    meta = meta[meta["ticker"] != ""].copy()
    meta = meta.sort_values(["ticker", "_source_priority"])

    def first_known(values: pd.Series, default: str = "Unknown") -> str:
        for value in values:
            text = str(value).strip()
            if text and text.lower() != "nan" and text != "Unknown":
                return text
        return default

    records: list[dict] = []
    for ticker, group in meta.groupby("ticker", sort=True):
        asset_type = "ETF" if group["type"].astype(str).str.upper().eq("ETF").any() else first_known(group["type"], "Stock")
        records.append({
            "ticker": ticker,
            # TOS exports do not include sector, so borrow swing sector when present.
            "sector": first_known(group["sector"]),
            # TOS usually has current market cap; swing fills names not in the scan.
            "market_cap_bucket": first_known(group["market_cap_bucket"]),
            "type": asset_type,
            "notes": first_known(group["notes"], ""),
        })
    meta = pd.DataFrame.from_records(records)
    meta["is_context_only"] = meta["ticker"].isin(_CONTEXT_ONLY_SET)
    return meta.reset_index(drop=True)


def get_candidate_pool(*, include_context_only: bool = False) -> list[str]:
    """
    Default candidate pool: union of the Thinkorswim scan, full swing universe
    including ETFs, and a small hand-curated liquid/high-beta supplement.

    Context-only regime ETFs are kept in metadata but excluded from the default
    training/trading pool. Fetch them with fetch_context_bars() instead.
    """
    meta = load_candidate_metadata()
    if meta.empty:
        return sorted(_HAND_CURATED_TICKERS)
    if not include_context_only and "is_context_only" in meta.columns:
        meta = meta[~meta["is_context_only"]].copy()
    return sorted(meta["ticker"].astype(str).tolist())


def filter_context_only_tickers(tickers: Iterable[str]) -> list[str]:
    """Remove regime/context-only instruments from a candidate ticker list."""
    return [t for t in dict.fromkeys(_clean_ticker(t) for t in tickers) if t and t not in _CONTEXT_ONLY_SET]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class TickerSnapshotMetrics:
    ticker:        str
    last_close:    float
    avg_dollar_vol: float
    rs_5:          float
    rs_20:         float
    rs_60:         float
    atr_expand:    float
    rvol:          float
    composite:     float


def _load_daily(ticker: str) -> pd.DataFrame | None:
    p = RAW_1D_DIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _normalized_rank(values: pd.Series) -> pd.Series:
    """Rank to [0, 1] with NaN preserved."""
    ranked = values.rank(pct=True, method="average")
    return ranked


def score_universe(
    *,
    as_of: pd.Timestamp,
    candidates: Iterable[str],
    benchmark_ticker: str = "SPY",
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Score every candidate that has enough daily history as of `as_of`.

    Returns a DataFrame indexed by ticker with metric columns and a
    `composite` score (higher = better momentum/expansion candidate).
    """
    cfg = {**UNIVERSE_CONFIG, **(cfg or {})}
    rs_lookbacks = tuple(cfg["rs_lookbacks"])
    min_history_days = int(cfg["min_history_days"])
    score_weights = cfg["score_weights"]

    bench = _load_daily(benchmark_ticker)
    if bench is None:
        raise FileNotFoundError(
            f"Benchmark daily bars missing for {benchmark_ticker}. Run bar download first."
        )
    bench = bench.loc[bench.index <= as_of]
    if len(bench) < max(rs_lookbacks) + 5:
        raise ValueError(f"Not enough benchmark history before {as_of}.")

    bench_close = bench["close"]
    bench_rs = {
        n: bench_close.iloc[-1] / bench_close.iloc[-1 - n] - 1.0
        for n in rs_lookbacks
        if len(bench_close) > n + 1
    }

    rows: list[dict] = []
    for ticker in candidates:
        df = _load_daily(ticker)
        if df is None:
            continue
        df = df.loc[df.index <= as_of]
        if len(df) < min_history_days:
            continue

        c = df["close"]
        v = df["volume"]
        last_close = float(c.iloc[-1])
        if not (cfg["min_price"] <= last_close <= cfg["max_price"]):
            continue

        dollar_vol = (c * v).rolling(30).mean().iloc[-1]
        if not np.isfinite(dollar_vol) or dollar_vol < cfg["min_avg_dollar_vol"]:
            continue

        # Relative strength vs benchmark over each lookback
        rs_vals: dict[int, float] = {}
        for n in rs_lookbacks:
            if len(c) <= n + 1 or n not in bench_rs:
                rs_vals[n] = np.nan
                continue
            stk_ret = c.iloc[-1] / c.iloc[-1 - n] - 1.0
            rs_vals[n] = stk_ret - bench_rs[n]

        # ATR expansion: ATR(14) / ATR(60)
        h, lo, c_p = df["high"], df["low"], c.shift(1)
        tr = pd.concat([(h - lo), (h - c_p).abs(), (lo - c_p).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        atr_60 = tr.rolling(60).mean()
        atr_expand = float(atr_14.iloc[-1] / atr_60.iloc[-1]) if atr_60.iloc[-1] > 0 else np.nan

        # RVOL: today vs 20-day mean
        rvol_window = int(cfg["rvol_window"])
        rvol = float(v.iloc[-1] / v.rolling(rvol_window).mean().iloc[-1]) if v.rolling(rvol_window).mean().iloc[-1] > 0 else np.nan

        rows.append({
            "ticker":         ticker,
            "last_close":     last_close,
            "avg_dollar_vol": float(dollar_vol),
            "rs_5":           rs_vals.get(rs_lookbacks[0], np.nan),
            "rs_20":          rs_vals.get(rs_lookbacks[1], np.nan) if len(rs_lookbacks) > 1 else np.nan,
            "rs_60":          rs_vals.get(rs_lookbacks[2], np.nan) if len(rs_lookbacks) > 2 else np.nan,
            "atr_expand":     atr_expand,
            "rvol":           rvol,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "last_close", "avg_dollar_vol",
            "rs_5", "rs_20", "rs_60", "atr_expand", "rvol", "composite",
        ]).set_index("ticker")

    df_metrics = pd.DataFrame(rows).set_index("ticker")

    # Convert each metric to a [0, 1] cross-sectional rank, then weighted sum
    rs5_r   = _normalized_rank(df_metrics["rs_5"])
    rs20_r  = _normalized_rank(df_metrics["rs_20"])
    rs60_r  = _normalized_rank(df_metrics["rs_60"])
    atr_r   = _normalized_rank(df_metrics["atr_expand"])
    dvol_r  = _normalized_rank(df_metrics["avg_dollar_vol"])

    composite = (
        score_weights["rs_5"]       * rs5_r.fillna(0.5)
        + score_weights["rs_20"]    * rs20_r.fillna(0.5)
        + score_weights["rs_60"]    * rs60_r.fillna(0.5)
        + score_weights["atr_expand"] * atr_r.fillna(0.5)
        + score_weights["dollar_vol"] * dvol_r.fillna(0.5)
    )
    df_metrics["composite"] = composite
    return df_metrics.sort_values("composite", ascending=False)


# ---------------------------------------------------------------------------
# Snapshot writer / reader
# ---------------------------------------------------------------------------

def _snapshot_path(as_of: pd.Timestamp) -> Path:
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    return UNIVERSE_DIR / f"universe_{as_of.strftime('%Y-%m-%d')}.csv"


def write_weekly_snapshot(
    *,
    as_of: pd.Timestamp,
    candidates: Iterable[str] | None = None,
    cfg: dict | None = None,
) -> Path:
    """
    Build and persist the weekly universe snapshot.

    `as_of` should be a Sunday-of-week (or any date — the snapshot is keyed
    by that date). Backtests look up the snapshot dated <= the bar.
    """
    cfg = {**UNIVERSE_CONFIG, **(cfg or {})}
    if candidates is None:
        candidates = get_candidate_pool()
    else:
        candidates = filter_context_only_tickers(candidates)
    scored = score_universe(as_of=as_of, candidates=candidates, cfg=cfg)
    if scored.empty:
        logger.warning("No candidates scored for as_of=%s — wrote empty snapshot", as_of)

    top = scored.head(int(cfg["max_universe_size"])).reset_index()
    top["as_of"] = as_of.strftime("%Y-%m-%d")
    out = _snapshot_path(as_of)
    top.to_csv(out, index=False)
    logger.info("Universe snapshot %s written (%d names)", out.name, len(top))
    return out


def list_snapshots() -> list[Path]:
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(UNIVERSE_DIR.glob("universe_*.csv"))


def load_snapshot_for(date_or_ts: pd.Timestamp | str) -> pd.DataFrame:
    """
    Load the most recent universe snapshot dated <= the given timestamp.

    Returns an empty DataFrame if no snapshot exists at or before that date.
    """
    # Snapshot filenames are tz-naive dates; drop any tz from the query so a
    # tz-aware live `bar_ts` (pd.Timestamp.now(tz="UTC")) can be compared to them
    # without raising "cannot compare tz-naive and tz-aware timestamps".
    target = pd.Timestamp(date_or_ts)
    if target.tzinfo is not None:
        target = target.tz_localize(None)
    target = target.normalize()
    candidates = list_snapshots()
    if not candidates:
        return pd.DataFrame()

    chosen: Path | None = None
    for p in candidates:
        # filename: universe_YYYY-MM-DD.csv
        try:
            stem_date = pd.Timestamp(p.stem.replace("universe_", ""))
        except Exception:
            continue
        if stem_date <= target:
            chosen = p
        else:
            break
    if chosen is None:
        return pd.DataFrame()
    return pd.read_csv(chosen)


def build_snapshots_over_range(
    *,
    start: str | pd.Timestamp,
    end:   str | pd.Timestamp,
    candidates: Iterable[str] | None = None,
    weekday: int = 6,
    cfg: dict | None = None,
) -> list[Path]:
    """
    Build a snapshot for every `weekday` (default Sunday) between start and end.

    Used for backtest setup — writes one snapshot per week so the backtester
    can read the appropriate one at each test bar.
    """
    if candidates is None:
        candidates = get_candidate_pool()
    else:
        candidates = filter_context_only_tickers(candidates)
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    out: list[Path] = []
    cur = start_ts
    while cur.weekday() != weekday:
        cur = cur + pd.Timedelta(days=1)
    while cur <= end_ts:
        try:
            p = write_weekly_snapshot(as_of=cur, candidates=candidates, cfg=cfg)
            out.append(p)
        except Exception as exc:
            logger.warning("snapshot %s failed: %s", cur.date(), exc)
        cur = cur + pd.Timedelta(days=7)
    return out
