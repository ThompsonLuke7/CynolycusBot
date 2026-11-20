"""Local inputs for the personal portfolio page.

Everything personal lives under ``Data/personal/`` (gitignored):
  watchlist.csv        ticker, note, added_at; edited from the page
  config.json          optional overrides of PersonalConfig (unknown keys fail)
  look_through.json    optional {"ETF": {"underlying": "XYZ", "leverage": 2}} overrides
  bars_1d/             Alpaca SIP daily bars, split+dividend adjusted, for held names,
                       their underlyings, the watchlist and SPY/QQQ/IWM only
  earnings.parquet     yfinance earnings history for those names (separate from the
                       shared calendar, whose history writer replaces ALL tickers' rows)
  flag_log.jsonl       one record per ticker per ET date, for later evaluation
  equity_log.csv       one net-liquidation row per ET date
  transactions.jsonl   append-only Schwab TRADE + RECEIVE_AND_DELIVER records (trade_history)
  symbol_map.json      optional {"CUSIP or retired symbol": "TICKER"} for trade-history chart bars
  history_bars/        raw + split-adjusted daily and per-trade intraday bars (trade_bars)
  reports/             trade_review_{date}.md (trade_review)

Read from research outputs (never written here):
  Data/research/shadow_cross_sections/{decision}.parquet  momentum ranking per decision
  research/long_horizon_discount_2026-09-25/shadow/ledger.csv + marks_latest.txt
  signals/meta_context/meta_ranker/meta_ranker_matrix.parquet  latest dynamic theme label
"""
from __future__ import annotations

import csv
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from core.calendar import is_trading_day, prev_trading_day

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
REPO = Path(__file__).resolve().parents[2]
PERSONAL_DIR = REPO / "Data" / "personal"
CROSS_SECTION_DIR = REPO / "Data" / "research" / "shadow_cross_sections"
SHADOW_DIR = REPO / "research" / "long_horizon_discount_2026-09-25" / "shadow"
META_MATRIX = REPO / "signals" / "meta_context" / "meta_ranker" / "meta_ranker_matrix.parquet"
BENCHMARKS = ("SPY", "QQQ", "IWM")
BAR_HISTORY_DAYS = 560          # 252 sessions for beta + 200 for the SMA, with room for holidays
BAR_FINAL_AFTER = time(16, 30)  # the daily bar is final 30 min after the close (as the shadow ledger)
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


@dataclass(frozen=True)
class PersonalConfig:
    """Review thresholds. They raise flags only; nothing acts on them."""

    max_underlying_pct: float = 0.25    # look-through exposure to one name, share of net liquidation
    max_cluster_pct: float = 0.50       # exposure to one correlated cluster
    max_gross_exposure: float = 1.50    # sum of |look-through exposure| / net liquidation
    cluster_corr: float = 0.60          # daily-return correlation that joins two names into a cluster
    option_expiry_warn_days: int = 7    # calendar days
    wide_spread_pct: float = 0.10       # option bid/ask spread as a share of the mark
    beta_window: int = 252
    vol_window: int = 63
    shrug_window_days: int = 91         # research 05/07: a shrugged miss stays a flag for 91 days
    earnings_soon_days: int = 7
    late_loss_frac: float = 0.5         # a long option at or below this share of its cost...
    late_loss_dte: int = 14             # ...with this few calendar days left (trade_timing.comeback_stats)


def personal_dir(base: Path | None = None) -> Path:
    return Path(base) if base is not None else PERSONAL_DIR


def load_config(base: Path | None = None) -> PersonalConfig:
    path = personal_dir(base) / "config.json"
    if not path.exists():
        return PersonalConfig()
    raw = json.loads(path.read_text())
    known = {f.name for f in fields(PersonalConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}; allowed {sorted(known)}")
    return PersonalConfig(**{**asdict(PersonalConfig()), **raw})


def load_look_through_overrides(base: Path | None = None) -> dict[str, dict]:
    path = personal_dir(base) / "look_through.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out = {}
    for etf, spec in raw.items():
        if not isinstance(spec, dict) or "underlying" not in spec or "leverage" not in spec:
            raise ValueError(f"{path}: {etf} needs both 'underlying' and 'leverage'")
        out[str(etf).upper()] = {"underlying": str(spec["underlying"]).upper(), "leverage": float(spec["leverage"])}
    return out


# ---------- watchlist ----------
def normalize_ticker(ticker: str) -> str:
    t = str(ticker or "").strip().upper().lstrip("$")
    if not TICKER_RE.match(t):
        raise ValueError(f"invalid ticker: {ticker!r}")
    return t


def read_watchlist(base: Path | None = None) -> list[dict]:
    path = personal_dir(base) / "watchlist.csv"
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return [{"ticker": r["ticker"], "note": r.get("note") or "", "added_at": r.get("added_at") or ""}
                for r in csv.DictReader(fh) if r.get("ticker")]


def _write_watchlist(rows: list[dict], base: Path | None) -> None:
    path = personal_dir(base) / "watchlist.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "note", "added_at"])
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def add_to_watchlist(ticker: str, note: str = "", base: Path | None = None, now: datetime | None = None) -> list[dict]:
    t = normalize_ticker(ticker)
    note = str(note or "").strip()[:200]
    rows = [r for r in read_watchlist(base) if r["ticker"] != t]
    stamp = (now or datetime.now(timezone.utc)).astimezone(ET).strftime("%Y-%m-%d")
    old = next((r for r in read_watchlist(base) if r["ticker"] == t), None)
    rows.append({"ticker": t, "note": note, "added_at": old["added_at"] if old else stamp})
    _write_watchlist(sorted(rows, key=lambda r: r["ticker"]), base)
    return rows


def remove_from_watchlist(ticker: str, base: Path | None = None) -> list[dict]:
    t = normalize_ticker(ticker)
    rows = read_watchlist(base)
    if not any(r["ticker"] == t for r in rows):
        raise KeyError(f"{t} is not on the watchlist")
    rows = [r for r in rows if r["ticker"] != t]
    _write_watchlist(rows, base)
    return rows


# ---------- daily bars ----------
def last_completed_session(now: datetime) -> date:
    et = now.astimezone(ET)
    d = et.date()
    if not is_trading_day(d) or et.time() < BAR_FINAL_AFTER:
        d = prev_trading_day(d)
    return d


def _bar_dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)


def _last_bar_date(path: Path) -> date | None:
    try:
        df = pd.read_parquet(path, columns=["timestamp"])
    except (OSError, ValueError):
        return None
    return None if df.empty else _bar_dates(df).max().date()


def _fetch_bars(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

    # limit=None: alpaca-py's limit caps TOTAL bars, not a page (2026-11-08 finding).
    return fetch_intraday(ticker=ticker, start=start, end=end, timeframe="1Day", limit=None,
                          adjustment="all", feed="sip", save_path="", skip_delisted=False)


def refresh_bars(tickers: list[str], now: datetime, base: Path | None = None, fetch=_fetch_bars) -> dict[str, str]:
    """Refetch each ticker whose file is missing or ends before the last completed session."""
    out_dir = personal_dir(base) / "bars_1d"
    out_dir.mkdir(parents=True, exist_ok=True)
    want = last_completed_session(now)
    # The SIP feed's newest 15 minutes need a paid entitlement; end 20 minutes back (as the research fetcher).
    end = now.astimezone(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=BAR_HISTORY_DAYS)

    def one(t: str) -> tuple[str, str]:
        path = out_dir / f"{t}.parquet"
        last = _last_bar_date(path) if path.exists() else None
        if last is not None and last >= want:
            return t, "current"
        try:
            df = fetch(t, start, end)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the rest
            return t, f"error: {exc}"[:160]
        if df is None or df.empty:
            return t, "empty"
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        return t, "fetched"

    with ThreadPoolExecutor(max_workers=4) as ex:
        return dict(ex.map(one, sorted(set(tickers))))


def load_bars(tickers: list[str], base: Path | None = None) -> pd.DataFrame:
    """Daily closes, one column per ticker, indexed by ET session date. Missing files are absent columns."""
    series = {}
    for t in sorted(set(tickers)):
        path = personal_dir(base) / "bars_1d" / f"{t}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["timestamp", "close", "volume"])
        if df.empty:
            continue
        s = pd.Series(df["close"].to_numpy(copy=True), index=_bar_dates(df), name=t)
        series[t] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(series).sort_index() if series else pd.DataFrame()


# ---------- earnings ----------
def _fetch_earnings(ticker: str) -> pd.DataFrame | None:
    from signals.events.earnings_calendar import _fetch_one

    return _fetch_one(ticker, retries=2)


def refresh_earnings(tickers: list[str], now: datetime, base: Path | None = None, fetch=_fetch_earnings) -> dict[str, str]:
    """Refetch a ticker's earnings history at most once per ET date. Empty answers are remembered too."""
    path = personal_dir(base) / "earnings.parquet"
    today = now.astimezone(ET).date().isoformat()
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ticker", "fetched_on"])
    fresh = set(old.loc[old["fetched_on"] == today, "ticker"])
    todo = [t for t in sorted(set(tickers)) if t not in fresh]
    if not todo:
        return {}
    status, frames = {}, []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for t, df in zip(todo, ex.map(fetch, todo)):
            if df is None or df.empty:
                status[t] = "empty"
                frames.append(pd.DataFrame({"ticker": [t], "fetched_on": [today]}))
            else:
                status[t] = "fetched"
                frames.append(df.assign(ticker=t, fetched_on=today))
    keep = old[~old["ticker"].isin(todo)]
    out = pd.concat([keep, *frames], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return status


def load_earnings(base: Path | None = None) -> pd.DataFrame:
    path = personal_dir(base) / "earnings.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "date", "eps_estimate", "reported_eps", "surprise_pct", "fetched_on"])
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


# ---------- research outputs ----------
def load_momentum_context(cross_section_dir: Path = CROSS_SECTION_DIR, shadow_dir: Path = SHADOW_DIR) -> dict:
    """The newest saved momentum ranking, the research sleeve picked from it, and the forward record."""
    files = sorted(cross_section_dir.glob("????-??-??.parquet")) if cross_section_dir.exists() else []
    xs = pd.read_parquet(files[-1]) if files else None
    decision = files[-1].stem if files else None
    sleeve: list[str] = []
    ledger = shadow_dir / "ledger.csv"
    if ledger.exists() and decision:
        led = pd.read_csv(ledger)
        sleeve = sorted(led.loc[(led["decision_date"] == decision) & (led["arm"] == "mom_top300"), "ticker"])
    record = {}
    marks = shadow_dir / "marks_latest.txt"
    if marks.exists():
        for line in marks.read_text().splitlines():
            if m := re.search(r"marked through (\d{4}-\d{2}-\d{2}).*next scheduled decision (\d{4}-\d{2}-\d{2})", line):
                record.update(marked_through=m[1], next_decision=m[2])
            if line.startswith("cumulative since the first entry:"):
                record["cumulative"] = dict(re.findall(r"(\w+) ([+-]\d+\.\d+)%", line))
    return {"decision_date": decision, "cross_section": xs, "sleeve": sleeve, "record": record}


def load_themes(tickers: list[str], matrix: Path = META_MATRIX) -> dict[str, str]:
    """Latest dynamic-theme label per ticker. Labels churn weekly and can be wrong; show them as context only."""
    if not tickers or not matrix.exists():
        return {}
    df = pd.read_parquet(matrix, columns=["theme"], filters=[("ticker", "in", sorted(set(tickers)))]).reset_index()
    if df.empty:
        return {}
    df = df.sort_values("timestamp").groupby("ticker").tail(1)
    return {str(r.ticker): str(r.theme) for r in df.itertuples() if r.theme and str(r.theme) != "None"}


# ---------- logs ----------
def append_flag_log(records: list[dict], day: str, base: Path | None = None) -> bool:
    """Write the day's records once. Returns False if that ET date is already logged."""
    path = personal_dir(base) / "flag_log.jsonl"
    if path.exists():
        last = ""
        with path.open() as fh:
            for last in fh:
                pass
        if last.strip() and json.loads(last).get("date") == day:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for rec in records:
            fh.write(json.dumps({"date": day, **rec}, default=str, allow_nan=False) + "\n")
    return True


def upsert_equity_log(day: str, net_liq: float, spy_close: float | None, base: Path | None = None) -> None:
    path = personal_dir(base) / "equity_log.csv"
    old = pd.read_csv(path, dtype={"date": str}).to_dict("records") if path.exists() else []
    rows = pd.DataFrame([r for r in old if r["date"] != day] + [{"date": day, "net_liq": net_liq, "spy_close": spy_close}])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    rows.sort_values("date").to_csv(tmp, index=False)
    tmp.replace(path)
