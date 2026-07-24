"""Dynamic ticker discovery + promotion gate for the shared universe.

The shared universe historically only *re-ranked* a static seed pool. This
module makes it *discover* new names and promote them when they earn their way
in:

  1. Broad nightly market screen (Alpaca active US equities -> daily price/ADV
     filter -> market-cap band). Any name that crosses into the liquidity band
     becomes a candidate -- this is how a small-cap becomes investable after a
     run.
  2. Catalyst-driven discovery (tickers mentioned in the news/catalyst ledger
     that are not yet in the universe) -- catches a name on day 1 of a catalyst
     before its average dollar volume has built.

Both feeds land in a *pending* queue (``pending_tickers.csv``). The promotion
gate evaluates the queue every run: a pending name is **released** (status
``promoted``) only once it meets the price / ADV / market-cap minimums *and* has
enough daily history to be scored (the natural quarantine -- momentum scoring
needs >= ``min_history_days`` bars). Promoted tickers are then folded into
``shared_universe.csv`` as eligible by ``core.shared_universe.universe``.

Network access (Alpaca, yfinance) is isolated in the ``fetch_*``/``run_*``
helpers; the filtering / merge / promotion logic is pure and unit-tested.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from core.shared_universe.universe import (
    DATA_DIR,
    PENDING_NOVEL_TICKERS_CSV,
    PENDING_TICKERS_CSV,
    clean_ticker,
    shared_tickers,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DAILY_DIR = REPO_ROOT / "Data" / "shared" / "bars" / "1d"
NEWS_RECORDS = REPO_ROOT / "signals" / "news" / "data" / "processed" / "news_records.parquet"
LIVE_CATALYST_RECORDS = REPO_ROOT / "signals" / "news" / "data" / "processed" / "live_catalyst_records.parquet"

VALID_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")

PENDING_COLUMNS = [
    "ticker",
    "source",
    "status",
    "first_seen",
    "last_seen",
    "last_price",
    "avg_dollar_volume_20d",
    "market_cap",
    "cap_tier",
    "stock_only",
    "history_days",
    "atr_expand",
    "catalyst_mentions",
    "passes_price",
    "passes_adv",
    "passes_market_cap",
    "passes_history",
    "passes_atr",
    "passes_observation",
    "meets_all",
    "promoted_date",
    "reject_reason",
]


@dataclass(frozen=True)
class DiscoveryConfig:
    """Thresholds a candidate must clear to be released from the pending queue.

    Defaults lower the market-cap floor to $100M so small caps are discoverable,
    while keeping a real liquidity floor so we do not promote names the option
    layer (and even stock fills) cannot handle.
    """

    min_price: float = 3.0
    max_price: float = 1000.0
    min_market_cap: float = 100_000_000.0
    min_avg_dollar_volume_20d: float = 4_000_000.0
    min_history_days: int = 200
    min_observation_days: int = 2
    adv_window: int = 20
    # Movement gate: don't promote names that barely move. atr_expand = ATR(14)/
    # ATR(60) — the recent 14-day true range vs the 60-day baseline. >1.0 means the
    # name is moving more than its own recent norm (an active mover); a quiet/dead
    # name sits at/under 1.0. Set 0.0 to disable. (This is the universe-level slow-
    # mover filter — it replaces the old momentum-snapshot ATR filter.)
    min_atr_expand: float = 1.10
    # Sub-$1B names route stock-only: listed options are unreliable / absent.
    options_market_cap_floor: float = 1_000_000_000.0
    catalyst_lookback_days: int = 30
    catalyst_min_mentions: int = 2
    # Cap yfinance enrichment on catalyst hits (highest-mention names first).
    catalyst_max_enrich: int = 150
    benchmark_exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX", "ARCA", "BATS")
    # Asset-class symbols to skip even if Alpaca lists them as equities.
    exclude_substrings: tuple[str, ...] = field(default=("TEST",))


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested, no IO)
# ---------------------------------------------------------------------------

def cap_tier(market_cap: float | None) -> str:
    """Coarse market-cap bucket aligned with the momentum bucketer naming."""
    if market_cap is None or not np.isfinite(market_cap) or market_cap <= 0:
        return "unknown"
    cap_m = float(market_cap) / 1e6
    if cap_m >= 200_000:
        return "mega"
    if cap_m >= 10_000:
        return "large"
    if cap_m >= 2_000:
        return "mid"
    if cap_m >= 300:
        return "small"
    if cap_m >= 50:
        return "micro"
    return "nano"


def is_stock_only(market_cap: float | None, cfg: DiscoveryConfig) -> bool:
    """Sub-floor caps should trade as stock (no reliable option chain)."""
    if market_cap is None or not np.isfinite(market_cap):
        return True
    return float(market_cap) < cfg.options_market_cap_floor


def compute_daily_metrics(bars: pd.DataFrame, *, adv_window: int = 20) -> dict[str, float]:
    """Last price, 20d average dollar volume, and history length from daily bars.

    Accepts a frame with ``close``/``volume`` columns (case-insensitive). Returns
    NaNs / 0 when the frame is empty so callers can filter uniformly.
    """
    _empty = {"last_price": float("nan"), "avg_dollar_volume_20d": float("nan"),
              "history_days": 0, "atr_expand": float("nan")}
    if bars is None or bars.empty:
        return dict(_empty)
    df = bars.rename(columns={c: str(c).lower() for c in bars.columns})
    if "close" not in df.columns or "volume" not in df.columns:
        return dict(_empty)
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    dollar_vol = (close * volume).tail(adv_window)
    return {
        "last_price": float(close.iloc[-1]) if len(close) else float("nan"),
        "avg_dollar_volume_20d": float(dollar_vol.mean()) if len(dollar_vol) else float("nan"),
        "history_days": int(close.notna().sum()),
        "atr_expand": _atr_expand(df, close),
    }


def _atr_expand(df: pd.DataFrame, close: pd.Series, *, fast: int = 14, slow: int = 60) -> float:
    """ATR(fast)/ATR(slow): recent true-range vs the longer baseline. >1 = moving
    more than its own norm. NaN when high/low missing or <slow bars of history."""
    if "high" not in df.columns or "low" not in df.columns:
        return float("nan")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    if int(tr.notna().sum()) < slow:
        return float("nan")
    atr_fast = tr.rolling(fast).mean().iloc[-1]
    atr_slow = tr.rolling(slow).mean().iloc[-1]
    if not np.isfinite(atr_slow) or atr_slow <= 0:
        return float("nan")
    return float(atr_fast / atr_slow)


def normalize_exchange(value: object) -> str:
    """Normalize Alpaca exchange enums/strings to stable short codes."""
    raw = getattr(value, "value", None) or getattr(value, "name", None) or value
    exchange = str(raw or "").strip().upper().replace("ASSETEXCHANGE.", "")
    aliases = {
        "NYSEARCA": "ARCA",
    }
    return aliases.get(exchange, exchange)


def screen_filters(row: dict, cfg: DiscoveryConfig) -> dict:
    """Evaluate the price / ADV / market-cap / history gates for one candidate."""
    price = row.get("last_price")
    adv = row.get("avg_dollar_volume_20d")
    mcap = row.get("market_cap")
    hist = row.get("history_days", 0)
    atr_expand = row.get("atr_expand")

    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    price, adv, mcap = _num(price), _num(adv), _num(mcap)
    passes_price = np.isfinite(price) and (cfg.min_price <= price <= cfg.max_price)
    passes_adv = np.isfinite(adv) and adv >= cfg.min_avg_dollar_volume_20d
    # Missing market cap is treated as a soft fail (cannot confirm the band).
    passes_market_cap = np.isfinite(mcap) and mcap >= cfg.min_market_cap
    try:
        hist_num = int(float(hist)) if hist is not None and np.isfinite(float(hist)) else 0
    except (TypeError, ValueError):
        hist_num = 0
    passes_history = hist_num >= cfg.min_history_days
    # Movement gate: require demonstrated range expansion. Disabled when
    # min_atr_expand <= 0. Fail-closed on a non-finite value (can't confirm it moves).
    atr_val = _num(atr_expand)
    passes_atr = (
        cfg.min_atr_expand <= 0.0
        or (np.isfinite(atr_val) and atr_val >= cfg.min_atr_expand)
    )
    first_seen = pd.to_datetime(row.get("first_seen"), errors="coerce")
    today = pd.Timestamp(row.get("_evaluation_date") or _today_str())
    observation_days = (
        int((today.normalize() - first_seen.normalize()).days)
        if pd.notna(first_seen)
        else cfg.min_observation_days
    )
    passes_observation = observation_days >= cfg.min_observation_days
    meets_all = bool(
        passes_price
        and passes_adv
        and passes_market_cap
        and passes_history
        and passes_atr
        and passes_observation
    )
    return {
        "passes_price": bool(passes_price),
        "passes_adv": bool(passes_adv),
        "passes_market_cap": bool(passes_market_cap),
        "passes_history": bool(passes_history),
        "passes_atr": bool(passes_atr),
        "passes_observation": bool(passes_observation),
        "meets_all": meets_all,
    }


def _today_str() -> str:
    return dt.date.today().isoformat()


def merge_pending(
    existing: pd.DataFrame,
    discovered: pd.DataFrame,
    *,
    today: str | None = None,
) -> pd.DataFrame:
    """Upsert freshly-discovered candidates into the pending queue.

    Preserves ``first_seen`` and an already-``promoted`` status; refreshes
    ``last_seen``, the latest metrics, and unions the discovery source.
    """
    today = today or _today_str()
    existing = existing.copy() if existing is not None else pd.DataFrame(columns=PENDING_COLUMNS)
    discovered = discovered.copy() if discovered is not None else pd.DataFrame(columns=["ticker"])
    if not existing.empty:
        existing["ticker"] = existing["ticker"].map(clean_ticker)
    by_ticker = {row["ticker"]: dict(row) for _, row in existing.iterrows()} if not existing.empty else {}

    for _, new in discovered.iterrows():
        ticker = clean_ticker(new.get("ticker"))
        if not ticker:
            continue
        prior = by_ticker.get(ticker, {})
        merged = dict(prior)
        merged["ticker"] = ticker
        merged["first_seen"] = prior.get("first_seen") or today
        merged["last_seen"] = today
        # Union of discovery sources (screener / catalyst / both).
        new_source = str(new.get("source") or "").strip()
        prior_source = str(prior.get("source") or "").strip()
        sources = sorted({s for s in (prior_source, new_source) if s})
        merged["source"] = "both" if len(sources) > 1 else (sources[0] if sources else "")
        # Latest metrics win (the nightly screen is authoritative).
        for col in ("last_price", "avg_dollar_volume_20d", "market_cap", "cap_tier", "stock_only", "history_days", "catalyst_mentions"):
            if col in new and pd.notna(new.get(col)):
                merged[col] = new.get(col)
            elif col not in merged:
                merged[col] = new.get(col, np.nan)
        merged.setdefault("status", "pending")
        by_ticker[ticker] = merged

    out = pd.DataFrame(list(by_ticker.values()))
    for col in PENDING_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[PENDING_COLUMNS].sort_values("ticker").reset_index(drop=True)


def evaluate_promotions(
    pending: pd.DataFrame,
    cfg: DiscoveryConfig,
    *,
    already_eligible: Iterable[str] = (),
    today: str | None = None,
) -> pd.DataFrame:
    """Apply the promotion gate to every pending row.

    A name is released (``promoted``) once it clears all gates and is not already
    in the universe. Names already promoted stay promoted (latched). Everything
    else stays ``pending`` with the failing gate recorded in ``reject_reason``.
    """
    today = today or _today_str()
    already = {clean_ticker(t) for t in already_eligible}
    pending = pending.copy()
    if pending.empty:
        return pending

    statuses, reasons, promoted_dates = [], [], []
    flag_cols = {
        k: []
        for k in (
            "passes_price",
            "passes_adv",
            "passes_market_cap",
            "passes_history",
            "passes_atr",
            "passes_observation",
            "meets_all",
        )
    }
    tiers, stock_only_flags = [], []
    for _, row in pending.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        mcap = row.get("market_cap")
        flags = screen_filters({**dict(row), "_evaluation_date": today}, cfg)
        for k in flag_cols:
            flag_cols[k].append(flags[k])
        tiers.append(cap_tier(mcap if pd.notna(mcap) else None))
        stock_only_flags.append(is_stock_only(mcap if pd.notna(mcap) else None, cfg))

        prior_status = str(row.get("status") or "pending")
        premature_self_promotion = bool(
            prior_status == "promoted"
            and not flags["passes_observation"]
            and str(row.get("first_seen") or "") == str(row.get("promoted_date") or "")
        )
        if (ticker in already or prior_status == "promoted") and not premature_self_promotion:
            statuses.append("promoted")
            reasons.append("")
            promoted_dates.append(row.get("promoted_date") if pd.notna(row.get("promoted_date")) else today)
        elif flags["meets_all"]:
            statuses.append("promoted")
            reasons.append("")
            promoted_dates.append(today)
        else:
            failed = [
                name.replace("passes_", "")
                for name in (
                    "passes_price",
                    "passes_adv",
                    "passes_market_cap",
                    "passes_history",
                    "passes_atr",
                    "passes_observation",
                )
                if not flags[name]
            ]
            statuses.append("pending")
            reasons.append("needs_" + "+".join(failed) if failed else "")
            promoted_dates.append(row.get("promoted_date") if pd.notna(row.get("promoted_date")) else "")

    pending["status"] = statuses
    pending["reject_reason"] = reasons
    pending["promoted_date"] = promoted_dates
    for k, vals in flag_cols.items():
        pending[k] = vals
    pending["cap_tier"] = tiers
    pending["stock_only"] = stock_only_flags
    return pending.reset_index(drop=True)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_pending(path: Path = PENDING_TICKERS_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=PENDING_COLUMNS)
    df = pd.read_csv(path)
    for col in PENDING_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    numeric_cols = ["last_price", "avg_dollar_volume_20d", "market_cap", "history_days", "catalyst_mentions"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[PENDING_COLUMNS]


def save_pending(df: pd.DataFrame, path: Path = PENDING_TICKERS_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_novel_pending_view(
    df: pd.DataFrame,
    *,
    known_tickers: Iterable[str],
    path: Path = PENDING_NOVEL_TICKERS_CSV,
) -> Path:
    """Persist pending candidates that are not already in the shared universe."""
    known = {clean_ticker(ticker) for ticker in known_tickers}
    novel = df[
        (df["status"].astype(str) == "pending")
        & (~df["ticker"].map(clean_ticker).isin(known))
    ].copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    novel.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Network feeds (isolated so the logic above stays testable offline)
# ---------------------------------------------------------------------------

def fetch_alpaca_assets(cfg: DiscoveryConfig, *, env_file: str = ".env") -> pd.DataFrame:
    """All active, tradable US-equity symbols from Alpaca (free endpoint)."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    from core.API.Alpaca_API.core.config import AlpacaConfig

    creds = AlpacaConfig.from_env(env_file)
    client = TradingClient(creds.key_id, creds.secret_key, paper=True)
    request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    rows = []
    for asset in client.get_all_assets(request):
        symbol = clean_ticker(asset.symbol)
        if not VALID_SYMBOL_RE.match(symbol):
            continue
        if not getattr(asset, "tradable", False):
            continue
        exchange = normalize_exchange(getattr(asset, "exchange", "") or "")
        if cfg.benchmark_exchanges and exchange not in cfg.benchmark_exchanges:
            continue
        if any(sub in symbol for sub in cfg.exclude_substrings):
            continue
        rows.append({"ticker": symbol, "name": getattr(asset, "name", ""), "exchange": exchange})
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    logger.info("Alpaca assets: %d tradable US equities after filter", len(df))
    return df


def _local_daily_metrics(ticker: str, cfg: DiscoveryConfig) -> dict[str, float] | None:
    path = SHARED_DAILY_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        bars = pd.read_parquet(path)
    except Exception:
        return None
    return compute_daily_metrics(bars, adv_window=cfg.adv_window)


def fetch_daily_metrics(
    tickers: Sequence[str],
    cfg: DiscoveryConfig,
    *,
    env_file: str = ".env",
    lookback_days: int = 420,
    batch_size: int = 200,
) -> pd.DataFrame:
    """Daily price / ADV20 / history for each symbol.

    Reuses the local shared daily-bar cache when present (free, instant) and
    falls back to a batched Alpaca daily-bar pull for unseen symbols.
    """
    tickers = [clean_ticker(t) for t in tickers if clean_ticker(t)]
    metrics: dict[str, dict[str, float]] = {}
    remote_needed: list[str] = []
    for ticker in tickers:
        local = _local_daily_metrics(ticker, cfg)
        if local is not None and local["history_days"] > 0:
            metrics[ticker] = local
        else:
            remote_needed.append(ticker)

    if remote_needed:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        from core.API.Alpaca_API.core.config import AlpacaConfig

        creds = AlpacaConfig.from_env(env_file)
        client = StockHistoricalDataClient(creds.key_id, creds.secret_key)
        start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=lookback_days)
        for i in range(0, len(remote_needed), batch_size):
            batch = remote_needed[i : i + batch_size]
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame(1, TimeFrameUnit.Day),
                    start=start,
                    adjustment=Adjustment.RAW,
                    feed=DataFeed.SIP,
                )
                frame = client.get_stock_bars(request).df
            except Exception as exc:  # pragma: no cover - network
                logger.warning("Alpaca daily batch %d failed: %s", i // batch_size, exc)
                continue
            if frame is None or frame.empty:
                continue
            frame = frame.reset_index()
            for symbol, grp in frame.groupby("symbol"):
                metrics[clean_ticker(symbol)] = compute_daily_metrics(grp, adv_window=cfg.adv_window)

    rows = [{"ticker": t, **m} for t, m in metrics.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "last_price", "avg_dollar_volume_20d", "history_days"])


def fetch_market_caps(
    tickers: Sequence[str], *, request_delay: float = 0.4, request_timeout: float = 10.0
) -> pd.DataFrame:
    """Market cap per ticker via yfinance fast_info (rate-limit friendly).

    ``request_timeout`` bounds each underlying HTTP call. yfinance's default
    session has no timeout, so a broken connection to Yahoo (observed
    2026-07-22: "Connection refused" on one ticker) can hang for tens of
    minutes on a single name -- the per-ticker try/except below only guards
    against *errors*, not a stall -- burning the whole nightly discovery
    step's time budget until the wrapper's hard `timeout` kills it and
    every ticker after the stuck one is silently never attempted. A shared
    session with a real timeout turns that into a fast per-ticker failure so
    the loop keeps moving.
    """
    import time

    import yfinance as yf
    from curl_cffi import requests as curl_requests

    # Match yfinance's own default session construction (browser
    # impersonation avoids Yahoo blocking non-browser-like clients) but with
    # an explicit timeout instead of none.
    session = curl_requests.Session(impersonate="chrome", timeout=request_timeout)

    rows = []
    for ticker in [clean_ticker(t) for t in tickers if clean_ticker(t)]:
        mcap = None
        try:
            yt = yf.Ticker(ticker, session=session)
            mcap = yt.fast_info.get("market_cap")
            if mcap is None:
                mcap = yt.info.get("marketCap")
        except Exception:
            mcap = None
        try:
            mcap = float(mcap) if mcap is not None else np.nan
        except (TypeError, ValueError):
            mcap = np.nan
        rows.append({"ticker": ticker, "market_cap": mcap})
        if request_delay:
            time.sleep(request_delay)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "market_cap"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_broad_screen(
    cfg: DiscoveryConfig,
    *,
    env_file: str = ".env",
    max_symbols: int | None = None,
) -> pd.DataFrame:
    """Alpaca assets -> price/ADV filter -> market-cap band -> candidate frame.

    Market-cap lookups (yfinance) run only on the price+ADV survivors to keep
    the call count bounded.
    """
    assets = fetch_alpaca_assets(cfg, env_file=env_file)
    if assets.empty:
        return pd.DataFrame(columns=["ticker", "source"])
    symbols = assets["ticker"].tolist()
    if max_symbols:
        symbols = symbols[:max_symbols]

    metrics = fetch_daily_metrics(symbols, cfg, env_file=env_file)
    if metrics.empty:
        return pd.DataFrame(columns=["ticker", "source"])

    # Cheap stage: price + ADV before any market-cap call.
    def _passes_liquidity(r: pd.Series) -> bool:
        flags = screen_filters({**r, "market_cap": np.nan, "history_days": cfg.min_history_days}, cfg)
        return flags["passes_price"] and flags["passes_adv"]

    liquid = metrics[metrics.apply(_passes_liquidity, axis=1)].copy()
    logger.info("Broad screen: %d/%d names pass price+ADV", len(liquid), len(metrics))
    if liquid.empty:
        return pd.DataFrame(columns=["ticker", "source"])

    caps = fetch_market_caps(liquid["ticker"].tolist())
    out = liquid.merge(caps, on="ticker", how="left")
    out["cap_tier"] = out["market_cap"].map(cap_tier)
    out["stock_only"] = out["market_cap"].map(lambda m: is_stock_only(m, cfg))
    out["source"] = "screener"
    return out


def run_catalyst_discovery(
    cfg: DiscoveryConfig,
    *,
    universe: Iterable[str] | None = None,
    today: dt.date | None = None,
) -> pd.DataFrame:
    """Tickers in the news/catalyst ledger that are not yet in the universe."""
    today = today or dt.date.today()
    cutoff = pd.Timestamp(today - dt.timedelta(days=cfg.catalyst_lookback_days), tz="UTC")
    known = {clean_ticker(t) for t in (universe if universe is not None else shared_tickers(eligible_only=False))}

    frames = []
    for path in (NEWS_RECORDS, LIVE_CATALYST_RECORDS):
        if not Path(path).exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["ticker", "timestamp"])
        except Exception:
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
        if "ticker" not in df.columns or "timestamp" not in df.columns:
            continue
        frames.append(df[["ticker", "timestamp"]])
    if not frames:
        return pd.DataFrame(columns=["ticker", "source", "catalyst_mentions"])

    news = pd.concat(frames, ignore_index=True)
    news["ticker"] = news["ticker"].map(clean_ticker)
    news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
    recent = news[(news["timestamp"] >= cutoff) & news["ticker"].map(lambda t: bool(VALID_SYMBOL_RE.match(t)))]
    counts = recent.groupby("ticker").size().rename("catalyst_mentions").reset_index()
    counts = counts[
        (~counts["ticker"].isin(known)) & (counts["catalyst_mentions"] >= cfg.catalyst_min_mentions)
    ].copy()
    counts = counts.sort_values("catalyst_mentions", ascending=False).reset_index(drop=True)
    counts["source"] = "catalyst"
    logger.info("Catalyst discovery: %d off-universe names with >= %d mentions", len(counts), cfg.catalyst_min_mentions)
    return counts


def run_discovery(
    cfg: DiscoveryConfig | None = None,
    *,
    env_file: str = ".env",
    do_screen: bool = True,
    do_catalyst: bool = True,
    enrich_catalyst_metrics: bool = True,
    max_symbols: int | None = None,
    pending_path: Path = PENDING_TICKERS_CSV,
) -> dict:
    """Run the feeds, merge into the pending queue, apply the promotion gate.

    Returns a summary dict; persists the updated queue to ``pending_path``.
    """
    cfg = cfg or DiscoveryConfig()
    already_eligible = set(shared_tickers(eligible_only=True))
    already_known = set(shared_tickers(eligible_only=False))

    discovered_frames: list[pd.DataFrame] = []
    if do_screen:
        try:
            discovered_frames.append(run_broad_screen(cfg, env_file=env_file, max_symbols=max_symbols))
        except Exception as exc:
            logger.error("Broad screen failed: %s", exc, exc_info=True)
    if do_catalyst:
        try:
            cat = run_catalyst_discovery(cfg, universe=already_eligible)
            if cfg.catalyst_max_enrich and len(cat) > cfg.catalyst_max_enrich:
                logger.info("Capping catalyst enrichment to top %d of %d hits", cfg.catalyst_max_enrich, len(cat))
                cat = cat.head(cfg.catalyst_max_enrich).copy()
            if enrich_catalyst_metrics and not cat.empty:
                metrics = fetch_daily_metrics(cat["ticker"].tolist(), cfg, env_file=env_file)
                caps = fetch_market_caps(cat["ticker"].tolist())
                cat = cat.merge(metrics, on="ticker", how="left").merge(caps, on="ticker", how="left")
                cat["cap_tier"] = cat["market_cap"].map(cap_tier)
                cat["stock_only"] = cat["market_cap"].map(lambda m: is_stock_only(m, cfg))
            discovered_frames.append(cat)
        except Exception as exc:
            logger.error("Catalyst discovery failed: %s", exc, exc_info=True)

    discovered = (
        pd.concat([f for f in discovered_frames if f is not None and not f.empty], ignore_index=True)
        if any(f is not None and not f.empty for f in discovered_frames)
        else pd.DataFrame(columns=["ticker", "source"])
    )
    if not discovered.empty:
        discovered = discovered[
            ~discovered["ticker"].map(clean_ticker).isin(already_known)
        ].reset_index(drop=True)

    pending = load_pending(pending_path)
    pending = merge_pending(pending, discovered)
    pending = evaluate_promotions(pending, cfg, already_eligible=already_eligible)
    save_pending(pending, pending_path)
    novel_path = save_novel_pending_view(pending, known_tickers=already_known)

    newly_promoted = pending[(pending["status"] == "promoted") & (~pending["ticker"].isin(already_eligible))]
    summary = {
        "discovered": int(len(discovered)),
        "pending_total": int(len(pending)),
        "pending_waiting": int((pending["status"] == "pending").sum()),
        "promoted_total": int((pending["status"] == "promoted").sum()),
        "newly_promoted": sorted(newly_promoted["ticker"].tolist()),
        "pending_path": str(pending_path),
        "novel_pending": int(
            (
                (pending["status"].astype(str) == "pending")
                & (~pending["ticker"].map(clean_ticker).isin(already_known))
            ).sum()
        ),
        "novel_pending_path": str(novel_path),
    }
    logger.info("Discovery summary: %s", summary)
    return summary
