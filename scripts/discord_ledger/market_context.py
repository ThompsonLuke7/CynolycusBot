"""Point-in-time-safe market context for the Discord managed-trade ledger.

This module is deliberately read-only.  The local underlying caches contain
historically retrieved final bars, not a record of when a vendor published
them, so their use is retrospective context only.  They are not certified
point-in-time market-data evidence.  Historical option bid/ask quotes are not
present locally; option follower pricing therefore fails closed.
"""
from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIN_DAILY_HISTORY = 21


def _utc(value: Any) -> pd.Timestamp:
    """Parse a timestamp and reject a timezone-free alert timestamp."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("alert_timestamp_utc must be timezone-aware")
    return ts.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("missing timestamp column/index")
        frame = frame.reset_index(names="timestamp")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    required = {"timestamp", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required bar columns: {sorted(missing)}")
    if frame["timestamp"].isna().any() or frame["close"].isna().any():
        raise ValueError("null timestamp or close")
    if frame["timestamp"].duplicated().any():
        raise ValueError("duplicate bar timestamps: no silent deduplication")
    if (frame["close"] <= 0).any() or frame["volume"].isna().any() or (frame["volume"] < 0).any():
        raise ValueError("invalid close or volume")
    return frame.sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=256)
def _cached_bars(path: str, mtime_ns: int, size_bytes: int) -> pd.DataFrame:
    return _read_bars(Path(path))


def _bars(path: Path) -> pd.DataFrame:
    stat = path.stat()
    return _cached_bars(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _cached_digest(path: str, mtime_ns: int, size_bytes: int) -> str:
    return _sha256(Path(path))


def _digest(path: Path) -> str:
    stat = path.stat()
    return _cached_digest(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _source_metadata(path: Path, frame: pd.DataFrame, *, timeframe: str) -> dict[str, Any]:
    """Return auditable source facts without inventing unavailable provenance."""

    return {
        "path": str(path),
        "sha256": _digest(path),
        "rows": len(frame),
        "timestamp_min_utc": frame["timestamp"].min().isoformat(),
        "timestamp_max_utc": frame["timestamp"].max().isoformat(),
        "timeframe": timeframe,
        "feed": "unknown: parquet carries no feed provenance",
        "adjustment": "unknown: parquet carries no adjustment provenance",
        "availability": (
            "retrospectively fetched final observations; local cache does not certify "
            "vendor publication/availability at the alert timestamp"
        ),
    }


def _default_paths(symbol: str, repo_root: Path) -> tuple[Path, Path]:
    daily = repo_root / "Data" / "shared" / "bars" / "1d" / f"{symbol}.parquet"
    raw_dir = repo_root / "Data" / "raw" / symbol.lower()
    minute = raw_dir / f"{symbol.lower()}_intraday_1min.parquet"
    # SPY is the only broad 1-minute archive in the raw tree; prefer it over
    # the small legacy file when it is available.
    if symbol == "SPY":
        archive = raw_dir / "spy_intraday_1min_runtime_rth_cache.parquet"
        if archive.exists():
            minute = archive
    return daily, minute


def _daily_completed_at(timestamps: pd.Series) -> pd.Series:
    """Map Alpaca-style daily timestamps to the corresponding RTH close.

    The provider stamps daily bars near 04:00 UTC, which is not the session
    close.  A day is usable only after 16:00 America/New_York on that trade
    date.  This rule is intentionally conservative and does not assert the
    vendor had published the final bar exactly at that instant.
    """

    ny = timestamps.dt.tz_convert("America/New_York")
    session_dates = ny.dt.normalize()
    return (session_dates + pd.Timedelta(hours=16)).dt.tz_convert("UTC")


def _blocked(symbol: str, alert: pd.Timestamp, reasons: Iterable[str], **extra: Any) -> dict[str, Any]:
    return {
        "status": "blocked",
        "symbol": symbol,
        "alert_timestamp_utc": alert.isoformat(),
        "reasons": list(reasons),
        **extra,
    }


def lookup_context(
    symbol: str,
    alert_timestamp_utc: Any,
    *,
    repo_root: str | Path | None = None,
    daily_path: str | Path | None = None,
    minute_path: str | Path | None = None,
    minimum_daily_history: int = _MIN_DAILY_HISTORY,
) -> dict[str, Any]:
    """Return strictly pre-alert, completed underlying-bar context.

    Minute bars use their *end* time (start + one minute), so a bar still in
    progress at the alert is excluded.  The latest completed minute must be
    the expected prior minute; an intraday hole blocks the result instead of
    silently carrying an older price forward.  Daily bars likewise require a
    completed US regular session.  No backfill or imputation occurs.
    """

    clean_symbol = str(symbol).strip().upper()
    if not clean_symbol:
        raise ValueError("symbol is required")
    alert = _utc(alert_timestamp_utc)
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    default_daily, default_minute = _default_paths(clean_symbol, root)
    daily_file = Path(daily_path) if daily_path is not None else default_daily
    minute_file = Path(minute_path) if minute_path is not None else default_minute

    daily_all = minute_all = None
    reasons: list[str] = []
    if daily_file.is_file():
        try:
            daily_all = _bars(daily_file)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            reasons.append(f"invalid_daily_underlying_cache:{type(exc).__name__}")
    else:
        reasons.append("missing_daily_underlying_cache")
    if minute_file.is_file():
        try:
            minute_all = _bars(minute_file)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            reasons.append(f"invalid_minute_underlying_cache:{type(exc).__name__}")
    else:
        reasons.append("missing_minute_underlying_cache")

    if daily_all is not None:
        daily_all = daily_all.assign(completed_at_utc=_daily_completed_at(daily_all["timestamp"]))
        daily = daily_all.loc[daily_all["completed_at_utc"] <= alert].copy()
    else:
        daily = pd.DataFrame()
    # Bar starts are timestamps.  A 10:30:00--10:31:00 bar may not influence
    # a 10:30:30 alert, hence the end-time comparison.
    if minute_all is not None:
        minute_all = minute_all.assign(completed_at_utc=minute_all["timestamp"] + pd.Timedelta(minutes=1))
        minute = minute_all.loc[minute_all["completed_at_utc"] <= alert].copy()
    else:
        minute = pd.DataFrame()

    if daily_all is not None and len(daily) < minimum_daily_history:
        reasons.append(f"insufficient_completed_daily_history:{len(daily)}<{minimum_daily_history}")
    if minute_all is not None and minute.empty:
        reasons.append("no_completed_minute_before_alert")
    elif minute_all is not None:
        expected_start = alert.floor("min") - pd.Timedelta(minutes=1)
        if minute["timestamp"].iloc[-1] != expected_start:
            reasons.append(
                "minute_gap_at_alert:expected_start="
                f"{expected_start.isoformat()},actual_start={minute['timestamp'].iloc[-1].isoformat()}"
            )
    daily_result = None
    if daily_all is not None and len(daily) >= minimum_daily_history:
        latest_daily = daily.iloc[-1]
        prior_daily = daily.iloc[-2]
        volume_window = daily["volume"].tail(minimum_daily_history)
        mean_volume = float(volume_window.iloc[:-1].mean())
        daily_result = {
            "completed_bars": len(daily),
            "latest_bar_start_utc": latest_daily["timestamp"].isoformat(),
            "latest_completed_at_utc": latest_daily["completed_at_utc"].isoformat(),
            "close": float(latest_daily["close"]),
            "return_1_session": float(latest_daily["close"] / prior_daily["close"] - 1.0),
            "return_5_sessions": float(latest_daily["close"] / daily.iloc[-6]["close"] - 1.0),
            "volume": float(latest_daily["volume"]),
            "volume_vs_prior_20_mean": float(latest_daily["volume"]) / mean_volume if mean_volume > 0 else None,
        }
    minute_result = None
    if minute_all is not None and not minute.empty and not any(reason.startswith("minute_gap_at_alert") for reason in reasons):
        latest_minute = minute.iloc[-1]
        minute_result = {
            "completed_bars": len(minute),
            "latest_bar_start_utc": latest_minute["timestamp"].isoformat(),
            "latest_completed_at_utc": latest_minute["completed_at_utc"].isoformat(),
            "close": float(latest_minute["close"]),
            "return_1_minute": float(latest_minute["close"] / minute.iloc[-2]["close"] - 1.0) if len(minute) >= 2 else None,
        }
    sources = {}
    if daily_all is not None and not daily_all.empty:
        sources["daily"] = _source_metadata(daily_file, daily_all, timeframe="1Day")
    if minute_all is not None and not minute_all.empty:
        sources["minute"] = _source_metadata(minute_file, minute_all, timeframe="1Min")
    return {
        "status": "ok" if daily_result and minute_result else "partial" if daily_result or minute_result else "blocked",
        "symbol": clean_symbol,
        "alert_timestamp_utc": alert.isoformat(),
        "certified_point_in_time": False,
        "reasons": reasons,
        "limitation": (
            "Underlying bars were retrieved historically and are final observations, not "
            "proof of vendor publication availability at alert time."
        ),
        "daily": daily_result,
        "minute": minute_result,
        "sources": sources,
    }


def follower_scenarios(
    symbol: str,
    alert_timestamp_utc: Any,
    instrument_type: str,
    *,
    delays_seconds: Iterable[int] = (30, 60, 300),
    repo_root: str | Path | None = None,
    daily_path: str | Path | None = None,
    minute_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build delayed follower scenarios without fabricating option execution.

    For options, every delay is blocked because historical two-sided quotes
    (bid/ask and quote timestamps) are unavailable locally.  For equities,
    the returned close is an underlying completed-bar proxy only--never a
    quote, fill, or estimate of an option fill.
    """

    clean_type = str(instrument_type).strip().lower()
    alert = _utc(alert_timestamp_utc)
    delays = tuple(int(delay) for delay in delays_seconds)
    if any(delay < 0 for delay in delays):
        raise ValueError("delays_seconds must be non-negative")
    if clean_type in {"option", "options"}:
        return {
            "status": "blocked",
            "symbol": str(symbol).strip().upper(),
            "instrument_type": "option",
            "alert_timestamp_utc": alert.isoformat(),
            "reason": "historical_option_bid_ask_unavailable_locally",
            "scenarios": [
                {
                    "delay_seconds": delay,
                    "target_timestamp_utc": (alert + timedelta(seconds=delay)).isoformat(),
                    "status": "blocked",
                    "price": None,
                    "reason": "no_historical_option_bid_ask_or_quote_timestamp",
                }
                for delay in delays
            ],
        }
    if clean_type not in {"equity", "stock", "underlying"}:
        raise ValueError("instrument_type must be option, equity, stock, or underlying")

    scenarios: list[dict[str, Any]] = []
    for delay in delays:
        target = alert + timedelta(seconds=delay)
        context = lookup_context(
            symbol,
            target,
            repo_root=repo_root,
            daily_path=daily_path,
            minute_path=minute_path,
        )
        if context["status"] != "ok":
            scenarios.append({
                "delay_seconds": delay,
                "target_timestamp_utc": target.isoformat(),
                "status": "blocked",
                "price": None,
                "reason": context["reasons"],
            })
            continue
        scenarios.append({
            "delay_seconds": delay,
            "target_timestamp_utc": target.isoformat(),
            "status": "proxy_only",
            "price": context["minute"]["close"],
            "proxy_bar_start_utc": context["minute"]["latest_bar_start_utc"],
            "proxy_bar_completed_at_utc": context["minute"]["latest_completed_at_utc"],
            "limitation": "completed underlying bar close; not a quote, fill, or option-price estimate",
        })
    return {
        "status": "ok",
        "symbol": str(symbol).strip().upper(),
        "instrument_type": "equity",
        "alert_timestamp_utc": alert.isoformat(),
        "scenarios": scenarios,
    }
