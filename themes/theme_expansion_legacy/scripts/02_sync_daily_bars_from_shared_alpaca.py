from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "theme_expansion"))

from strategies.momentum_expansion.config.momentum_config import RAW_1D_DIR
from core.shared_universe.universe import shared_tickers
from theme_expansion.config import BENCHMARK_TICKERS, DAILY_BARS_PATH, ensure_dirs

_DAILY_DOWNLOAD_PATH = Path(__file__).resolve().parent / "02_download_daily_bars.py"
_SPEC = importlib.util.spec_from_file_location("theme_daily_download", _DAILY_DOWNLOAD_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load {_DAILY_DOWNLOAD_PATH}")
daily_download = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(daily_download)


def _load_one(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = RAW_1D_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize(),
            "ticker": ticker,
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "adj_close": df.get("close"),
            "volume": df["volume"],
        }
    )
    return out.loc[(out["date"] >= start) & (out["date"] < end)].dropna(subset=["date", "close"])


def main() -> None:
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2026-06-02")
    ensure_dirs()
    tickers = sorted(set(shared_tickers(eligible_only=True, rebuild=False)) | set(BENCHMARK_TICKERS))
    frames = [_load_one(ticker, start, end) for ticker in tickers]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise SystemExit("no shared Alpaca daily bars found")
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["date", "ticker"], keep="last")
        .sort_values(["date", "ticker"])
    )
    combined.to_parquet(DAILY_BARS_PATH, index=False)
    universe_filter = daily_download.build_universe_filter(combined)
    print(f"synced {len(combined):,} rows for {combined['ticker'].nunique()} tickers -> {DAILY_BARS_PATH}")
    print(f"eligible universe: {int(universe_filter['is_eligible'].sum())}/{len(universe_filter)}")


if __name__ == "__main__":
    main()
