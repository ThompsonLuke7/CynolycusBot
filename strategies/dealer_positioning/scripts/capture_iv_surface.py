"""Nightly forward capture of implied-volatility surface features.

The dealer snapshots keep open-interest x gamma positioning with expirations
collapsed per strike, and the CBOE stage keeps one iv30 level per name. Neither
can produce the IV-shape signals that survived 2023-2026 (see
``strategies/dealer_positioning/iv_surface.py``). This captures them, plus the
raw per-contract quotes they came from, from the same 120-day Schwab chain the
dealer grid requests.

Writes Data/dealer_positioning/iv_surface/YYYYMMDD/:
  iv_surface_features.parquet  one row per symbol
  iv_contracts.parquet         raw contract quotes (bid/ask/mark/IV/delta/OI)
  errors.jsonl                 one line per symbol that failed

There is deliberately no --snapshot-date: a live chain can only be today's.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from strategies.dealer_positioning.iv_surface import surface_features
from strategies.dealer_positioning.scripts.capture_historical_snapshots import (
    DEFAULT_MIN_ADV,
    DEFAULT_MIN_MARKET_CAP,
    UNIVERSE,
    _write_daily_frame,
    _write_jsonl,
    load_symbols,
)

_ET = ZoneInfo("America/New_York")
OUT_ROOT = REPO / "Data" / "dealer_positioning" / "iv_surface"
# Same window UI.dealer_positioning_dashboard.chain_grid requests.
CHAIN_HORIZON_DAYS = 120
CONTRACT_KEY = ["symbol", "expiration", "strike", "option_type"]

logger = logging.getLogger(__name__)


class ChainClient(Protocol):
    def get_option_chain(self, symbol: str, ref_date: Any, *, from_date: Any, to_date: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class IvSurfaceCaptureResult:
    symbols: int
    feature_rows: int
    contract_rows: int
    errors: int
    output_dir: Path


def capture_iv_surface(
    *,
    symbols: list[str],
    client: ChainClient,
    output_root: Path = OUT_ROOT,
    sleep_seconds: float = 0.25,
    keep_contracts: bool = True,
    now: datetime | None = None,
) -> IvSurfaceCaptureResult:
    now = now or datetime.now(tz=_ET)
    snapshot_date = now.date().isoformat()
    output_dir = output_root / snapshot_date.replace("-", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, Any]] = []
    archives: list[pd.DataFrame] = []
    error_rows: list[dict[str, Any]] = []
    for idx, symbol in enumerate(symbols, start=1):
        try:
            chain = client.get_option_chain(
                symbol,
                now.date(),
                from_date=now.date(),
                to_date=now.date() + timedelta(days=CHAIN_HORIZON_DAYS),
            )
            features, archive = surface_features(
                chain, symbol=symbol, snapshot_date=snapshot_date, captured_at=now.isoformat()
            )
            feature_rows.append(features)
            if keep_contracts:
                archives.append(archive)
        except Exception as exc:  # noqa: BLE001 - one bad chain must not stop the sweep
            error_rows.append(
                {
                    "captured_at": now.isoformat(),
                    "snapshot_date": snapshot_date,
                    "symbol": symbol,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if idx % 50 == 0 or idx == len(symbols):
            logger.info(
                "iv surface capture %d/%d: features=%d errors=%d", idx, len(symbols), len(feature_rows), len(error_rows)
            )
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    _write_daily_frame(output_dir / "iv_surface_features.parquet", feature_rows, ["symbol", "snapshot_date"])
    contract_rows = _write_contracts(output_dir / "iv_contracts.parquet", archives)
    if error_rows:
        _write_jsonl(output_dir / "errors.jsonl", error_rows)
    return IvSurfaceCaptureResult(
        symbols=len(symbols),
        feature_rows=len(feature_rows),
        contract_rows=contract_rows,
        errors=len(error_rows),
        output_dir=output_dir,
    )


def _write_contracts(path: Path, archives: list[pd.DataFrame]) -> int:
    if not archives:
        return 0
    frame = pd.concat(archives, ignore_index=True)
    if path.exists():
        # A same-day re-run replaces that symbol's contracts rather than doubling them.
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        frame = frame.drop_duplicates(CONTRACT_KEY, keep="last")
    float_cols = frame.select_dtypes("float64").columns
    frame[float_cols] = frame[float_cols].astype("float32")
    frame.to_parquet(path, index=False, compression="zstd")
    return int(len(frame))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture implied-volatility surface features.")
    parser.add_argument("--universe", default=str(UNIVERSE))
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--min-adv", type=float, default=DEFAULT_MIN_ADV)
    parser.add_argument("--min-market-cap", type=float, default=DEFAULT_MIN_MARKET_CAP)
    parser.add_argument("--no-liquidity-prefilter", action="store_true")
    parser.add_argument("--no-contracts", action="store_true", help="Skip the raw per-contract archive.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    symbols = load_symbols(
        universe_path=Path(args.universe),
        tickers=args.tickers,
        liquidity_prefilter=not args.no_liquidity_prefilter and not args.tickers,
        min_adv=float(args.min_adv),
        min_market_cap=float(args.min_market_cap),
        limit=args.limit,
    )
    if not symbols:
        raise SystemExit("no symbols selected")

    from strategies.dealer_positioning.config import DealerPositioningConfig
    from strategies.dealer_positioning.schwab_adapter import SchwabDealerDataClient

    result = capture_iv_surface(
        symbols=symbols,
        client=SchwabDealerDataClient(DealerPositioningConfig.from_env()),
        output_root=Path(args.output_root),
        sleep_seconds=float(args.sleep_seconds),
        keep_contracts=not args.no_contracts,
    )
    print(
        "iv surface capture complete: "
        f"symbols={result.symbols} features={result.feature_rows} contracts={result.contract_rows} "
        f"errors={result.errors} output={result.output_dir}"
    )
    return 0 if result.feature_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
