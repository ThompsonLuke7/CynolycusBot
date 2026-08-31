#!/usr/bin/env python
"""Price-only baseline for the Intraday Structure engine over real 1-minute bars.

WHY. The live engine's ledger fills forward from today, so the first honest
answer to "does the level-interaction layer work?" is otherwise weeks away.
This runs the SAME engine, detectors, levels, targets and ledger builder over
stored 1-minute history, which gives an unbiased read now.

It is deliberately the WEAKEST arm of the eventual ablation:

* no ranker, no theme, no catalyst -- every session simply seeds one long and
  one short candidate at the first bar, so nothing selects the name;
* no dealer/options context (``NullOptionsProvider``), because stored dealer
  snapshots do not reach back and a current one would leak.

So it measures the price-structure machinery alone.  Anything the rankers add
has to be measured on top of this, not instead of it.

Causality: candidates are registered at a session's first bar with
``available_at`` equal to that bar, and the engine's existing one-bar entry
delay applies.  Nothing here reads a bar the engine has not already seen.

Warm-up.  The engine starts with an empty history, so for the first session it
has no prior-day levels and an ATR measured over a handful of one-minute bars.
Left alone that produces a burst of degenerate entries -- a real run confirmed
seven setups on the first morning with stops three cents wide on a $644
instrument, every one of them stopped within minutes.  Those are an artifact of
the cold start, not signal, so no candidate is registered until
``--warmup-sessions`` complete sessions are in history.  This is a
data-availability guard, not a tuned parameter: it withholds candidates until
the inputs those candidates depend on exist.

Writes to its own output directory; it never touches the live ledger.

    python -m scripts.run_intraday_structure_baseline \
        --bars Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet \
        --symbol SPY --start 2024-08-26 --end 2026-08-24 \
        --output Data/analysis/intraday_structure_baseline/spy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from strategies.intraday_structure.config import DEFAULT_CONFIG_PATH, load_config
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.ledger import ledger_sink
from strategies.intraday_structure.models import Bar, Candidate, Direction
from strategies.intraday_structure.options import NullOptionsProvider
from strategies.intraday_structure.regime import abstention_sink
from strategies.intraday_structure.reporting import build_report, read_jsonl, render_report


logger = logging.getLogger("intraday_structure_baseline")

ET = "America/New_York"


def load_bars(path: Path, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        frame = frame.reset_index()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str).str.upper() == symbol.upper()]
    if start:
        frame = frame[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        frame = frame[frame["timestamp"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if frame.empty:
        raise SystemExit(f"no bars for {symbol} in {path} over the requested window")

    # Guard the one assumption everything else rests on: these must be true
    # 1-minute bars, not a higher timeframe someone resampled down.
    gaps = frame["timestamp"].diff().dropna().dt.total_seconds()
    intraday = gaps[gaps < 6 * 3600]
    if not intraday.empty and float(intraday.median()) > 90.0:
        raise SystemExit(
            f"median intraday spacing is {float(intraday.median()):.0f}s; this is not 1-minute data"
        )
    return frame


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "closed_setups.jsonl"
    abstention_path = output / "abstentions.jsonl"
    for stale in (ledger_path, abstention_path):
        if stale.exists():
            stale.unlink()

    frame = load_bars(Path(args.bars), args.symbol, args.start, args.end)
    sessions = frame["timestamp"].dt.tz_convert(ET).dt.date
    first_bar_of_session = ~sessions.duplicated()
    logger.info(
        "%s: %d bars, %d sessions, %s -> %s",
        args.symbol, len(frame), int(first_bar_of_session.sum()),
        frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1],
    )

    config = load_config(args.config)
    engine = IntradayStructureEngine(
        config,
        options_provider=NullOptionsProvider(),
        ledger_sink=ledger_sink(ledger_path),
        abstention_sink=abstention_sink(abstention_path),
    )

    started = time.time()
    directions = (Direction.LONG, Direction.SHORT)
    session_ordinal = first_bar_of_session.cumsum()
    warmed_up_at = None
    for index, row in enumerate(frame.itertuples(index=False)):
        timestamp = row.timestamp.to_pydatetime()
        warm = int(session_ordinal.iloc[index]) > args.warmup_sessions
        if warm and warmed_up_at is None:
            warmed_up_at = timestamp
            logger.info("warm-up complete after %d sessions; candidates start %s",
                        args.warmup_sessions, timestamp)
        if warm and first_bar_of_session.iloc[index]:
            for direction in directions:
                engine.register_candidate(Candidate(
                    args.symbol, timestamp, direction, ("price_only_baseline",),
                    score=0.5, available_at=timestamp,
                ))
        engine.on_bar(Bar(args.symbol, timestamp, row.open, row.high, row.low, row.close, row.volume))
        if index and index % 20_000 == 0:
            logger.info("  %d/%d bars (%.0f%%, %.0f min elapsed)",
                        index, len(frame), 100 * index / len(frame), (time.time() - started) / 60)

    elapsed = (time.time() - started) / 60
    ledger_rows = read_jsonl(ledger_path)
    abstention_rows = read_jsonl(abstention_path)
    report = build_report(ledger_rows, abstention_rows)
    report["run"] = {
        "symbol": args.symbol,
        "bars": len(frame),
        "sessions": int(first_bar_of_session.sum()),
        "first_bar": str(frame["timestamp"].iloc[0]),
        "last_bar": str(frame["timestamp"].iloc[-1]),
        "engine_version": config.version,
        "options_provider": "NullOptionsProvider (price-only arm)",
        "candidate_source": "one long + one short at each session's first bar; no ranker",
        "warmup_sessions": args.warmup_sessions,
        "first_candidate_at": str(warmed_up_at) if warmed_up_at else None,
        "minutes_elapsed": round(elapsed, 1),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    text = render_report(report)
    (output / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    logger.info("done in %.0f min: %d closed setups, %d abstentions -> %s",
                elapsed, len(ledger_rows), len(abstention_rows), output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bars", required=True, help="Parquet of true 1-minute OHLCV bars.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default=None, help="Inclusive UTC date, e.g. 2024-08-26.")
    parser.add_argument("--end", default=None, help="Inclusive UTC date.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--warmup-sessions", type=int, default=2,
                        help="Sessions of history to build before any candidate is registered.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
