"""Stage 3a — cache SIP 1-minute bars for every traded ticker.

Stage 0 measured IEX at a median 22% of RTH minutes missing on these very names
and SIP at 0.2%. Excursion metrics on the IEX tape would understate MFE/MAE by
construction, so this study uses SIP. (The LIVE system still streams IEX — that
is a separate finding, see 01_stage0_feasibility.md.)

Batched multi-symbol request, same idiom as scripts/fetch_shared_swing_30m_batched.py.
Idempotent: a ticker already cached is skipped unless --force.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.API.Alpaca_API.core.config import AlpacaConfig

DATA = REPO_ROOT / "research/execution_quality/data"
BARS = DATA / "bars_1m"


def chunks(values, size):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-07-08T00:00:00Z")
    ap.add_argument("--end", default="2026-08-28T20:30:00Z")  # SIP historical is 15-min delayed on this plan
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Union of TRADED tickers (needed for entry/exit timing) and every RANKED
    # ticker (needed for Analysis A's forward returns, which must be measured
    # from the exact availability minute — a daily bar dated the same session
    # would leak the morning of the decision day).
    tickers = set()
    for line in (DATA / "stage1_trade_spine.jsonl").open():
        r = json.loads(line)
        if r.get("module"):
            tickers.add(r["ticker"])
    for line in (DATA / "stage2_signal_spine.jsonl").open():
        tickers.add(json.loads(line)["ticker"])
    tickers = sorted(tickers)
    BARS.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if args.force or not (BARS / f"{t}.parquet").exists()]
    logging.info("tickers=%d to fetch=%d", len(tickers), len(todo))
    if not todo:
        return 0

    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(api_key=cfg.key_id, secret_key=cfg.secret_key)
    tf = TimeFrame(1, TimeFrameUnit.Minute)
    start = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = dt.datetime.fromisoformat(args.end.replace("Z", "+00:00"))

    done = 0
    for ci, batch in enumerate(chunks(todo, args.chunk_size), 1):
        # NO `limit`: in this alpaca-py version `limit` caps the ENTIRE
        # auto-paginated response and `next_page_token` comes back empty, so
        # limit=10_000 silently returned the first symbol only. Omitting it lets
        # the SDK page through to completion.
        req = StockBarsRequest(
            symbol_or_symbols=batch, timeframe=tf, start=start, end=end,
            adjustment=Adjustment("split"), feed=DataFeed.SIP,
        )
        resp = client.get_stock_bars(req)
        frames = [resp.df.reset_index()] if (resp.df is not None and not resp.df.empty) else []
        if not frames:
            logging.warning("batch %d: no bars for %s..%s", ci, batch[0], batch[-1])
            continue
        df = pd.concat(frames, ignore_index=True)
        for sym in batch:
            sub = df[df["symbol"].astype(str).str.upper() == sym]
            if sub.empty:
                continue
            sub = (sub.sort_values("timestamp")
                      .drop_duplicates(subset=["timestamp"], keep="last")
                      .reset_index(drop=True))
            sub.to_parquet(BARS / f"{sym}.parquet", index=False)
            done += 1
        logging.info("batch %d/%d done, cached=%d",
                     ci, (len(todo) + args.chunk_size - 1) // args.chunk_size, done)
    logging.info("complete cached=%d", done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
