"""Stage 4C5 — the intraday structure engine: is the invalidation inside the noise?

Also the test promised in 01_stage0_feasibility.md: the engine decides on the IEX
tape (median 22% of RTH minutes missing on these names); how different is the
price path it acted on from the consolidated one?

The engine writes paper setups with no broker orders, so it is analysed from its
own ledger against SIP 1-minute bars rather than from the trade spine.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/execution_quality"))
from stage3_metrics import P, bars, window  # noqa: E402

LEDGER = REPO_ROOT / "Data/inference/intraday_structure/closed_setups.jsonl"


def med(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def main() -> None:
    rows = [json.loads(l) for l in LEDGER.open() if l.strip()]
    print(f"closed setups: {len(rows)}")

    have, noise, widths, mfe_r, mae_r = 0, [], [], [], []
    fwd_after_invalid = []
    for r in rows:
        t = r.get("ticker")
        et, xt = P(r.get("entry_time")), P(r.get("exit_time"))
        entry, risk = r.get("entry_price"), r.get("risk_points")
        if not (t and et and entry and risk and risk > 0):
            continue
        sign = 1 if str(r.get("direction", "long")).lower() == "long" else -1
        # Noise band: median 1-minute absolute excursion over the 60 minutes
        # BEFORE entry, in the same units as the invalidation distance.
        w_pre = window(t, et - timedelta(minutes=60), et)
        if w_pre is None or len(w_pre) < 15:
            continue
        have += 1
        rng = (w_pre["high"] - w_pre["low"]).to_numpy(dtype=float)
        band = float(np.median(rng))
        noise.append(band)
        widths.append(risk / band if band > 0 else float("nan"))
        # What did price do in the hour after the setup was closed?
        if xt:
            w_post = window(t, xt, xt + timedelta(minutes=180))
            if w_post is not None and len(w_post) > 5:
                px = float(r.get("exit_price") or entry)
                fav = ((float(w_post["high"].max()) - px) if sign > 0
                       else (px - float(w_post["low"].min())))
                fwd_after_invalid.append(fav / risk)
        mfe_r.append((r.get("mfe_points") or 0) / risk)
        mae_r.append((r.get("mae_points") or 0) / risk)

    print(f"setups with a SIP 1m pre-entry window: {have}")
    print(f"\n  median 1-minute high-low range before entry : {med(noise):.4f} price units")
    print(f"  median invalidation distance (risk_points)  : {med([r.get('risk_points') for r in rows]):.4f}")
    print(f"  --> invalidation width in 1-minute noise units: {med(widths):.2f}x")
    print(f"      share of setups whose stop is INSIDE one 1-minute bar's range: "
          f"{np.mean([1.0 if w < 1 else 0.0 for w in widths if math.isfinite(w)]):.0%}")
    print(f"\n  ledger MFE/risk median {med(mfe_r):.2f}R   MAE/risk median {med(mae_r):.2f}R")
    print(f"  favourable move in the 3h AFTER the setup closed: median {med(fwd_after_invalid):.2f}R"
          f"  (n={len(fwd_after_invalid)})")

    inval = [r for r in rows if str(r.get("exit_reason", "")).startswith("invalidation")]
    print(f"\n  invalidation-touched setups: {len(inval)} of {len(rows)}")

    # Tape comparison: how much of the path did the live IEX feed not see?
    print("\n" + "=" * 90)
    print("IEX vs SIP on the setups themselves — what the engine could not see")
    print("=" * 90)
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from core.API.Alpaca_API.core.config import AlpacaConfig
    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(api_key=cfg.key_id, secret_key=cfg.secret_key)

    sample = [r for r in rows if r.get("entry_time") and r.get("ticker")][-40:]
    miss_share, extra_range = [], []
    for r in sample:
        t = r["ticker"]
        et = P(r["entry_time"])
        lo, hi = et - timedelta(minutes=45), et + timedelta(minutes=45)
        try:
            resp = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=[t], timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=lo, end=hi, adjustment=Adjustment("split"), feed=DataFeed.IEX))
        except Exception:
            continue
        if resp.df is None or resp.df.empty:
            continue
        iex = resp.df.reset_index()
        sip = window(t, lo, hi)
        if sip is None or len(sip) < 10:
            continue
        miss_share.append(1.0 - len(iex) / len(sip))
        # Did SIP see a wider range than IEX in the same window?
        i_rng = float(iex["high"].max() - iex["low"].min())
        s_rng = float(sip["high"].max() - sip["low"].min())
        if i_rng > 0:
            extra_range.append(s_rng / i_rng - 1.0)
    print(f"  sampled {len(miss_share)} setups, +/-45 min around entry")
    print(f"  minutes the live IEX feed did NOT have: median {med(miss_share):.0%}")
    print(f"  price range SIP saw beyond IEX in the same window: median +{med(extra_range):.1%}")


if __name__ == "__main__":
    main()
