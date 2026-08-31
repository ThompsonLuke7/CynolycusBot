"""Stage 0c — is the 1-minute tape dense enough on the names we actually trade?

AGENTS.md "trade prints vs marks": a sparse tape makes MFE/MAE understate, and a
"price" at an arbitrary timestamp on an illiquid name is a stale print, not a
mark. The traded set includes microcaps, so this has to be measured before any
excursion metric is computed on it. Also compares IEX vs SIP to establish the
entitlement actually in use rather than the one the endpoint appears to offer.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

MATCH = REPO_ROOT / "research/execution_quality/data/stage0_entry_match.jsonl"
RTH_MINUTES = 390


def main() -> None:
    rows = [json.loads(l) for l in MATCH.open() if l.strip()]
    tick = Counter(r["ticker"] for r in rows)
    # 20 names spanning the traded liquidity range: most-traded + a random tail.
    ranked = [t for t, _ in tick.most_common()]
    sample = ranked[:10] + ranked[-10:]
    sample = list(dict.fromkeys(sample))

    start = datetime(2026, 8, 18, tzinfo=timezone.utc)
    end = datetime(2026, 8, 26, tzinfo=timezone.utc)
    print(f"1m coverage probe, {start.date()} -> {end.date()}, {len(sample)} tickers\n")
    print(f"{'ticker':8s} {'IEX bars/sess':>14s} {'SIP bars/sess':>14s} {'IEX gap%':>9s} {'SIP gap%':>9s}")
    out = []
    for t in sample:
        rec = {"ticker": t}
        for feed in ("IEX", "SIP"):
            try:
                df = fetch_intraday(ticker=t, start=start, end=end, timeframe="1Min", feed=feed)
            except Exception as exc:  # noqa: BLE001
                rec[feed] = {"error": f"{type(exc).__name__}: {exc}"[:120]}
                continue
            if df is None or len(df) == 0:
                rec[feed] = {"bars_per_session": 0.0, "gap_pct": 1.0, "sessions": 0}
                continue
            et = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
            mins = et.dt.hour * 60 + et.dt.minute
            in_rth = (mins >= 570) & (mins < 960)
            nsess = et[in_rth].dt.date.nunique() or 1
            bps = int(in_rth.sum()) / nsess
            rec[feed] = {
                "bars_per_session": round(bps, 1),
                "gap_pct": round(max(0.0, 1 - bps / RTH_MINUTES), 3),
                "sessions": int(nsess),
            }
        i, s = rec.get("IEX", {}), rec.get("SIP", {})
        print(f"{t:8s} {i.get('bars_per_session', i.get('error','')):>14} "
              f"{s.get('bars_per_session', s.get('error','')):>14} "
              f"{i.get('gap_pct',''):>9} {s.get('gap_pct',''):>9}")
        out.append(rec)

    dest = REPO_ROOT / "research/execution_quality/data/stage0_bar_coverage.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
