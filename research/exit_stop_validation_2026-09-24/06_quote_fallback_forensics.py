"""Step 6 — why does core/risk_prices.py fall back to the premium stop?

For every "premium fallback active" log line, rebuild what the IEX latest-quote
endpoint would have returned at fetch time (historical IEX quotes, last one at or
before the log timestamp) and classify the rejection:

  future      quote timestamp is AFTER the pass's now_et  -> age < 0 -> rejected
              (now_et is taken once at pass start; modules run sequentially)
  stale       last quote is more than 60s older than now_et
  one_sided   bid or ask is 0 / crossed
  ok          would have passed -> cause unexplained

For every event it also records what the alternatives would have used:
  * last IEX 1-min bar close before fetch, and its age      (option A)
  * the same for SIP 1-min bars (historical SIP >15m old is free) — ground truth
Read-only; network calls to Alpaca market data only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
LOGS = sorted((REPO / "logs/live_server").glob("server_*.log"))
PASS_RE = re.compile(r"^risk pass (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) E[DS]T")
FB_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) WARNING \[core\.risk_prices\] "
                   r"(\S+) current underlying unavailable")
MAX_AGE = 60.0


def parse_events() -> pd.DataFrame:
    rows, now_et = [], None
    for p in LOGS:
        for line in p.read_text(errors="replace").splitlines():
            m = PASS_RE.match(line)
            if m:
                now_et = pd.Timestamp(m.group(1)).tz_localize("America/New_York")
                continue
            m = FB_RE.match(line)
            if m and now_et is not None:
                t = pd.Timestamp(f"{m.group(1)}.{m.group(2)}").tz_localize("America/New_York")
                rows.append({"ticker": m.group(3), "fetch": t, "now_et": now_et})
    return pd.DataFrame(rows)


def last_before(client, kind, sym, feed, end, lookback_s):
    start = end - pd.Timedelta(seconds=lookback_s)
    params = {"symbols": sym, "feed": feed, "start": start.tz_convert("UTC").isoformat(),
              "end": end.tz_convert("UTC").isoformat(), "limit": 1, "sort": "desc"}
    if kind == "bars":
        params["timeframe"] = "1Min"
    r = client._request("GET", f"{client._data_base}/v2/stocks/{kind}", params=params)
    xs = (r or {}).get(kind, {}).get(sym) or []
    return xs[0] if xs else None


def main():
    ev = parse_events()
    print(f"events {len(ev)}  unique (ticker,fetch-second) "
          f"{ev.assign(s=ev.fetch.dt.floor('s')).drop_duplicates(['ticker','s']).shape[0]}")
    client = AlpacaOptionsClient()
    cache, out = {}, []
    for r in ev.itertuples():
        key = (r.ticker, r.fetch.floor("s"))
        if key not in cache:
            q = last_before(client, "quotes", r.ticker, "iex", r.fetch, 3600)
            qn = last_before(client, "quotes", r.ticker, "iex", r.now_et, 3600)
            bi = last_before(client, "bars", r.ticker, "iex", r.fetch, 6 * 3600)
            bs = last_before(client, "bars", r.ticker, "sip", r.fetch, 6 * 3600)
            cache[key] = (q, qn, bi, bs)
        q, qn, bi, bs = cache[key]
        rec = {"ticker": r.ticker, "fetch": r.fetch.isoformat(), "now_et": r.now_et.isoformat()}
        if q is None:
            rec["cause"] = "no_quote_1h"
        else:
            t = pd.Timestamp(q["t"])
            rec["q_age_vs_now"] = (r.now_et - t).total_seconds()
            one_sided = not (0 < float(q["bp"]) <= float(q["ap"]))
            rec["cause"] = ("future" if rec["q_age_vs_now"] < 0 else
                            "stale" if rec["q_age_vs_now"] > MAX_AGE else
                            "one_sided" if one_sided else "ok")
            rec["q_mid"] = (float(q["bp"]) + float(q["ap"])) / 2 if not one_sided else None
        # what a fixed check (quote at or before now_et) would have seen
        if qn is not None:
            rec["qn_age"] = (r.now_et - pd.Timestamp(qn["t"])).total_seconds()
            rec["qn_valid"] = 0 < float(qn["bp"]) <= float(qn["ap"])
        for tag, b in (("iex_bar", bi), ("sip_bar", bs)):
            if b is not None:
                # bar t is the bar OPEN; its close is known at t+60s
                rec[f"{tag}_age"] = (r.fetch - pd.Timestamp(b["t"]) - pd.Timedelta(seconds=60)).total_seconds()
                rec[f"{tag}_close"] = float(b["c"])
        out.append(rec)
    df = pd.DataFrame(out)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "quote_fallback_forensics.csv", index=False)
    summ = {
        "events": len(df),
        "cause_counts": df["cause"].value_counts().to_dict(),
        "cause_by_ticker": df.groupby("ticker")["cause"].value_counts().unstack(fill_value=0).to_dict("index"),
        "future_age_s_quantiles": df.loc[df.cause == "future", "q_age_vs_now"].quantile([.1, .5, .9]).to_dict(),
        "fixed_check_pass_rate": float(((df.qn_age <= MAX_AGE) & df.qn_valid.fillna(False).astype(bool)).mean()),
        "iex_bar_age_s_quantiles": df["iex_bar_age"].quantile([.5, .9, .99]).to_dict(),
        "sip_bar_age_s_quantiles": df["sip_bar_age"].quantile([.5, .9, .99]).to_dict(),
    }
    (OUT / "quote_fallback_forensics.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()
