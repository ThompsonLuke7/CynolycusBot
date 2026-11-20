"""Score extracted AfterHour callouts on the UNDERLYING (no option P&L).

Rules (fixed before any outcome was inspected):
- availability time = post createdAt (UTC). Entry = open of the first regular
  session that starts after the post: same day if posted before 09:30 ET on a
  trading day, otherwise the next trading day. Never a price from before the post.
- direction hit at horizon h (1/5/20 sessions): sign(close[entry+h-1]/open[entry]-1)
  matches the call's direction. Also reported as excess over SPY on the same window.
- strike + expiry calls: strict = expiry-day close beyond strike; lenient = strike
  touched (high for calls, low for puts) at any session from entry to expiry.
  Expiry after the last bar = censored; expiry before entry = unscorable.
- price targets: hit if touched from entry through the stated deadline, else
  through 60 sessions (flagged `default_deadline`).
- levels are compared in the post-date share basis: split-adjusted bars rescaled
  by raw/adjusted at the entry session, so later splits do not fake a hit.
- control: per-symbol base rate = share of ALL h-session windows (same symbol,
  2023-01-01..last bar) whose return has the call's sign. Expected hits are the
  sum of per-call base rates, so a long-only caller in a bull market gets no credit
  for drift alone.
Returns are gross of costs. This is retrospective research, not a signal.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.afterhour.collect import read_jsonl, write_jsonl

ET = "America/New_York"
HORIZONS = (1, 5, 20)
DEFAULT_TARGET_SESSIONS = 60
BASE_RATE_START = "2023-01-01"
BENCH = "SPY"


# ---------------------------------------------------------------- price data
def fetch_daily(symbols: list[str], start: str, end: str, cache: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Daily SIP bars, split-adjusted and raw, cached per symbol as parquet."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from core.API.Alpaca_API.core.config import AlpacaConfig

    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(cfg.key_id, cfg.secret_key)
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for adj_name, adj in (("split", Adjustment.SPLIT), ("raw", Adjustment.RAW)):
        missing = [s for s in symbols if not (cache / adj_name / f"{s}.parquet").exists()]
        for i in range(0, len(missing), 50):
            chunk = missing[i:i + 50]
            try:
                df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                                            start=pd.Timestamp(start, tz="UTC").to_pydatetime(),
                                                            end=pd.Timestamp(end, tz="UTC").to_pydatetime(),
                                                            feed=DataFeed.SIP, adjustment=adj, limit=None)).df
            except Exception as exc:  # e.g. one invalid symbol rejects the batch: retry singly
                if len(chunk) == 1:
                    print(f"bars {adj_name} {chunk[0]}: {str(exc)[:120]}")
                    df = pd.DataFrame()
                else:
                    for s in chunk:
                        fetch_daily([s], start, end, cache)
                    continue
            (cache / adj_name).mkdir(parents=True, exist_ok=True)
            got = set()
            if len(df):
                for sym, g in df.reset_index().groupby("symbol"):
                    g = g.assign(date=pd.to_datetime(g["timestamp"]).dt.tz_convert(ET).dt.date.astype(str))
                    g.set_index("date")[["open", "high", "low", "close", "volume"]].to_parquet(cache / adj_name / f"{sym}.parquet")
                    got.add(sym)
            for s in set(chunk) - got:  # record "no bars" so reruns do not refetch
                pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).to_parquet(cache / adj_name / f"{s}.parquet")
    for s in symbols:
        frames = clean_bars({a: pd.read_parquet(cache / a / f"{s}.parquet") for a in ("split", "raw")})
        if len(frames["split"]):
            out[s] = frames
    return out


def clean_bars(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Drop zero-volume rows: Alpaca pads reused/dead symbols with flat 0-volume bars
    (e.g. `BTC` sat at 495.55 with volume 0 until the Grayscale mini trust listed 2024-07-31)."""
    return {k: v[v["volume"] > 0] for k, v in frames.items()}


# ---------------------------------------------------------------- scoring
def entry_index(dates: pd.Index, posted_utc: pd.Timestamp) -> int | None:
    """Index of the first session whose 09:30 ET open is after the post."""
    et = posted_utc.tz_convert(ET)
    day = et.date().isoformat()
    before_open = et.time() < pd.Timestamp("09:30").time()
    pos = int(dates.searchsorted(day, side="left" if before_open else "right"))
    return pos if pos < len(dates) else None


def base_rates(adj: pd.DataFrame) -> dict[int, float]:
    a = adj[adj.index >= BASE_RATE_START]
    rates = {}
    for h in HORIZONS:
        if len(a) <= h:
            rates[h] = float("nan")
            continue
        r = a["close"].shift(-(h - 1)).to_numpy() / a["open"].to_numpy() - 1
        r = r[~np.isnan(r)]
        rates[h] = float((r > 0).mean()) if len(r) else float("nan")
    return rates


def strike_base(adj_full: pd.DataFrame, ratio: float, sessions: int, sign: int, instrument: str) -> tuple[float, float]:
    """Control for a strike call: the share of ALL same-length windows on the same symbol (2023+) that would have
    met the same strike, expressed as strike/entry-open, under the strict and lenient rules."""
    a = adj_full[adj_full.index >= BASE_RATE_START]
    n = len(a) - sessions
    if n <= 0:
        return float("nan"), float("nan")
    o, h, lo, c = (a[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    end_ratio = c[sessions:sessions + n] / o[:n]
    if instrument == "short_put" or sign > 0:
        strict = float(np.mean(end_ratio > ratio))
    else:
        strict = float(np.mean(end_ratio < ratio))
    if instrument == "short_put":
        return strict, strict
    win = np.lib.stride_tricks.sliding_window_view(h if sign > 0 else lo, sessions + 1)[:n]
    ext = (win.max(axis=1) if sign > 0 else win.min(axis=1)) / o[:n]
    lenient = float(np.mean(ext >= ratio)) if sign > 0 else float(np.mean(ext <= ratio))
    return strict, lenient


def score_call(call: dict, bars: dict[str, pd.DataFrame], bench: pd.DataFrame, rates: dict[int, float]) -> dict:
    out = {k: call.get(k) for k in ("call_id", "username", "posted_at_utc", "symbol", "direction", "instrument",
                                    "strike", "expiry", "target", "deadline", "tier", "rule", "source",
                                    "post_quotes_tweets", "share_url")}
    adj, raw = bars["split"], bars["raw"]
    adj_full = adj
    dates = adj.index
    i = entry_index(dates, pd.Timestamp(call["posted_at_utc"]))
    if i is None:
        return {**out, "status": "censored_no_entry_bar"}
    sign = 1 if call["direction"] == "long" else -1
    entry_date = dates[i]
    # a >10-calendar-day hole after entry = delisting/suspension (AKTS died 2024-12, its symbol was reused
    # 2026-01): everything after the hole is another instrument, so the series ends there (censored)
    cal = pd.to_datetime(dates)
    holes = np.flatnonzero((cal[i + 1:] - cal[i:-1]).days > 10)
    if len(holes):
        stop = i + 1 + int(holes[0])
        adj, raw, dates = adj.iloc[:stop], raw.iloc[:stop], dates[:stop]
        out_gap = str(cal[stop].date())
    else:
        out_gap = None
    entry = float(adj["open"].iloc[i])
    scale = float(raw.loc[entry_date, "open"]) / entry if entry_date in raw.index and entry else float("nan")
    out.update({"status": "scored", "entry_date": entry_date, "entry_open_raw": entry * scale, "series_ends_at_gap": out_gap})
    for h in HORIZONS:
        j = i + h - 1
        if j >= len(dates):
            out[f"hit_{h}d"] = None
            continue
        ret = float(adj["close"].iloc[j]) / entry - 1
        out[f"ret_{h}d"] = ret
        out[f"hit_{h}d"] = bool(sign * ret > 0)
        p_up = rates.get(h, float("nan"))
        out[f"base_{h}d"] = p_up if sign > 0 else 1 - p_up
        if entry_date in bench.index and dates[j] in bench.index:
            b = float(bench.loc[dates[j], "close"]) / float(bench.loc[entry_date, "open"]) - 1
            out[f"xs_spy_{h}d"] = sign * (ret - b)
    level_hi = adj["high"] * scale
    level_lo = adj["low"] * scale
    level_close = adj["close"] * scale
    if call.get("strike") and call.get("expiry"):
        exp = call["expiry"]
        if exp < entry_date:
            out["strike_status"] = "unscorable_expired_before_entry"
        elif exp > dates[-1]:
            out["strike_status"] = "censored_unexpired"
        else:
            k = int(dates.searchsorted(exp, side="right")) - 1  # last session on/before expiry
            window = slice(i, k + 1)
            touched = (level_hi.iloc[window].max() >= call["strike"]) if sign > 0 else (level_lo.iloc[window].min() <= call["strike"])
            close_k = float(level_close.iloc[k])
            strict = bool(close_k > call["strike"] if sign > 0 else close_k < call["strike"])
            if call.get("instrument") == "short_put":
                touched = strict  # a sold put "works" only if it expires above the strike; touching means nothing
            ratio = call["strike"] / (entry * scale)
            base_strict, base_lenient = strike_base(adj_full, ratio, k - i, sign, call.get("instrument") or "")
            out.update({"strike_status": "scored", "expiry_close_raw_basis": close_k, "strike_moneyness": ratio - 1,
                        "strike_sessions": k - i, "strike_hit_strict": strict, "strike_hit_lenient": bool(touched),
                        "strike_base_strict": base_strict, "strike_base_lenient": base_lenient})
    if call.get("target"):
        if call.get("deadline"):
            end = int(dates.searchsorted(call["deadline"], side="right")) - 1
            out["target_deadline_basis"] = "stated"
            censored = call["deadline"] > dates[-1]
        else:
            end = i + DEFAULT_TARGET_SESSIONS - 1
            out["target_deadline_basis"] = "default_deadline"
            censored = end >= len(dates)
        end = min(end, len(dates) - 1)
        window = slice(i, end + 1)
        reached = (level_hi.iloc[window].max() >= call["target"]) if sign > 0 else (level_lo.iloc[window].min() <= call["target"])
        out["target_status"] = "scored" if (reached or not censored) else "censored_open"
        out["target_hit"] = bool(reached) if out["target_status"] == "scored" else None
        out["target_distance_at_entry"] = call["target"] / (entry * scale) - 1
    return out


# ---------------------------------------------------------------- summaries
def mde_hit_rate(n: int, p: float, z_alpha: float = 1.96, z_power: float = 0.84) -> float:
    return (z_alpha + z_power) * math.sqrt(p * (1 - p) / n) if n else float("nan")


def clustered_t(values: pd.Series, clusters: pd.Series) -> float:
    """Mean / cluster-robust SE (CR0) - calls in the same week are not independent."""
    e = values.astype(float)
    n = len(e)
    if n < 10 or clusters.nunique() < 5:  # tiny samples give a near-zero SE and absurd t (e.g. 6 calls, all misses)
        return float("nan")
    dev = (e - e.mean()).groupby(clusters.to_numpy()).sum()
    se = math.sqrt(float((dev ** 2).sum())) / n
    return float(e.mean() / se) if se else float("nan")


def summarize_group(g: pd.DataFrame) -> dict:
    week = pd.to_datetime(g["entry_date"]).dt.strftime("%G-%V")
    rec = {"calls_scored": len(g), "share_long": float((g["direction"] == "long").mean()),
           "first_call": g["posted_at_utc"].min()[:10], "last_call": g["posted_at_utc"].max()[:10],
           "distinct_weeks": int(week.nunique())}
    for h in HORIZONS:
        s = g[g[f"hit_{h}d"].notna()]
        n = len(s)
        if not n:
            continue
        hits = s[f"hit_{h}d"].astype(bool)
        base = s[f"base_{h}d"].astype(float)
        var = float((base * (1 - base)).sum())
        wk = week.loc[s.index]
        rec.update({f"n_{h}d": n, f"hit_{h}d": float(hits.mean()), f"base_{h}d": float(base.mean()),
                    f"excess_hit_{h}d": float(hits.mean() - base.mean()),
                    f"z_{h}d": float((hits.sum() - base.sum()) / math.sqrt(var)) if var else float("nan"),
                    f"t_week_{h}d": clustered_t(hits.astype(float) - base, wk),
                    f"mde_{h}d": mde_hit_rate(n, float(base.mean()))})
        if f"xs_spy_{h}d" in s:
            x = s[f"xs_spy_{h}d"].dropna()
            rec.update({f"mean_xs_spy_{h}d": float(x.mean()) if len(x) else float("nan"),
                        f"median_xs_spy_{h}d": float(x.median()) if len(x) else float("nan"),
                        f"t_week_xs_spy_{h}d": clustered_t(x, week.loc[x.index]) if len(x) else float("nan")})
    if "strike_status" in g:
        k = g[g["strike_status"] == "scored"]
        rec.update({"n_strike": len(k), "strike_strict": float(k["strike_hit_strict"].mean()) if len(k) else None,
                    "strike_strict_base": float(k["strike_base_strict"].mean()) if len(k) else None,
                    "strike_lenient": float(k["strike_hit_lenient"].mean()) if len(k) else None,
                    "strike_lenient_base": float(k["strike_base_lenient"].mean()) if len(k) else None})
    if "target_status" in g:
        t = g[g["target_status"] == "scored"]
        rec.update({"n_target": len(t), "target_hit": float(t["target_hit"].mean()) if len(t) else None})
    return rec


def confidence(row: dict) -> str:
    """Audited 2026-11-17 (run_03, seed 4242): tier A 13/20 and strike-bearing tier B 8/8 correct ->
    'high' (~75%); bare calls/puts 3/12 and opinions 12/20 -> 'low' (~47%)."""
    strike = row.get("strike")
    has_strike = strike is not None and not (isinstance(strike, float) and math.isnan(strike))  # NaN is truthy
    return "high" if row.get("tier") == "A" or (row.get("tier") == "B" and has_strike) else "low"


def summarize(rows: list[dict]) -> pd.DataFrame:
    """Per user x tier, per user x confidence, per user (all), and the same pooled over users."""
    df = pd.DataFrame([r for r in rows if r.get("status") == "scored"])
    if df.empty:
        return df
    df["confidence"] = [confidence(r) for r in df.to_dict("records")]
    out = []
    for (user, conf), g in df.groupby(["username", "confidence"]):
        out.append({"username": user, "tier": f"conf_{conf}", **summarize_group(g)})
    for conf, g in df.groupby("confidence"):
        out.append({"username": "ALL", "tier": f"conf_{conf}", **summarize_group(g)})
    for (user, tier), g in df.groupby(["username", "tier"]):
        out.append({"username": user, "tier": tier, **summarize_group(g)})
    for user, g in df.groupby("username"):
        out.append({"username": user, "tier": "ALL", **summarize_group(g)})
    for tier, g in df.groupby("tier"):
        out.append({"username": "ALL", "tier": tier, **summarize_group(g)})
    out.append({"username": "ALL", "tier": "ALL", **summarize_group(df)})
    return pd.DataFrame(out).sort_values(["username", "tier"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--bars-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--end", default=pd.Timestamp.now(tz="UTC").date().isoformat())
    args = ap.parse_args()
    calls = read_jsonl(args.calls)
    symbols = sorted({c["symbol"] for c in calls} | {BENCH})
    bars = fetch_daily(symbols, BASE_RATE_START, args.end, args.bars_cache)
    if BENCH not in bars:
        raise RuntimeError("no SPY bars; cannot score benchmark-relative returns")
    rates = {s: base_rates(b["split"]) for s, b in bars.items()}
    rows = []
    for c in calls:
        if c["symbol"] not in bars:
            rows.append({"call_id": c["call_id"], "username": c["username"], "symbol": c["symbol"], "status": "no_bars"})
            continue
        if c["posted_at_utc"][:10] < BASE_RATE_START:
            rows.append({"call_id": c["call_id"], "username": c["username"], "symbol": c["symbol"], "status": "before_bar_window"})
            continue
        rows.append(score_call(c, bars[c["symbol"]], bars[BENCH]["split"], rates[c["symbol"]]))
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "scored_calls.jsonl", rows)
    summary = summarize(rows)
    summary.to_csv(args.out / "summary_by_user.csv", index=False)
    status = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    (args.out / "manifest.json").write_text(json.dumps({
        "schema": "afterhour_callout_scoring_v1", "calls_path": str(args.calls), "end": args.end,
        "bars": "alpaca_sip_daily_split_and_raw", "horizons_sessions": HORIZONS,
        "base_rate_window_start": BASE_RATE_START, "status_counts": status}, indent=2) + "\n")
    print(json.dumps({"calls": len(calls), "status": status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
