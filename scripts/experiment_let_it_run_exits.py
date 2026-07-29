"""Does letting options RUN fix them? (filter + option-native exits)

Motivation (`research/options_experiment/06_parabolic_filter.md`):
momentum takes profit at +2.0 ATR while the median trade's true 20-bar MFE is
3.07 ATR and the p90 is 9.58 ATR. A long option's whole edge is convexity in a
large move, so a 2-ATR take-profit truncates exactly the tail being paid for.
Phase 3 priced every option at the MODULE's exit, so it never tested the
combination the strategy actually intends.

This reprices the SAME option entries under alternative exits, using the already
cached daily contract bars (`Data/options_history/bars/1Day/...`), so it needs
essentially no new API calls.

Exit policies compared:
  module_exit    : baseline -- exit at the module's own exit timestamp (Phase 3)
  hold_to_expiry : never exit early
  time_20d       : fixed 20 calendar-day stop
  atr_target_4   : exit the first bar the underlying reaches +4 ATR from entry,
                   else fall through to expiry
  trail_50pct    : exit when the option's mark retraces 50% from its running peak,
                   else expiry

All exits are evaluated on DAILY bars. Costs use the Phase 3 fill model
(`research.options_lab.fills`) at the pessimistic bound, per the G1 verdict.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
P3 = REPO / "research/options_experiment/data/phase3_counterfactual.parquet"
BARS_DIR = REPO / "Data/options_history/bars/1Day"
UNDER = REPO / "Data/shared/bars/1d"
OUT = REPO / "research/options_experiment/data/let_it_run.parquet"

ROUND_TRIP_SPREAD = 0.2556  # calibrated median from G1c; pessimistic basis
COMMISSION = 0.65


def _contract_bars(ticker: str, expiry: str) -> pd.DataFrame | None:
    p = BARS_DIR / ticker / f"{expiry}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if d.empty:
        return None
    tcol = next((c for c in ("ts", "timestamp", "t") if c in d.columns), None)
    if tcol is None or "osi_symbol" not in d.columns:
        return None
    # cache stores raw Alpaca short field names: o/h/l/c/v/n/vw/t
    d = d.rename(columns={tcol: "ts", "c": "close", "h": "high", "l": "low", "o": "open"})
    if "close" not in d.columns:
        return None
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    return d.sort_values("ts")


def _under_bars(ticker: str) -> pd.DataFrame | None:
    p = UNDER / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p)
    if "timestamp" not in b.columns:
        b = b.reset_index()
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    return b.sort_values("timestamp")


def run(limit: int | None) -> pd.DataFrame:
    p3 = pd.read_parquet(P3)
    # long calls that were actually executable, near DTE, matched notional
    m = p3[(p3.executable) & (p3.strategy == "long_call_atm") & (p3.direction > 0)
           & (p3.sizing_mode == "matched_notional") & (p3.dte_bucket == "near")].copy()
    m = m.dropna(subset=["expiry_date", "entry_ts", "exit_ts", "entry_cost"])
    if limit:
        m = m.head(limit)
    print(f"candidate long-call trades: {len(m)}")

    rows = []
    for i, r in enumerate(m.itertuples(index=False)):
        if i % 200 == 0:
            print(f"  {i}/{len(m)}")
        exp = str(pd.Timestamp(r.expiry_date).date())
        cb = _contract_bars(r.ticker, exp)
        if cb is None:
            continue
        # identify the specific contract actually used: the one priced at entry
        ent = pd.Timestamp(r.entry_ts)
        cand = cb[cb.ts >= ent.normalize()]
        if cand.empty:
            continue
        # pick contract whose first close best matches the recorded entry cost per contract
        target = float(r.entry_cost) / 100.0
        first = cand.groupby("osi_symbol").first().reset_index()
        if first.empty:
            continue
        first["err"] = (first["close"] - target).abs()
        sym = first.nsmallest(1, "err").osi_symbol.iloc[0]
        s = cb[cb.osi_symbol == sym].sort_values("ts")
        s = s[s.ts >= ent.normalize()]
        if len(s) < 2:
            continue
        entry_px = float(s.close.iloc[0])
        if entry_px <= 0:
            continue

        ub = _under_bars(r.ticker)
        exits: dict[str, float | None] = {}

        # baseline: module exit
        xt = pd.Timestamp(r.exit_ts).normalize()
        b = s[s.ts <= xt]
        exits["module_exit"] = float(b.close.iloc[-1]) if len(b) else None
        exits["hold_to_expiry"] = float(s.close.iloc[-1])
        t20 = s[s.ts <= ent.normalize() + pd.Timedelta(days=20)]
        exits["time_20d"] = float(t20.close.iloc[-1]) if len(t20) else None

        # +4 ATR underlying target
        px = None
        if ub is not None and np.isfinite(r.atr_at_entry) and r.atr_at_entry > 0:
            tgt = r.entry_px_underlying + 4.0 * r.atr_at_entry
            fu = ub[(ub.timestamp >= ent) & (ub.high >= tgt)]
            if len(fu):
                hit = fu.timestamp.iloc[0].normalize()
                hb = s[s.ts >= hit]
                px = float(hb.close.iloc[0]) if len(hb) else None
        exits["atr_target_4"] = px if px is not None else float(s.close.iloc[-1])

        # 50% trail off the running peak of the option mark
        c = s.close.to_numpy(dtype=float)
        peak = np.maximum.accumulate(c)
        trig = np.where(c <= 0.5 * peak)[0]
        exits["trail_50pct"] = float(c[trig[0]]) if len(trig) else float(c[-1])

        # THE DEPLOYED LIVE POLICY -- premium-based tail-rider (see simulate_deployed)
        dep = simulate_deployed(c)
        if dep is not None:
            mult, reason, bars = dep
            exits["deployed_tailrider"] = entry_px * mult

        for pol, xpx in exits.items():
            if xpx is None:
                continue
            gross = (xpx - entry_px) * 100.0
            cost = (entry_px + xpx) * 100.0 * (ROUND_TRIP_SPREAD / 2.0) + 2 * COMMISSION
            rows.append(dict(
                trade_id=r.trade_id, module=r.module, ticker=r.ticker, week_key=r.week_key,
                score=r.score, policy=pol, entry_px=entry_px, exit_px=xpx,
                capital=entry_px * 100.0, gross_pnl=gross, net_pnl=gross - cost,
                shares_pnl=r.net_pnl_pessimistic if r.strategy == "long_shares" else np.nan,
                deployed_exit_reason=(dep[1] if pol == "deployed_tailrider" and dep else None),
            ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    df = run(args.limit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {len(df)} rows -> {OUT}")
    if df.empty:
        return
    g = df.groupby("policy").apply(lambda x: pd.Series({
        "n": len(x),
        "net_pnl": x.net_pnl.sum(),
        "capital": x.capital.sum(),
        "return_on_capital_%": 100 * x.net_pnl.sum() / x.capital.sum(),
        "win_rate": (x.net_pnl > 0).mean(),
        "median_pnl": x.net_pnl.median(),
    }), include_groups=False).sort_values("return_on_capital_%", ascending=False)
    print("\n=== OPTION EXIT POLICY (long calls, pessimistic costs) ===")
    print(g.round(2).to_string())
    if "deployed_exit_reason" in df.columns:
        dep = df[df.policy == "deployed_tailrider"]
        if len(dep):
            print("\ndeployed policy exit reasons:",
                  dep.deployed_exit_reason.value_counts().to_dict())


# ---------------------------------------------------------------------------
# Deployed live policy ("tail-rider", core/live_4h_exec.py ExecPolicy defaults).
# Added 2026-07-27 after the user correctly pointed out the first pass tested the
# BACKTEST's 2-ATR take-profit, not the policy actually running live. The live
# policy is premium-based: for an option position `gain` is the premium change.
# Its own code comment notes it was selected on a SHARES-ONLY backtest with "no
# option-premium path modeled ... not yet paper-validated live" -- so this is the
# first evaluation of it against real option price paths.
# ---------------------------------------------------------------------------

def simulate_deployed(close: np.ndarray, *, stop=0.39, tp=0.30, scale=0.16, horizon=21):
    """Return (realized_multiple, exit_reason, bars_held) for the live policy.

    realized_multiple is the weighted exit value / entry premium, accounting for
    the partial scale-out at take-profit (16% sold, remainder rides on).
    """
    entry = close[0]
    if entry <= 0:
        return None
    remaining = 1.0
    proceeds = 0.0
    trimmed = False
    for i in range(1, len(close)):
        gain = close[i] / entry - 1.0
        if gain <= -stop:                      # 1) premium stop -- full exit
            proceeds += remaining * close[i]
            return proceeds / entry, "stop", i
        if (not trimmed) and gain >= tp:       # 3) take-profit scale-out
            proceeds += scale * close[i]
            remaining -= scale
            trimmed = True
        if i >= horizon:                       # 4) horizon -- full exit
            proceeds += remaining * close[i]
            return proceeds / entry, "horizon", i
    proceeds += remaining * close[-1]          # expiry
    return proceeds / entry, "expiry", len(close) - 1


if __name__ == "__main__":
    main()
