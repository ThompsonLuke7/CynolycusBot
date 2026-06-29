"""
HTF Swing live runner — standalone long-only top-K harness on the HTF scorer.

Mirrors the Meta Ranker runner (signals/meta_context/meta_ranker/live_runner.py)
but the signal is the HTF scorer's ``htf_score`` — the SAME per-(timestamp,ticker)
value that feeds the Meta matrix's htf_score column. This is the "take the base-
model output and trade it through its own order policy" harness: it reads htf_score
off the shared Meta matrix (no re-scoring), ranks, and reconciles a paper (default)
Alpaca account to an equal-weight long-only top-K target portfolio.

Exit policy + defaults mirror Meta (validated in backtest_exits.py): scale 50% out
at +20%, let the rest ride to a 25-bar horizon with a 3-bar drop-out grace. The
backtest (holdout 2025-07-01+, top-K by htf_score, exact stock paths) showed:
  * top-K width (5..20) barely changes per-trade quality — htf_score is flat across
    the top, so top-10 is chosen for diversification.
  * scale-out + let-rest-run beats pure rebalance (which churns at ~1-2 bar holds)
    and beats a hard +20% full exit on monster capture (the point of this harness),
    while banking the pop for theta protection on the options path.

SAFETY:
  * DRY-RUN by default (orders only with --submit); PAPER account unless --live.
  * Scoped: only ever sells symbols it bought (tracked in htf_live_state.json).
  * Staleness guard: refuses to trade on a matrix bar older than --max-staleness-days.

Run:
  PYTHONPATH=. python strategies/multi_ticker_swing_htf/live/runner.py            # dry-run paper
  PYTHONPATH=. python strategies/multi_ticker_swing_htf/live/runner.py --submit   # submit paper
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from signals.meta_context.meta_ranker.options_exec import select_option

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
BARS_4H = REPO / "Data/shared/bars/4h"
STATE_PATH = HERE / "htf_live_state.json"
BLACKLIST_PATH = HERE / "blacklist.txt"
MIN_FULL_BAR = 50  # ignore degenerate edge bars with fewer rows
SIGNAL = "htf_score"


def _load_blacklist() -> set[str]:
    if not BLACKLIST_PATH.exists():
        return set()
    out = set()
    for line in BLACKLIST_PATH.read_text().splitlines():
        t = line.split("#", 1)[0].strip().upper()
        if t:
            out.add(t)
    return out


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"managed": {}, "history": []}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _ref_price(ticker: str) -> float | None:
    p = BARS_4H / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p, columns=["close"])
    if b.empty:
        return None
    return float(b["close"].iloc[-1])


def _latest_full_bar(df: pd.DataFrame) -> pd.Timestamp:
    counts = df.groupby("timestamp").size()
    full = counts[counts >= max(MIN_FULL_BAR, int(0.25 * counts.max()))].index
    return full.max()


def _exit_action(gain, runs_held, bars_out, trimmed, args) -> tuple[str, str]:
    """Hold-based exit: horizon cap, drop-out (with grace), then take-profit scale-out."""
    if runs_held >= args.horizon_bars:
        return "exit", "horizon"
    if bars_out > args.grace_bars:
        return "exit", "dropped_out"
    if not trimmed and gain is not None and gain >= args.take_profit:
        return "trim", f"take_profit_+{int(args.take_profit * 100)}%"
    return "hold", ""


def main():
    ap = argparse.ArgumentParser(description="HTF Swing long-only top-K live runner (paper by default).")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--mode", choices=["equity", "options"], default="equity",
                    help="equity = buy shares; options = delta-filtered monthly calls on qualifying names.")
    ap.add_argument("--max-spread-pct", type=float, default=0.15, help="Options: max bid/ask spread to trade.")
    ap.add_argument("--roll-trading-days", type=int, default=5,
                    help="Options: roll to next monthly when nearest is within this many trading days.")
    ap.add_argument("--top-k", type=int, default=10, help="Top-K by htf_score (width barely matters 5..20).")
    ap.add_argument("--liquidity-floor", type=float, default=0.6,
                    help="Min dollar_vol_pctile_252 to be eligible.")
    ap.add_argument("--htf-rank-floor", type=float, default=0.85,
                    help="Min within-bar htf_xs_rank to be eligible (confidence gate).")
    # --- fixed sizing ---
    ap.add_argument("--shares", type=int, default=100, help="Equity: shares per new position.")
    ap.add_argument("--contracts", type=int, default=10, help="Options: contracts per new position.")
    # --- exit policy (mirrors Meta; validated in backtest_exits.py) ---
    ap.add_argument("--take-profit", type=float, default=0.20, help="Scale out at this gain.")
    ap.add_argument("--scale-frac", type=float, default=0.5, help="Fraction to sell at take-profit.")
    ap.add_argument("--horizon-bars", type=int, default=25, help="Exit remainder after this many bars (~10d).")
    ap.add_argument("--grace-bars", type=int, default=3, help="Exit only after this many consecutive bars out of top-K.")
    ap.add_argument("--max-staleness-days", type=float, default=0.5)
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--live", action="store_true", help="Target the LIVE account (default: paper).")
    ap.add_argument("--submit", action="store_true", help="Actually place orders (default: dry-run).")
    args = ap.parse_args()

    profile = "LIVE" if args.live else "PAPER"
    env_file = f".env#{profile}"
    run_mode = "SUBMIT" if args.submit else "DRY-RUN"
    print(f"=== HTF Swing {args.mode} runner | account={profile} | mode={run_mode} | top_k={args.top_k} ===")

    # --- signal: read htf_score straight off the shared matrix (no re-scoring) ---
    df = pd.read_parquet(args.matrix).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if SIGNAL not in df.columns:
        raise SystemExit(f"ABORT: matrix has no '{SIGNAL}' column — rebuild via update_meta_matrix.py")
    bar = _latest_full_bar(df)
    staleness = (datetime.now(timezone.utc) - bar.to_pydatetime()).total_seconds() / 86400.0
    print(f"latest full bar: {bar}  ({staleness:.1f} days old)")
    if staleness > args.max_staleness_days and not args.allow_stale:
        raise SystemExit(
            f"ABORT: matrix is {staleness:.1f}d stale (> {args.max_staleness_days}d). "
            f"Refresh via nightly_data_readiness.sh or pass --allow-stale."
        )

    cur = df[df["timestamp"] == bar].dropna(subset=[SIGNAL]).copy()
    cur["htf_rank_pct"] = cur[SIGNAL].rank(pct=True)
    n0 = len(cur)
    elig = cur[cur["htf_rank_pct"] >= args.htf_rank_floor]
    if "dollar_vol_pctile_252" in elig.columns:
        elig = elig[elig["dollar_vol_pctile_252"].fillna(0) >= args.liquidity_floor]
    blacklist = _load_blacklist()
    if blacklist:
        elig = elig[~elig["ticker"].str.upper().isin(blacklist)]
    top = elig.sort_values(SIGNAL, ascending=False).head(args.top_k)
    targets = list(top["ticker"])
    print(f"\neligible {len(elig)}/{n0} after filters "
          f"(htf_rank>={args.htf_rank_floor}, liq>={args.liquidity_floor}, blacklist={len(blacklist)})")
    print(f"htf top-{args.top_k}: {targets}")

    # --- account + positions ---
    client = AlpacaOptionsClient(env_file=env_file)
    acct = client.get_account()
    equity = float(acct["equity"])
    pos_info = {
        p["symbol"]: {"qty": int(float(p["qty"])),
                      "avg_entry": float(p.get("avg_entry_price", 0) or 0),
                      "current": float(p.get("current_price", 0) or 0)}
        for p in (client.get_positions() or [])
    }
    state = _load_state()
    managed = state.get("managed", {})
    print(f"equity=${equity:,.0f}  shares/name={args.shares}  TP +{int(args.take_profit*100)}% "
          f"(sell {int(args.scale_frac*100)}%)  horizon {args.horizon_bars}b  grace {args.grace_bars}b")
    print(f"managed held: {sorted(managed)}")

    if args.mode == "options":
        return _run_options(args, client, targets, state, managed, pos_info, bar)

    # --- equity reconciliation (only ever SELL symbols we manage) ---
    plan: list[tuple[str, str, int, str]] = []
    new_managed: dict[str, dict] = {}
    for sym, st in managed.items():
        held = pos_info.get(sym, {}).get("qty", 0)
        if held <= 0:
            continue
        in_tgt = sym in targets
        st["runs_held"] = st.get("runs_held", 0) + 1
        st["bars_out"] = 0 if in_tgt else st.get("bars_out", 0) + 1
        info = pos_info.get(sym, {})
        gain = (info["current"] / info["avg_entry"] - 1) if info.get("avg_entry") else None
        action, reason = _exit_action(gain, st["runs_held"], st["bars_out"], st.get("trimmed", False), args)
        if action == "exit":
            plan.append((sym, "sell", held, reason))
            continue
        if action == "trim":
            q = int(math.floor(args.scale_frac * held))
            if q >= 1:
                plan.append((sym, "sell", q, reason))
                st["trimmed"] = True
        new_managed[sym] = st
    for t in targets:
        if t in new_managed or pos_info.get(t, {}).get("qty", 0) > 0:
            continue
        plan.append((t, "buy", args.shares, "entry"))
        new_managed[t] = {"qty": args.shares, "runs_held": 0, "bars_out": 0, "trimmed": False, "entry_bar": str(bar)}

    print(f"\n--- order plan ({len(plan)} orders) ---")
    for sym, side, qty, reason in plan:
        px = _ref_price(sym) or 0.0
        print(f"  {side.upper():4} {qty:>4} {sym:<6} (~${qty*px:,.0f} @ {px:.2f})  [{reason}]")
    if not plan:
        print("  (nothing to do — positions within policy)")

    _execute(args, client, plan, state, new_managed, bar, targets, is_option=False)


def _run_options(args, client, targets, state, managed, pos_info, bar):
    """Options path: same hold-based exit + scale-out on delta-filtered monthly calls."""
    plan: list[tuple[str, str, int, str]] = []
    limits: dict[str, float] = {}
    new_managed: dict[str, dict] = {}
    for tkr, st in managed.items():
        occ = st.get("occ")
        held = pos_info.get(occ, {}).get("qty", 0) if occ else 0
        if held <= 0:
            continue
        in_tgt = tkr in targets
        st["runs_held"] = st.get("runs_held", 0) + 1
        st["bars_out"] = 0 if in_tgt else st.get("bars_out", 0) + 1
        info = pos_info.get(occ, {})
        gain = (info["current"] / info["avg_entry"] - 1) if info.get("avg_entry") else None
        action, reason = _exit_action(gain, st["runs_held"], st["bars_out"], st.get("trimmed", False), args)
        if action == "exit":
            plan.append((occ, "sell", held, reason))
            continue
        if action == "trim":
            q = int(math.floor(args.scale_frac * held))
            if q >= 1:
                plan.append((occ, "sell", q, reason))
                st["trimmed"] = True
        new_managed[tkr] = st
    print("\n--- contract selection (monthly expiry, delta 0.35-0.60, spread filter) ---")
    for t in targets:
        if t in new_managed:
            continue
        px = _ref_price(t)
        if not px or px <= 0:
            print(f"  ! {t:<6} skip: no price"); continue
        order, reason = select_option(client, t, px, 1e12, max_spread_pct=args.max_spread_pct,
                                      roll_trading_days=args.roll_trading_days)
        if order is None:
            print(f"  - {t:<6} SKIP ({reason})"); continue
        occ = order["occ"]
        if pos_info.get(occ, {}).get("qty", 0) > 0:
            continue
        plan.append((occ, "buy", args.contracts, "entry"))
        limits[occ] = order["limit"]
        new_managed[t] = {"occ": occ, "contracts": args.contracts, "runs_held": 0, "bars_out": 0,
                          "trimmed": False, "entry_bar": str(bar), "expiry": order.get("expiry")}
        print(f"  + {t:<6} {occ:<20} x{args.contracts} exp={order.get('expiry')} "
              f"delta={order.get('delta'):.2f} mid={order['mid']:.2f}")

    print(f"\n--- option order plan ({len(plan)} orders) ---")
    for occ, side, qty, reason in plan:
        print(f"  {side.upper():4} {qty:>3} {occ}" + (f" @limit {limits[occ]}" if occ in limits else " @market") + f"  [{reason}]")
    if not plan:
        print("  (nothing to do — positions within policy)")

    _execute(args, client, plan, state, new_managed, bar, targets, is_option=True, limits=limits)


def _execute(args, client, plan, state, new_managed, bar, targets, *, is_option: bool, limits=None):
    """Submit a market/limit order plan (paper/live) and persist managed state. Dry-run by default."""
    limits = limits or {}
    if args.submit and plan:
        print("\nsubmitting...")
        for sym, side, qty, _reason in plan:
            lim = limits.get(sym)
            try:
                if is_option:
                    resp = client.submit_option_order(symbol=sym, qty=qty, side=side,
                                                      order_type="limit" if lim else "market",
                                                      time_in_force="day", limit_price=lim)
                else:
                    resp = client.submit_order(symbol=sym, qty=qty, side=side,
                                               order_type="market", time_in_force="day")
                print(f"  OK {side} {qty} {sym}  id={resp.get('id', '?')}")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {side} {qty} {sym}: {exc}")
        state["managed"] = new_managed
        state.setdefault("history", []).append(
            {"ts": _now(), "bar": str(bar), "mode": args.mode, "targets": targets, "orders": len(plan)})
        _save_state(state)
        print(f"state updated -> {STATE_PATH}")
    elif not args.submit:
        print("\n(dry-run: no orders submitted, state unchanged. Add --submit to execute.)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
