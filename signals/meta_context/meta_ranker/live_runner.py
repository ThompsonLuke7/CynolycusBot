"""
Meta Ranker live equity runner — long-only, equal-weight top-K combo portfolio.

Scores the meta-ranker matrix with the confluence (combo) signal, takes the top-K
names on the latest full 4H bar, and reconciles a paper (default) Alpaca account to
an equal-weight long-only target portfolio.

SAFETY (read before running live):
  * DRY-RUN by default. Orders are only submitted with --submit.
  * PAPER account by default (.env#PAPER). --live is required to target the live account.
  * Scoped holdings: this runner ONLY ever sells symbols it previously bought (tracked in
    live_state.json). It never touches positions opened by other strategies sharing the account.
  * Staleness guard: refuses to trade if the matrix's latest bar is older than
    --max-staleness-days (override with --allow-stale). The shipped matrix is label-limited;
    rebuild via build_meta_ranker_matrix.py for current signals.

Run:
  # dry run on paper (no orders), see the plan
  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/live_runner.py
  # actually submit on paper
  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/live_runner.py --submit
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from signals.meta_context.meta_ranker.score import score_frame
from signals.meta_context.meta_ranker.options_exec import select_option

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_MATRIX = HERE / "meta_ranker_matrix.parquet"
BARS_4H = REPO / "Data/shared/bars/4h"
STATE_PATH = HERE / "live_state.json"
BLACKLIST_PATH = HERE / "blacklist.txt"  # optional hard exclusions, one ticker per line (# comments ok)
MIN_FULL_BAR = 50  # ignore degenerate edge bars with fewer rows


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
    """Decide what to do with a held position. Returns (action, reason).

    Order: time-horizon exit, dropped-out-of-topK (with grace), then take-profit scale-out.
    Pure rebalance-only (grace=0) churns at ~1.8 bars and only nets ~+1%/trade — see
    backtest_exits.py — so the default holds to the horizon with a grace buffer.
    """
    if runs_held >= args.horizon_bars:
        return "exit", "horizon"
    if bars_out > args.grace_bars:
        return "exit", "dropped_out"
    if not trimmed and gain is not None and gain >= args.take_profit:
        return "trim", f"take_profit_+{int(args.take_profit * 100)}%"
    return "hold", ""


def main():
    ap = argparse.ArgumentParser(description="Meta Ranker long-only equity live runner (paper by default).")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--mode", choices=["equity", "options"], default="equity",
                    help="equity = buy shares; options = buy delta-filtered calls on qualifying names.")
    ap.add_argument("--max-spread-pct", type=float, default=0.15, help="Options: max bid/ask spread to trade.")
    ap.add_argument("--roll-trading-days", type=int, default=5,
                    help="Options: roll to next monthly when nearest is within this many trading days.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--liquidity-floor", type=float, default=0.6,
                    help="Min dollar_vol_pctile_252 to be eligible (generalizable; raises holdout return).")
    ap.add_argument("--combo-floor", type=float, default=0.90,
                    help="Min combo rank-pct to be eligible (confidence gate; generalizable).")
    ap.add_argument("--quality-floor", type=float, default=0.4,
                    help="Min raw s_quality to be eligible for a NEW entry. Backtested (Sep25-May26): "
                         "gating top-5 cross-in entries on s_quality>=0.4 lifts mean forward-close "
                         "from ~8.9%% to ~13.4%% and blocks pyramiding into blow-off tops (e.g. CAR). "
                         "Set to a very negative number to disable.")
    # --- fixed sizing ---
    ap.add_argument("--shares", type=int, default=100, help="Equity: shares per new position.")
    ap.add_argument("--contracts", type=int, default=10, help="Options: contracts per new position.")
    # --- exit policy (hold-based; rebalance-only churns — see backtest_exits.py) ---
    ap.add_argument("--take-profit", type=float, default=0.20, help="Scale out at this gain.")
    ap.add_argument("--scale-frac", type=float, default=0.5, help="Fraction to sell at take-profit.")
    ap.add_argument("--horizon-bars", type=int, default=25, help="Exit remainder after this many bars (~10d).")
    ap.add_argument("--grace-bars", type=int, default=3, help="Exit only after this many consecutive bars out of top-K.")
    # Refuse to trade on stale data: if the latest full bar is older than this,
    # abort. Tight (half a day) so a missed nightly refresh can't trade yesterday.
    ap.add_argument("--max-staleness-days", type=float, default=0.5)
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--live", action="store_true", help="Target the LIVE account (default: paper).")
    ap.add_argument("--submit", action="store_true", help="Actually place orders (default: dry-run).")
    args = ap.parse_args()

    profile = "LIVE" if args.live else "PAPER"
    env_file = f".env#{profile}"
    run_mode = "SUBMIT" if args.submit else "DRY-RUN"
    print(f"=== Meta Ranker {args.mode} runner | account={profile} | mode={run_mode} | top_k={args.top_k} ===")

    # --- score + select ---
    df = pd.read_parquet(args.matrix).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    bar = _latest_full_bar(df)
    staleness = (datetime.now(timezone.utc) - bar.to_pydatetime()).total_seconds() / 86400.0
    print(f"latest full bar: {bar}  ({staleness:.1f} days old)")
    if staleness > args.max_staleness_days and not args.allow_stale:
        raise SystemExit(
            f"ABORT: matrix is {staleness:.1f}d stale (> {args.max_staleness_days}d). "
            f"Rebuild via build_meta_ranker_matrix.py or pass --allow-stale."
        )

    scored = score_frame(df[df["timestamp"] == bar].copy())
    # Eligibility filters (validated to generalize out-of-sample; see analyze_policy notes):
    #   liquidity floor + combo confidence floor + optional manual hard-exclusions.
    n0 = len(scored)
    elig = scored[scored["s_combo"] >= args.combo_floor]
    if "dollar_vol_pctile_252" in elig.columns:
        elig = elig[elig["dollar_vol_pctile_252"].fillna(0) >= args.liquidity_floor]
    blacklist = _load_blacklist()
    if blacklist:
        elig = elig[~elig["ticker"].str.upper().isin(blacklist)]
    top = elig.sort_values("s_combo", ascending=False).head(args.top_k)
    targets = list(top["ticker"])
    # Quality gate is applied to NEW ENTRIES only (held names exit via horizon/grace,
    # not a quality dip). A combo top-K name is bought only if its s_quality clears
    # the floor — this is the backtested "cross-in + quality" entry rule.
    quality_by_ticker = scored.set_index("ticker")["s_quality"].to_dict()
    entry_ok = {t: float(quality_by_ticker.get(t, float("-inf"))) >= args.quality_floor for t in targets}
    print(f"\neligible {len(elig)}/{n0} after filters "
          f"(combo>={args.combo_floor}, liq>={args.liquidity_floor}, blacklist={len(blacklist)})")
    print(f"combo top-{args.top_k}: {targets}")
    gated_out = [t for t in targets if not entry_ok[t]]
    if gated_out:
        print(f"quality-gated OUT of new entries (s_quality<{args.quality_floor}): {gated_out}")

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
    managed = state.get("managed", {})  # symbols THIS strategy owns
    print(f"equity=${equity:,.0f}  shares/name={args.shares}  TP +{int(args.take_profit*100)}% "
          f"(sell {int(args.scale_frac*100)}%)  horizon {args.horizon_bars}b  grace {args.grace_bars}b")
    print(f"managed held: {sorted(managed)}")

    if args.mode == "options":
        return _run_options(args, client, targets, state, managed, pos_info, bar, entry_ok)

    # --- hold-based reconciliation (only ever SELL symbols we manage) ---
    plan: list[tuple[str, str, int, str]] = []  # (symbol, side, qty, reason)
    new_managed: dict[str, dict] = {}
    for sym, st in managed.items():
        held = pos_info.get(sym, {}).get("qty", 0)
        if held <= 0:
            continue  # position gone (closed elsewhere) — drop from state
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
    # entries: new top-K names not already held, that clear the quality gate
    for t in targets:
        if t in new_managed or pos_info.get(t, {}).get("qty", 0) > 0:
            continue
        if not entry_ok.get(t, False):
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


def _run_options(args, client, targets, state, managed, pos_info, bar, entry_ok=None):
    """Options path: same hold-based exit + scale-out state machine, on delta-filtered monthly calls."""
    plan: list[tuple[str, str, int, str]] = []  # (occ, side, contracts, reason)
    limits: dict[str, float] = {}
    new_managed: dict[str, dict] = {}
    # 1) manage existing option positions (managed: ticker -> {occ, ...})
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
    # 2) entries: new top-K names not already managed (fixed contract count)
    print("\n--- contract selection (monthly expiry, delta 0.35-0.60, spread filter) ---")
    entry_ok = entry_ok or {}
    for t in targets:
        if t in new_managed:
            continue
        if not entry_ok.get(t, False):
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
