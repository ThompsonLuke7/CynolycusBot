"""
Two-sleeve exit-policy shadow tracker — Momentum / HTF Swing / Meta Ranker.

Paper-only observation, matching the intraday_structure module's own stated
convention ("Existing broker execution is untouched. There is deliberately no
submit_order path in this module."). This script:

  1. Reads each module's live_state.json `managed` dict (already written by the
     real runners) to see real open positions — never calls a broker, never
     imports an order-submission path.
  2. For any real position not yet shadow-tracked, assigns it to the tail-rider
     (id4) or harvester (g284) sleeve using the SAME val-selected, test-frozen
     feature/threshold found in scripts/capstone/two_sleeve_cross_module.py
     (momentum/htf: atr_14 bottom-tercile; meta: signal_agreement top-tercile).
     Looked up with the SAME bounded, row-group-chunked parquet reader used
     there — momentum's features_4h.parquet is 4.0GB on disk; a plain
     pd.read_parquet() of it is what caused the 2026-07-21 WSL OOM crash.
  3. For harvester-sleeve entries, records a hypothetical call-debit and
     put-credit spread using the LATEST real chain snapshot
     (Data/dealer_positioning/historical_snapshots/<date>/dealer_strike_ladder.parquet
     — actual strikes/deltas/IV, not the backtest's assumed debit/credit
     fractions since that data doesn't exist for this forward-looking case).
  4. On every run, evaluates already-tracked shadow positions against current
     bars for their assigned exit policy (stop/target-scale/horizon) and logs
     a shadow exit when triggered. Shadow positions are independent of the
     real position's lifecycle — a shadow position keeps running on its own
     policy even after the real position (on the OLD id4-everywhere policy)
     exits, since that's the entire point of the comparison.

Output (own directory, never touches core.live_4h_exec or any runner state):
  Data/inference/shadow_two_sleeve/<module>_shadow_state.json   (open positions)
  Data/inference/shadow_two_sleeve/<module>_shadow_audit.jsonl  (entry/exit events)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/shadow/two_sleeve_shadow_tracker.py
  PYTHONPATH=. .venv/bin/python scripts/shadow/two_sleeve_shadow_tracker.py --module meta
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.live_signal_audit import append_jsonl, json_safe  # noqa: E402

OUT_DIR = REPO / "Data/inference/shadow_two_sleeve"
SHARED_BARS = REPO / "Data/shared/bars/4h"
SNAPSHOT_ROOT = REPO / "Data/dealer_positioning/historical_snapshots"

def _is_replayed(shadow_opened_at: str | None, exit_ts) -> bool:
    """True when the exit bar had already printed before the shadow book opened
    the position, i.e. the result was read out of history rather than tracked
    forward. Unknown open times are treated as replayed (conservative)."""
    if not shadow_opened_at:
        return True
    try:
        opened = pd.Timestamp(shadow_opened_at)
        exited = pd.Timestamp(exit_ts)
        if opened.tzinfo is None:
            opened = opened.tz_localize("UTC")
        if exited.tzinfo is None:
            exited = exited.tz_localize("UTC")
    except Exception:
        return True
    return bool(exited <= opened)


TAIL_POLICY = dict(name="tail_rider_id4", stop=0.39, trail=None, target=0.30, scale_frac=0.16, horizon=53)
HARVEST_POLICY = dict(name="harvester_g284", stop=0.59, trail=None, target=0.07, scale_frac=1.0, horizon=60)

MODULES = {
    # momentum/htf: split feature is atr_14, val-screened against each module's own
    # features_4h.parquet (fine for that backtest — 2025-07-01..2026-05-14 is within
    # both files' coverage). For LIVE lookups those files are the wrong source: only
    # momentum's copy is kept current (rebuilt by the readiness job); HTF's own local
    # copy tops out at 2026-06-02 and nothing refreshes it — confirmed via grep, no
    # live code reads it at all. HTF's live runner actually scores off
    # meta_ranker_matrix.parquet (DEFAULT_MATRIX in its own runner.py), which doesn't
    # carry raw atr_14 either. So for both, atr_14 is computed directly from the
    # always-current Data/shared/bars/4h cache (same True-Range formula as
    # feature_matrix_4h.py's _atr()) instead of depending on either matrix file.
    "momentum": dict(
        live_state=REPO / "strategies/momentum_expansion/live/momentum_live_state.json",
        source="bars_atr", split_feature="atr_14", split_positive=False, split_thresh=0.4149,
    ),
    "htf": dict(
        live_state=REPO / "strategies/multi_ticker_swing_htf/live/htf_live_state.json",
        source="bars_atr", split_feature="atr_14", split_positive=False, split_thresh=0.3269,
    ),
    # meta: signal_agreement = mom_xs_rank * htf_xs_rank, a cross-sectional feature
    # that needs the whole-universe distribution at that bar — can't be computed
    # from one ticker's bars. meta_ranker_matrix.parquet already carries it and is
    # confirmed fresh (max timestamp = today) since meta's own live pipeline rebuilds
    # it every run; it's also small (67MB), so a direct read is safe as-is.
    "meta": dict(
        live_state=REPO / "signals/meta_context/meta_ranker/live_state.json",
        source="matrix", feature_matrix=REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet",
        split_feature="signal_agreement", split_positive=True, split_thresh=0.9451,
    ),
}


def compute_atr_14(bars: pd.DataFrame, at_ts: pd.Timestamp) -> float | None:
    """Same formula as feature_matrix_4h.py's _atr(): True Range, simple 14-bar
    rolling mean. Uses bars up to and including at_ts from the always-current
    shared 4H cache — no dependency on either module's (stale/wrong) matrix file."""
    hist = bars[bars.index <= at_ts]
    if len(hist) < 15:
        return None
    h, lo, c_p = hist["high"], hist["low"], hist["close"].shift(1)
    tr = pd.concat([(h - lo), (h - c_p).abs(), (lo - c_p).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    val = atr.iloc[-1]
    return float(val) if pd.notna(val) else None


# ---- memory-safe single/few-key parquet lookup (same technique as the backtest) ----
def lookup_feature_values(path: Path, feature: str, needed_keys: set[tuple]) -> dict:
    """Row-group-chunked read, columns pruned to [timestamp, ticker, feature] only.
    needed_keys is normally 0-10 entries (new shadow entries since last run) against
    a file that can be multiple GB — never materialize the whole thing."""
    if not needed_keys:
        return {}
    pf = pq.ParquetFile(path)
    names = pf.schema.names
    if feature not in names:
        print(f"  [warn] {feature} not found in {path.name}; skipping shadow-entry lookup this pass")
        return {}
    found = {}
    for i in range(pf.num_row_groups):
        if len(found) == len(needed_keys):
            break
        tbl = pf.read_row_group(i, columns=["timestamp", "ticker", feature])
        df = tbl.to_pandas()
        del tbl
        if df.index.names and df.index.names[0] is not None:
            df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        keys = list(zip(df["timestamp"], df["ticker"]))
        for k, v in zip(keys, df[feature]):
            if k in needed_keys and k not in found and pd.notna(v):
                found[k] = float(v)
        del df, keys
        gc.collect()
    return found


def load_bars(ticker: str) -> pd.DataFrame | None:
    p = SHARED_BARS / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p, columns=["timestamp", "close", "high", "low"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    return b.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def resolve_underlying_entry_price(pos: dict) -> tuple[float | None, str]:
    """Underlying price at entry, for a managed position of either route.

    `evaluate_exit` walks the UNDERLYING's 4H bars, so entry_price must be an
    underlying price. For equity-routed positions `entry_avg_price` already is
    one. For option-routed positions it is the OPTION PREMIUM, and feeding that
    to a bar walk compares a $2-30 premium against a $15-530 share price: every
    option position clears the harvest sleeve's +7% target on its very first bar
    (observed 2026-08-03: WDC +1599%, BE +915%, MLTX +831%, CIFR +701%, FCEL
    +652%, BLZE +588% implied on bar one). The option audit carries the real
    underlying price at order time, so prefer that; momentum also stamps the
    signal bar's close. Returns (price, source) with price None when unresolvable.
    """
    route = str(pos.get("route") or "").strip().lower()
    audit = pos.get("order_audit") or {}
    signal_extra = ((pos.get("signal_audit") or {}).get("extra") or {})
    if route == "option":
        candidates = [
            (audit.get("underlying_price"), "order_audit.underlying_price"),
            (signal_extra.get("bar_close"), "signal_audit.extra.bar_close"),
        ]
    else:
        candidates = [
            (pos.get("entry_avg_price"), "entry_avg_price"),
            (audit.get("reference_price"), "order_audit.reference_price"),
            (signal_extra.get("bar_close"), "signal_audit.extra.bar_close"),
        ]
    for raw, source in candidates:
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val) and val > 0:
            return val, source
    return None, "unresolved"


def entry_price_is_plausible(entry_price: float, bars: pd.DataFrame,
                             entry_ts: pd.Timestamp, tol: float = 0.5) -> bool:
    """True when `entry_price` is on the same scale as the underlying bar at entry.

    A backstop for resolve_underlying_entry_price: any future route or audit-schema
    change that reintroduces a premium-vs-share mix-up shows up here as an
    order-of-magnitude gap rather than as silently fabricated +7% target exits.
    Tolerance is deliberately loose (±50%) — it catches scale errors, not slippage.
    """
    prior = bars[bars.index <= entry_ts]
    if prior.empty:
        return True  # no reference bar; evaluate_exit will simply find no path
    ref = float(prior["close"].iloc[-1])
    if not math.isfinite(ref) or ref <= 0:
        return True
    return abs(entry_price / ref - 1.0) <= tol


def latest_snapshot_dir() -> Path | None:
    dirs = sorted([d for d in SNAPSHOT_ROOT.iterdir() if d.is_dir()]) if SNAPSHOT_ROOT.exists() else []
    return dirs[-1] if dirs else None


def price_hypothetical_spreads(ticker: str, entry_price: float) -> dict:
    """Real strikes/deltas/IV from the latest chain snapshot — this is the live
    forward case, so (unlike the backtest) actual chain data exists WHEN the
    ticker is covered. It usually isn't: the dealer-positioning snapshot only
    covers ~700 relatively liquid names, confirmed to overlap only 0-23% of
    Momentum/HTF/Meta's actual open book (small-cap/expansion-stage names
    mostly lack a real chain at all — the same reason ~100% of this book
    routes to shares live, per route_option_or_shares' liquidity gate). Always
    returns a dict — either the priced spreads or an explicit "unavailable"
    reason, so the gap is visible in the audit trail rather than a silent null.
    Picks the near-ATM call and the call nearest entry_price*(1+7%) for a
    debit spread; a put ~5% OTM and one ~20% OTM for a credit spread."""
    snap_dir = latest_snapshot_dir()
    if snap_dir is None:
        return {"available": False, "reason": "no_snapshot_directory"}
    ladder_path = snap_dir / "dealer_strike_ladder.parquet"
    if not ladder_path.exists():
        return {"available": False, "reason": "no_ladder_file", "snapshot_date": snap_dir.name}
    needed_cols = ["symbol", "scope", "spot", "strike", "call_delta", "call_iv", "put_delta", "put_iv"]
    # A day where the dealer capture itself failed (e.g. a Schwab auth outage,
    # as on 2026-07-22: 0/2790 requests succeeded) still writes a ladder file,
    # but an empty one — none of these columns exist on it, only capture
    # bookkeeping. Check the on-disk schema before the typed column read
    # instead of letting pyarrow raise, so a bad capture day degrades to
    # "unavailable" like every other gap here rather than crashing the run.
    existing_cols = set(pq.ParquetFile(ladder_path).schema_arrow.names)
    if not set(needed_cols).issubset(existing_cols):
        return {"available": False, "reason": "ladder_file_missing_columns", "snapshot_date": snap_dir.name}
    df = pd.read_parquet(ladder_path, columns=needed_cols)
    sub = df[(df["symbol"] == ticker) & (df["scope"] == "daily_week")]
    if sub.empty:
        sub = df[(df["symbol"] == ticker) & (df["scope"] == "two_months")]
    del df
    if sub.empty:
        return {"available": False, "reason": "ticker_not_in_chain_snapshot",
                "snapshot_date": snap_dir.name}
    spot = float(sub["spot"].iloc[0])

    def nearest(target_strike):
        return sub.iloc[(sub["strike"] - target_strike).abs().argsort()[:1]].iloc[0]

    long_call = nearest(spot)
    short_call = nearest(spot * (1 + HARVEST_POLICY["target"]))
    short_put = nearest(spot * 0.95)
    long_put = nearest(spot * 0.80)
    return {
        "available": True,
        "snapshot_date": snap_dir.name,
        "spot_at_snapshot": spot,
        "call_debit_spread": {
            "long_strike": float(long_call["strike"]), "long_delta": float(long_call["call_delta"]),
            "long_iv": float(long_call["call_iv"]),
            "short_strike": float(short_call["strike"]), "short_delta": float(short_call["call_delta"]),
            "short_iv": float(short_call["call_iv"]),
        },
        "put_credit_spread": {
            "short_strike": float(short_put["strike"]), "short_delta": float(short_put["put_delta"]),
            "short_iv": float(short_put["put_iv"]),
            "long_strike": float(long_put["strike"]), "long_delta": float(long_put["put_delta"]),
            "long_iv": float(long_put["put_iv"]),
        },
    }


def _load_json(path: Path, default):
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return default


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_safe(obj), f, indent=2, default=str)


def evaluate_exit(entry_price: float, bars: pd.DataFrame, entry_ts: pd.Timestamp,
                   policy: dict) -> dict | None:
    """Same exit mechanic as core.live_4h_exec / the backtest harness, walked
    forward over actual bars since entry. Returns exit info or None if still open."""
    path_bars = bars[bars.index > entry_ts]
    if path_bars.empty:
        return None
    peak = entry_price
    trimmed = False
    realized = 0.0
    remaining = 1.0
    for n, (ts, row) in enumerate(path_bars.iterrows(), start=1):
        peak = max(peak, row["high"])
        lo_ret = row["low"] / entry_price - 1
        hi_ret = row["high"] / entry_price - 1
        if policy["stop"] is not None and lo_ret <= -policy["stop"]:
            return dict(exit_ts=ts, bars_held=n, reason="stop",
                        underlying_ret=realized + remaining * (-policy["stop"]))
        if policy["trail"] is not None and row["low"] <= peak * (1 - policy["trail"]):
            exit_ret = peak * (1 - policy["trail"]) / entry_price - 1
            return dict(exit_ts=ts, bars_held=n, reason="trail",
                        underlying_ret=realized + remaining * exit_ret)
        if policy["target"] is not None and not trimmed and hi_ret >= policy["target"]:
            if policy["scale_frac"] >= 1.0:
                return dict(exit_ts=ts, bars_held=n, reason="target",
                            underlying_ret=policy["target"])
            realized += policy["scale_frac"] * policy["target"]
            remaining = 1.0 - policy["scale_frac"]
            trimmed = True
        if policy["horizon"] is not None and n >= policy["horizon"]:
            exit_ret = row["close"] / entry_price - 1
            return dict(exit_ts=ts, bars_held=n, reason="horizon",
                        underlying_ret=realized + remaining * exit_ret)
    return None  # still open


def bull_call_debit_spread_realized(underlying_ret: float, spread: dict, entry_price: float) -> float | None:
    long_k, short_k = spread["long_strike"], spread["short_strike"]
    width_pct = (short_k - long_k) / entry_price
    if width_pct <= 0:
        return None
    frac = max(0.0, min(1.0, underlying_ret / width_pct))
    # crude entry debit proxy from delta spread (no IV-surface repricing at exit —
    # flagged approximate, same caveat as the backtest): wider delta gap ~ pricier debit.
    debit_frac = max(0.15, min(0.85, 1 - (spread["long_delta"] - spread["short_delta"])))
    return (frac - debit_frac) / debit_frac if debit_frac > 0 else None


def bull_put_credit_spread_realized(underlying_ret: float, spread: dict, entry_price: float) -> float | None:
    short_k, long_k = spread["short_strike"], spread["long_strike"]
    width_pct = (short_k - long_k) / entry_price
    if width_pct <= 0:
        return None
    short_buffer = 1 - short_k / entry_price
    loss_frac = max(0.0, min(1.0, (-short_buffer - underlying_ret) / width_pct))
    credit_frac = max(0.10, min(0.60, abs(spread["short_delta"])))
    credit = credit_frac * width_pct
    max_loss = width_pct - credit
    if max_loss <= 0:
        return None
    return (credit - loss_frac * width_pct) / max_loss


def run_module(module: str, cfg: dict) -> None:
    print(f"\n=== {module} ===")
    state_path = OUT_DIR / f"{module}_shadow_state.json"
    audit_path = OUT_DIR / f"{module}_shadow_audit.jsonl"
    live_state = _load_json(cfg["live_state"], {"managed": {}})
    shadow_state = _load_json(state_path, {"managed": {}, "closed": []})

    real_managed = live_state.get("managed", {})
    shadow_managed = shadow_state.get("managed", {})
    # A shadow key that already exited must never be re-entered. The real module
    # keeps holding the position long after the shadow policy would have exited
    # it, so without this registry the same (ticker, entry_bar) was shadow-entered
    # and shadow-exited on EVERY run — booking one trade dozens of times and
    # making the closed-trade statistics meaningless.
    closed_keys = set(shadow_state.get("closed") or [])

    # ---- 1) new shadow entries: real positions not yet shadow-tracked ----
    new_keys = []
    for ticker, pos in real_managed.items():
        shadow_key = f"{ticker}:{pos.get('entry_bar')}"
        if shadow_key in closed_keys or shadow_key in shadow_managed:
            continue
        entry_ts = pd.Timestamp(pos["entry_bar"])
        new_keys.append((shadow_key, ticker, entry_ts, pos))

    if new_keys:
        if cfg["source"] == "matrix":
            needed = {(ts, tk) for _, tk, ts, _ in new_keys}
            feat_vals = lookup_feature_values(cfg["feature_matrix"], cfg["split_feature"], needed)
        else:  # "bars_atr" — computed per-ticker from the shared bars cache, no matrix file
            feat_vals = {}
            for _, ticker, entry_ts, _ in new_keys:
                bars = load_bars(ticker)
                if bars is None:
                    continue
                v = compute_atr_14(bars, entry_ts)
                if v is not None:
                    feat_vals[(entry_ts, ticker)] = v
        for shadow_key, ticker, entry_ts, pos in new_keys:
            fval = feat_vals.get((entry_ts, ticker))
            if fval is None:
                print(f"  [{ticker}] no {cfg['split_feature']} value yet "
                      f"(no shared bars / insufficient history) — will retry next run")
                continue
            # A position opened THIS cycle by the real runner doesn't get
            # `entry_avg_price` until its next management pass (that field is
            # only backfilled from the broker's avg_entry on already-held
            # positions — see core/live_4h_exec.py's build_mixed_plan). The
            # resolver falls back to the order-time reference price, and for
            # option-routed positions reads the underlying price rather than the
            # premium; skip and retry once the runner's next cycle fills it in —
            # never crash the whole module's shadow run over one ticker.
            raw_entry_price, price_source = resolve_underlying_entry_price(pos)
            if raw_entry_price is None:
                print(f"  [{ticker}] no underlying entry price yet (route="
                      f"{pos.get('route')}, position opened this cycle) — will retry next run")
                continue
            entry_price = float(raw_entry_price)
            entry_bars = load_bars(ticker)
            if entry_bars is not None and not entry_price_is_plausible(entry_price, entry_bars, entry_ts):
                ref = entry_bars[entry_bars.index <= entry_ts]["close"].iloc[-1]
                print(f"  [{ticker}] SKIP: entry price {entry_price:.4f} from {price_source} is off "
                      f"the underlying scale (bar close {ref:.4f}) — refusing to fabricate an exit")
                continue
            is_tail = (fval >= cfg["split_thresh"]) == cfg["split_positive"]
            policy = TAIL_POLICY if is_tail else HARVEST_POLICY
            record = {
                "ticker": ticker, "entry_bar": pos["entry_bar"], "entry_price": entry_price,
                # Always an UNDERLYING price (see resolve_underlying_entry_price);
                # recorded so a future audit can tell premium from share price.
                "entry_price_source": price_source,
                "sleeve": "tail" if is_tail else "harvest", "policy": policy["name"],
                "split_feature": cfg["split_feature"], "split_value": fval,
                "split_thresh": cfg["split_thresh"], "real_route": pos.get("route"),
                "real_signal_audit": pos.get("signal_audit"),
                # When the shadow book first saw this position. `entry_bar` can be
                # days older, in which case the exit below is resolved over bars
                # that had already printed — a replay, not a forward-tracked
                # result. Recorded so research can filter those out.
                "shadow_opened_at": datetime.now(timezone.utc).isoformat(),
            }
            spread_note = ""
            if not is_tail:
                spreads = price_hypothetical_spreads(ticker, entry_price)
                record["hypothetical_spreads"] = spreads
                spread_note = "" if spreads["available"] else f", no chain data ({spreads['reason']})"
            shadow_managed[shadow_key] = record
            append_jsonl(audit_path, {
                "event": "shadow_entry", "module": module, "paper_shadow": True,
                "submitted": False, **record,
            })
            print(f"  [{ticker}] NEW shadow entry -> {record['sleeve']} sleeve "
                  f"({cfg['split_feature']}={fval:.4f}{spread_note})")

    # ---- 2) evaluate existing shadow positions for exit ----
    still_open = {}
    for shadow_key, pos in shadow_managed.items():
        ticker = pos["ticker"]
        bars = load_bars(ticker)
        if bars is None:
            still_open[shadow_key] = pos
            continue
        policy = TAIL_POLICY if pos["sleeve"] == "tail" else HARVEST_POLICY
        entry_ts = pd.Timestamp(pos["entry_bar"])
        entry_price = pos["entry_price"]
        result = evaluate_exit(entry_price, bars, entry_ts, policy)
        if result is None:
            still_open[shadow_key] = pos
            continue
        underlying_ret = result["underlying_ret"]
        exit_record = {
            "event": "shadow_exit", "module": module, "paper_shadow": True, "submitted": False,
            "ticker": ticker, "entry_bar": pos["entry_bar"], "sleeve": pos["sleeve"],
            "policy": pos["policy"], "exit_ts": str(result["exit_ts"]),
            "bars_held": result["bars_held"], "exit_reason": result["reason"],
            "underlying_ret": underlying_ret,
            "entry_price": entry_price,
            "entry_price_source": pos.get("entry_price_source", "legacy_unknown"),
            "real_route": pos.get("real_route"),
            "shadow_opened_at": pos.get("shadow_opened_at"),
            "replayed": _is_replayed(pos.get("shadow_opened_at"), result["exit_ts"]),
        }
        if pos["sleeve"] == "harvest" and pos.get("hypothetical_spreads", {}).get("available"):
            sp = pos["hypothetical_spreads"]
            cs = sp.get("call_debit_spread")
            ps = sp.get("put_credit_spread")
            if cs:
                exit_record["call_debit_spread_ret"] = bull_call_debit_spread_realized(
                    underlying_ret, cs, entry_price)
            if ps:
                exit_record["put_credit_spread_ret"] = bull_put_credit_spread_realized(
                    underlying_ret, ps, entry_price)
        append_jsonl(audit_path, exit_record)
        closed_keys.add(shadow_key)
        print(f"  [{ticker}] shadow EXIT ({pos['sleeve']}, {result['reason']}) "
              f"underlying_ret={underlying_ret*100:+.2f}%"
              f"{' [replayed]' if exit_record['replayed'] else ''}")

    shadow_state["managed"] = still_open
    shadow_state["closed"] = sorted(closed_keys)
    _save_json(state_path, shadow_state)
    print(f"  shadow book: {len(still_open)} open, {len(shadow_managed) - len(still_open)} closed this run")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=list(MODULES) + ["all"], default="all")
    args = ap.parse_args()
    modules = list(MODULES) if args.module == "all" else [args.module]
    for module in modules:
        run_module(module, MODULES[module])


if __name__ == "__main__":
    main()
