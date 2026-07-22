"""Dealer-positioning end-of-day ranked options runner.

This turns the nightly/near-close dealer swing-potential ranking into a paper-first
execution pass:

  1. optionally refresh option-chain snapshots and rebuild rankings,
  2. take the top N dealer_swing_rank names,
  3. buy the closest ATM non-0DTE option contract,
  4. manage exits with the shared 4H execution policy.

It is an experiment harness, not a proof of edge. Live routing requires --live and
--submit, and the combined-server dashboard keeps paper as the default account.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
# An absolute script launch puts ``strategies/dealer_positioning`` first on
# sys.path.  That directory contains signals.py, which otherwise shadows the
# repository's top-level ``signals`` package whenever PYTHONPATH already
# contains REPO (the combined-server scheduler does exactly that).
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.live_4h_exec import (
    ExecPolicy,
    build_mixed_plan,
    defer_entries_if_market_closed,
    execute_plan,
    now_utc_iso,
    order_plan_audit_record,
    submit_pending_open_entries,
)
from core.live_signal_audit import append_jsonl, build_signal_audit
from signals.meta_context.meta_ranker.options_exec import equity_order_tif
from strategies.dealer_positioning.scripts.build_dealer_rankings import (
    RANKING_ROOT,
    SNAPSHOT_ROOT,
    build_rankings,
    latest_snapshot_path,
)
from strategies.dealer_positioning.scripts.capture_historical_snapshots import (
    OUT_ROOT,
    SCOPES,
    capture_snapshots,
    load_symbols,
)
from strategies.multi_ticker_swing.live.runner import (
    _contract_expiry,
    _contract_strike,
    _contract_symbol,
    _is_standard_100_contract,
)

logger = logging.getLogger(__name__)

BARS_4H = REPO / "Data/shared/bars/4h"
STATE_PATH = REPO / "Data/inference/dealer_ranker/live_state.json"
AUDIT_LOG = REPO / "Data/inference/dealer_ranker/live_signal_audit.jsonl"
MODULE = "dealer_ranker"
_ET = ZoneInfo("America/New_York")


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"managed": {}, "history": []}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _build_pos_info(client: AlpacaOptionsClient) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in client.get_positions() or []:
        try:
            out[str(p["symbol"]).upper()] = {
                "qty": int(float(p["qty"])),
                "avg_entry": float(p.get("avg_entry_price", 0) or 0),
                "current": float(p.get("current_price", 0) or 0),
            }
        except Exception:
            continue
    return out


def _latest_4h_price(ticker: str) -> float | None:
    path = BARS_4H / f"{ticker.upper()}.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path, columns=["close"])
    except Exception:
        return None
    if frame.empty:
        return None
    try:
        return float(frame["close"].iloc[-1])
    except Exception:
        return None


def _snapshot_spot_map() -> dict[str, float]:
    try:
        path = latest_snapshot_path(SNAPSHOT_ROOT)
        frame = pd.read_parquet(path, columns=["symbol", "scope", "spot"])
    except Exception:
        return {}
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    out: dict[str, float] = {}
    for symbol, group in frame.groupby("symbol"):
        preferred = group[group["scope"].astype(str).eq("daily_week")]
        row = preferred.iloc[-1] if not preferred.empty else group.iloc[-1]
        try:
            spot = float(row.get("spot"))
        except Exception:
            continue
        if math.isfinite(spot) and spot > 0:
            out[symbol] = spot
    return out


def _ref_price_fn(spot_map: dict[str, float]):
    def _ref(ticker: str) -> float | None:
        return spot_map.get(ticker.upper()) or _latest_4h_price(ticker)
    return _ref


def refresh_rankings(
    *,
    workers: int,
    limit: int | None,
    sleep_seconds: float,
    snapshot_date: str | None,
    ref_date: str | None,
) -> Path:
    symbols = load_symbols(limit=limit)
    result = capture_snapshots(
        symbols=symbols,
        scopes=SCOPES,
        output_root=OUT_ROOT,
        sleep_seconds=sleep_seconds,
        snapshot_date=snapshot_date,
        ref_date=ref_date,
        workers=workers,
    )
    summary_path = result.output_dir / "dealer_level_summary.parquet"
    frame = pd.read_parquet(summary_path)
    rankings = build_rankings(frame)
    if rankings.empty:
        raise RuntimeError(f"no dealer rankings built from {summary_path}")
    snapshot_key = str(rankings["snapshot_date"].dropna().iloc[0]).replace("-", "")
    RANKING_ROOT.mkdir(parents=True, exist_ok=True)
    out = RANKING_ROOT / f"dealer_swing_rankings_{snapshot_key}.parquet"
    latest = RANKING_ROOT / "dealer_swing_rankings_latest.parquet"
    rankings.to_parquet(out, index=False)
    rankings.to_parquet(latest, index=False)
    logger.info("dealer rankings refreshed: rows=%d latest=%s", len(rankings), latest)
    return latest


def _load_rankings(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else RANKING_ROOT / "dealer_swing_rankings_latest.parquet"
    if not path.exists():
        raise FileNotFoundError(f"dealer ranking file not found: {path}")
    frame = pd.read_parquet(path)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame.sort_values("dealer_swing_rank").reset_index(drop=True)


def _signal_audits(top: pd.DataFrame, *, bar: str, side_mode: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    total = max(1, len(top))
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ticker = str(row["symbol"]).upper()
        out[ticker] = build_signal_audit(
            module=MODULE,
            ticker=ticker,
            score=row.get("dealer_swing_potential_score"),
            side="long",
            rank=rank,
            rank_pct=1.0 - ((rank - 1) / total),
            signal_ts=bar,
            extra={
                "dealer_swing_rank": row.get("dealer_swing_rank"),
                "dealer_direction": row.get("dealer_direction"),
                "dealer_direction_bias": row.get("dealer_direction_bias"),
                "side_mode": side_mode,
                "max_scope_score": row.get("max_scope_score"),
                "avg_vacuum_component": row.get("avg_vacuum_component"),
                "avg_sparse_gamma_component": row.get("avg_sparse_gamma_component"),
                "avg_pinning_room_component": row.get("avg_pinning_room_component"),
                "avg_wall_room_component": row.get("avg_wall_room_component"),
                "scope_scores_json": row.get("scope_scores_json"),
            },
        )
    return out


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _quote_from_snapshot(snapshot: dict | None) -> tuple[float | None, float | None]:
    quote = (snapshot or {}).get("latestQuote") or {}
    bid = _as_float(quote.get("bp", quote.get("bid_price")))
    ask = _as_float(quote.get("ap", quote.get("ask_price")))
    return bid, ask


def _latest_quote(client: AlpacaOptionsClient, occ: str) -> tuple[float | None, float | None]:
    try:
        resp = client.get_option_quotes(symbols=occ)
    except Exception:
        return None, None
    quotes = resp.get("quotes", resp) if isinstance(resp, dict) else {}
    q = quotes.get(occ) if isinstance(quotes, dict) else None
    if q is None and isinstance(quotes, dict) and quotes:
        q = next(iter(quotes.values()))
    if not isinstance(q, dict):
        return None, None
    return _as_float(q.get("bp", q.get("bid_price"))), _as_float(q.get("ap", q.get("ask_price")))


def _has_tradable_contracts(
    client: AlpacaOptionsClient,
    ticker: str,
    *,
    option_type: str,
    min_dte: int,
    max_dte: int,
    now_et: datetime | None = None,
) -> bool:
    """Cheap existence check: does this underlying list ANY standard 100-share,
    non-0DTE contract of ``option_type`` in [min_dte, max_dte]?

    No strike bound (unlike ``_select_atm_option``) — this only answers whether
    a name is worth a priced ATM lookup at all, so it can pre-filter ranking
    candidates before they consume a top-K slot. Some names (illiquid ADRs,
    small caps) simply have no Alpaca-listed chain in the window; when several
    of them land in the same day's top-K together the module can otherwise
    place zero orders for the whole session.
    """
    now_et = now_et or datetime.now(_ET)
    start = now_et.date() + timedelta(days=max(1, int(min_dte)))
    end = now_et.date() + timedelta(days=max(int(min_dte), int(max_dte)))
    cp = "call" if option_type == "call" else "put"
    try:
        resp = client.get_option_contracts(
            underlying_symbol=ticker.upper(),
            expiration_date_gte=start.isoformat(),
            expiration_date_lte=end.isoformat(),
            type=cp,
            status="active",
        )
    except Exception:
        return False
    page = resp.get("option_contracts") if isinstance(resp, dict) else resp
    for c in (page or []):
        if not isinstance(c, dict) or not c.get("tradable", True):
            continue
        if not _is_standard_100_contract(c, ticker):
            continue
        exp = _contract_expiry(c)
        if exp is not None and start <= exp <= end:
            return True
    return False


def _select_optionable_targets(
    client: AlpacaOptionsClient,
    rankings: pd.DataFrame,
    *,
    top_k: int,
    side_mode: str,
    min_dte: int,
    max_dte: int,
    scan_multiple: int = 5,
    now_et: datetime | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Walk the rank-sorted table and keep the best ``top_k`` names that
    actually have a listed non-0DTE chain, instead of freezing on a fixed
    top-K slice that can be entirely non-optionable (observed 7/17: 6/10
    skipped; 7/20: 10/10 skipped -> zero orders placed for the whole day).
    Rank priority is preserved; the scan is capped at ``top_k * scan_multiple``
    candidates so a systemically non-optionable universe can't turn one pass
    into an unbounded number of API calls. Skipped tickers are returned so the
    caller can record them in the signal-decision audit instead of the module
    going silently idle.
    """
    top_k = max(1, int(top_k))
    scan_cap = top_k * max(1, int(scan_multiple))
    keep_idx: list[int] = []
    skipped: list[str] = []
    for scanned, (idx, row) in enumerate(rankings.iterrows()):
        if len(keep_idx) >= top_k or scanned >= scan_cap:
            break
        ticker = str(row["symbol"]).upper()
        opt_type = _side_for_ticker(ticker, side_mode=side_mode, top=rankings)
        if _has_tradable_contracts(client, ticker, option_type=opt_type, min_dte=min_dte, max_dte=max_dte,
                                   now_et=now_et):
            keep_idx.append(idx)
        else:
            skipped.append(ticker)
    return rankings.loc[keep_idx], skipped


def _select_atm_option(
    client: AlpacaOptionsClient,
    ticker: str,
    current_price: float,
    *,
    option_type: str,
    min_dte: int,
    max_dte: int,
    now_et: datetime | None = None,
) -> tuple[dict | None, str]:
    now_et = now_et or datetime.now(_ET)
    start = now_et.date() + timedelta(days=max(1, int(min_dte)))
    end = now_et.date() + timedelta(days=max(int(min_dte), int(max_dte)))
    cp = "call" if option_type == "call" else "put"
    strike_lo = int(round(current_price * 0.90, 0))
    strike_hi = int(round(current_price * 1.10, 0))
    contracts: list[dict] = []
    page_token: str | None = None
    try:
        for _ in range(10):
            resp = client.get_option_contracts(
                underlying_symbol=ticker.upper(),
                expiration_date_gte=start.isoformat(),
                expiration_date_lte=end.isoformat(),
                type=cp,
                strike_price_gte=strike_lo,
                strike_price_lte=strike_hi,
                status="active",
                page_token=page_token,
            )
            page = resp.get("option_contracts") if isinstance(resp, dict) else resp
            contracts.extend(c for c in (page or []) if isinstance(c, dict))
            page_token = resp.get("next_page_token") if isinstance(resp, dict) else None
            if not page_token:
                break
    except Exception as exc:  # noqa: BLE001
        return None, f"contracts_error({exc})"

    tradable = [
        c for c in contracts
        if c.get("tradable", True)
        and _is_standard_100_contract(c, ticker)
        and (_contract_expiry(c) is not None)
        and start <= _contract_expiry(c) <= end
    ]
    if not tradable:
        return None, f"no_non_0dte_{cp}_contracts"
    expiry = min(_contract_expiry(c) for c in tradable if _contract_expiry(c) is not None)
    same_exp = [c for c in tradable if _contract_expiry(c) == expiry]
    best = min(same_exp, key=lambda c: abs(_contract_strike(c) - current_price))
    occ = _contract_symbol(best)
    strike = _contract_strike(best)
    exp_str = expiry.isoformat() if isinstance(expiry, date) else str(best.get("expiration_date"))

    snap = {}
    try:
        snaps = client.get_option_snapshots(ticker.upper(), expiration_date=exp_str, type=cp)
        snap = (snaps or {}).get(occ) or {}
    except Exception:
        snap = {}
    bid, ask = _quote_from_snapshot(snap)
    if not (bid and ask and bid > 0 and ask > 0):
        bid, ask = _latest_quote(client, occ)
    if not (bid and ask and bid > 0 and ask > 0):
        return None, "no_two_sided_quote"
    mid = (bid + ask) / 2.0
    greeks = (snap or {}).get("greeks") or {}
    dte = max(0, (expiry - now_et.date()).days) if isinstance(expiry, date) else None
    return (
        {
            "ticker": ticker.upper(),
            "occ": occ,
            "mid": mid,
            "limit": round(ask, 2),
            "delta": _as_float(greeks.get("delta")),
            "expiry": exp_str,
            "strike": strike,
            "open_interest": int(_as_float(snap.get("openInterest")) or 0),
            "volume": int(_as_float(snap.get("dailyVolume")) or 0),
            "spread": ((ask - bid) / mid) if mid > 0 else None,
            "option_type": cp,
            "selection_method": "nearest_atm_non_0dte",
            "dte": dte,
        },
        "ok",
    )


def _side_for_ticker(ticker: str, *, side_mode: str, top: pd.DataFrame) -> str:
    if side_mode != "dealer_direction":
        return "call"
    row = top[top["symbol"].astype(str).str.upper().eq(ticker.upper())]
    direction = str(row.iloc[0].get("dealer_direction", "")) if not row.empty else ""
    return "put" if direction == "bearish" else "call"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run dealer-ranked ATM options pass (paper by default).")
    ap.add_argument("--refresh-chain", action="store_true", help="Capture chains and rebuild rankings before trading.")
    ap.add_argument("--ranking-path", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--target-notional", type=float, default=5000.0,
                    help="Dollar size per new option entry; contracts = round(target/(premium*100)).")
    ap.add_argument("--min-dte", type=int, default=1, help="Minimum calendar DTE; 1 avoids 0DTE.")
    ap.add_argument("--max-dte", type=int, default=21)
    ap.add_argument("--side-mode", choices=["call", "dealer_direction"], default="call")
    ap.add_argument("--workers", type=int, default=8, help="Parallel chain-capture workers when --refresh-chain is set.")
    ap.add_argument("--scan-limit", type=int, default=None)
    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    ap.add_argument("--snapshot-date", default=None)
    ap.add_argument("--ref-date", default=None)
    ap.add_argument("--take-profit", type=float, default=0.20)
    ap.add_argument("--scale-frac", type=float, default=0.5)
    ap.add_argument("--horizon-bars", type=int, default=25)
    ap.add_argument("--grace-bars", type=int, default=None)
    ap.add_argument("--stop-loss", type=float, default=0.50)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if args.refresh_chain:
        ranking_path = refresh_rankings(
            workers=max(1, int(args.workers)),
            limit=args.scan_limit,
            sleep_seconds=float(args.sleep_seconds),
            snapshot_date=args.snapshot_date,
            ref_date=args.ref_date,
        )
    else:
        ranking_path = Path(args.ranking_path) if args.ranking_path else None

    rankings = _load_rankings(ranking_path)
    profile = "LIVE" if args.live else "PAPER"
    client = AlpacaOptionsClient(env_file=f".env#{profile}")
    top, skipped_not_optionable = _select_optionable_targets(
        client, rankings, top_k=args.top_k, side_mode=args.side_mode,
        min_dte=args.min_dte, max_dte=args.max_dte,
    )
    targets = top["symbol"].astype(str).str.upper().tolist()
    bar = now_utc_iso()
    signal_audits = _signal_audits(top, bar=bar, side_mode=args.side_mode)
    append_jsonl(AUDIT_LOG, {"event": "signal_decision", "module": MODULE, "bar": bar,
                             "targets": targets, "signal_audits": signal_audits,
                             "skipped_not_optionable": skipped_not_optionable})
    pos_info = _build_pos_info(client)
    state = _load_state()
    managed = state.get("managed", {})

    spot_map = _snapshot_spot_map()

    def _route(client_: AlpacaOptionsClient, ticker: str, px: float, **_kwargs) -> tuple[str, dict | None, str]:
        opt_type = _side_for_ticker(ticker, side_mode=args.side_mode, top=top)
        order, reason = _select_atm_option(
            client_,
            ticker,
            px,
            option_type=opt_type,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
        )
        if order is None:
            return "skip", {"option_type": opt_type}, reason
        return "option", order, "ok"

    policy = ExecPolicy(
        take_profit=float(args.take_profit),
        scale_frac=float(args.scale_frac),
        horizon_bars=int(args.horizon_bars),
        grace_bars=args.grace_bars,
        stop_loss=float(args.stop_loss) if args.stop_loss else None,
        # Pinned explicitly: this module wasn't part of the 2026-07-18 cross-module
        # exit-policy search (Momentum/HTF/Meta only), so it keeps its own prior
        # behavior rather than silently inheriting ExecPolicy's new default.
        trail_stop=0.35,
        target_notional=float(args.target_notional),
    )
    plan = build_mixed_plan(
        client,
        targets=targets,
        managed=managed,
        pos_info=pos_info,
        bar=bar,
        signal_audits=signal_audits,
        policy=policy,
        route_fn=_route,
        ref_price_fn=_ref_price_fn(spot_map),
        verbose=True,
    )

    if args.submit:
        pending = submit_pending_open_entries(
            client,
            MODULE,
            targets,
            equity_tif_fn=equity_order_tif,
            pos_lookup=pos_info,
        )
        if pending.get("submitted"):
            plan.new_managed.update(pending["submitted"])
        active_plan = defer_entries_if_market_closed(
            MODULE,
            bar,
            plan.plan,
            plan.new_managed,
            plan.limits,
        )
        execute_plan(
            client,
            plan=active_plan,
            limits=plan.limits,
            submit=True,
            equity_tif_fn=equity_order_tif,
            new_managed=plan.new_managed,
            exit_context=plan.exit_context,
            module=MODULE,
            pos_lookup=pos_info,
            bar=bar,
        )
        state["managed"] = plan.new_managed
        state.setdefault("history", []).append(
            {"ts": bar, "targets": targets, "orders": len(active_plan), "profile": profile}
        )
        _save_state(state)
    else:
        print("\n(dry-run: no orders submitted, state unchanged. Add --submit to execute.)")

    append_jsonl(
        AUDIT_LOG,
        order_plan_audit_record(
            module=MODULE,
            bar=bar,
            mode="options",
            submit=bool(args.submit),
            targets=targets,
            plan=plan.plan,
            signal_audits=signal_audits,
            order_audits=plan.order_audits,
            contract_selection=plan.contract_selection,
            dropped=plan.dropped,
        ),
    )
    print(f"dealer ranker done: targets={targets} orders={len(plan.plan)} submit={args.submit} account={profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
