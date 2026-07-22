"""
Live runner for momentum_expansion.

What it does at each tick:
  1. Sunday refresh: rebuild weekly universe snapshot.
  2. Every 4H bar close (RTH only): re-score active universe.
  3. Every 1H bar close: for each name in current top-N, evaluate
     entry rules. If a trigger fires, hand to MomentumOptionPolicy.
  4. Manage open positions every 4H bar (trail / exit).

Auto-trade is OFF by default — emits alerts to a JSONL file. Flip
LIVE_CONFIG["auto_trade"] = True to wire orders through the policy.
"""
from __future__ import annotations

import json
import logging
import os
import time as time_mod
from datetime import datetime
from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.config.momentum_config import (
    CONTEXT_TICKERS,
    LIVE_CONFIG,
    SECTOR_ETFS,
)
from strategies.momentum_expansion.data.bars import (
    fetch_context_bars,
    fetch_universe_bars,
)
from core.live_4h_exec import (
    ExecPolicy,
    build_mixed_plan,
    execute_plan,
    now_utc_iso,
    order_plan_audit_record,
    submit_pending_open_entries,
)
from core.live_signal_audit import append_jsonl, build_signal_audit
from signals.meta_context.meta_ranker.options_exec import equity_order_tif, route_option_or_shares
from strategies.momentum_expansion.data.load_bars import load_1d, load_1h, load_4h
from strategies.momentum_expansion.data.universe import (
    get_candidate_pool,
    list_snapshots,
    load_snapshot_for,
    write_weekly_snapshot,
)
from strategies.momentum_expansion.features.feature_matrix_4h import build_ticker_features_4h
from strategies.momentum_expansion.features.live_feature_panel_4h import (
    assert_manifest_coverage,
    build_live_feature_panel_4h,
)
from strategies.momentum_expansion.inference.entry_rules import evaluate_entry
from strategies.momentum_expansion.inference.ranker import ExpansionRanker, top_n_at
from strategies.momentum_expansion.policy.momentum_option_policy import (
    MomentumOptionConfig,
    MomentumOptionPolicy,
)

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SIGNAL_AUDIT_LOG = REPO_ROOT / "Data/inference/momentum_expansion/live_signal_audit.jsonl"
STATE_PATH = Path(__file__).resolve().parent / "momentum_live_state.json"
SIGNAL_MAX_STALENESS_DAYS = float(os.getenv("MOMENTUM_MAX_SIGNAL_STALENESS_DAYS", "0.5"))


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"managed": {}, "history": []}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _ref_price_4h(ticker: str) -> float | None:
    """Underlying reference price = latest 4H close (shared-engine ref_price_fn)."""
    try:
        df = load_4h(ticker)
        return float(df["close"].iloc[-1]) if not df.empty else None
    except Exception:
        return None


def _build_pos_info(client) -> dict:
    out: dict[str, dict] = {}
    for p in (client.get_positions() or []):
        try:
            out[p["symbol"]] = {
                "qty": int(float(p["qty"])),
                "avg_entry": float(p.get("avg_entry_price", 0) or 0),
                "current": float(p.get("current_price", 0) or 0),
            }
        except Exception:
            continue
    return out

# Max staleness (days) tolerated for a 4H context series. VIXY's 4H cache is
# dead since 2022; feeding it makes regime_vix_z NaN, and the ranker's
# dropna(subset=feature_columns) then discards every row -> module goes dark.
# Dropping the stale series lets build_ticker_features_4h fall back to a neutral
# regime_vix_z (0.0). TODO: refresh the VIXY 4H cache for a live VIX regime.
_CTX_MAX_STALE_DAYS = 5


def _load_context_4h() -> dict[str, pd.DataFrame]:
    """Load 4H context/sector series, skipping any whose latest bar is stale."""
    now = pd.Timestamp.now(tz="UTC")
    ctx: dict[str, pd.DataFrame] = {}
    for sym in list(CONTEXT_TICKERS) + list(SECTOR_ETFS):
        try:
            df = load_4h(sym)
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        if (now - df.index[-1]).days > _CTX_MAX_STALE_DAYS:
            logger.warning("Momentum: dropping stale context %s (last=%s)", sym, df.index[-1])
            continue
        ctx[sym] = df
    return ctx


def _load_daily(ticker: str) -> pd.DataFrame | None:
    """Daily bars for the daily/weekly feature block; None if uncached.

    Required for the 15 daily_*/weekly_* model features; without it the ranker
    drops every row and the module produces no ranking.
    """
    try:
        return load_1d(ticker)
    except (FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def _emit_alert(payload: dict, *, log_path: Path | None = None) -> None:
    log_path = Path(log_path or LIVE_CONFIG["alert_log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "ts": datetime.utcnow().isoformat() + "Z"}
    with open(log_path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    logger.info("alert: %s", payload)


# ---------------------------------------------------------------------------
# Universe refresh
# ---------------------------------------------------------------------------

def refresh_weekly_universe(*, as_of: pd.Timestamp | None = None) -> Path:
    as_of = as_of or pd.Timestamp.now(tz="UTC").normalize()
    snapshot_path = write_weekly_snapshot(as_of=as_of, candidates=get_candidate_pool())
    _refresh_dynamic_theme_taxonomy(as_of=as_of)
    return snapshot_path


def _refresh_dynamic_theme_taxonomy(*, as_of: pd.Timestamp | None = None) -> None:
    """Run the dynamic theme weekly pipeline (recluster + Claude labeling + features)."""
    try:
        from themes.dynamic_theme.pipeline import weekly_run
        weekly_run(as_of=as_of)
    except Exception as exc:
        logger.error("Dynamic theme weekly run failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Live cycle
# ---------------------------------------------------------------------------

class MomentumLiveRunner:
    def __init__(
        self,
        *,
        ranker: ExpansionRanker | None = None,
        policy: MomentumOptionPolicy | None = None,
        auto_trade: bool | None = None,
        exec_policy: ExecPolicy | None = None,
    ):
        self.ranker = ranker or ExpansionRanker()
        cfg = MomentumOptionConfig(submit_orders=bool(auto_trade if auto_trade is not None else LIVE_CONFIG["auto_trade"]))
        self.policy = policy or MomentumOptionPolicy(cfg=cfg)
        self.auto_trade = bool(cfg.submit_orders)
        # Shared 4H engine hold/scale-out sizing (configurable via CLI in main(),
        # same as Meta/HTF); defaults match the engine's own defaults.
        self.exec_policy = exec_policy or ExecPolicy()
        # Populated by evaluate_now for run_pass (shared 4H engine).
        self._last_gate: dict | None = None

    # ---- one-shot scoring + alerting at a 4H/1H bar ----
    def evaluate_now(self, *, bar_ts: pd.Timestamp | None = None) -> list[dict]:
        bar_ts = bar_ts or pd.Timestamp.now(tz="UTC")
        self._last_gate = None
        snapshot = load_snapshot_for(bar_ts)
        if snapshot.empty:
            logger.warning("No universe snapshot available — refresh required")
            return []
        tickers = snapshot["ticker"].astype(str).tolist()

        # Build live features for the candidate universe on the latest 4H bars.
        # build_live_feature_panel_4h bundles both batch-only post-processing
        # steps (xsec_* cross-sectional ranks + earnings-proximity columns)
        # unconditionally -- previously skipped live entirely at this exact
        # call site, silently starving the booster of manifest columns (see
        # 2026-07-19 audit in LIVING_SUMMARY.md). assert_manifest_coverage is
        # the structural check that would have caught it immediately.
        ctx_4h = _load_context_4h()
        result = build_live_feature_panel_4h(
            tickers=tickers, load_4h_bars=load_4h, load_1d_bars=_load_daily,
            ctx_4h=ctx_4h, tail=1,
        )
        if result.panel.empty:
            logger.warning("Momentum: no feature rows built (bar cache empty?)")
            return []
        logger.info("Momentum: panel built %d/%d tickers in %.1fs",
                    result.tickers_built, result.tickers_requested, result.build_seconds)
        live_features = result.panel
        assert_manifest_coverage(scorer=self.ranker, panel=live_features, label="momentum/evaluate_now")

        # Rank on the MODAL latest bar (the timestamp shared by the bulk of the
        # universe), not the global max: a single ticker with a stray newer/partial
        # 4H bar would otherwise hijack `.max()` and leave top_n_at scoring 1 row.
        ts_counts = live_features.index.get_level_values(0).value_counts()
        ts_used = ts_counts.idxmax()
        if ts_used != live_features.index.get_level_values(0).max():
            logger.info("Momentum: ranking on modal bar %s (%d names); ignoring stray newer bars",
                        ts_used, int(ts_counts.max()))
        ts_utc = pd.Timestamp(ts_used)
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.tz_localize("UTC")
        else:
            ts_utc = ts_utc.tz_convert("UTC")
        staleness_days = (pd.Timestamp.now(tz="UTC") - ts_utc).total_seconds() / 86400.0
        if staleness_days > SIGNAL_MAX_STALENESS_DAYS:
            msg = (
                f"Momentum: ABORT stale signal bar {ts_utc} "
                f"({staleness_days:.1f}d > {SIGNAL_MAX_STALENESS_DAYS:.1f}d)"
            )
            logger.warning(msg)
            print(msg)
            append_jsonl(DEFAULT_SIGNAL_AUDIT_LOG, {
                "event": "signal_decision",
                "module": "momentum_expansion",
                "bar": ts_utc,
                "ticker": None,
                "entry_triggered": False,
                "skip_reason": "stale_signal_bar",
                "staleness_days": staleness_days,
                "max_staleness_days": SIGNAL_MAX_STALENESS_DAYS,
            })
            return []
        ranked = top_n_at(bar_ts=ts_used, features=live_features, ranker=self.ranker)
        if ranked.empty:
            print(f"momentum: no candidates after ranking (scored {len(live_features)} names)")
            append_jsonl(DEFAULT_SIGNAL_AUDIT_LOG, {
                "event": "signal_decision", "module": "momentum_expansion", "bar": ts_used,
                "ticker": None, "entry_triggered": False, "skip_reason": "no_candidates_after_ranking",
                "scored": int(len(live_features)),
            })
            return []

        top_syms = list(ranked.head(10)["ticker"]) if "ticker" in ranked.columns else []
        print(f"momentum top-{len(top_syms)}: {top_syms}")
        logger.info("momentum top-%d: %s", len(top_syms), top_syms)

        # Trigger check. Momentum's distinctive entry: a name must be top-ranked AND
        # print a 1H entry trigger. We record targets/entry_ok/signal_audits so
        # run_pass can drive the shared option/share engine (entry_ok gates entries;
        # held positions are still managed regardless).
        targets = [str(t) for t in ranked["ticker"]]
        entry_ok: dict[str, bool] = {}
        sa_map: dict[str, dict] = {}
        emitted: list[dict] = []
        for _, row in ranked.iterrows():
            ticker = row["ticker"]
            rank_count = max(1, len(ranked))
            signal_audit = build_signal_audit(
                module="momentum_expansion",
                ticker=ticker,
                score=row["expansion_score"],
                side="long",
                rank=row["rank"],
                rank_pct=1.0 - ((int(row["rank"]) - 1) / rank_count),
                signal_ts=ts_used,
                extra={"entry_triggered": False},
            )
            sa_map[ticker] = signal_audit
            try:
                df_1h = load_1h(ticker)
            except FileNotFoundError:
                entry_ok[ticker] = False
                append_jsonl(DEFAULT_SIGNAL_AUDIT_LOG, {
                    "event": "signal_decision", "module": "momentum_expansion", "bar": ts_used,
                    "ticker": ticker, "entry_triggered": False, "skip_reason": "missing_1h_bars",
                    "signal_audit": signal_audit,
                })
                continue
            triggers = evaluate_entry(df_1h=df_1h.tail(60))
            if not triggers:
                entry_ok[ticker] = False
                append_jsonl(DEFAULT_SIGNAL_AUDIT_LOG, {
                    "event": "signal_decision", "module": "momentum_expansion", "bar": ts_used,
                    "ticker": ticker, "entry_triggered": False, "skip_reason": "no_entry_trigger",
                    "signal_audit": signal_audit,
                })
                continue
            trig = triggers[0]
            signal_audit = build_signal_audit(
                module="momentum_expansion",
                ticker=ticker,
                score=row["expansion_score"],
                side="long",
                rank=row["rank"],
                rank_pct=1.0 - ((int(row["rank"]) - 1) / rank_count),
                signal_ts=ts_used,
                extra={
                    "trigger_rule": trig.rule, "bar_close": trig.close,
                    "stop_atr": trig.suggested_stop_atr, "note": trig.note,
                    "entry_triggered": True,
                },
            )
            sa_map[ticker] = signal_audit
            entry_ok[ticker] = True
            append_jsonl(DEFAULT_SIGNAL_AUDIT_LOG, {
                "event": "signal_decision", "module": "momentum_expansion", "bar": ts_used,
                "ticker": ticker, "entry_triggered": True, "signal_audit": signal_audit,
            })
            payload = {
                "kind":            "entry_trigger",
                "ticker":          ticker,
                "rank":            int(row["rank"]),
                "expansion_score": float(row["expansion_score"]),
                "trigger_rule":    trig.rule,
                "bar_close":       trig.close,
                "stop_atr":        trig.suggested_stop_atr,
                "note":            trig.note,
                "auto_trade":      self.auto_trade,
                "signal_audit":    signal_audit,
            }
            _emit_alert(payload)
            emitted.append(payload)

        # Execution + exits now run through the shared 4H engine in run_pass();
        # this method stays display/alert-only (used by the dashboard).
        self._last_gate = {"targets": targets, "entry_ok": entry_ok,
                           "signal_audits": sa_map, "bar": ts_used}
        return emitted

    # ---- exit management ----
    # KNOWN ISSUE (flagged, not fixed -- out of scope for the 2026-07-19 shared
    # feature-panel refactor; see LIVING_SUMMARY.md): this loop scores ONE
    # ticker at a time via a direct build_ticker_features_4h call, with no
    # _add_cross_sectional_features/_add_earnings_features_4h step, so
    # `score` below is missing the same manifest columns the two live scoring
    # paths were before this session's fix. It is also structurally unable to
    # get xsec_* right even if post-processed: those ranks need the whole
    # scored universe at one timestamp, and this method only ever builds one
    # ticker. Do NOT route this through build_live_feature_panel_4h as-is --
    # a single-ticker panel would produce well-formed-looking but meaningless
    # xsec_* values (every rank = 1.0). Currently DEAD CODE on the scheduled
    # path: run_pass() never calls this; the only caller in the repo is the
    # manual "Run now" button in UI/momentum_dashboard.py. Needs its own
    # design (e.g. score off the last full panel/matrix row for this ticker
    # instead of rebuilding) before it's reachable again.
    def manage_open_positions(self, *, bar_ts: pd.Timestamp | None = None) -> list[dict]:
        bar_ts = bar_ts or pd.Timestamp.now(tz="UTC")
        out: list[dict] = []
        for ticker in list(self.policy.positions.keys()):
            try:
                df_4h = load_4h(ticker)
            except FileNotFoundError:
                continue
            if df_4h.empty:
                continue
            last_close = float(df_4h["close"].iloc[-1])
            ema_slow = float(df_4h["close"].ewm(span=20, adjust=False).mean().iloc[-1])
            atr_pct = float(((df_4h["high"] - df_4h["low"]).rolling(14).mean() / df_4h["close"]).iloc[-1])
            atr_abs = atr_pct * last_close
            # Score the latest bar
            ctx_4h = _load_context_4h()
            feats = build_ticker_features_4h(
                ticker=ticker, df_4h=df_4h, df_1d=_load_daily(ticker), ctx_4h=ctx_4h
            )
            if feats is None:
                continue
            logger.warning(
                "manage_open_positions: single-ticker score for %s lacks xsec_*/earnings "
                "manifest columns (known issue, unfixed -- see comment above method)", ticker,
            )
            score_row = feats.tail(1)
            score = float(self.ranker.score(score_row).iloc[0]) if not score_row.empty else 0.0
            should_exit, reason = self.policy.update_and_check_exit(
                ticker=ticker,
                underlying_price=last_close,
                atr=atr_abs,
                ema_slow=ema_slow,
                expansion_score=score,
            )
            if should_exit:
                self.policy.record_campaign_exit(
                    ticker=ticker,
                    exit_underlying=last_close,
                    as_of=bar_ts,
                )
                self.policy.close(ticker, reason=reason)
                payload = {
                    "kind":     "exit",
                    "ticker":   ticker,
                    "reason":   reason,
                    "underlying": last_close,
                }
                _emit_alert(payload)
                out.append(payload)
        return out

    # ---- shared 4H option/share execution + exits ----
    def run_pass(self, client, *, submit: bool, bar_ts: pd.Timestamp | None = None,
                 flush_pending_open: bool = False) -> dict:
        """One scheduled pass through the SHARED 4H engine (same as Meta/HTF):
        rank top-10 + 1H trigger gate -> route each name to options or shares ->
        manage held positions with the hold-based TP/horizon/grace machine ->
        execute + persist state + audit. Replaces MomentumOptionPolicy execution."""
        bar_ts = bar_ts or pd.Timestamp.now(tz="UTC")
        self.evaluate_now(bar_ts=bar_ts)  # ranks, gates, audits, alerts -> self._last_gate
        gate = self._last_gate
        if not gate:
            # No ranking this pass (no snapshot / no feature rows / nothing survived
            # ranking). There are no new candidates to route, but held positions
            # still need their TP/horizon/grace exits evaluated every pass — do not
            # skip build_mixed_plan just because there is nothing to enter.
            print("momentum: no ranking this pass — still managing held positions")
        targets = gate["targets"] if gate else []
        bar = gate["bar"] if gate else bar_ts
        signal_audits = gate["signal_audits"] if gate else {}
        entry_ok = gate["entry_ok"] if gate else {}

        state = _load_state()
        managed = state.get("managed", {})
        pos_info = _build_pos_info(client)

        # Pre-open flush: submit after-close-queued entries still in TODAY's top-K
        # (re-rank against the freshly-ranked `targets`), then exit — no position mgmt.
        if flush_pending_open:
            if submit:
                r = submit_pending_open_entries(client, "momentum_expansion", targets,
                                                equity_tif_fn=equity_order_tif, pos_lookup=pos_info)
                managed.update(r["submitted"])
                state["managed"] = managed
                _save_state(state)
                print(f"pending-open flush: submitted {r['count']} / skipped {len(r['skipped'])}")
                return {"orders": r["count"]}
            print("pending-open flush (dry-run): add --submit to place queued entries")
            return {"orders": 0}

        res = build_mixed_plan(
            client, targets=targets, managed=managed, pos_info=pos_info,
            bar=bar, signal_audits=signal_audits, policy=self.exec_policy,
            route_fn=route_option_or_shares, ref_price_fn=_ref_price_4h,
            entry_ok=entry_ok, gate_reason="no_entry_trigger",
        )
        execute_plan(client, plan=res.plan, limits=res.limits, submit=submit,
                     equity_tif_fn=equity_order_tif,
                     new_managed=res.new_managed, exit_context=res.exit_context,
                     module="momentum_expansion", pos_lookup=pos_info, bar=bar)
        audit = order_plan_audit_record(
            module="momentum_expansion", bar=bar, mode="options", submit=submit,
            targets=targets, plan=res.plan, signal_audits=signal_audits,
            order_audits=res.order_audits, contract_selection=res.contract_selection,
            dropped=res.dropped,
        )
        if submit:
            # Persist managed state every submit pass (not only when an order was
            # placed) — hold/grace counters (runs_held/bars_out) must advance on
            # quiet passes too, or horizon/grace exits stall indefinitely.
            state["managed"] = res.new_managed
            if res.plan:
                state.setdefault("history", []).append(
                    {"ts": now_utc_iso(), "bar": str(bar), "orders": len(res.plan)})
            _save_state(state)
            print(f"state updated -> {STATE_PATH}")
        append_jsonl(DEFAULT_SIGNAL_AUDIT_LOG, audit)
        return {"orders": len(res.plan)}


def main() -> None:
    """One scheduled Momentum pass: score (ExpansionRanker, same as Meta's mom_score)
    -> 1H entry trigger -> the SHARED 4H option/share engine (same as Meta & HTF).
    Dry-run/paper by default; --submit places orders, --live targets real money."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

    ap = argparse.ArgumentParser(description="Standalone Momentum Expansion harness — one pass.")
    ap.add_argument("--submit", action="store_true", help="Place orders (default: dry-run / alerts only).")
    ap.add_argument("--flush-pending-open", action="store_true", help="Pre-open: submit after-close queued entries still in the top-K (re-rank), then exit.")
    ap.add_argument("--live", action="store_true", help="Target the LIVE account (default: paper).")
    # --- exit policy + sizing (mirrors Meta/HTF runners; same shared engine) ---
    ap.add_argument("--take-profit", type=float, default=0.30, help="Scale out scale_frac at this gain, then ride the rest.")
    ap.add_argument("--scale-frac", type=float, default=0.16, help="Fraction to sell at take-profit.")
    ap.add_argument("--horizon-bars", type=int, default=53, help="Full exit after this many managed bars (~21d).")
    ap.add_argument("--grace-bars", type=int, default=None, help="Rank drop-out backstop: exit after N bars out of top-K. Default None = ride to horizon (backtest-preferred).")
    ap.add_argument("--stop-loss", type=float, default=0.39, help="Hard stop: full exit if gain <= -this (premium for options). 0 disables.")
    ap.add_argument("--trail-stop", type=float, default=None, help="Trailing stop: full exit if value gives back this fraction from its peak. Default None = disabled (2026-07-18 cross-module search: no-trail beat trail on mean return per trade).")
    ap.add_argument("--contracts", type=int, default=10, help="Options: contracts per new position.")
    ap.add_argument("--shares", type=int, default=100, help="Equity: shares per new position.")
    ap.add_argument("--roll-trading-days", type=int, default=5,
                    help="Options: roll to next monthly when nearest is within this many trading days.")
    args = ap.parse_args()

    profile = "LIVE" if args.live else "PAPER"
    env_file = f".env#{profile}"
    mode = "SUBMIT" if args.submit else "DRY-RUN"
    print(f"=== Momentum Expansion harness | account={profile} | mode={mode} ===")

    client = AlpacaOptionsClient(env_file=env_file)
    exec_policy = ExecPolicy(take_profit=args.take_profit, scale_frac=args.scale_frac,
                             horizon_bars=args.horizon_bars, grace_bars=args.grace_bars,
                             stop_loss=args.stop_loss, trail_stop=args.trail_stop,
                             contracts=args.contracts, shares=args.shares,
                             roll_trading_days=args.roll_trading_days)
    runner = MomentumLiveRunner(auto_trade=bool(args.submit), exec_policy=exec_policy)
    result = runner.run_pass(client, submit=bool(args.submit),
                             flush_pending_open=bool(getattr(args, "flush_pending_open", False)))
    print(f"orders: {result['orders']}")


if __name__ == "__main__":
    main()
