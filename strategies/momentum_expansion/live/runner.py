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
from strategies.momentum_expansion.data.load_bars import load_1h, load_4h
from strategies.momentum_expansion.data.universe import (
    get_candidate_pool,
    list_snapshots,
    load_snapshot_for,
    write_weekly_snapshot,
)
from strategies.momentum_expansion.features.feature_matrix_4h import (
    FEATURE_COLUMNS_4H,
    build_ticker_features_4h,
)
from strategies.momentum_expansion.inference.entry_rules import evaluate_entry
from strategies.momentum_expansion.inference.ranker import ExpansionRanker, top_n_at
from strategies.momentum_expansion.policy.momentum_option_policy import (
    MomentumOptionConfig,
    MomentumOptionPolicy,
)

logger = logging.getLogger(__name__)


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
    ):
        self.ranker = ranker or ExpansionRanker()
        cfg = MomentumOptionConfig(submit_orders=bool(auto_trade if auto_trade is not None else LIVE_CONFIG["auto_trade"]))
        self.policy = policy or MomentumOptionPolicy(cfg=cfg)
        self.auto_trade = bool(cfg.submit_orders)

    # ---- one-shot scoring + alerting at a 4H/1H bar ----
    def evaluate_now(self, *, bar_ts: pd.Timestamp | None = None) -> list[dict]:
        bar_ts = bar_ts or pd.Timestamp.now(tz="UTC")
        snapshot = load_snapshot_for(bar_ts)
        if snapshot.empty:
            logger.warning("No universe snapshot available — refresh required")
            return []
        tickers = snapshot["ticker"].astype(str).tolist()

        # Build live features for each ticker on the latest 4H bars
        ctx_4h: dict[str, pd.DataFrame] = {}
        for sym in list(CONTEXT_TICKERS) + list(SECTOR_ETFS):
            try:
                ctx_4h[sym] = load_4h(sym)
            except FileNotFoundError:
                continue

        feature_rows: list[pd.DataFrame] = []
        for t in tickers:
            try:
                df_4h = load_4h(t)
            except FileNotFoundError:
                continue
            try:
                df_1d = None  # daily features can be NaN in live mode if not cached
            except Exception:
                df_1d = None
            feats = build_ticker_features_4h(
                ticker=t, df_4h=df_4h, df_1d=df_1d, ctx_4h=ctx_4h
            )
            if feats is None:
                continue
            feats = feats.tail(1).copy()
            feats["ticker"] = t
            feature_rows.append(feats)

        if not feature_rows:
            return []
        live_features = pd.concat(feature_rows, axis=0)
        live_features = live_features.reset_index().rename(columns={"index": "timestamp"})
        live_features = live_features.set_index(["timestamp", "ticker"])

        # Rank
        ts_used = live_features.index.get_level_values(0).max()
        ranked = top_n_at(bar_ts=ts_used, features=live_features, ranker=self.ranker)
        if ranked.empty:
            return []

        # Trigger check
        emitted: list[dict] = []
        for _, row in ranked.iterrows():
            ticker = row["ticker"]
            try:
                df_1h = load_1h(ticker)
            except FileNotFoundError:
                continue
            triggers = evaluate_entry(df_1h=df_1h.tail(60))
            if not triggers:
                continue
            trig = triggers[0]
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
            }
            _emit_alert(payload)
            emitted.append(payload)

            if self.auto_trade:
                # Caller-provided ATR from latest 4H feature row
                atr_pct = float(live_features.loc[(ts_used, ticker), "atr_pct_14"])
                close_4h = float(load_4h(ticker)["close"].iloc[-1])
                atr_abs = atr_pct * close_4h
                self.policy.consider_entry(
                    ticker=ticker, direction=+1,
                    underlying_price=close_4h, atr=atr_abs,
                    expansion_score=float(row["expansion_score"]),
                    suggested_stop_atr=trig.suggested_stop_atr,
                    as_of=ts_used,
                )
        return emitted

    # ---- exit management ----
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
            ctx_4h: dict[str, pd.DataFrame] = {}
            for sym in list(CONTEXT_TICKERS) + list(SECTOR_ETFS):
                try:
                    ctx_4h[sym] = load_4h(sym)
                except FileNotFoundError:
                    continue
            feats = build_ticker_features_4h(ticker=ticker, df_4h=df_4h, df_1d=None, ctx_4h=ctx_4h)
            if feats is None:
                continue
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


def main() -> None:
    """One scheduled Momentum pass: score (ExpansionRanker, same as Meta's mom_score)
    -> entry triggers -> this module's own MomentumOptionPolicy. Dry-run/paper by
    default; --submit places orders, --live targets the real-money account."""
    import argparse

    from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
    from strategies.momentum_expansion.policy.momentum_option_policy import (
        MomentumOptionConfig,
        MomentumOptionPolicy,
    )

    ap = argparse.ArgumentParser(description="Standalone Momentum Expansion harness — one pass.")
    ap.add_argument("--submit", action="store_true", help="Place orders (default: dry-run / alerts only).")
    ap.add_argument("--live", action="store_true", help="Target the LIVE account (default: paper).")
    args = ap.parse_args()

    profile = "LIVE" if args.live else "PAPER"
    env_file = f".env#{profile}"
    mode = "SUBMIT" if args.submit else "DRY-RUN"
    print(f"=== Momentum Expansion harness | account={profile} | mode={mode} ===")

    policy = MomentumOptionPolicy(
        cfg=MomentumOptionConfig(submit_orders=bool(args.submit)),
        client=AlpacaOptionsClient(env_file=env_file),
    )
    runner = MomentumLiveRunner(policy=policy, auto_trade=bool(args.submit))
    entries = runner.evaluate_now()
    exits = runner.manage_open_positions()
    print(f"entries/triggers: {len(entries)}  exits: {len(exits)}")


if __name__ == "__main__":
    main()
