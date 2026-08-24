"""
Meta Ranker live equity runner — long-only, equal-weight top-K combo portfolio.

Scores the meta-ranker matrix with the confluence (combo) signal, takes the top-K
names on the latest full 4H bar, and reconciles a paper (default) Alpaca account to
an equal-weight long-only target portfolio.

SAFETY (read before running live):
  * DRY-RUN by default. Orders are only submitted with --submit.
  * PAPER account only. --live is rejected: the governed path has no live route.
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
import json
import logging
import sys
import math
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.nervous_system.contracts.enums import InstrumentFamily
from core.live_signal_audit import (
    append_jsonl,
    build_equity_order_audit,
    build_option_order_audit,
    build_signal_audit,
)
from core.nervous_system.contracts.enums import PolicyMode
from signals.meta_context.meta_ranker.gateway_execution import (
    GovernedPathUnavailable,
    build_router,
)
from signals.meta_context.meta_ranker.nervous_system_adapter import MetaIntentConfig
from signals.meta_context.meta_ranker.score import score_frame
from signals.meta_context.meta_ranker.ticker_state_publication import (
    publish_ticker_states,
)
from core.live_4h_exec import (
    ExecPolicy,
    build_mixed_plan,
    defer_entries_if_market_closed,
    defer_exits_if_opg_unavailable,
    drop_failed_entry,
    exit_action as _shared_exit_action,
    mark_entry_unconfirmed,
    record_exit_realized_pnl,
    shares_for_notional,
    submit_pending_exit_orders,
    submit_pending_open_entries,
)
from core.live_readiness import filter_entry_orders_for_readiness
from signals.meta_context.meta_ranker.options_exec import (
    equity_order_tif,
    route_option_or_shares,
    select_option,
)

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_MATRIX = HERE / "meta_ranker_matrix.parquet"
BARS_4H = REPO / "Data/shared/bars/4h"
STATE_PATH = HERE / "live_state.json"
BLACKLIST_PATH = HERE / "blacklist.txt"  # optional hard exclusions, one ticker per line (# comments ok)
MIN_FULL_BAR = 50  # ignore degenerate edge bars with fewer rows
AUDIT_MODULE = "meta_ranker"
DEFAULT_AUDIT_LOG = REPO / "Data/inference/meta_ranker/live_signal_audit.jsonl"


@dataclass(frozen=True)
class MetaRankingConfig:
    """Pure Meta selection policy and the injectable model boundary."""

    top_k: int = 10
    liquidity_floor: float = 0.6
    combo_floor: float = 0.90
    blacklist: frozenset[str] = field(default_factory=frozenset)
    booster_loader: Callable[[str], tuple[Any, list[str]]] | None = None

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be positive")
        for field_name in ("liquidity_floor", "combo_floor"):
            value = getattr(self, field_name)
            if isinstance(value, (bool, str, bytes)) or not isinstance(value, (Real, Decimal)):
                raise TypeError(f"{field_name} must be a finite numeric value")
            normalized = float(value)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{field_name} must be finite and within [0, 1]")
            object.__setattr__(self, field_name, normalized)
        object.__setattr__(
            self,
            "blacklist",
            _canonical_ticker_set(self.blacklist, field_name="blacklist"),
        )


@dataclass(frozen=True)
class MetaRankingResult:
    """Immutable operational result for the runner; ranking remains a DataFrame API."""

    ranked: pd.DataFrame
    scored_count: int
    eligible_count: int
    decision_bar: pd.Timestamp
    # Every scored name on this bar, not just the eligible top-K. A held name
    # that dropped out of the top-K still has a current score here, so its exit
    # can be explained instead of recorded blind.
    scored: pd.DataFrame = field(default_factory=pd.DataFrame)


def _canonical_ticker(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a ticker string")
    ticker = value.strip().upper()
    if not ticker:
        raise ValueError(f"{field_name} must be non-empty")
    return ticker


def _canonical_ticker_set(value: object, *, field_name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of ticker strings")
    try:
        tickers = [_canonical_ticker(item, field_name=field_name) for item in value]  # type: ignore[union-attr]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of ticker strings") from exc
    return frozenset(tickers)


def _normalize_timestamp_series(values: pd.Series, *, field_name: str) -> pd.Series:
    normalized: list[pd.Timestamp] = []
    for value in values.tolist():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} contains an invalid timestamp") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"{field_name} must contain only timezone-aware timestamps")
        normalized.append(timestamp.tz_convert("UTC"))
    return pd.Series(normalized, index=values.index, name=values.name)


def _decision_timestamp(value: datetime | pd.Timestamp, *, field_name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def rank_meta_candidates_with_diagnostics(
    feature_matrix: pd.DataFrame,
    *,
    bar: pd.Timestamp,
    config: MetaRankingConfig,
) -> MetaRankingResult:
    """Score and rank exactly one supplied Meta decision bar.

    The model boundary remains ``score_frame``; all selection behavior below
    mirrors the existing paper runner.  No latest-row inference or file/clock
    access occurs here.
    """

    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError("feature_matrix must be a pandas DataFrame")
    if not isinstance(config, MetaRankingConfig):
        raise TypeError("config must be a MetaRankingConfig")
    if "timestamp" not in feature_matrix.columns:
        raise KeyError("feature_matrix requires a timestamp column")

    decision_bar = _decision_timestamp(bar, field_name="bar")
    timestamps = _normalize_timestamp_series(feature_matrix["timestamp"], field_name="feature_matrix timestamps")
    selected_mask = timestamps == decision_bar
    if not bool(selected_mask.any()):
        raise ValueError(f"feature_matrix has no rows for exact decision bar {decision_bar.isoformat()}")

    selected = feature_matrix.loc[selected_mask].copy()
    selected["timestamp"] = timestamps.loc[selected_mask]
    selected["ticker"] = selected["ticker"].map(
        lambda value: _canonical_ticker(value, field_name="feature_matrix ticker")
    )
    if selected["ticker"].duplicated().any():
        raise ValueError("duplicate canonical ticker in selected ranking bar")
    scored = score_frame(selected, booster_loader=config.booster_loader)

    n0 = len(scored)
    finite_combo = pd.to_numeric(scored["s_combo"], errors="coerce").map(math.isfinite)
    eligible = scored.loc[finite_combo & (scored["s_combo"] >= config.combo_floor)]
    if "dollar_vol_pctile_252" in eligible.columns:
        eligible = eligible[eligible["dollar_vol_pctile_252"].fillna(0) >= config.liquidity_floor]
    if config.blacklist:
        eligible = eligible[~eligible["ticker"].isin(config.blacklist)]
    ranked = eligible.sort_values("s_combo", ascending=False, kind="stable").head(config.top_k)
    return MetaRankingResult(
        ranked=ranked,
        scored_count=n0,
        eligible_count=len(eligible),
        decision_bar=decision_bar,
        scored=scored,
    )


def rank_meta_candidates(
    feature_matrix: pd.DataFrame,
    *,
    bar: pd.Timestamp,
    config: MetaRankingConfig,
) -> pd.DataFrame:
    """Public DataFrame ranking interface retained for existing callers."""

    return rank_meta_candidates_with_diagnostics(feature_matrix, bar=bar, config=config).ranked


# --- governed-intent lineage -------------------------------------------------
# The decision-relevant runner settings. A change to any of them is a change to
# the policy that produced the decision, so it must change the recorded
# config_version and therefore the intent identity.
RUNNER_CONFIG_FIELDS = (
    "mode", "top_k", "liquidity_floor", "combo_floor", "quality_floor",
    "target_notional", "take_profit", "scale_frac", "horizon_bars",
    "grace_bars", "stop_loss", "trail_stop", "roll_trading_days",
)


def _short_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


def runner_config_version(args) -> str:
    """Content-derived version of the settings that shaped this decision."""

    return "meta-runner@" + _short_hash(
        {name: getattr(args, name, None) for name in RUNNER_CONFIG_FIELDS}
    )


def model_version() -> str:
    """Identify the deployed booster set by its on-disk content.

    A constant here would keep reporting the same version after a retrain,
    which is exactly the lineage error that makes a persisted decision
    unreproducible.
    """

    models_dir = HERE / "models"
    if not models_dir.exists():
        return "meta-combo@absent"
    entries = sorted(
        (p.name, p.stat().st_size) for p in models_dir.rglob("*") if p.is_file()
    )
    return "meta-combo@" + _short_hash(entries)


def feature_version(matrix_path: str | Path) -> str:
    return "meta-matrix@" + Path(matrix_path).name


# The scoring context carried onto a governed intent. Matches what
# nervous_system_adapter persists, so an entry and an exit describe the same
# three things. s_combo is already the combo rank-percentile; the ordinal rank
# stays in the signal audit trail where it has always lived.
INTENT_SCORE_FIELDS = ("s_combo", "s_upside", "s_quality")


def scores_by_ticker(result: MetaRankingResult) -> dict[str, dict[str, float]]:
    """Per-ticker scoring context for every scored name on the decision bar.

    This reads the full scored frame rather than the eligible top-K, so a held
    name that just dropped out of the ranking can still have its exit
    explained instead of recorded blind.
    """

    out: dict[str, dict[str, float]] = {}
    frame = result.scored
    if frame is None or frame.empty:
        return out
    for _idx, row in frame.iterrows():
        ticker = str(row["ticker"]).upper()
        entry: dict[str, float] = {}
        for name in INTENT_SCORE_FIELDS:
            if name not in frame.columns:
                continue
            try:
                numeric = float(row.get(name))
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                entry[name] = numeric
        if "s_combo" in entry:
            out[ticker] = entry
    return out


def intent_config(args, *, held_tickers: frozenset[str] = frozenset()) -> MetaIntentConfig:
    """Versioned metadata for every intent this pass produces."""

    families = (
        (InstrumentFamily.SINGLE_OPTION, InstrumentFamily.EQUITY)
        if args.mode == "options"
        else (InstrumentFamily.EQUITY,)
    )
    return MetaIntentConfig(
        quality_floor=args.quality_floor,
        held_tickers=held_tickers,
        requested_notional=Decimal(str(args.target_notional)),
        model_version=model_version(),
        feature_version=feature_version(args.matrix),
        config_version=runner_config_version(args),
        instrument_preferences=families,
        expected_holding_period=f"{args.horizon_bars}x4h",
    )


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


def _ref_price(ticker: str, *, decision_bar: datetime | pd.Timestamp) -> float:
    """Read one exact completed-bar close for compatibility callers.

    This is deliberately strict: an unscoped or stale last print is not a
    reference price for a decision.
    """

    selected_bar = _decision_timestamp(decision_bar, field_name="decision_bar")
    p = BARS_4H / f"{ticker}.parquet"
    if not p.exists():
        raise ValueError(f"no reference-bar Parquet exists for {ticker}")
    b = pd.read_parquet(p)
    if "timestamp" not in b.columns:
        if b.index.name == "timestamp":
            b = b.reset_index()
        else:
            raise ValueError(f"{ticker} reference Parquet has no timestamp column")
    if "close" not in b.columns:
        raise ValueError(f"{ticker} reference Parquet has no close column")
    timestamps = _normalize_timestamp_series(b["timestamp"], field_name=f"{ticker} reference-bar timestamps")
    matches = b.loc[timestamps == selected_bar]
    if len(matches) == 0:
        raise ValueError(f"{ticker} reference Parquet has no exact decision-bar match")
    if len(matches) != 1:
        raise ValueError(f"{ticker} reference Parquet has duplicate exact decision-bar matches")
    try:
        close = float(matches.iloc[0]["close"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ticker} exact decision-bar close is not finite") from exc
    if not math.isfinite(close):
        raise ValueError(f"{ticker} exact decision-bar close is not finite")
    return close


def _latest_full_bar(df: pd.DataFrame) -> pd.Timestamp:
    counts = df.groupby("timestamp").size()
    full = counts[counts >= max(MIN_FULL_BAR, int(0.25 * counts.max()))].index
    return full.max()


def validate_latest_full_bar(df: pd.DataFrame) -> pd.Timestamp:
    """Return the latest full bar and fail closed if the data is from the future."""

    bar = _latest_full_bar(df)
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("UTC")
    if bar > now:
        raise ValueError(f"latest full bar {bar.isoformat()} is in the future")
    return bar


def _signal_audits(top: pd.DataFrame, *, bar: pd.Timestamp, entry_ok: dict[str, bool]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    total = max(1, len(top))
    for rank, (_idx, row) in enumerate(top.iterrows(), start=1):
        ticker = str(row["ticker"]).upper()
        out[ticker] = build_signal_audit(
            module=AUDIT_MODULE,
            ticker=ticker,
            score=row.get("s_combo"),
            side="long",
            rank=rank,
            rank_pct=1.0 - ((rank - 1) / total),
            signal_ts=bar,
            extra={
                "s_quality": row.get("s_quality"),
                "s_upside": row.get("s_upside"),
                "mom_score": row.get("mom_score"),
                "htf_score": row.get("htf_score"),
                "news_catalyst_score": row.get("news_catalyst_score"),
                "dollar_vol_pctile_252": row.get("dollar_vol_pctile_252"),
                "quality_entry_ok": bool(entry_ok.get(ticker, False)),
            },
        )
    return out


def _option_dte(expiry, bar) -> int | None:
    try:
        exp = pd.Timestamp(expiry).date()
        base = pd.Timestamp(bar).date()
        return max(0, int((exp - base).days))
    except Exception:
        return None


def _exit_action(gain, runs_held, bars_out, trimmed, args) -> tuple[str, str]:
    """Hold-based exit + scale-out (delegates to the shared 4H engine)."""
    policy = ExecPolicy(take_profit=args.take_profit, scale_frac=args.scale_frac,
                        horizon_bars=args.horizon_bars, grace_bars=args.grace_bars,
                        stop_loss=args.stop_loss, trail_stop=args.trail_stop)
    return _shared_exit_action(gain, runs_held, bars_out, trimmed, policy)


def main() -> int:
    ap = argparse.ArgumentParser(description="Meta Ranker long-only equity live runner (paper by default).")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--mode", choices=["equity", "options"], default="equity",
                    help="equity = buy shares; options = buy delta-filtered calls on qualifying names.")
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
    # --- sizing ---
    ap.add_argument("--target-notional", type=float, default=5000.0,
                    help="Dollar size per new entry; shares/contracts are computed from the "
                         "current price/premium so exposure is comparable across tickers.")
    # --- exit policy (hold-based; rebalance-only churns — see backtest_exits.py) ---
    ap.add_argument("--take-profit", type=float, default=0.30, help="Scale out scale_frac at this gain, then ride the rest.")
    ap.add_argument("--scale-frac", type=float, default=0.16, help="Fraction to sell at take-profit.")
    ap.add_argument("--horizon-bars", type=int, default=53, help="Full exit after this many managed bars (~21d).")
    ap.add_argument("--grace-bars", type=int, default=None, help="Rank drop-out backstop: exit after N bars out of top-K. Default None = ride to horizon (backtest-preferred).")
    ap.add_argument("--stop-loss", type=float, default=0.39, help="Hard stop: full exit if gain <= -this from entry (premium for options). 0 disables.")
    ap.add_argument("--trail-stop", type=float, default=None, help="Trailing stop: full exit if value gives back this fraction from its peak. Default None = disabled (2026-07-18 cross-module search: no-trail beat trail on mean return per trade).")
    # Refuse to trade on stale data: if the latest full bar is older than this,
    # abort. Tight (half a day) so a missed nightly refresh can't trade yesterday.
    ap.add_argument("--max-staleness-days", type=float, default=0.5)
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--signal-audit-log", default=str(DEFAULT_AUDIT_LOG),
                    help="Append-only JSONL path for signal/order audit events; set empty to disable.")
    # Kept only so an old invocation fails loudly instead of being ignored.
    ap.add_argument("--live", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--submit", action="store_true", help="Actually place orders (default: dry-run).")
    ap.add_argument("--flush-pending-open", action="store_true", help="Pre-open: submit after-close queued entries still in the top-K (re-rank), then exit.")
    args = ap.parse_args()

    if args.live:
        # The governed Meta path is paper-only. Accepting this flag would
        # route real money around every environment check the nervous system
        # performs, so it fails loudly rather than being silently ignored.
        print(
            "ERROR: --live is not accepted by the Meta Ranker runner; "
            "live trading is not enabled in this MVP.",
            file=sys.stderr,
        )
        return 2

    profile = "PAPER"
    env_file = f".env#{profile}"
    run_mode = "SUBMIT" if args.submit else "DRY-RUN"
    print(f"=== Meta Ranker {args.mode} runner | account={profile} | mode={run_mode} | top_k={args.top_k} ===")

    # --- score + select ---
    df = pd.read_parquet(args.matrix).reset_index()
    df["timestamp"] = _normalize_timestamp_series(df["timestamp"], field_name="feature-matrix timestamps")
    bar = validate_latest_full_bar(df)
    staleness = (datetime.now(timezone.utc) - bar.to_pydatetime()).total_seconds() / 86400.0
    print(f"latest full bar: {bar}  ({staleness:.1f} days old)")
    if staleness > args.max_staleness_days and not args.allow_stale:
        raise SystemExit(
            f"ABORT: matrix is {staleness:.1f}d stale (> {args.max_staleness_days}d). "
            f"Rebuild via build_meta_ranker_matrix.py or pass --allow-stale."
        )

    # Eligibility filters (validated to generalize out-of-sample; see analyze_policy notes):
    #   liquidity floor + combo confidence floor + optional manual hard-exclusions.
    blacklist = _load_blacklist()
    ranking_config = MetaRankingConfig(
        top_k=args.top_k,
        liquidity_floor=args.liquidity_floor,
        combo_floor=args.combo_floor,
        blacklist=frozenset(blacklist),
    )
    ranking_result = rank_meta_candidates_with_diagnostics(df, bar=bar, config=ranking_config)
    # Carried to the governed path: an intent cannot be built without the
    # scores that justify it, and an entry cannot be sized without an exact
    # decision-bar price. Passed explicitly rather than stashed on `args`.
    meta_scores = scores_by_ticker(ranking_result)
    reference_prices: dict[str, float] = {}
    top = ranking_result.ranked
    n0 = ranking_result.scored_count
    eligible_count = ranking_result.eligible_count
    targets = list(top["ticker"])
    # Quality gate is applied to NEW ENTRIES only (held names exit via horizon/grace,
    # not a quality dip). A combo top-K name is bought only if its s_quality clears
    # the floor — this is the backtested "cross-in + quality" entry rule.
    quality_by_ticker = top.set_index("ticker")["s_quality"].to_dict()
    entry_ok = {t: float(quality_by_ticker.get(t, float("-inf"))) >= args.quality_floor for t in targets}
    signal_audits = _signal_audits(top, bar=bar, entry_ok=entry_ok)
    print(f"\neligible {eligible_count}/{n0} after filters "
          f"(combo>={args.combo_floor}, liq>={args.liquidity_floor}, blacklist={len(blacklist)})")
    print(f"combo top-{args.top_k}: {targets}")
    if signal_audits:
        print("signal audit buckets: " + ", ".join(
            f"{t}:{signal_audits[t].get('score_bucket')}" for t in targets if t in signal_audits
        ))
    audit_log = Path(args.signal_audit_log) if str(args.signal_audit_log or "").strip() else None
    append_jsonl(
        audit_log,
        {
            "event": "signal_decision",
            "module": AUDIT_MODULE,
            "bar": bar,
            "targets": targets,
            "signal_audits": signal_audits,
            "filters": {
                "combo_floor": args.combo_floor,
                "liquidity_floor": args.liquidity_floor,
                "quality_floor": args.quality_floor,
                "blacklist_count": len(blacklist),
            },
        },
    )
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
    print(f"equity=${equity:,.0f}  target_notional/name=${args.target_notional:,.0f}  TP +{int(args.take_profit*100)}% "
          f"(sell {int(args.scale_frac*100)}%)  horizon {args.horizon_bars}b  grace {args.grace_bars}b")
    print(f"managed held: {sorted(managed)}")

    # Publish this bar's TICKER states BEFORE anything can submit. `policy.rule.
    # snapshot` requires one per name and `policy.rule.liquidity` reads its
    # dollar_volume_20d metric, so without this every governed submission is
    # refused for want of state — which is what happened on 2026-08-20/21.
    #
    # Held names are published alongside today's targets on purpose: the
    # liquidity and snapshot rules gate EXITS too (broker_vetoes carries
    # applies_to_risk_reducing=True), so publishing only the top-K would leave
    # every exit vetoed while entries went through.
    if args.submit:
        publishable = sorted(set(targets) | {str(t).upper() for t in managed})
        pub = publish_ticker_states(
            ranking_result.scored,
            tickers=publishable,
            decision_bar=bar.to_pydatetime() if hasattr(bar, "to_pydatetime") else bar,
            matrix_path=Path(args.matrix),
        )
        print(f"ticker states: {pub['status']} published={pub['published']}/{len(publishable)}")
        skipped_states = pub.get("skipped") or {}
        if skipped_states:
            # Named, not counted: a name with no state will be refused by the
            # governed path, and "which names" is the first question that asks.
            preview = sorted(skipped_states)[:10]
            print(f"  no ticker state for {len(skipped_states)}: {preview}"
                  f"{' ...' if len(skipped_states) > len(preview) else ''}")

    # Pre-open flush: submit after-close-queued entries still in TODAY's top-K
    # (re-rank against the freshly-scored `targets`), then exit — no position mgmt.
    if getattr(args, "flush_pending_open", False):
        if args.submit:
            # One governed submitter for both flushes: the pre-open pass must
            # not reach the broker any more directly than the 4H pass does.
            # The scored frame goes with it — a queued entry still has to
            # explain itself, and this pass has already re-scored every name.
            submitter = governed_submitter(args, bar=bar, scores_by_ticker=meta_scores)
            # Exits first: a queued exit is an already-made decision on a position
            # we still hold, and flushing it before entries frees the buying power
            # the queued entries are about to use.
            ex = submit_pending_exit_orders(client, AUDIT_MODULE,
                                            equity_tif_fn=equity_order_tif, pos_lookup=pos_info,
                                            managed=managed, submit_fn=submitter)
            if ex["count"] or ex["skipped"]:
                print(f"pending-exit flush: submitted {ex['count']} / skipped {len(ex['skipped'])}")
            res = submit_pending_open_entries(
                client, AUDIT_MODULE, targets,
                equity_tif_fn=equity_order_tif, pos_lookup=pos_info,
                submit_fn=submitter,
            )
            managed.update(res["submitted"])
            state["managed"] = managed
            _save_state(state)
            print(f"pending-open flush: submitted {res['count']} / skipped {len(res['skipped'])}")
        else:
            print("pending-open flush (dry-run): add --submit to place queued entries")
        return

    if args.mode == "options":
        return _run_options(args, client, targets, state, managed, pos_info, bar, entry_ok,
                            signal_audits, meta_scores=meta_scores,
                            reference_prices=reference_prices)

    # --- hold-based reconciliation (only ever SELL symbols we manage) ---
    plan: list[tuple[str, str, int, str]] = []  # (symbol, side, qty, reason)
    order_audits: dict[str, dict] = {}
    new_managed: dict[str, dict] = {}
    exit_context: dict[str, tuple[str, dict]] = {}
    dropped: dict[str, dict] = {}
    for sym, st in managed.items():
        info_present = sym in pos_info
        held = pos_info.get(sym, {}).get("qty", 0)
        if held <= 0:
            status = "confirmed_flat" if info_present else "not_found"
            logger.warning("equity reconcile: dropping %s from managed — %s", sym, status)
            dropped[sym] = {"symbol": sym, "route": "equity", "status": status}
            continue  # position gone (closed elsewhere) — drop from state
        in_tgt = sym in targets
        st["runs_held"] = st.get("runs_held", 0) + 1
        st["bars_out"] = 0 if in_tgt else st.get("bars_out", 0) + 1
        info = pos_info.get(sym, {})
        gain = (info["current"] / info["avg_entry"] - 1) if info.get("avg_entry") else None
        action, reason = _exit_action(gain, st["runs_held"], st["bars_out"], st.get("trimmed", False), args)
        if action == "exit":
            plan.append((sym, "sell", held, reason))
            order_audits[sym] = build_equity_order_audit(
                signal_audit=signal_audits.get(sym),
                symbol=sym,
                side="sell",
                qty=held,
                reason=reason,
                reference_price=info.get("current"),
            )
            exit_context[sym] = (sym, dict(st))
            continue
        if action == "trim":
            q = int(math.floor(args.scale_frac * held))
            if q >= 1:
                plan.append((sym, "sell", q, reason))
                order_audits[sym] = build_equity_order_audit(
                    signal_audit=signal_audits.get(sym),
                    symbol=sym,
                    side="sell",
                    qty=q,
                    reason=reason,
                    reference_price=info.get("current"),
                )
                st["trimmed"] = True
        new_managed[sym] = st
    # entries: new top-K names not already held, that clear the quality gate
    for t in targets:
        if t in new_managed or pos_info.get(t, {}).get("qty", 0) > 0:
            continue
        if not entry_ok.get(t, False):
            continue
        try:
            entry_price = _ref_price(t, decision_bar=bar)
        except Exception as exc:  # noqa: BLE001 - isolate one bad data lookup from the plan
            logger.warning("equity entry %s dropped: exact selected-bar price unavailable (%s)", t, exc)
            continue
        reference_prices[t] = entry_price
        qty = shares_for_notional(entry_price, args.target_notional)
        plan.append((t, "buy", qty, "entry"))
        order_audits[t] = build_equity_order_audit(
            signal_audit=signal_audits.get(t),
            symbol=t,
            side="buy",
            qty=qty,
            reason="entry",
            reference_price=entry_price,
        )
        new_managed[t] = {"qty": qty, "runs_held": 0, "bars_out": 0, "trimmed": False, "entry_bar": str(bar)}

    print(f"\n--- order plan ({len(plan)} orders) ---")
    for sym, side, qty, reason in plan:
        if side == "sell":
            px = pos_info.get(sym, {}).get("current")
        else:
            px = order_audits.get(sym, {}).get("reference_price")
        if isinstance(px, (Real, Decimal)) and not isinstance(px, bool) and math.isfinite(float(px)) and float(px) > 0:
            print(f"  {side.upper():4} {qty:>4} {sym:<6} (~${qty*float(px):,.0f} @ {float(px):.2f})  [{reason}]")
        else:
            print(f"  {side.upper():4} {qty:>4} {sym:<6} (mark unavailable)  [{reason}]")
    if not plan:
        print("  (nothing to do — positions within policy)")

    _execute(
        args, client, plan, state, new_managed, bar, targets,
        is_option=False, signal_audits=signal_audits, order_audits=order_audits,
        exit_context=exit_context, dropped=dropped, module="meta_ranker", pos_lookup=pos_info,
        meta_scores=meta_scores, reference_prices=reference_prices,
    )


def _run_options(args, client, targets, state, managed, pos_info, bar, entry_ok=None,
                 signal_audits=None, meta_scores=None, reference_prices=None):
    """Options path: shared 4H route + hold-based exit engine (mixed option/share)."""
    signal_audits = signal_audits or {}
    reference_prices = {} if reference_prices is None else reference_prices
    policy = ExecPolicy(take_profit=args.take_profit, scale_frac=args.scale_frac,
                        horizon_bars=args.horizon_bars, grace_bars=args.grace_bars,
                        stop_loss=args.stop_loss, trail_stop=args.trail_stop,
                        target_notional=args.target_notional,
                        roll_trading_days=args.roll_trading_days)
    def _safe_entry_ref_price(ticker: str) -> float | None:
        try:
            price = _ref_price(ticker, decision_bar=bar)
            reference_prices[ticker] = price
            return price
        except Exception as exc:  # noqa: BLE001 - isolate one bad data lookup from managed exits
            logger.warning("options entry %s dropped: exact selected-bar price unavailable (%s)", ticker, exc)
            return None

    res = build_mixed_plan(
        client, targets=targets, managed=managed, pos_info=pos_info, bar=bar,
        signal_audits=signal_audits, policy=policy, route_fn=route_option_or_shares,
        ref_price_fn=_safe_entry_ref_price,
        entry_ok=entry_ok, gate_reason="quality_gated",
        module="meta_ranker",
    )
    _execute(
        args, client, res.plan, state, res.new_managed, bar, targets,
        is_option=True, limits=res.limits, signal_audits=signal_audits,
        order_audits=res.order_audits, contract_selection=res.contract_selection,
        exit_context=res.exit_context, dropped=res.dropped, module="meta_ranker", pos_lookup=pos_info,
        meta_scores=meta_scores, reference_prices=reference_prices,
    )


def _execute(
    args, client, plan, state, new_managed, bar, targets, *, is_option: bool,
    limits=None, signal_audits=None, order_audits=None, contract_selection=None,
    exit_context=None, dropped=None, module="meta_ranker", pos_lookup=None,
    meta_scores=None, reference_prices=None,
):
    """Submit a market/limit order plan (paper/live) and persist managed state. Dry-run by default.

    Managed state (runs_held/bars_out/trimmed) is persisted on every --submit
    pass, not only when the plan has orders — otherwise hold/grace counters
    freeze on quiet passes and horizon exits stall indefinitely. A failed exit
    order's pre-exit state is restored into new_managed (via exit_context) so a
    position that Alpaca never actually closed stays tracked instead of being
    silently orphaned.
    """
    limits = limits or {}
    if args.submit:
        # What the module DECIDED, captured before anything prunes it. The audit
        # append at the end of this function runs on the surviving plan, so a run
        # that deferred every row logged `plan: []` — both 2026-08-20 16:20 and
        # 2026-08-21 16:20 recorded an empty plan while queueing 5-8 orders, and
        # reconstructing what was intended meant reading the pending files and
        # hoping they had not been rewritten since.
        planned = list(plan)
        planned_disposition: dict[str, str] = {}

        def _mark(before, after, label: str) -> None:
            gone = {row[0] for row in before} - {row[0] for row in after}
            for symbol in gone:
                planned_disposition[symbol] = label

        # After the close, queue entries for the next open instead of erroring on
        # them — BEFORE the readiness gate, which is re-applied at flush time by
        # submit_pending_open_entries. See core.live_4h_exec.execute_plan for why
        # the reverse order silently discarded every after-close entry.
        before = list(plan)
        plan = defer_entries_if_market_closed(module, bar, plan, new_managed, limits)
        _mark(before, plan, "deferred_entry_market_closed")
        # Exits are deferred separately and AFTER entries — see
        # core.live_4h_exec.execute_plan. Without this an after-close exit is
        # submitted into a window the broker refuses (equity opg 403 / options
        # 422) and just fails; the HTF 16:25 run lost both its exits that way on
        # 2026-08-05, and Meta's CRWV take-profit hit the same 403 on 2026-08-03.
        # new_managed/exit_context: build_mixed_plan already dropped the position
        # when it planned the exit, and only a submit FAILURE puts it back — a
        # deferred exit never reaches submission, so without this the position is
        # held at the broker and claimed by nobody, and a sibling reconcile can
        # adopt it (the 2026-08-11 VSH case; see defer_exits_if_opg_unavailable).
        before = list(plan)
        plan = defer_exits_if_opg_unavailable(module, bar, plan, limits,
                                              new_managed=new_managed, exit_context=exit_context)
        _mark(before, plan, "deferred_exit_opg_unavailable")
        # No per-ticker fallback here: this module scores meta_ranker_matrix.parquet
        # (readiness stage 5), not the shared 4H bar cache, so current bars are not
        # evidence that *this* module's input is current. Momentum and HTF build
        # their features from bars at decision time and do take the fallback.
        before = list(plan)
        plan, skipped, reason = filter_entry_orders_for_readiness(
            plan, new_managed=new_managed, per_ticker_fallback=False
        )
        _mark(before, plan, f"readiness_gate:{reason}" if skipped else "readiness_gate")
        if skipped:
            print(f"\nreadiness gate: skipped {len(skipped)} entry orders ({reason})")
        if plan:
            print("\nsubmitting through the governed path...")
            try:
                _submit_via_gateway(
                    args, plan, state, new_managed, bar, is_option=is_option,
                    limits=limits, exit_context=exit_context, module=module,
                    pos_lookup=pos_lookup, client=client,
                    contract_selection=contract_selection,
                    meta_scores=meta_scores or {},
                    reference_prices=reference_prices or {},
                )
            except GovernedPathUnavailable as exc:
                # An unreachable governed path means "cannot submit now", never
                # "lose the plan". On 2026-08-20 at 14:20 ET a refused Postgres
                # connection propagated out of the gateway as SystemExit and
                # took the process down mid-_execute, so the audit append, the
                # managed-state save and the deferral files below all never
                # ran: a 9-order plan vanished without leaving a single record
                # that it had ever existed. Queue it instead, and let the
                # pre-open flush retry against a governed path that is up.
                #
                # No direct-broker fallback: that is precisely the bypass the
                # cutover to the governed path removed.
                logger.error(
                    "%s: governed path unavailable (%s) — queueing %d order(s) "
                    "for the next flush instead of submitting",
                    module, type(exc).__name__, len(plan),
                )
                print(f"\n  GOVERNED PATH UNAVAILABLE: queueing {len(plan)} order(s) "
                      f"for the next flush — nothing was submitted")
                before = list(plan)
                plan = defer_entries_if_market_closed(
                    module, bar, plan, new_managed, limits,
                    force=True, reason="governed path unavailable",
                )
                plan = defer_exits_if_opg_unavailable(
                    module, bar, plan, limits,
                    new_managed=new_managed, exit_context=exit_context,
                )
                _mark(before, plan, "queued_governed_path_unavailable")
        state["managed"] = new_managed
        _append_order_plan_audit(args, bar, targets, plan, signal_audits, order_audits,
                                 contract_selection, dropped,
                                 planned=planned, disposition=planned_disposition)
        if plan:
            state.setdefault("history", []).append(
                {
                    "ts": _now(),
                    "bar": str(bar),
                    "mode": args.mode,
                    "targets": targets,
                    "orders": len(plan),
                    "signal_audits": signal_audits or {},
                    "order_audits": order_audits or {},
                }
            )
        _save_state(state)
        print(f"state updated -> {STATE_PATH}")
    else:
        _append_order_plan_audit(args, bar, targets, plan, signal_audits, order_audits, contract_selection, dropped)
        print("\n(dry-run: no orders submitted, state unchanged. Add --submit to execute.)")


def governed_submitter(args, *, bar, module: str = AUDIT_MODULE,
                       scores_by_ticker: Mapping[str, Mapping[str, float]] | None = None):
    """A submit_fn for the shared 4H engine that routes through the gateway.

    The shared engine is used by several modules; injecting a submitter lets
    Meta be governed without changing anyone else's execution path. If the
    governed path cannot be built this raises, so the caller records a skip
    rather than silently falling back to a direct broker call.

    ``scores_by_ticker`` is the decision bar's scored frame. Entries need it:
    the adapter refuses to open a position it cannot explain, so a submitter
    built without scores turns every queued entry into a skip. The pre-open
    flush only submits names that are still in today's top-K, so today's scores
    are the evidence the decision is actually being re-made on.
    """

    router = build_router(intent_config=intent_config(args))
    decision_bar = bar.to_pydatetime() if hasattr(bar, "to_pydatetime") else bar
    scores = dict(scores_by_ticker or {})
    quote_client: list = []  # lazily built; the flush may never need a quote

    def _entry_quote(symbol: str, ticker: str | None) -> tuple[Any, str]:
        """Observe the market for one option we are about to BUY.

        An opening option order is refused without a quote
        (``NO_OPTION_QUOTE_FOR_ENTRY``) because opening risk we cannot price is
        never acceptable. This submitter used to pass an empty quote map, so the
        refusal was unconditional: *every* option entry in the pre-open flush was
        rejected regardless of the market. On 2026-08-19 that silently discarded
        four queued Meta entries (P, CRDO, DUOT, SMTC) which had nothing wrong
        with them. Exits are untouched — they are allowed to go unquoted.
        """
        from signals.meta_context.meta_ranker.nervous_system_adapter import underlying_for
        from signals.meta_context.meta_ranker.options_exec import (
            _latest_quote,
            _option_quote,
        )

        if not quote_client:
            from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

            quote_client.append(AlpacaOptionsClient())
        observed, why = _latest_quote(quote_client[0], symbol)
        if observed is None:
            return None, why
        bid, ask, quote_at = observed
        try:
            return _option_quote(
                symbol,
                underlying=str(ticker or "").upper() or underlying_for(symbol),
                bid=bid,
                ask=ask,
                quote_at=quote_at,
                delta=None,
                open_interest=None,
                volume=None,
            ), "ok"
        except Exception as exc:  # noqa: BLE001 - an undescribable contract is a skip
            return None, f"invalid_quote({exc.__class__.__name__})"

    def _submit(*, symbol, side, qty, route, limit=None,
                reason: str | None = None, full_exit: bool = False,
                ticker: str | None = None):
        # `full_exit` is how the engine tells a close from a trim; the adapter
        # reads it off exit_context membership, so an unnamed sell would be
        # recorded as an ADJUSTMENT even when it closed the position.
        quotes_by_symbol: dict = {}
        quote_failures: dict = {}
        if str(route) == "option" and str(side).lower() == "buy":
            quote, why = _entry_quote(symbol, ticker)
            if quote is not None:
                quotes_by_symbol[symbol] = quote
            else:
                quote_failures[symbol] = why
        rows = router.route(
            [(symbol, side, qty, reason or "pending_open", route)],
            exit_context={symbol: (None, None)} if full_exit else {},
            # Queued records carry their ticker, so an OCC symbol is mapped
            # rather than having its root inferred.
            ticker_by_symbol={symbol: str(ticker).upper()} if ticker else {},
            scores_by_ticker=scores,
            decision_bar=decision_bar,
            reference_prices={},
            position_keys={symbol: f"paper:{symbol}"},
            policy_mode=PolicyMode.ENFORCE,
            submit=True,
            quotes_by_symbol=quotes_by_symbol,
            quote_failures=quote_failures,
        )
        row = rows[0]
        if row.refusal is not None or not row.submitted:
            detail = row.refusal.value if row.refusal is not None else "not submitted"
            # Name the rules that refused, so the log says why rather than only
            # that something did.
            if getattr(row, "policy_vetoes", ()):
                detail = f"{detail} ({', '.join(row.policy_vetoes)})"
            raise RuntimeError(f"governed path refused {side} {qty} {symbol}: {detail}")
        result = getattr(row.outcome, "execution_result", None)
        return {"id": getattr(result, "broker_order_id", None) or "?"}

    return _submit


def _submit_via_gateway(
    args, plan, state, new_managed, bar, *, is_option: bool, limits, exit_context,
    module: str, pos_lookup, client, contract_selection=None,
    meta_scores=None, reference_prices=None,
) -> None:
    """Submit one plan through DecisionCoordinator -> ExecutionGateway.

    The runner no longer talks to a broker. If the governed path cannot be
    built the submission stops: falling back to a direct broker call would
    reintroduce exactly the bypass this cutover removes.

    All of the existing per-order bookkeeping is preserved -- realized-PnL
    recording on sells, restoring a failed exit's pre-exit state so a position
    the broker never closed stays tracked, dropping a failed entry, and saving
    state after every order rather than at the end of the plan.
    """

    exit_context = exit_context or {}
    contract_selection = contract_selection or {}
    router = build_router(intent_config=intent_config(args))

    quotes_by_symbol = {}
    for selection in contract_selection.values():
        occ, quote = selection.get("occ"), selection.get("quote")
        if occ and quote is not None:
            quotes_by_symbol[occ] = quote

    def _record(row) -> None:
        symbol = row.symbol
        if row.refusal is not None or not row.submitted:
            detail = row.refusal.value if row.refusal is not None else "not submitted"
            if getattr(row, "policy_vetoes", ()):
                detail = f"{detail} ({', '.join(row.policy_vetoes)})"
            print(f"  REFUSED {row.side} {row.quantity} {symbol}: {detail}")
            if symbol in exit_context:
                ticker, previous = exit_context[symbol]
                new_managed[ticker] = previous
                logger.warning(
                    "_execute: exit refused for %s (%s) — restoring to managed state",
                    ticker, symbol,
                )
            else:
                drop_failed_entry(new_managed, symbol)
        else:
            result = getattr(row.outcome, "execution_result", None)
            broker_id = getattr(result, "broker_order_id", None) or "?"
            print(f"  OK {row.side} {row.quantity} {symbol}  id={broker_id}")
            if str(row.side).strip().lower() == "buy":
                # The gateway confirms the broker ACCEPTED the order, not that it
                # filled. See core.live_4h_exec.mark_entry_unconfirmed: an
                # accepted-but-unfilled entry otherwise persists as a position
                # the account does not hold.
                mark_entry_unconfirmed(new_managed, symbol, {"id": broker_id})
            if str(row.side).strip().lower() == "sell":
                entry_state = exit_context.get(symbol, (None, None))[1]
                item = (symbol, row.side, row.quantity,
                        row.intent.reason_codes[0] if row.intent.reason_codes else "exit",
                        "option" if is_option else "equity")
                record_exit_realized_pnl(
                    client, module=module, item=item,
                    resp={"id": broker_id}, entry_state=entry_state,
                    pos_lookup=pos_lookup, bar=bar,
                )
        # Saved after every order, not at the end of the plan: a sibling
        # module's broker reconcile must never find a fresh position missing
        # from this module's on-disk state (the 2026-07-23 IOT incident).
        state["managed"] = new_managed
        _save_state(state)

    router.route(
        plan,
        exit_context=exit_context,
        ticker_by_symbol={
            selection["occ"]: ticker
            for ticker, selection in contract_selection.items()
            if selection.get("occ")
        },
        scores_by_ticker=meta_scores or {},
        decision_bar=bar.to_pydatetime() if hasattr(bar, "to_pydatetime") else bar,
        reference_prices=reference_prices or {},
        position_keys={
            symbol: f"paper:{symbol}" for symbol, *_ in ((p[0],) for p in plan)
        },
        policy_mode=PolicyMode.ENFORCE,
        submit=True,
        quotes_by_symbol=quotes_by_symbol,
        quote_failures={},
        on_row=_record,
    )


def _append_order_plan_audit(args, bar, targets, plan, signal_audits, order_audits,
                             contract_selection=None, dropped=None, *,
                             planned=None, disposition=None) -> None:
    audit_log = Path(args.signal_audit_log) if str(args.signal_audit_log or "").strip() else None
    append_jsonl(
        audit_log,
        {
            "event": "order_plan",
            "module": AUDIT_MODULE,
            "bar": bar,
            "mode": args.mode,
            "submit": bool(args.submit),
            "targets": targets,
            # What was actually submitted on this pass.
            "plan": [
                {"symbol": p[0], "side": p[1], "qty": p[2], "reason": p[3],
                 "route": p[4] if len(p) > 4 else args.mode}
                for p in plan
            ],
            # What the module DECIDED, and what became of each row. `plan` is the
            # residue after deferral and the readiness gate, so a run that
            # queued everything logged `plan: []` and the decision survived only
            # in the pending files. `submitted` here means "reached the submit
            # call", not "filled".
            "planned": [
                {"symbol": p[0], "side": p[1], "qty": p[2], "reason": p[3],
                 "route": p[4] if len(p) > 4 else args.mode,
                 "disposition": (disposition or {}).get(p[0], "submitted")}
                for p in (planned if planned is not None else plan)
            ],
            "signal_audits": signal_audits or {},
            "order_audits": order_audits or {},
            "contract_selection": contract_selection or {},
            "dropped": dropped or {},
        },
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
