#!/usr/bin/env python
"""Assemble the pre-open trade plan: MARKET, BEST SETUPS, AVOID.

Reuses the live candidate feeds (ranker audits, dealer rankings, liquidity
universe) and the engine's own level fusion and runway scoring, then publishes
the full target ladder up front so reward:risk can be judged before the open
rather than one rung at a time after entry.

The AVOID list is the part that does not exist anywhere else today: names that
were candidates and were declined, WITH the reason. That is the artifact the
engine has been producing internally and discarding since it went live.

Read-only. It submits nothing and touches no live state.

    python -m scripts.build_premarket_plan --top 25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from strategies.intraday_structure.candidate_sources import (
    AuditCandidateFeed,
    DealerRankingCandidateFeed,
    LiquidityCandidateFeed,
)
from strategies.intraday_structure.config import DEFAULT_CONFIG_PATH, load_config
from strategies.intraday_structure.models import Candidate, Direction
from strategies.intraday_structure.options import (
    CompositeOptionsProvider,
    DealerLevelSummaryOptionsProvider,
    DealerSnapshotOptionsProvider,
)
from strategies.intraday_structure.premarket import (
    OVERNIGHT_GAP_WARNING,
    PLAN_SCHEMA_VERSION,
    PremarketPlan,
    build_trade_plan,
)


logger = logging.getLogger("premarket_plan")
ET = ZoneInfo("America/New_York")

DAILY_ROOT = Path("Data/shared/bars/1d")
HOURLY_ROOT = Path("Data/shared/bars/4h")
HOURLY_1H_ROOT = Path("Data/shared/bars/1h")
MARKET_SYMBOLS = ("SPY", "QQQ")


def _read_bars(root: Path, ticker: str, *, as_of: datetime) -> pd.DataFrame:
    path = root / f"{ticker.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if "timestamp" not in frame.columns:
        frame = frame.reset_index()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    # Nothing after the decision time, ever. A stale cache is fine; a future
    # bar is not.
    frame = frame[frame["timestamp"] <= as_of]
    return frame.sort_values("timestamp").reset_index(drop=True)


def collect_candidates(config, *, now: datetime, top: int) -> list[Candidate]:
    """Whatever the live feeds would hand the engine at this moment."""
    found: dict[tuple[str, str], Candidate] = {}

    def absorb(items):
        for candidate in items:
            key = (candidate.ticker, candidate.direction.value)
            existing = found.get(key)
            if existing is None or candidate.score > existing.score:
                merged_sources = tuple(sorted(set(candidate.sources) | set(existing.sources if existing else ())))
                found[key] = candidate.__class__(
                    ticker=candidate.ticker, timestamp=candidate.timestamp,
                    direction=candidate.direction, sources=merged_sources,
                    score=candidate.score, pivot=candidate.pivot,
                    sector_etf=candidate.sector_etf,
                    average_dollar_volume=candidate.average_dollar_volume,
                    available_at=candidate.available_at, metadata=candidate.metadata,
                )

    try:
        absorb(AuditCandidateFeed(min_poll_interval_seconds=0.0).poll())
    except Exception:
        logger.warning("ranker audit feed unavailable", exc_info=True)
    if config.dealer_plate.enabled:
        try:
            absorb(DealerRankingCandidateFeed(
                config.dealer_plate.ranking_path,
                top_structural=config.dealer_plate.candidate_top_structural,
                top_change=config.dealer_plate.candidate_top_change,
                max_age_hours=config.dealer_plate.candidate_max_age_hours,
            ).poll(now=now))
        except Exception:
            logger.warning("dealer ranking feed unavailable", exc_info=True)
    if config.liquidity_universe.enabled and len(found) < top:
        try:
            absorb(LiquidityCandidateFeed(
                config.liquidity_universe.universe_path,
                top_n=config.liquidity_universe.top_n,
            ).poll(now=now))
        except Exception:
            logger.warning("liquidity universe feed unavailable", exc_info=True)

    ordered = sorted(found.values(), key=lambda c: (-c.score, c.ticker))
    return ordered[: max(1, top)]


def build(args: argparse.Namespace) -> PremarketPlan:
    config = load_config(args.config)
    now = datetime.now(timezone.utc) if args.as_of is None else pd.Timestamp(args.as_of, tz="UTC").to_pydatetime()
    session = now.astimezone(ET).date().isoformat()

    options_provider = CompositeOptionsProvider(
        DealerLevelSummaryOptionsProvider(
            config.dealer_plate.snapshot_root, max_age_minutes=args.dealer_max_age_minutes,
        ),
        DealerSnapshotOptionsProvider(max_age_minutes=args.dealer_max_age_minutes),
    )

    candidates = collect_candidates(config, now=now, top=args.top)
    logger.info("planning %d candidates for %s", len(candidates), session)

    setups: list[dict] = []
    avoid: list[dict] = []
    for candidate in candidates:
        plan = _plan_for(candidate.ticker, candidate.direction, candidate.sources, candidate.score,
                         now=now, options_provider=options_provider, config=config)
        if plan is None:
            continue
        (setups if plan.actionable else avoid).append(plan.to_dict())

    market: list[dict] = []
    for symbol in MARKET_SYMBOLS:
        for direction in (Direction.LONG, Direction.SHORT):
            plan = _plan_for(symbol, direction, ("market_context",), 0.5,
                             now=now, options_provider=options_provider, config=config)
            if plan is not None:
                market.append(plan.to_dict())

    setups.sort(key=lambda row: (-(row["reward_risk"] or 0), -(row["runway_score"] or 0)))
    avoid.sort(key=lambda row: row["ticker"])

    return PremarketPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        generated_at=now.isoformat(),
        session=session,
        engine_version=config.version,
        market=market, setups=setups, avoid=avoid,
        warnings=[OVERNIGHT_GAP_WARNING],
        inputs={
            "daily_bars": str(DAILY_ROOT),
            "hourly_bars": str(HOURLY_1H_ROOT),
            "dealer_snapshots": config.dealer_plate.snapshot_root,
            "candidates_considered": len(candidates),
            "note": (
                "Levels come from stored daily/hourly bars and the PRIOR session's "
                "dealer snapshot. Nothing here reads a bar later than generated_at."
            ),
        },
    )


def _plan_for(ticker, direction, sources, score, *, now, options_provider, config):
    daily = _read_bars(DAILY_ROOT, ticker, as_of=now)
    if daily.empty:
        return None
    hourly = _read_bars(HOURLY_1H_ROOT, ticker, as_of=now)
    spot = float(daily.iloc[-1]["close"])
    try:
        options = options_provider.context(ticker, now, spot)
    except Exception:
        options = None
    return build_trade_plan(
        ticker=ticker, direction=direction, sources=sources, score=score,
        daily=daily, hourly=hourly, options=options, config=config,
        reference_as_of=str(daily.iloc[-1]["timestamp"]),
    )


def render(plan: PremarketPlan) -> str:
    out = [
        "=" * 78,
        f"PRE-OPEN PLAN — {plan.session}   (generated {plan.generated_at})",
        "=" * 78,
        "",
        "MARKET",
    ]
    for row in plan.market:
        if not row["actionable"]:
            continue
        arrow = ">" if row["direction"] == "long" else "<"
        ladder = " / ".join(f"{x:.2f}" for x in row["targets"])
        out.append(f"  {row['ticker']} {row['direction']:<5} {arrow} {row['trigger']:.2f}"
                   f"  ->  {ladder}   (stop {row['invalidation']:.2f}, R:R {row['reward_risk']:.2f})")
    if not any(r["actionable"] for r in plan.market):
        out.append("  no actionable market plan (see AVOID)")

    out += ["", f"BEST SETUPS ({len(plan.setups)})"]
    for row in plan.setups[:20]:
        arrow = ">" if row["direction"] == "long" else "<"
        ladder = " / ".join(f"{x:.2f}" for x in row["targets"])
        out.append(f"  {row['ticker']:<7} {row['direction']:<5} {arrow} {row['trigger']:.2f}  ->  {ladder}")
        out.append(f"          stop {row['invalidation']:.2f}  R:R {row['reward_risk']:.2f}"
                   f"  runway {row['runway_score']:.2f}  {row['context_regime']}")
        out.append(f"          trigger backed by: {', '.join(row['trigger_level_sources'])}")
    if not plan.setups:
        out.append("  none — every candidate was declined")

    out += ["", f"AVOID ({len(plan.avoid)}) — candidate, but declined"]
    reasons: dict[str, list[str]] = {}
    for row in plan.avoid:
        reasons.setdefault(row["no_trade_reason"], []).append(f"{row['ticker']}({row['direction'][0]})")
    for reason, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        out.append(f"  {reason} ({len(names)}):")
        for i in range(0, len(names), 10):
            out.append("      " + " ".join(names[i:i + 10]))

    out += ["", "WARNINGS"]
    out += [f"  {w}" for w in plan.warnings]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--top", type=int, default=25, help="Maximum candidates to plan.")
    parser.add_argument("--as-of", default=None, help="UTC timestamp to plan as of (for replay/testing).")
    parser.add_argument("--dealer-max-age-minutes", type=int, default=48 * 60)
    parser.add_argument("--output", default="Data/inference/intraday_structure/premarket_plan.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    plan = build(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(output)
    if not args.quiet:
        print(render(plan))
    logger.info("wrote %s (%d setups, %d avoid)", output, len(plan.setups), len(plan.avoid))
    return 0


if __name__ == "__main__":
    sys.exit(main())
