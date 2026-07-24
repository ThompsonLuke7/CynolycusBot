from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterable

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.dealer_plate import evaluate_dealer_plate
from strategies.intraday_structure.detectors import (
    BreakoutContinuationDetector,
    ExhaustionDetector,
    StructuralRejectionDetector,
    TrendPullbackDetector,
    VReversalDetector,
    VwapReclaimDetector,
)
from strategies.intraday_structure.detectors.base import DetectionContext, DetectionDecision
from strategies.intraday_structure.features import compute_features
from strategies.intraday_structure.levels import StructuralLevelProvider
from strategies.intraday_structure.market import MarketContextProvider
from strategies.intraday_structure.models import (
    Bar,
    Candidate,
    Direction,
    PriceUpdate,
    SetupRecord,
    SetupState,
    SetupType,
    StateTransition,
    StructureSignal,
)
from strategies.intraday_structure.options import NullOptionsProvider, OptionsProvider
from strategies.intraday_structure.state_store import JsonStateStore, append_jsonl
from strategies.intraday_structure.target_manager import build_target_plan, evaluate_extension, manage_running_setup


logger = logging.getLogger(__name__)


class IntradayStructureEngine:
    """Persistent bar-close state machine. No broker/order methods exist here."""

    def __init__(
        self,
        config: IntradayStructureConfig,
        *,
        options_provider: OptionsProvider | None = None,
        state_store: JsonStateStore | None = None,
        transition_sink: Callable[[StateTransition], None] | None = None,
        max_history_bars: int = 780,
    ) -> None:
        if not config.paper_only:
            raise ValueError("intraday structure v1 is paper-only")
        self.config = config
        self.options_provider = options_provider or NullOptionsProvider()
        self.state_store = state_store
        self.transition_sink = transition_sink
        self.max_history_bars = max_history_bars
        self.candidates: dict[tuple[str, Direction], Candidate] = {}
        self.setups: dict[str, SetupRecord] = {}
        self.histories: dict[str, list[Bar]] = defaultdict(list)
        self.market_provider = MarketContextProvider(max_bars=max_history_bars)
        self.level_provider = StructuralLevelProvider(config.levels)
        self.transitions: list[StateTransition] = []
        self._bar_counts: dict[str, int] = defaultdict(int)
        self._detectors = {
            SetupType.V_REVERSAL: VReversalDetector(),
            SetupType.BREAKOUT: BreakoutContinuationDetector(),
            SetupType.VWAP_RECLAIM: VwapReclaimDetector(),
            SetupType.STRUCTURAL_REJECTION: StructuralRejectionDetector(),
            SetupType.TREND_PULLBACK: TrendPullbackDetector(),
        }
        self._exhaustion = ExhaustionDetector()

    def register_candidate(self, candidate: Candidate) -> bool:
        if self.config.supported_tickers and candidate.ticker not in set(self.config.supported_tickers):
            return False
        if (
            candidate.average_dollar_volume is not None
            and candidate.average_dollar_volume < self.config.min_average_dollar_volume
        ):
            return False
        key = (candidate.ticker, candidate.direction)
        current = self.candidates.get(key)
        if current and candidate.timestamp < current.timestamp:
            # Older dealer/audit context can still enrich a newer broad-universe
            # seed; never let it roll the candidate's availability time back.
            merged = replace(
                current, sources=tuple(sorted(set(current.sources) | set(candidate.sources))),
                score=max(current.score, candidate.score),
                pivot=candidate.pivot if candidate.pivot is not None else current.pivot,
                sector_etf=candidate.sector_etf or current.sector_etf,
                metadata={**current.metadata, **candidate.metadata},
            )
            changed = merged != current
            self.candidates[key] = merged
            return changed
        if current:
            merged = replace(
                candidate, sources=tuple(sorted(set(current.sources) | set(candidate.sources))),
                score=max(current.score, candidate.score),
                pivot=candidate.pivot if candidate.pivot is not None else current.pivot,
                sector_etf=candidate.sector_etf or current.sector_etf,
                metadata={**current.metadata, **candidate.metadata},
            )
            changed = merged != current
            self.candidates[key] = merged
            if changed:
                for setup in self._candidate_setups(merged):
                    if setup.state != SetupState.CLOSED:
                        setup.candidate = merged
                self.persist()
            return changed
        self.candidates[key] = candidate
        self._enforce_candidate_limit()
        for setup_type in self._detectors:
            if setup_type == SetupType.V_REVERSAL and candidate.direction != Direction.LONG:
                continue
            setup_id = self._setup_id(candidate, setup_type)
            existing = self.setups.get(setup_id)
            if existing is None or existing.state == SetupState.CLOSED and self._cooldown_complete(existing):
                available_at = candidate.available_at or candidate.timestamp
                self.setups[setup_id] = SetupRecord(
                    setup_id=setup_id, ticker=candidate.ticker, setup_type=setup_type,
                    direction=candidate.direction, candidate=candidate,
                    created_at=available_at, updated_at=available_at,
                    state_entered_at=available_at, confidence=float(max(0.0, min(1.0, candidate.score * 0.35))),
                )
                if candidate.average_dollar_volume is None:
                    self.setups[setup_id].warnings.append("average_dollar_volume_unverified")
            elif existing.state not in {SetupState.CLOSED}:
                existing.candidate = candidate
        self.persist()
        return True

    def on_bar(self, bar: Bar) -> list[StateTransition]:
        if self._is_context_symbol(bar.symbol):
            self.market_provider.update(bar)
        active_candidates = [c for (ticker, _), c in self.candidates.items() if ticker == bar.symbol]
        if not active_candidates:
            return []
        if not self._append_bar(bar):
            return []
        self._expire_candidates(bar)
        emitted_before = len(self.transitions)
        for candidate in active_candidates:
            if (candidate.ticker, candidate.direction) not in self.candidates:
                continue
            self._evaluate_candidate(candidate, bar)
        # persist() serializes the WHOLE engine state (candidates/setups/histories),
        # which grows all session; calling it on every bar (not just when
        # something actually changed) made the per-bar cost scale with total
        # accumulated state instead of O(1), and was the real driver behind the
        # shared bar queue backing up in the last ~40min of RTH (2026-07-20/21).
        # on_price_update() below already only persists on a real transition --
        # this matches that existing pattern.
        if len(self.transitions) > emitted_before:
            self.persist()
        return self.transitions[emitted_before:]

    def on_price_update(self, update: PriceUpdate) -> list[StateTransition]:
        """Use trades/quotes for faster stop/target management, never detection.

        Detector features and confirmations remain one-minute bar-close based.
        """
        if update.symbol not in {ticker for ticker, _direction in self.candidates}:
            return []
        emitted_before = len(self.transitions)
        marker = Bar(update.symbol, update.timestamp, update.price, update.price, update.price, update.price, 0.0)
        for setup in self.setups.values():
            if setup.ticker != update.symbol or setup.state not in {SetupState.RUNNING, SetupState.EXTENDED}:
                continue
            if setup.updated_at and update.timestamp < setup.updated_at:
                continue
            setup.spot = update.price
            setup.updated_at = update.timestamp
            if setup.invalidation is not None:
                invalid = update.price <= setup.invalidation if setup.direction == Direction.LONG else update.price >= setup.invalidation
                if invalid:
                    setup.metadata["exit_price"] = setup.invalidation
                    setup.metadata["exit_reason"] = f"{update.event_type}_update_invalidation"
                    self._transition(
                        setup, SetupState.INVALIDATED, marker,
                        f"{update.event_type} update crossed invalidation",
                        ("intrabar_invalidation",), phase="INVALIDATED",
                    )
                    continue
            target = setup.active_target
            reached = target is not None and (update.price >= target if setup.direction == Direction.LONG else update.price <= target)
            if reached:
                self._transition(
                    setup, SetupState.TARGET_REACHED, marker,
                    f"{update.event_type} update reached active target",
                    ("intrabar_target_reached",), phase="TARGET_REACHED",
                )
        if len(self.transitions) > emitted_before:
            self.persist()
        return self.transitions[emitted_before:]

    def active_signals(self, *, include_watching: bool = False) -> list[StructureSignal]:
        records = [s for s in self.setups.values() if s.state != SetupState.CLOSED and (include_watching or s.state != SetupState.WATCHING)]
        records.sort(key=lambda s: (s.updated_at or s.candidate.timestamp, s.confidence), reverse=True)
        return [StructureSignal.from_setup(s, self.config.version) for s in records]

    def snapshot(self) -> dict:
        return {
            "version": self.config.version,
            "candidates": [candidate.to_dict() for candidate in self.candidates.values()],
            "setups": [setup.to_dict() for setup in self.setups.values()],
            "histories": {symbol: [bar.to_dict() for bar in bars] for symbol, bars in self.histories.items()},
            "market_histories": self.market_provider.snapshot(),
            "bar_counts": dict(self._bar_counts),
        }

    def restore(self, raw: dict) -> None:
        if raw.get("version") != self.config.version:
            raise ValueError(f"state version {raw.get('version')} does not match {self.config.version}")
        self.candidates.clear()
        for item in raw.get("candidates", []):
            candidate = Candidate.from_mapping(item)
            self.candidates[(candidate.ticker, candidate.direction)] = candidate
        self.setups = {item["setup_id"]: SetupRecord.from_mapping(item) for item in raw.get("setups", [])}
        self.histories = defaultdict(list, {
            symbol: [Bar.from_mapping(row) for row in rows][-self.max_history_bars:]
            for symbol, rows in raw.get("histories", {}).items()
        })
        self.market_provider.restore(raw.get("market_histories", {}))
        self._bar_counts = defaultdict(int, {str(k): int(v) for k, v in raw.get("bar_counts", {}).items()})

    def restore_from_store(self) -> bool:
        if self.state_store is None:
            return False
        raw = self.state_store.load()
        if raw is None:
            return False
        self.restore(raw)
        return True

    def persist(self) -> None:
        if self.state_store is not None:
            self.state_store.save(self.snapshot())

    def write_active_signals(self, path: str | Path | None = None) -> None:
        output = Path(path or self.config.signal_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(json.dumps([signal.to_dict() for signal in self.active_signals()], indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        temp.replace(output)

    def _evaluate_candidate(self, candidate: Candidate, bar: Bar) -> None:
        bars = self.histories[bar.symbol]
        if bar.close < self.config.min_price:
            for setup in self._candidate_setups(candidate):
                if setup.state not in {SetupState.CLOSED, SetupState.INVALIDATED, SetupState.EXHAUSTED}:
                    self._transition(setup, SetupState.CLOSED, bar, "price below configured minimum", ("minimum_price_failed",), phase="CLOSED")
            return
        if len(bars) < self.config.detector.min_history_bars:
            return
        features = compute_features(
            bars,
            spy_bars=self.market_provider.bars("SPY", bar.timestamp),
            qqq_bars=self.market_provider.bars("QQQ", bar.timestamp),
            sector_bars=self.market_provider.bars(candidate.sector_etf or "", bar.timestamp) if candidate.sector_etf else (),
        )
        market = self.market_provider.context(candidate, bars)
        options = self.options_provider.context(bar.symbol, bar.timestamp, bar.close)
        levels = self.level_provider.levels(
            bars=bars, candidate=candidate, options=options, features=features.to_dict(),
        )
        ctx = DetectionContext(bar, bars, features, levels, market, options, self.config)
        for setup in self._candidate_setups(candidate):
            setup.bars_alive += 1
            setup.bars_in_state += 1
            setup.spot = bar.close
            setup.updated_at = bar.timestamp
            setup.market_alignment_score = market.market_alignment_score
            setup.options_context = options.to_dict()
            setup.warnings = list(dict.fromkeys([*setup.warnings, *market.warnings, *options.warnings]))
            if setup.state in {SetupState.INVALIDATED, SetupState.EXHAUSTED}:
                terminal_count = int(setup.metadata.get("terminal_bar_count", self._bar_counts[bar.symbol]))
                if self._bar_counts[bar.symbol] > terminal_count:
                    self._transition(setup, SetupState.CLOSED, bar, "terminal state archived", ("closed",), phase="CLOSED")
                continue
            if setup.state == SetupState.CLOSED:
                continue
            if setup.state == SetupState.CONFIRMED:
                setup.entry_price = bar.open
                setup.entry_time = bar.timestamp
                self._transition(setup, SetupState.RUNNING, bar, "next-bar entry delay elapsed", ("running",), phase="RUNNING")
                decision = manage_running_setup(setup, ctx)
                self._apply_decision(setup, decision, ctx)
                continue
            if setup.state == SetupState.TARGET_REACHED:
                self._apply_decision(setup, evaluate_extension(setup, ctx), ctx)
                continue
            if setup.state in {SetupState.RUNNING, SetupState.EXTENDED}:
                exhaustion = self._exhaustion.evaluate(setup, ctx)
                if exhaustion.state is not None:
                    self._apply_decision(setup, exhaustion, ctx)
                else:
                    self._apply_decision(setup, manage_running_setup(setup, ctx), ctx)
                continue
            detector = self._detectors[setup.setup_type]
            self._apply_decision(setup, detector.evaluate(setup, ctx), ctx)

    def _apply_decision(self, setup: SetupRecord, decision: DetectionDecision, ctx: DetectionContext) -> None:
        if decision.phase:
            setup.phase = decision.phase
        if decision.pivot is not None and math.isfinite(decision.pivot):
            setup.pivot = float(decision.pivot)
        if decision.invalidation is not None and math.isfinite(decision.invalidation):
            setup.invalidation = float(decision.invalidation)
        if decision.confidence is not None:
            base = 0.65 * decision.confidence + 0.35 * setup.candidate.score
            directional_market = ctx.market.market_alignment_score if setup.direction == Direction.LONG else 1.0 - ctx.market.market_alignment_score
            penalty = self.config.market_alignment_penalty * max(0.0, 0.5 - directional_market) * 2.0
            setup.confidence = float(max(0.0, min(1.0, base - penalty)))
        setup.metadata.update(decision.metadata)
        setup.evidence = list(dict.fromkeys([*setup.evidence, *decision.evidence]))[-20:]
        setup.warnings = list(dict.fromkeys([*setup.warnings, *decision.warnings]))[-20:]
        next_state = decision.state
        if next_state == SetupState.CONFIRMED:
            plan = build_target_plan(setup, ctx)
            if plan is None:
                setup.warnings = list(dict.fromkeys([*setup.warnings, "risk_or_target_plan_unavailable"]))
                return
            setup.invalidation = plan.invalidation
            setup.metadata["initial_invalidation"] = plan.invalidation
            setup.targets = list(plan.targets)
            setup.runway_score = plan.runway.runway_score
            setup.expected_reward_risk = plan.reward_risk
            setup.metadata["runway_components"] = plan.runway.components
            setup.metadata["intermediate_obstacles"] = list(plan.runway.intermediate_obstacles)
            setup.evidence = list(dict.fromkeys([*setup.evidence, *plan.runway.explanation]))[-20:]
            plate = evaluate_dealer_plate(
                direction=setup.direction, spot=ctx.bar.close, atr=ctx.features.get("atr"),
                options=ctx.options, policy=self.config.dealer_plate,
            )
            setup.metadata["dealer_plate"] = plate.to_dict()
            setup.evidence = list(dict.fromkeys([*setup.evidence, *plate.evidence]))[-20:]
            setup.warnings = list(dict.fromkeys([*setup.warnings, *plate.warnings]))[-20:]
            if plan.runway.runway_score < self.config.target.min_runway_score:
                setup.warnings = list(dict.fromkeys([*setup.warnings, "runway_below_threshold"]))
                return
            if plan.reward_risk < self.config.target.min_reward_risk:
                setup.warnings = list(dict.fromkeys([*setup.warnings, "reward_risk_below_threshold"]))
                return
        if next_state is not None and next_state != setup.state:
            if next_state == SetupState.INVALIDATED:
                setup.metadata["exit_price"] = setup.invalidation if setup.invalidation is not None else ctx.bar.close
                setup.metadata["exit_reason"] = decision.reason
            elif next_state in {SetupState.EXHAUSTED, SetupState.CLOSED}:
                setup.metadata["exit_price"] = ctx.bar.close
                setup.metadata["exit_reason"] = decision.reason
            self._transition(setup, next_state, ctx.bar, decision.reason, decision.evidence, phase=decision.phase)

    def _transition(self, setup: SetupRecord, state: SetupState, bar: Bar, reason: str, evidence: Iterable[str], *, phase: str | None = None) -> None:
        _validate_transition(setup.state, state)
        previous = setup.state
        setup.state = state
        setup.phase = phase or state.value
        setup.state_entered_at = bar.timestamp
        setup.updated_at = bar.timestamp
        setup.bars_in_state = 0
        if state in {SetupState.INVALIDATED, SetupState.EXHAUSTED}:
            setup.metadata["terminal_bar_count"] = self._bar_counts[bar.symbol]
        transition = StateTransition(
            setup_id=setup.setup_id, ticker=setup.ticker, setup_type=setup.setup_type.value,
            timestamp=bar.timestamp, from_state=previous, to_state=state,
            phase=setup.phase, spot=bar.close, reason=reason, evidence=tuple(evidence),
        )
        self.transitions.append(transition)
        if self.transition_sink:
            self.transition_sink(transition)

    def _append_bar(self, bar: Bar) -> bool:
        history = self.histories[bar.symbol]
        if history and bar.timestamp < history[-1].timestamp:
            logger.warning("ignoring out-of-order bar %s %s", bar.symbol, bar.timestamp)
            return False
        if history and bar.timestamp == history[-1].timestamp:
            history[-1] = bar
            return False
        history.append(bar)
        if len(history) > self.max_history_bars:
            del history[:-self.max_history_bars]
        self._bar_counts[bar.symbol] += 1
        return True

    def _candidate_setups(self, candidate: Candidate) -> list[SetupRecord]:
        prefix = f"{candidate.ticker}:{candidate.direction.value}:"
        return [setup for key, setup in self.setups.items() if key.startswith(prefix)]

    def _expire_candidates(self, bar: Bar) -> None:
        ttl = timedelta(minutes=self.config.candidate_ttl_minutes)
        expired = [key for key, candidate in self.candidates.items() if bar.timestamp - (candidate.available_at or candidate.timestamp) > ttl]
        for key in expired:
            candidate = self.candidates.pop(key)
            for setup in self._candidate_setups(candidate):
                if setup.state not in {SetupState.CLOSED, SetupState.INVALIDATED, SetupState.EXHAUSTED}:
                    self._transition(setup, SetupState.CLOSED, bar, "candidate expired", ("candidate_ttl_expired",), phase="CLOSED")

    def _enforce_candidate_limit(self) -> None:
        if len(self.candidates) <= self.config.candidate_limit:
            return
        ordered = sorted(self.candidates.items(), key=lambda item: (item[1].score, item[1].timestamp))
        for key, _candidate in ordered[: len(self.candidates) - self.config.candidate_limit]:
            self.candidates.pop(key, None)

    def _is_context_symbol(self, symbol: str) -> bool:
        sectors = {c.sector_etf for c in self.candidates.values() if c.sector_etf}
        return symbol in set(self.config.context_symbols) | sectors

    def _setup_id(self, candidate: Candidate, setup_type: SetupType) -> str:
        return f"{candidate.ticker}:{candidate.direction.value}:{setup_type.value}"

    def _cooldown_complete(self, setup: SetupRecord) -> bool:
        terminal = int(setup.metadata.get("terminal_bar_count", -10_000))
        return self._bar_counts[setup.ticker] - terminal >= self.config.failure_cooldown_bars


def transition_log_sink(path: str | Path) -> Callable[[StateTransition], None]:
    return lambda transition: append_jsonl(path, transition.to_dict())


def _validate_transition(current: SetupState, target: SetupState) -> None:
    allowed = {
        SetupState.WATCHING: {SetupState.SETUP_DETECTED, SetupState.ARMED, SetupState.CONFIRMED, SetupState.INVALIDATED, SetupState.CLOSED},
        SetupState.SETUP_DETECTED: {SetupState.ARMED, SetupState.CONFIRMED, SetupState.INVALIDATED, SetupState.CLOSED},
        SetupState.ARMED: {SetupState.CONFIRMED, SetupState.INVALIDATED, SetupState.CLOSED},
        SetupState.CONFIRMED: {SetupState.RUNNING, SetupState.INVALIDATED, SetupState.CLOSED},
        SetupState.RUNNING: {SetupState.TARGET_REACHED, SetupState.EXHAUSTED, SetupState.INVALIDATED, SetupState.CLOSED},
        SetupState.TARGET_REACHED: {SetupState.EXTENDED, SetupState.EXHAUSTED, SetupState.CLOSED},
        SetupState.EXTENDED: {SetupState.TARGET_REACHED, SetupState.EXHAUSTED, SetupState.INVALIDATED, SetupState.CLOSED},
        SetupState.EXHAUSTED: {SetupState.CLOSED},
        SetupState.INVALIDATED: {SetupState.CLOSED},
        SetupState.CLOSED: set(),
    }
    if target not in allowed[current]:
        raise ValueError(f"invalid setup transition {current.value} -> {target.value}")
