from __future__ import annotations

import functools
import json
import logging
import math
import threading
from collections import defaultdict, namedtuple
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

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
from strategies.intraday_structure.ledger import (
    LEDGERED_FLAG,
    ClosedSetupRecord,
    build_closed_setup_record,
)
from strategies.intraday_structure.levels import StructuralLevelProvider
from strategies.intraday_structure.market import MarketContextProvider
from strategies.intraday_structure.models import (
    TERMINAL_STATES,
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
from strategies.intraday_structure.regime import (
    AbstentionRecord,
    build_abstention_record,
    classify_context,
    regime_conflicts,
)
from strategies.intraday_structure.regime import RegimeAssessment
from strategies.intraday_structure.state_store import JsonStateStore, append_jsonl
from strategies.intraday_structure.target_manager import TargetPlan, build_target_plan, evaluate_extension, manage_running_setup


logger = logging.getLogger(__name__)

#: Price observed on a setup's own tape, and when. Distinct from the arriving
#: bar, which may belong to a different ticker entirely.
Observed = namedtuple("Observed", ("price", "as_of"))


def _synchronized(method):
    """Serialize a public engine entry point on the engine's own lock.

    WHY: the runner ingests bars on its own thread while HTTP threads read
    ``snapshot()``/``active_signals()`` and POST manual candidates. Unlocked,
    ``snapshot()`` iterated ``self.setups`` while the runner inserted into it —
    ``RuntimeError: dictionary changed size during iteration``, which reproduced
    about one run in three and killed whichever thread hit it (the runner's own
    ``persist()`` included, i.e. state loss, not just a failed page load).

    Re-entrant because the mutating paths call ``persist()`` -> ``snapshot()``
    on their way out, and ``write_active_signals()`` calls ``active_signals()``.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class IntradayStructureEngine:
    """Persistent bar-close state machine. No broker/order methods exist here."""

    def __init__(
        self,
        config: IntradayStructureConfig,
        *,
        options_provider: OptionsProvider | None = None,
        state_store: JsonStateStore | None = None,
        transition_sink: Callable[[StateTransition], None] | None = None,
        ledger_sink: Callable[[ClosedSetupRecord], None] | None = None,
        execution_sink: Any | None = None,
        abstention_sink: Callable[[AbstentionRecord], None] | None = None,
        candidate_event_sink: Callable[[dict], None] | None = None,
        max_history_bars: int = 780,
    ) -> None:
        if not config.paper_only:
            raise ValueError("intraday structure v1 is paper-only")
        # Guards every mutable attribute below. Public methods take it; the
        # private helpers they call must not, so they can nest freely.
        self._lock = threading.RLock()
        self.config = config
        self.options_provider = options_provider or NullOptionsProvider()
        self.state_store = state_store
        self.transition_sink = transition_sink
        self.ledger_sink = ledger_sink
        # Optional broker execution. A sink, like every other output here, so
        # the engine keeps no broker imports and execution stays removable.
        # Anything with on_entry/on_exit works; failures are the sink's problem.
        self.execution_sink = execution_sink
        self.abstention_sink = abstention_sink
        self.candidate_event_sink = candidate_event_sink
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

    @_synchronized
    def register_candidate(self, candidate: Candidate, *, persist: bool = True) -> bool:
        if self.config.supported_tickers and candidate.ticker not in set(self.config.supported_tickers):
            self._emit_candidate_event("candidate_rejected", candidate, reason="unsupported_ticker")
            return False
        if (
            candidate.average_dollar_volume is not None
            and candidate.average_dollar_volume < self.config.min_average_dollar_volume
        ):
            self._emit_candidate_event("candidate_rejected", candidate, reason="below_minimum_liquidity")
            return False
        key = (candidate.ticker, candidate.direction)
        current = self.candidates.get(key)
        is_new_key = current is None
        if current is not None and candidate.timestamp < current.timestamp:
            # Older dealer/audit context can still enrich a newer broad-universe
            # seed; never let it roll the candidate's availability time back.
            active = self._merge_candidates(base=current, other=candidate)
        elif current is not None:
            active = self._merge_candidates(base=candidate, other=current)
        else:
            active = candidate
        changed = active != current
        self.candidates[key] = active
        evicted: list[Candidate] = []
        if is_new_key:
            evicted = self._enforce_candidate_limit()
            for removed in evicted:
                if (removed.ticker, removed.direction) == key:
                    continue
                self._emit_candidate_event(
                    "candidate_evicted", removed, reason="candidate_capacity",
                    retained_candidate_count=len(self.candidates),
                )
        if key not in self.candidates:
            self._emit_candidate_event(
                "candidate_rejected", active, reason="candidate_capacity",
                retained_candidate_count=len(self.candidates),
            )
            return False
        # Setup creation and revival must NOT depend on the candidate key being
        # absent. It used to: this loop sat inside the `current is None` branch,
        # so a candidate refreshed more often than its TTL (LiquidityCandidateFeed
        # re-emits the same top-ADV names every session, TTL is 1440 min) kept its
        # CLOSED setups permanently dark and made _cooldown_complete unreachable.
        # Measured 2026-08-26: 204 of 630 setups under a live candidate were
        # CLOSED and could never come back.
        revived = self._ensure_setups(active)
        if persist and (changed or revived):
            self.persist()
        self._emit_candidate_event(
            "candidate_registered" if is_new_key else (
                "candidate_refreshed" if changed or revived else "candidate_duplicate"
            ),
            active,
            reason="new" if is_new_key else ("merged_or_revived" if changed or revived else "unchanged"),
            retained_candidate_count=len(self.candidates),
            setup_created_or_revived=revived,
        )
        return changed or revived

    def _emit_candidate_event(self, event_type: str, candidate: Candidate, *, reason: str, **extra) -> None:
        if self.candidate_event_sink is None:
            return
        try:
            self.candidate_event_sink({
                "event_type": event_type,
                "reason": reason,
                "candidate": candidate.to_dict(),
                **extra,
            })
        except Exception:
            logger.exception("intraday structure candidate audit write failed for %s", candidate.candidate_id)

    @staticmethod
    def _merge_candidates(*, base: Candidate, other: Candidate) -> Candidate:
        """Union the two candidates' context onto ``base``'s identity and clocks."""
        return replace(
            base,
            sources=tuple(sorted(set(base.sources) | set(other.sources))),
            score=max(base.score, other.score),
            pivot=base.pivot if base.pivot is not None else other.pivot,
            sector_etf=base.sector_etf or other.sector_etf,
            average_dollar_volume=base.average_dollar_volume if base.average_dollar_volume is not None else other.average_dollar_volume,
            metadata={**other.metadata, **base.metadata},
        )

    def _ensure_setups(self, candidate: Candidate) -> bool:
        """Create missing setups, revive cooled-down closed ones, refresh the rest.

        Returns whether any setup was created or revived, which is the only part
        the caller needs to persist for.
        """
        created = False
        for setup_type in self._detectors:
            if setup_type == SetupType.V_REVERSAL and candidate.direction != Direction.LONG:
                continue
            setup_id = self._setup_id(candidate, setup_type)
            existing = self.setups.get(setup_id)
            if existing is None or (existing.state == SetupState.CLOSED and self._cooldown_complete(existing)):
                available_at = candidate.available_at or candidate.timestamp
                self.setups[setup_id] = SetupRecord(
                    setup_id=setup_id, ticker=candidate.ticker, setup_type=setup_type,
                    direction=candidate.direction, candidate=candidate,
                    created_at=available_at, updated_at=available_at,
                    state_entered_at=available_at, confidence=float(max(0.0, min(1.0, candidate.score * 0.35))),
                )
                if candidate.average_dollar_volume is None:
                    self.setups[setup_id].warnings.append("average_dollar_volume_unverified")
                created = True
            elif existing.state != SetupState.CLOSED:
                existing.candidate = candidate
        return created

    @_synchronized
    def seed_history(self, bars: Iterable[Bar]) -> int:
        """Warm a newly promoted candidate with bars already observed upstream.

        This path runs no detector and creates no decision. It only prevents a
        late candidate promotion from waiting another ``min_history_bars`` after
        the opportunity has already appeared.
        """
        added = 0
        for bar in sorted(bars, key=lambda value: value.timestamp):
            history = self.histories.get(bar.symbol)
            if history and bar.timestamp <= history[-1].timestamp:
                continue
            if self._is_context_symbol(bar.symbol):
                self.market_provider.update(bar)
            if self._append_bar(bar):
                added += 1
        return added

    @_synchronized
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

    @_synchronized
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

    def read_lock(self) -> threading.RLock:
        """The engine's own lock, for callers composing a multi-read view.

        ``IntradayStructureRunner.snapshot()`` mixes several engine reads into
        one payload; taking this around them keeps the counts and the signals
        describing the same instant.
        """
        return self._lock

    @_synchronized
    def active_signals(self, *, include_watching: bool = False) -> list[StructureSignal]:
        records = [s for s in self.setups.values() if s.state != SetupState.CLOSED and (include_watching or s.state != SetupState.WATCHING)]
        records.sort(key=lambda s: (s.updated_at or s.candidate.timestamp, s.confidence), reverse=True)
        return [StructureSignal.from_setup(s, self.config.version) for s in records]

    @_synchronized
    def snapshot(self) -> dict:
        return {
            "version": self.config.version,
            "candidates": [candidate.to_dict() for candidate in self.candidates.values()],
            "setups": [setup.to_dict() for setup in self.setups.values()],
            "histories": {symbol: [bar.to_dict() for bar in bars] for symbol, bars in self.histories.items()},
            "market_histories": self.market_provider.snapshot(),
            "bar_counts": dict(self._bar_counts),
        }

    @_synchronized
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

    @_synchronized
    def restore_from_store(self) -> bool:
        if self.state_store is None:
            return False
        raw = self.state_store.load()
        if raw is None:
            return False
        self.restore(raw)
        return True

    @_synchronized
    def persist(self) -> None:
        if self.state_store is not None:
            self.state_store.save(self.snapshot())

    @_synchronized
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
                self._submit_entry(setup, bar)
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
            # The context label is recorded on every confirmation decision,
            # taken or not, so a later ablation can ask whether standing down
            # in a given regime was the right call.
            regime = classify_context(
                spot=ctx.bar.close, atr=ctx.features.get("atr"),
                features=ctx.features.to_dict(), levels=ctx.levels,
                policy=self.config.regime,
            )
            setup.metadata["context_regime"] = regime.regime
            outcome = build_target_plan(setup, ctx)
            plan = outcome.plan
            if plan is None:
                self._abstain(setup, ctx, outcome.reason or "target_plan_unavailable", regime)
                return
            setup.invalidation = plan.invalidation
            setup.metadata["initial_invalidation"] = plan.invalidation
            setup.targets = list(plan.targets)
            setup.runway_score = plan.runway.runway_score
            setup.expected_reward_risk = plan.reward_risk
            setup.metadata["runway_components"] = plan.runway.components
            # The ledger needs to know WHICH level the target came from and how
            # many independent mechanisms backed it; the ablation in step D
            # slices on exactly this.
            setup.metadata["target_level_type"] = plan.runway.target_level_type
            setup.metadata["target_level_sources"] = list(plan.runway.target_level_sources)
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
                self._abstain(setup, ctx, "runway_below_threshold", regime, plan=plan)
                return
            if plan.reward_risk < self.config.target.min_reward_risk:
                self._abstain(setup, ctx, "reward_risk_below_threshold", regime, plan=plan)
                return
            # Default OFF. Recording the regime changes nothing; letting it veto
            # is a trading change, so it stays behind a switch until step D has
            # measured it.
            if self.config.regime.veto_enabled and regime_conflicts(regime.regime, setup.direction.value):
                self._abstain(setup, ctx, f"regime_conflict_{regime.regime.lower()}", regime, plan=plan)
                return
        if next_state is not None and next_state != setup.state:
            if next_state == SetupState.CONFIRMED:
                setup.metadata["confirmed_at"] = ctx.bar.timestamp.isoformat()
                setup.metadata.pop("no_trade_reason", None)
            if next_state == SetupState.INVALIDATED:
                setup.metadata["exit_price"] = setup.invalidation if setup.invalidation is not None else ctx.bar.close
                setup.metadata["exit_reason"] = decision.reason
            elif next_state in {SetupState.EXHAUSTED, SetupState.CLOSED}:
                setup.metadata["exit_price"] = ctx.bar.close
                setup.metadata["exit_reason"] = decision.reason
            self._transition(setup, next_state, ctx.bar, decision.reason, decision.evidence, phase=decision.phase)

    def _abstain(
        self, setup: SetupRecord, ctx: DetectionContext, reason: str,
        regime: RegimeAssessment, *, plan: TargetPlan | None = None,
    ) -> None:
        """Record a decision NOT to confirm, instead of dropping it on the floor.

        The setup stays in whatever state it was in and may still confirm on a
        later bar; this notes only that on THIS bar the engine looked at a setup
        its detector had cleared and declined it, and why.
        """
        setup.warnings = list(dict.fromkeys([*setup.warnings, reason]))
        setup.metadata["no_trade_reason"] = reason
        if self.abstention_sink is None:
            return
        record = build_abstention_record(
            setup, reason=reason, regime=regime, timestamp=ctx.bar.timestamp,
            spot=ctx.bar.close, atr=ctx.features.get("atr"),
            engine_version=self.config.version,
            min_runway_score=self.config.target.min_runway_score,
            min_reward_risk=self.config.target.min_reward_risk,
            runway_score=plan.runway.runway_score if plan else None,
            reward_risk=plan.reward_risk if plan else None,
            proposed_invalidation=plan.invalidation if plan else None,
            proposed_target=plan.targets[0] if plan and plan.targets else None,
        )
        try:
            self.abstention_sink(record)
        except Exception:
            logger.exception("intraday structure abstention write failed for %s", setup.setup_id)

    def _transition(
        self, setup: SetupRecord, state: SetupState, bar: Bar, reason: str,
        evidence: Iterable[str], *, phase: str | None = None,
        observed: Observed | None = None,
    ) -> None:
        """Move a setup to ``state``, log it, and ledger it if it is terminal.

        ``bar`` supplies the DECISION time.  ``observed`` supplies the price and
        when that price was seen, and defaults to the arriving bar.  The two are
        the same thing only when this ticker's own bar drove the transition; see
        ``_expire_candidates`` for the case where they are not.
        """
        _validate_transition(setup.state, state)
        price = bar.close if observed is None else observed.price
        price_as_of = bar.timestamp if observed is None else observed.as_of
        previous = setup.state
        setup.state = state
        setup.phase = phase or state.value
        setup.state_entered_at = bar.timestamp
        setup.updated_at = bar.timestamp
        setup.bars_in_state = 0
        if price is not None:
            setup.spot = price
        if state in {SetupState.INVALIDATED, SetupState.EXHAUSTED}:
            setup.metadata["terminal_bar_count"] = self._bar_counts[bar.symbol]
        transition = StateTransition(
            setup_id=setup.setup_id, ticker=setup.ticker, setup_type=setup.setup_type.value,
            timestamp=bar.timestamp, from_state=previous, to_state=state,
            phase=setup.phase, spot=price, reason=reason, evidence=tuple(evidence),
            spot_as_of=price_as_of,
        )
        self.transitions.append(transition)
        if self.transition_sink:
            self.transition_sink(transition)
        self._ledger_if_terminal(setup, state, reason)

    def _ledger_if_terminal(self, setup: SetupRecord, state: SetupState, reason: str) -> None:
        """Emit exactly one ledger row, on the setup's FIRST terminal state.

        INVALIDATED and EXHAUSTED are archived to CLOSED on a later bar, so
        keying off CLOSED alone would both double-count and lose the real exit
        reason.  The guard rides in ``metadata`` and therefore survives a
        restart, which is what stops a mid-session restore from re-emitting.
        """
        if self.ledger_sink is None or state not in TERMINAL_STATES:
            return
        if setup.metadata.get(LEDGERED_FLAG):
            return
        setup.metadata.setdefault("exit_reason", reason)
        if setup.metadata.get("exit_price") is None and setup.spot is not None:
            setup.metadata["exit_price"] = setup.spot
        record = build_closed_setup_record(
            setup, replay_policy=self.config.replay,
            engine_version=self.config.version, terminal_state=state,
        )
        # A setup that closed before its entry bar arrived has no fill to
        # record.  Flag it anyway so a later archival CLOSED does not retry.
        setup.metadata[LEDGERED_FLAG] = (setup.updated_at or setup.candidate.timestamp).isoformat()
        if record is None:
            return
        try:
            self.ledger_sink(record)
        except Exception:
            logger.exception("intraday structure ledger write failed for %s", setup.setup_id)
        self._submit_exit(setup, reason)

    def _submit_entry(self, setup: SetupRecord, bar: Bar) -> None:
        """Hand a confirmed setup to the executor, if one is attached."""
        if self.execution_sink is None:
            return
        try:
            self.execution_sink.on_entry(
                setup, spot=bar.open,
                # Risk distance stands in for ATR: it is this engine's own unit
                # of risk (entry to invalidation) and is what its R-multiples are
                # quoted in. SetupRecord has no `risk_points` field — it is
                # derived at ledger time — so it is computed here.
                atr=self._risk_distance(setup, bar),
            )
        except Exception:  # noqa: BLE001 - detection must survive a broker outage
            logger.exception("intraday structure execution entry failed for %s", setup.setup_id)

    @staticmethod
    def _risk_distance(setup: SetupRecord, bar: Bar) -> float | None:
        """Entry-to-invalidation distance, the engine's own risk unit."""
        entry = setup.entry_price if setup.entry_price is not None else bar.open
        if entry is None or setup.invalidation is None:
            return None
        gap = abs(float(entry) - float(setup.invalidation))
        return gap if gap > 0 else None

    def _submit_exit(self, setup: SetupRecord, reason: str) -> None:
        if self.execution_sink is None:
            return
        try:
            self.execution_sink.on_exit(setup, exit_reason=reason, spot=setup.spot)
        except Exception:  # noqa: BLE001
            logger.exception("intraday structure execution exit failed for %s", setup.setup_id)

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
        """Expire TTL-lapsed candidates across all tickers on the arriving bar.

        The TTL sweep is global — it has to be, or a candidate whose ticker
        stopped printing would never expire — but the arriving bar belongs to
        whichever symbol happened to print first, so its price says nothing
        about the setup being closed.  Stamping it anyway is what wrote GS
        ($1,040), APTV ($50) and FIVE ($250) all at spot=133.45 on
        2026-08-24T13:07Z.  The decision time is the arriving bar's; the price
        must come from the closing setup's own tape.
        """
        ttl = timedelta(minutes=self.config.candidate_ttl_minutes)
        expired = [key for key, candidate in self.candidates.items() if bar.timestamp - (candidate.available_at or candidate.timestamp) > ttl]
        for key in expired:
            candidate = self.candidates.pop(key)
            for setup in self._candidate_setups(candidate):
                if setup.state not in {SetupState.CLOSED, SetupState.INVALIDATED, SetupState.EXHAUSTED}:
                    self._transition(
                        setup, SetupState.CLOSED, bar, "candidate expired",
                        ("candidate_ttl_expired",), phase="CLOSED",
                        observed=self._last_observed(setup.ticker),
                    )

    def _last_observed(self, ticker: str) -> Observed:
        """Last price actually seen for ``ticker``, and when it was seen.

        ``Observed(None, None)`` when this ticker has never printed a bar here —
        an honest gap, not a price to be invented.
        """
        history = self.histories.get(ticker)
        if history:
            return Observed(history[-1].close, history[-1].timestamp)
        return Observed(None, None)

    def _enforce_candidate_limit(self) -> list[Candidate]:
        """Retain active setups and reserved event/opening capacity.

        Unused reserved slots flow back to the globally ranked pool. Manual
        candidates remain highest priority through the configured source order.
        """
        if len(self.candidates) <= self.config.candidate_limit:
            return []
        capacity = self.config.candidate_capacity
        ranked = sorted(self.candidates.items(), key=self._candidate_retention_key, reverse=True)

        protected_keys: set[tuple[str, Direction]] = set()
        for source, reserve in (
            ("opening_momentum", capacity.opening_reserve),
            ("validated_catalyst", capacity.catalyst_reserve),
        ):
            source_rows = [item for item in ranked if source in item[1].sources]
            protected_keys.update(key for key, _candidate in source_rows[: max(0, int(reserve))])
        protected = [item for item in ranked if item[0] in protected_keys]
        if len(protected) > self.config.candidate_limit:
            protected = protected[: self.config.candidate_limit]
        keep_keys = {key for key, _candidate in protected}
        for key, _candidate in ranked:
            if len(keep_keys) >= self.config.candidate_limit:
                break
            keep_keys.add(key)

        evicted: list[Candidate] = []
        for key, candidate in tuple(self.candidates.items()):
            if key in keep_keys:
                continue
            evicted.append(candidate)
            self.candidates.pop(key, None)
            # Passive setup shells have no signal or outcome to manage. Drop
            # them so capacity eviction cannot inflate dashboard counts forever.
            for setup in self._candidate_setups(candidate):
                if setup.state == SetupState.WATCHING:
                    self.setups.pop(setup.setup_id, None)
        return evicted

    def _candidate_retention_key(self, item: tuple[tuple[str, Direction], Candidate]) -> tuple:
        key, candidate = item
        live_setup = any(
            setup.state not in {
                SetupState.WATCHING, SetupState.CLOSED,
                SetupState.INVALIDATED, SetupState.EXHAUSTED,
            }
            for setup in self._candidate_setups(candidate)
        )
        priorities = {
            source: len(self.config.candidate_capacity.source_priority) - index
            for index, source in enumerate(self.config.candidate_capacity.source_priority)
        }
        source_priority = max((priorities.get(source, 0) for source in candidate.sources), default=0)
        return (
            1 if live_setup else 0,
            source_priority,
            float(candidate.score),
            candidate.timestamp,
            key[0],
            key[1].value,
        )

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
