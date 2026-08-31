from __future__ import annotations

import logging
import queue
import threading
from dataclasses import asdict
from zoneinfo import ZoneInfo

from strategies.intraday_structure.candidate_sources import (
    AuditCandidateFeed,
    CatalystCandidateFeed,
    DealerRankingCandidateFeed,
    LiquidityCandidateFeed,
    ManualCandidateFeed,
    OpeningMomentumCandidateFeed,
)
from strategies.intraday_structure.bar_archive import BarArchive
from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.engine import IntradayStructureEngine, transition_log_sink
from strategies.intraday_structure.evidence import (
    CandidateOutcomeTracker,
    DecisionEventLedger,
    composite_sink,
)
from strategies.intraday_structure.ledger import ledger_sink
from strategies.intraday_structure.regime import abstention_sink
from strategies.intraday_structure.models import Bar, utc_datetime
from strategies.intraday_structure.options import (
    CompositeOptionsProvider,
    DealerLevelSummaryOptionsProvider,
    DealerSnapshotOptionsProvider,
)
from strategies.intraday_structure.state_store import JsonStateStore


logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


class IntradayStructureRunner:
    """Queue consumer for the shared Alpaca 1-minute stream; read-only in v1."""

    def __init__(self, config: IntradayStructureConfig, bar_queue: queue.Queue) -> None:
        self.config = config
        self.bar_queue = bar_queue
        self.feed = AuditCandidateFeed()
        self.dealer_feed = DealerRankingCandidateFeed(
            config.dealer_plate.ranking_path,
            top_structural=config.dealer_plate.candidate_top_structural,
            top_change=config.dealer_plate.candidate_top_change,
            max_age_hours=config.dealer_plate.candidate_max_age_hours,
        ) if config.dealer_plate.enabled else None
        self.liquidity_feed = LiquidityCandidateFeed(
            config.liquidity_universe.universe_path,
            top_n=config.liquidity_universe.top_n,
        ) if config.liquidity_universe.enabled else None
        self.opening_feed = (
            OpeningMomentumCandidateFeed(config.opening_discovery)
            if config.opening_discovery.enabled else None
        )
        discovery_symbols = self.opening_feed.symbols() if self.opening_feed is not None else []
        self.catalyst_feed = (
            CatalystCandidateFeed(
                config.catalyst_discovery,
                allowed_symbols=discovery_symbols,
            )
            if config.catalyst_discovery.enabled else None
        )
        self.evidence_ledger = (
            DecisionEventLedger(config.evidence.event_path) if config.evidence.enabled else None
        )
        # Broker execution is opt-in and constructed here, not in the engine, so
        # a config with execution disabled builds no client at all.
        self.executor = self._build_executor(config)
        transition_sink = transition_log_sink(config.transition_log_path)
        closed_sink = ledger_sink(config.ledger_path)
        declined_sink = abstention_sink(config.abstention_path)
        if self.evidence_ledger is not None:
            transition_sink = composite_sink(transition_sink, self.evidence_ledger.transition)
            closed_sink = composite_sink(closed_sink, self.evidence_ledger.closed_setup)
            declined_sink = composite_sink(declined_sink, self.evidence_ledger.abstention)
        self.engine = IntradayStructureEngine(
            config,
            options_provider=CompositeOptionsProvider(
                DealerLevelSummaryOptionsProvider(
                    config.dealer_plate.snapshot_root,
                    max_age_minutes=config.levels.options_max_age_minutes,
                ),
                DealerSnapshotOptionsProvider(max_age_minutes=config.levels.options_max_age_minutes),
            ),
            state_store=JsonStateStore(config.state_path),
            transition_sink=transition_sink,
            ledger_sink=closed_sink,
            abstention_sink=declined_sink,
            candidate_event_sink=(
                self.evidence_ledger.candidate_event if self.evidence_ledger is not None else None
            ),
            execution_sink=self.executor,
        )
        self.outcome_tracker = (
            CandidateOutcomeTracker(
                self.evidence_ledger,
                horizons_minutes=config.evidence.outcome_horizons_minutes,
            )
            if self.evidence_ledger is not None else None
        )
        self.manual_feed = ManualCandidateFeed(config.manual_watchlist)
        self.bar_archive = BarArchive(config.bar_archive_root) if config.archive_bars else None
        self._archive_session: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _flatten_expiring(self) -> None:
        """Close any position expiring today once the cut-off passes."""
        if self.executor is None:
            return
        try:
            self.executor.maybe_flatten_expiring()
        except Exception:  # noqa: BLE001 - never take the loop down
            logger.exception("Intraday Structure expiring flatten failed")

    @staticmethod
    def _build_executor(config: IntradayStructureConfig):
        """Construct the paper option executor, or None when it is disabled.

        Returns None on ANY failure. This engine ran for weeks as a pure
        simulation and must stay able to: a missing credential or an unreachable
        broker degrades execution back to modelled fills rather than stopping
        setup detection.
        """
        policy = getattr(config, "execution", None)
        if policy is None or not policy.enabled:
            return None
        try:
            from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
            from strategies.intraday_structure.execution import IntradayOptionExecutor

            executor = IntradayOptionExecutor(AlpacaOptionsClient(), policy)
            logger.info(
                "intraday structure: paper option execution ENABLED "
                "(dte %d-%d, max %d concurrent, $%.0f notional)",
                policy.min_dte, policy.max_dte,
                policy.max_concurrent_positions, policy.target_notional,
            )
            return executor
        except Exception:  # noqa: BLE001
            logger.exception("intraday structure: execution disabled — could not build executor")
            return None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        try:
            restored = self.engine.restore_from_store()
            if restored:
                logger.info("Intraday Structure restored %d setups", len(self.engine.setups))
        except Exception as exc:
            logger.error("Intraday Structure state restore failed closed: %s", exc)
            raise
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="intraday-structure-runner")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.engine.persist()
        self.engine.write_active_signals()
        if self.bar_archive is not None:
            self.bar_archive.flush()
            if self._archive_session is not None:
                self._write_archive_manifest(self._archive_session)

    def snapshot(self) -> dict:
        # One lock for the whole read: the runner thread is ingesting bars while
        # this runs on an HTTP thread, and a torn view would report counts from
        # one instant against signals from another.
        with self.engine.read_lock():
            active = [signal.to_dict() for signal in self.engine.active_signals()]
            return {
                "running": bool(self._thread and self._thread.is_alive()),
                "paper_only": True,
                "candidate_count": len(self.engine.candidates),
                "setup_count": len(self.engine.setups),
                "active_signals": active,
                "qualified_dealer_plate_signals": [
                    signal for signal in active if bool((signal.get("dealer_plate") or {}).get("qualified"))
                ],
                "recent_transitions": [transition.to_dict() for transition in self.engine.transitions[-100:]],
                "evidence": self.outcome_tracker.snapshot() if self.outcome_tracker is not None else {
                    "enabled": False,
                },
                "bar_archive": (
                    asdict(self.bar_archive.stats()) if self.bar_archive is not None else {"enabled": False}
                ),
            }

    def _register(self, candidate, *, persist: bool = True) -> bool:
        registered = self.engine.register_candidate(candidate, persist=False)
        seeded = 0
        if (
            self.opening_feed is not None
            and (candidate.ticker, candidate.direction) in self.engine.candidates
        ):
            seeded = self.engine.seed_history(
                self.opening_feed.history(candidate.ticker, before=candidate.timestamp)
            )
        if persist and (registered or seeded):
            self.engine.persist()
        if registered and self.outcome_tracker is not None:
            self.outcome_tracker.track(candidate)
        return bool(registered or seeded)

    def _record_archive(self, payload: dict) -> None:
        if self.bar_archive is None:
            return
        try:
            session = utc_datetime(payload["timestamp"]).astimezone(_ET).date().isoformat()
        except (KeyError, TypeError, ValueError):
            self.bar_archive.record(payload)
            return
        if self._archive_session is not None and session != self._archive_session:
            previous = self._archive_session
            self.bar_archive.flush()
            self._write_archive_manifest(previous)
        self._archive_session = session
        self.bar_archive.record(payload)

    def _write_archive_manifest(self, session: str) -> None:
        if self.bar_archive is None:
            return
        try:
            self.bar_archive.write_manifest(session, extra={
                "engine_version": self.config.version,
                "candidate_limit": self.config.candidate_limit,
                "opening_discovery_top_n": self.config.opening_discovery.top_n,
                "archive_stats_at_manifest": asdict(self.bar_archive.stats()),
            })
        except Exception:
            logger.exception("Intraday Structure archive manifest failed for %s", session)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                candidate_state_changed = False
                for candidate in self.feed.poll():
                    candidate_state_changed |= self._register(candidate, persist=False)
                if self.dealer_feed is not None:
                    for candidate in self.dealer_feed.poll():
                        candidate_state_changed |= self._register(candidate, persist=False)
                if self.liquidity_feed is not None:
                    for candidate in self.liquidity_feed.poll():
                        candidate_state_changed |= self._register(candidate, persist=False)
                if self.catalyst_feed is not None:
                    for candidate in self.catalyst_feed.poll():
                        candidate_state_changed |= self._register(candidate, persist=False)
                    if self.evidence_ledger is not None:
                        for decision in self.catalyst_feed.last_decisions:
                            self.evidence_ledger.feed_decision(decision)
                for candidate in self.manual_feed.poll():
                    candidate_state_changed |= self._register(candidate, persist=False)
                if candidate_state_changed:
                    self.engine.persist()
                # Runs on the CLOCK, not on a setup event. A same-day contract
                # whose setup never reaches a terminal state would otherwise
                # ride into expiry and be assigned — the one hazard 0DTE adds.
                # Also runs on the queue-empty path below: a quiet tape must not
                # be the reason the flatten never fires.
                self._flatten_expiring()
                payload = self.bar_queue.get(timeout=1.0)
            except queue.Empty:
                self._flatten_expiring()
                continue
            except Exception as exc:
                logger.exception("Intraday Structure candidate poll failed: %s", exc)
                continue
            if payload.get("_sentinel"):
                logger.warning("Intraday Structure stream sentinel: %s", payload.get("_error"))
                continue
            if self.bar_archive is not None:
                # Archive what ARRIVED, before the engine gets a chance to reject
                # it. A bar the engine refused is still evidence about the feed.
                self._record_archive(payload)
            try:
                bar = Bar.from_mapping(payload)
                if self.opening_feed is not None:
                    for candidate in self.opening_feed.observe(bar):
                        self._register(candidate)
                if self.outcome_tracker is not None:
                    self.outcome_tracker.on_bar(bar)
                transitions = self.engine.on_bar(bar)
                if transitions:
                    self.engine.write_active_signals()
            except Exception as exc:
                logger.exception("Intraday Structure rejected bar %s: %s", payload, exc)
