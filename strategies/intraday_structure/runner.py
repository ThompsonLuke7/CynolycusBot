from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timezone

from strategies.intraday_structure.candidate_sources import (
    AuditCandidateFeed,
    DealerRankingCandidateFeed,
    LiquidityCandidateFeed,
    manual_candidates,
)
from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.engine import IntradayStructureEngine, transition_log_sink
from strategies.intraday_structure.models import Bar
from strategies.intraday_structure.options import (
    CompositeOptionsProvider,
    DealerLevelSummaryOptionsProvider,
    DealerSnapshotOptionsProvider,
)
from strategies.intraday_structure.state_store import JsonStateStore


logger = logging.getLogger(__name__)


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
            transition_sink=transition_log_sink(config.transition_log_path),
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._manual_registered = False

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

    def snapshot(self) -> dict:
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
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for candidate in self.feed.poll():
                    self.engine.register_candidate(candidate)
                if self.dealer_feed is not None:
                    for candidate in self.dealer_feed.poll():
                        self.engine.register_candidate(candidate)
                if self.liquidity_feed is not None:
                    for candidate in self.liquidity_feed.poll():
                        self.engine.register_candidate(candidate)
                if not self._manual_registered:
                    for candidate in manual_candidates(self.config.manual_watchlist, timestamp=datetime.now(timezone.utc)):
                        self.engine.register_candidate(candidate)
                    self._manual_registered = True
                payload = self.bar_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.exception("Intraday Structure candidate poll failed: %s", exc)
                continue
            if payload.get("_sentinel"):
                logger.warning("Intraday Structure stream sentinel: %s", payload.get("_error"))
                continue
            try:
                transitions = self.engine.on_bar(Bar.from_mapping(payload))
                if transitions:
                    self.engine.write_active_signals()
            except Exception as exc:
                logger.exception("Intraday Structure rejected bar %s: %s", payload, exc)
