from __future__ import annotations

from datetime import datetime, timedelta, timezone
import queue
import unittest
from unittest.mock import Mock

from strategies.multi_ticker_swing.live.runner import SwingLiveRunner


class LiveRunnerBacklogTests(unittest.TestCase):
    def test_drop_stale_external_backlog_fast_forwards_to_fresh_bar(self) -> None:
        runner = SwingLiveRunner.__new__(SwingLiveRunner)
        runner._bar_queue = queue.Queue()
        runner._dropped_stale_bars = 0
        runner._last_queue_size = None
        runner._emit_backlog_event = Mock()
        runner._maybe_emit_heartbeat = Mock()
        runner._reset_ticker_accumulators = Mock()
        runner._queue_size = lambda: runner._bar_queue.qsize()
        runner._bar_is_too_stale = lambda lag_secs: lag_secs is not None and lag_secs >= 600.0

        now = datetime.now(timezone.utc)
        stale_one = {"symbol": "AAA", "timestamp": now - timedelta(minutes=20)}
        stale_two = {"symbol": "BBB", "timestamp": now - timedelta(minutes=12)}
        fresh = {"symbol": "CCC", "timestamp": now - timedelta(minutes=1)}

        runner._bar_queue.put_nowait(stale_two)
        runner._bar_queue.put_nowait(fresh)

        out = SwingLiveRunner._drop_stale_external_backlog(runner, stale_one)

        self.assertIs(out, fresh)
        self.assertEqual(runner._dropped_stale_bars, 2)
        self.assertEqual(runner._bar_queue.qsize(), 0)
        runner._reset_ticker_accumulators.assert_any_call("AAA")
        runner._reset_ticker_accumulators.assert_any_call("BBB")
        runner._emit_backlog_event.assert_called_once()
        runner._maybe_emit_heartbeat.assert_called_once_with(reason="stale_drop")


if __name__ == "__main__":
    unittest.main()
