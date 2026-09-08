from __future__ import annotations

import queue
import unittest

from UI.shared_stream import SharedBarStream


class SharedStreamTests(unittest.TestCase):
    def test_shared_stream_fanout_and_unregister(self) -> None:
        stream = SharedBarStream()
        left: queue.Queue = queue.Queue(maxsize=4)
        right: queue.Queue = queue.Queue(maxsize=4)

        stream.register(left, name="left")
        stream.register(right, name="right")
        stream._fanout_bar({"symbol": "SPY", "close": 1.0})

        self.assertEqual(left.get_nowait()["symbol"], "SPY")
        self.assertEqual(right.get_nowait()["symbol"], "SPY")
        self.assertEqual(stream.snapshot()["registered_queues"], 2)

        stream.unregister(right)
        stream._fanout_bar({"symbol": "QQQ", "close": 2.0})

        self.assertEqual(left.get_nowait()["symbol"], "QQQ")
        self.assertTrue(right.empty())
        self.assertEqual(stream.snapshot()["registered_queues"], 1)


    def test_shared_stream_records_drops_for_full_queue(self) -> None:
        stream = SharedBarStream()
        q: queue.Queue = queue.Queue(maxsize=1)
        stream.register(q, name="tiny")

        stream._fanout_bar({"symbol": "SPY"})
        stream._fanout_bar({"symbol": "QQQ"})

        stats = stream.snapshot()
        self.assertEqual(q.get_nowait()["symbol"], "QQQ")
        self.assertEqual(stats["delivered_count"], 2)
        self.assertEqual(stats["dropped_count"], 1)

    def test_shared_stream_filters_symbols_per_subscriber(self) -> None:
        stream = SharedBarStream()
        spy_only: queue.Queue = queue.Queue(maxsize=4)
        all_symbols: queue.Queue = queue.Queue(maxsize=4)
        stream.register(spy_only, name="spy-only", symbols=("SPY",))
        stream.register(all_symbols, name="all")

        stream._fanout_bar({"symbol": "QQQ", "close": 1.0})
        stream._fanout_bar({"symbol": "SPY", "close": 2.0})

        self.assertEqual(spy_only.get_nowait()["symbol"], "SPY")
        self.assertEqual(all_symbols.get_nowait()["symbol"], "QQQ")
        self.assertEqual(all_symbols.get_nowait()["symbol"], "SPY")
        self.assertEqual(stream.snapshot()["queue_symbols"]["spy-only"], ["SPY"])


if __name__ == "__main__":
    unittest.main()


class WatchdogLivenessTests(unittest.TestCase):
    """A dead stream thread must be revived at any hour, not only inside RTH.

    On 2026-09-03 the Alpaca streamer thread exited at 07:05:44 ET on
    `connection limit exceeded`. The watchdog's only trigger was bar staleness
    inside its 09:35-16:05 window, so nothing reconnected until 09:35:30 —
    two and a half hours later and five minutes into the session.
    """

    def _stream_with_dead_thread(self):
        stream = SharedBarStream()

        class _DeadStreamer:
            thread_error = "Alpaca WebSocket fatal error: connection limit exceeded"

            @staticmethod
            def is_alive() -> bool:
                return False

        stream._started = True
        stream._streamer = _DeadStreamer()
        stream._symbols = ["SPY"]
        return stream

    def test_a_dead_thread_is_revived_outside_the_rth_window(self) -> None:
        stream = self._stream_with_dead_thread()
        reasons: list[str] = []
        stream._reconnect = lambda *, reason: reasons.append(reason)
        stream._is_rth_watch_window = staticmethod(lambda: False)

        self._drive_once(stream)

        self.assertTrue(reasons, "a dead thread outside RTH was not reconnected")
        self.assertIn("stream thread not alive", reasons[0])
        self.assertIn("connection limit exceeded", reasons[0])

    def test_reviving_a_dead_thread_backs_off(self) -> None:
        """Reconnecting every 30s would keep re-triggering the connection limit."""

        stream = self._stream_with_dead_thread()
        reasons: list[str] = []
        stream._reconnect = lambda *, reason: reasons.append(reason)
        stream._is_rth_watch_window = staticmethod(lambda: False)

        self._drive_once(stream)
        self._drive_once(stream)

        self.assertEqual(len(reasons), 1, "second immediate attempt was not backed off")

    @staticmethod
    def _drive_once(stream) -> None:
        """Run exactly one watchdog iteration without the 30s wait."""

        original_wait = stream._stop_watchdog.wait
        calls = {"n": 0}

        def _wait(_timeout):
            calls["n"] += 1
            return calls["n"] > 1  # False first (run body), True second (exit)

        stream._stop_watchdog.wait = _wait
        try:
            stream._watchdog_loop()
        finally:
            stream._stop_watchdog.wait = original_wait
