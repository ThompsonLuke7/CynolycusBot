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
        self.assertEqual(stats["delivered_count"], 1)
        self.assertEqual(stats["dropped_count"], 1)


if __name__ == "__main__":
    unittest.main()
