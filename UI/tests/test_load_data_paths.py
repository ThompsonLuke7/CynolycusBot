from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Data.load_data import resolve_intraday_parquet_path


class ResolveIntradayParquetPathTests(unittest.TestCase):
    def test_prefers_runtime_context_files_over_snapshot_intraday_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            (raw_dir / "qqq_intraday_10min.parquet").touch()
            (raw_dir / "qqq_10min_live_runtime.parquet").touch()

            with patch("Data.load_data.get_ticker_raw_dir", return_value=raw_dir):
                resolved = resolve_intraday_parquet_path("QQQ")

        self.assertEqual(resolved.name, "qqq_10min_live_runtime.parquet")

    def test_prefers_runtime_rth_cache_for_spy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            (raw_dir / "spy_intraday_1min.parquet").touch()
            (raw_dir / "spy_intraday_1min_runtime_rth_cache.parquet").touch()

            with patch("Data.load_data.get_ticker_raw_dir", return_value=raw_dir):
                resolved = resolve_intraday_parquet_path("SPY")

        self.assertEqual(resolved.name, "spy_intraday_1min_runtime_rth_cache.parquet")


if __name__ == "__main__":
    unittest.main()
