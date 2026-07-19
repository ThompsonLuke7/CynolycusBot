from __future__ import annotations

import pytest

from core.live_job_guard import heavy_job_guard


pytestmark = pytest.mark.safe


def test_heavy_job_guard_clears_owner_metadata_after_release(tmp_path):
    lock_path = tmp_path / "heavy.lock"

    with heavy_job_guard(
        "test-owner",
        lock_path=lock_path,
        block_live_window=False,
        min_available_mb=0,
        min_swap_free_mb=0,
    ) as guard:
        assert guard.ok
        assert "test-owner" in lock_path.read_text()

    assert lock_path.read_text() == ""
